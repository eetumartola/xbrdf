use crate::math::Vec3;
use fbx::{Node, Property};
use std::path::{Path, PathBuf};

const MIN_TILE_SIZE: f32 = 1.0e-6;
const WHITE: Vec3 = Vec3::new(1.0, 1.0, 1.0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub triangles: Vec<Triangle>,
    pub bounds: Bounds,
    pub original_bounds: Bounds,
    pub y_offset_to_zero: f32,
    pub tile_min_x: f32,
    pub tile_min_z: f32,
    pub tile_width: f32,
    pub tile_depth: f32,
    pub color_source: ColorSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSource {
    None,
    ObjVertexColor,
    ObjMaterialDiffuse,
    FbxLayerElementColor,
}

impl ColorSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ObjVertexColor => "obj_vertex_color",
            Self::ObjMaterialDiffuse => "obj_material_diffuse",
            Self::FbxLayerElementColor => "fbx_layer_element_color",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GeometryError {
    #[error("failed to load OBJ {path}: {source}")]
    LoadObj {
        path: PathBuf,
        source: tobj::LoadError,
    },
    #[error("failed to load FBX {path}: {source}")]
    LoadFbx { path: PathBuf, source: fbx::Error },
    #[error("unsupported input extension for {0}; expected .obj or .fbx")]
    UnsupportedExtension(PathBuf),
    #[error("OBJ contains no triangles")]
    EmptyMesh,
    #[error("OBJ contains a non-finite position")]
    NonFinitePosition,
    #[error("OBJ contains a degenerate triangle")]
    DegenerateTriangle,
    #[error("tile size must be non-zero in X and Z; geometry bounds are used as the tile period")]
    InvalidTileSize,
}

impl Bounds {
    pub fn from_points(points: &[Vec3]) -> Option<Self> {
        let mut iter = points.iter().copied();
        let first = iter.next()?;
        let mut bounds = Bounds {
            min: first,
            max: first,
        };

        for point in iter {
            bounds.min = bounds.min.min(point);
            bounds.max = bounds.max.max(point);
        }

        Some(bounds)
    }
}

impl Mesh {
    pub fn load(path: &Path) -> Result<Self, GeometryError> {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("obj") => Self::load_obj(path),
            Some("fbx") => Self::load_fbx(path),
            _ => Err(GeometryError::UnsupportedExtension(path.to_path_buf())),
        }
    }

    pub fn load_obj(path: &Path) -> Result<Self, GeometryError> {
        let options = tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        };
        let (models, materials) =
            tobj::load_obj(path, &options).map_err(|source| GeometryError::LoadObj {
                path: path.to_path_buf(),
                source,
            })?;
        let materials = materials.unwrap_or_default();

        let mut positions = Vec::new();
        let mut indexed_faces: Vec<IndexedFace> = Vec::new();
        let mut color_source = ColorSource::None;

        for model in models {
            let mesh = model.mesh;
            let base_index = positions.len();
            for coords in mesh.positions.chunks_exact(3) {
                let position = Vec3::new(coords[0], coords[1], coords[2]);
                if !position.is_finite() {
                    return Err(GeometryError::NonFinitePosition);
                }
                positions.push(position);
            }

            let material_color = mesh
                .material_id
                .and_then(|id| materials.get(id))
                .and_then(|material| material.diffuse)
                .map(Vec3::from_array);
            if material_color.is_some() && color_source == ColorSource::None {
                color_source = ColorSource::ObjMaterialDiffuse;
            }

            let has_vertex_colors = !mesh.vertex_color.is_empty();
            if has_vertex_colors {
                color_source = ColorSource::ObjVertexColor;
            }

            for face in mesh.indices.chunks_exact(3) {
                let vertex_indices = [
                    base_index + face[0] as usize,
                    base_index + face[1] as usize,
                    base_index + face[2] as usize,
                ];
                let color = if has_vertex_colors {
                    average_color([
                        read_obj_color(&mesh.vertex_color, face[0] as usize),
                        read_obj_color(&mesh.vertex_color, face[1] as usize),
                        read_obj_color(&mesh.vertex_color, face[2] as usize),
                    ])
                } else {
                    material_color.unwrap_or(WHITE)
                };

                indexed_faces.push(IndexedFace {
                    indices: vertex_indices,
                    color,
                });
            }
        }

        Self::from_positions_and_faces(positions, indexed_faces, color_source)
    }

    pub fn load_fbx(path: &Path) -> Result<Self, GeometryError> {
        let file = std::fs::File::open(path).map_err(|source| GeometryError::LoadFbx {
            path: path.to_path_buf(),
            source: fbx::Error::Io(source),
        })?;
        let reader = std::io::BufReader::new(file);
        let fbx = fbx::File::read_from(reader).map_err(|source| GeometryError::LoadFbx {
            path: path.to_path_buf(),
            source,
        })?;

        let mut all_positions = Vec::new();
        let mut all_faces = Vec::new();
        let mut color_source = ColorSource::None;
        for geometry in fbx.children.iter().flat_map(geometry_nodes) {
            let Some((positions, faces, source)) = parse_fbx_geometry(geometry) else {
                continue;
            };
            let base_index = all_positions.len();
            all_positions.extend(positions);
            if source != ColorSource::None {
                color_source = source;
            }
            all_faces.extend(faces.into_iter().map(|face| IndexedFace {
                indices: [
                    base_index + face.indices[0],
                    base_index + face.indices[1],
                    base_index + face.indices[2],
                ],
                color: face.color,
            }));
        }

        Self::from_positions_and_faces(all_positions, all_faces, color_source)
    }

    fn from_positions_and_faces(
        mut positions: Vec<Vec3>,
        indexed_faces: Vec<IndexedFace>,
        color_source: ColorSource,
    ) -> Result<Self, GeometryError> {
        if indexed_faces.is_empty() {
            return Err(GeometryError::EmptyMesh);
        }
        let original_bounds = Bounds::from_points(&positions).ok_or(GeometryError::EmptyMesh)?;
        let y_offset_to_zero = -original_bounds.max.y;
        for position in &mut positions {
            position.y += y_offset_to_zero;
        }

        let mut triangles = Vec::with_capacity(indexed_faces.len());
        for face in indexed_faces {
            let v0 = positions[face.indices[0]];
            let v1 = positions[face.indices[1]];
            let v2 = positions[face.indices[2]];
            let Some(normal) = (v1 - v0).cross(v2 - v0).normalize() else {
                continue;
            };

            triangles.push(Triangle {
                v0,
                v1,
                v2,
                normal,
                color: face.color,
            });
        }
        if triangles.is_empty() {
            return Err(GeometryError::EmptyMesh);
        }

        let bounds = Bounds::from_points(&positions).ok_or(GeometryError::EmptyMesh)?;
        let tile_width = bounds.max.x - bounds.min.x;
        let tile_depth = bounds.max.z - bounds.min.z;

        if !tile_width.is_finite()
            || !tile_depth.is_finite()
            || tile_width.abs() <= MIN_TILE_SIZE
            || tile_depth.abs() <= MIN_TILE_SIZE
        {
            return Err(GeometryError::InvalidTileSize);
        }

        Ok(Self {
            triangles,
            bounds,
            original_bounds,
            y_offset_to_zero,
            tile_min_x: bounds.min.x,
            tile_min_z: bounds.min.z,
            tile_width: tile_width.abs(),
            tile_depth: tile_depth.abs(),
            color_source,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct IndexedFace {
    indices: [usize; 3],
    color: Vec3,
}

fn read_obj_color(colors: &[f32], index: usize) -> Vec3 {
    let offset = index * 3;
    if offset + 2 < colors.len() {
        Vec3::new(colors[offset], colors[offset + 1], colors[offset + 2])
    } else {
        WHITE
    }
}

fn average_color(colors: [Vec3; 3]) -> Vec3 {
    (colors[0] + colors[1] + colors[2]) / 3.0
}

fn geometry_nodes<'a>(node: &'a Node) -> Vec<&'a Node> {
    let mut nodes = Vec::new();
    collect_geometry_nodes(node, &mut nodes);
    nodes
}

fn collect_geometry_nodes<'a>(node: &'a Node, nodes: &mut Vec<&'a Node>) {
    if node.name == "Geometry" {
        nodes.push(node);
    }
    for child in &node.children {
        collect_geometry_nodes(child, nodes);
    }
}

fn parse_fbx_geometry(node: &Node) -> Option<(Vec<Vec3>, Vec<IndexedFace>, ColorSource)> {
    let vertices = child_f64_array(node, "Vertices")?;
    let polygon_indices = child_i32_array(node, "PolygonVertexIndex")?;
    let positions: Vec<_> = vertices
        .chunks_exact(3)
        .map(|chunk| Vec3::new(chunk[0] as f32, chunk[1] as f32, chunk[2] as f32))
        .collect();
    let color_layer = fbx_color_layer(node);
    let color_source = if color_layer.is_some() {
        ColorSource::FbxLayerElementColor
    } else {
        ColorSource::None
    };

    let mut faces = Vec::new();
    let mut polygon = Vec::new();
    let mut polygon_vertex_start = 0usize;
    let mut polygon_index = 0usize;

    for raw_index in polygon_indices {
        let end = raw_index < 0;
        let vertex_index = if end {
            (-raw_index - 1) as usize
        } else {
            raw_index as usize
        };
        polygon.push(vertex_index);

        if end {
            if polygon.len() >= 3 {
                for local in 1..polygon.len() - 1 {
                    let color = color_layer
                        .as_ref()
                        .and_then(|layer| {
                            layer.polygon_color(
                                polygon_index,
                                [0, local, local + 1],
                                polygon_vertex_start,
                            )
                        })
                        .unwrap_or(WHITE);
                    faces.push(IndexedFace {
                        indices: [polygon[0], polygon[local], polygon[local + 1]],
                        color,
                    });
                }
            }
            polygon_vertex_start += polygon.len();
            polygon_index += 1;
            polygon.clear();
        }
    }

    Some((positions, faces, color_source))
}

#[derive(Debug, Clone)]
struct FbxColorLayer {
    mapping: String,
    reference: String,
    colors: Vec<Vec3>,
    indices: Vec<i32>,
}

impl FbxColorLayer {
    fn polygon_color(
        &self,
        polygon_index: usize,
        local_indices: [usize; 3],
        polygon_vertex_start: usize,
    ) -> Option<Vec3> {
        match self.mapping.as_str() {
            "ByPolygonVertex" => Some(average_color([
                self.color_at(polygon_vertex_start + local_indices[0])?,
                self.color_at(polygon_vertex_start + local_indices[1])?,
                self.color_at(polygon_vertex_start + local_indices[2])?,
            ])),
            "ByPolygon" => self.color_at(polygon_index),
            "AllSame" => self.color_at(0),
            _ => None,
        }
    }

    fn color_at(&self, index: usize) -> Option<Vec3> {
        let direct_index = if self.reference == "IndexToDirect" || self.reference == "Index" {
            *self.indices.get(index)? as usize
        } else {
            index
        };
        self.colors.get(direct_index).copied()
    }
}

fn fbx_color_layer(node: &Node) -> Option<FbxColorLayer> {
    let layer = node
        .children
        .iter()
        .find(|child| child.name == "LayerElementColor")?;
    let mapping = child_string(layer, "MappingInformationType")
        .unwrap_or_else(|| "ByPolygonVertex".to_string());
    let reference =
        child_string(layer, "ReferenceInformationType").unwrap_or_else(|| "Direct".to_string());
    let raw_colors =
        child_f64_array(layer, "Colors").or_else(|| child_f64_array(layer, "Color"))?;
    let colors = raw_colors
        .chunks_exact(4)
        .map(|chunk| Vec3::new(chunk[0] as f32, chunk[1] as f32, chunk[2] as f32))
        .collect();
    let indices = child_i32_array(layer, "ColorIndex")
        .or_else(|| child_i32_array(layer, "ColorsIndex"))
        .unwrap_or_default();

    Some(FbxColorLayer {
        mapping,
        reference,
        colors,
        indices,
    })
}

fn child_f64_array(node: &Node, name: &str) -> Option<Vec<f64>> {
    node.children
        .iter()
        .find(|child| child.name == name)
        .and_then(|child| match child.properties.first()? {
            Property::F64Array(values) => Some(values.clone()),
            Property::F32Array(values) => Some(values.iter().map(|value| *value as f64).collect()),
            _ => None,
        })
}

fn child_i32_array(node: &Node, name: &str) -> Option<Vec<i32>> {
    node.children
        .iter()
        .find(|child| child.name == name)
        .and_then(|child| match child.properties.first()? {
            Property::I32Array(values) => Some(values.clone()),
            _ => None,
        })
}

fn child_string(node: &Node, name: &str) -> Option<String> {
    node.children
        .iter()
        .find(|child| child.name == name)
        .and_then(|child| match child.properties.first()? {
            Property::String(value) => Some(value.clone()),
            _ => None,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_period_uses_bounds() {
        let obj = "\
v -1 0 -2
v 1 0 -2
v 1 0 2
v -1 0 2
f 1 3 2
f 1 4 3
";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plane.obj");
        std::fs::write(&path, obj).unwrap();

        let mesh = Mesh::load_obj(&path).unwrap();

        assert_eq!(mesh.triangles.len(), 2);
        assert_eq!(mesh.tile_width, 2.0);
        assert_eq!(mesh.tile_depth, 4.0);
        assert_eq!(mesh.bounds.min, Vec3::new(-1.0, 0.0, -2.0));
        assert_eq!(mesh.bounds.max.y, 0.0);
        assert_eq!(mesh.triangles[0].color, WHITE);
    }

    #[test]
    fn highest_point_is_shifted_to_zero() {
        let obj = "\
v 0 2 0
v 1 4 0
v 0 3 1
f 1 2 3
";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raised.obj");
        std::fs::write(&path, obj).unwrap();

        let mesh = Mesh::load_obj(&path).unwrap();

        assert_eq!(mesh.original_bounds.max.y, 4.0);
        assert_eq!(mesh.y_offset_to_zero, -4.0);
        assert_eq!(mesh.bounds.max.y, 0.0);
        assert_eq!(mesh.bounds.min.y, -2.0);
    }

    #[test]
    fn obj_vertex_colors_are_averaged_per_triangle() {
        let obj = "\
v 0 0 0 1 0 0
v 1 0 0 0 1 0
v 0 0 1 0 0 1
f 1 2 3
";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("colors.obj");
        std::fs::write(&path, obj).unwrap();

        let mesh = Mesh::load_obj(&path).unwrap();

        assert_eq!(mesh.color_source, ColorSource::ObjVertexColor);
        assert_eq!(
            mesh.triangles[0].color,
            Vec3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        );
    }
}
