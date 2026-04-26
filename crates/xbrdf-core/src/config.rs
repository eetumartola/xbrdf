use crate::math::Vec3;
use crate::sampling::hemisphere_latlong_direction;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

const DEFAULT_WIDTH: u32 = 256;
const DEFAULT_HEIGHT: u32 = 64;
const DEFAULT_SAMPLES: u32 = 64;
const DEFAULT_LIGHT_WIDTH: u32 = 8;
const DEFAULT_LIGHT_HEIGHT: u32 = 4;
const DEFAULT_LIGHT: [f32; 3] = [0.0, 1.0, -1.0];
const DEFAULT_COLOR: [f32; 3] = [1.0, 1.0, 1.0];
const DEFAULT_SPECULAR_ROUGHNESS: f32 = 0.05;
const DEFAULT_MAX_REPEAT_RADIUS: u32 = 2;
const MAX_REPEAT_RADIUS_LIMIT: u32 = 16;
const MIN_LIGHT_COS: f32 = 1.0e-4;
const MAX_PHONG_EXPONENT: f32 = 1_000_000.0;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BakeConfigFile {
    pub obj: Option<PathBuf>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub mode: Option<BakeMode>,
    pub light_width: Option<u32>,
    pub light_height: Option<u32>,
    pub samples: Option<u32>,
    pub tile_width: Option<f32>,
    pub tile_depth: Option<f32>,
    pub light: Option<[f32; 3]>,
    pub max_repeat_radius: Option<u32>,
    pub sampler: Option<SamplerKind>,
    pub enable_shadows: Option<bool>,
    pub material: MaterialConfigFile,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct MaterialConfigFile {
    pub kind: Option<MaterialKind>,
    pub color: Option<[f32; 3]>,
    pub roughness: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct BakeOverrides {
    pub obj: Option<PathBuf>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub mode: Option<BakeMode>,
    pub light_width: Option<u32>,
    pub light_height: Option<u32>,
    pub samples: Option<u32>,
    pub tile_width: Option<f32>,
    pub tile_depth: Option<f32>,
    pub light: Option<[f32; 3]>,
    pub max_repeat_radius: Option<u32>,
    pub sampler: Option<SamplerKind>,
    pub enable_shadows: Option<bool>,
    pub material_kind: Option<MaterialKind>,
    pub material_color: Option<[f32; 3]>,
    pub material_roughness: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedBakeConfig {
    pub obj: PathBuf,
    pub width: u32,
    pub height: u32,
    pub mode: BakeMode,
    pub light_width: u32,
    pub light_height: u32,
    pub samples: u32,
    pub tile_width_override: Option<f32>,
    pub tile_depth_override: Option<f32>,
    pub light: [f32; 3],
    pub max_repeat_radius: u32,
    pub sampler: SamplerKind,
    pub enable_shadows: bool,
    pub material: ResolvedMaterial,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MaterialKind {
    Lambertian,
    #[serde(alias = "specular")]
    SpecularPhong,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BakeMode {
    Single,
    Full,
    Isotropic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SamplerKind {
    Halton,
    Random,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResolvedMaterial {
    pub kind: MaterialKind,
    pub color: [f32; 3],
    pub roughness: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub tool: ToolInfo,
    pub input: InputInfo,
    pub output: OutputInfo,
    pub convention: ConventionInfo,
    pub bake: BakeInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputInfo {
    pub obj: PathBuf,
    pub triangle_count: usize,
    pub color_source: String,
    pub height_offset_to_zero: f32,
    pub tile_min: [f32; 2],
    pub tile_size: [f32; 2],
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub original_bounds_min: [f32; 3],
    pub original_bounds_max: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutputInfo {
    pub image: String,
    pub format: String,
    pub channels: String,
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConventionInfo {
    pub coordinate_system: String,
    pub macro_normal: [f32; 3],
    pub normal_shading: String,
    pub pano_domain: String,
    pub pano_center: [f32; 3],
    pub azimuth_positive_toward: [f32; 3],
    pub row_order: String,
    pub normalization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BakeInfo {
    pub mode: BakeMode,
    pub samples_per_direction: u32,
    pub max_repeat_radius: u32,
    pub camera_width: u32,
    pub camera_height: u32,
    pub light_width: u32,
    pub light_height: u32,
    pub light_count: u32,
    pub sampler: SamplerKind,
    pub enable_shadows: bool,
    pub light_direction_surface_to_light: [f32; 3],
    pub material: MaterialInfo,
    pub transport: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterialInfo {
    pub kind: MaterialKind,
    pub color: [f32; 3],
    pub roughness: Option<f32>,
    pub phong_exponent: Option<f32>,
    pub model: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config {path}: {source}")]
    ReadConfig {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse config {path}: {source}")]
    ParseConfig {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("missing required OBJ path; set `obj` in the config or pass --obj")]
    MissingObj,
    #[error("width and height must both be greater than zero")]
    InvalidResolution,
    #[error("samples must be greater than zero")]
    InvalidSamples,
    #[error("tile width and depth overrides must be finite and greater than zero")]
    InvalidTileOverride,
    #[error("light direction must be finite, non-zero, and above the +Y macro plane")]
    InvalidLight,
    #[error("max_repeat_radius must be in the 0..=16 range")]
    InvalidMaxRepeatRadius,
    #[error("material color must contain finite non-negative values")]
    InvalidMaterialColor,
    #[error("material roughness must be finite and in the 0..=1 range")]
    InvalidMaterialRoughness,
}

impl MaterialKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MaterialKind::Lambertian => "lambertian",
            MaterialKind::SpecularPhong => "specular_phong",
        }
    }
}

impl SamplerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SamplerKind::Halton => "halton",
            SamplerKind::Random => "random",
        }
    }
}

impl BakeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            BakeMode::Single => "single",
            BakeMode::Full => "full",
            BakeMode::Isotropic => "isotropic",
        }
    }
}

impl fmt::Display for BakeMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for BakeMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "single" | "fixed" | "fixed_light" | "fixed-light" => Ok(Self::Single),
            "full" | "anisotropic" | "4d" => Ok(Self::Full),
            "isotropic" | "iso" => Ok(Self::Isotropic),
            _ => Err(format!("expected one of: single, full, isotropic")),
        }
    }
}

impl fmt::Display for SamplerKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for SamplerKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "halton" | "qmc" | "low_discrepancy" | "low-discrepancy" => Ok(Self::Halton),
            "random" | "hashed" | "hash" => Ok(Self::Random),
            _ => Err(format!("expected one of: halton, random")),
        }
    }
}

impl fmt::Display for MaterialKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for MaterialKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "lambertian" | "diffuse" => Ok(Self::Lambertian),
            "specular" | "specular_phong" | "specular-phong" | "phong" => Ok(Self::SpecularPhong),
            _ => Err(format!(
                "expected one of: lambertian, specular_phong, specular"
            )),
        }
    }
}

impl ResolvedMaterial {
    pub fn phong_exponent(&self) -> Option<f32> {
        if self.kind != MaterialKind::SpecularPhong {
            return None;
        }

        let roughness = self
            .roughness
            .unwrap_or(DEFAULT_SPECULAR_ROUGHNESS)
            .clamp(0.0, 1.0);
        let effective_roughness = roughness.max(0.001);
        Some(
            ((2.0 / (effective_roughness * effective_roughness)) - 2.0)
                .clamp(1.0, MAX_PHONG_EXPONENT),
        )
    }

    pub fn model_description(&self) -> String {
        match self.kind {
            MaterialKind::Lambertian => "Lambertian diffuse BRDF".to_string(),
            MaterialKind::SpecularPhong => {
                "normalized Phong reflection lobe around the mirror direction".to_string()
            }
        }
    }
}

impl ResolvedBakeConfig {
    pub fn camera_tile_width(&self) -> u32 {
        match self.mode {
            BakeMode::Isotropic => 1,
            BakeMode::Single | BakeMode::Full => self.width,
        }
    }

    pub fn camera_tile_height(&self) -> u32 {
        self.height
    }

    pub fn atlas_width(&self) -> u32 {
        self.camera_tile_width() * self.effective_light_width()
    }

    pub fn atlas_height(&self) -> u32 {
        self.camera_tile_height() * self.effective_light_height()
    }

    pub fn effective_light_width(&self) -> u32 {
        match self.mode {
            BakeMode::Single => 1,
            BakeMode::Full | BakeMode::Isotropic => self.light_width,
        }
    }

    pub fn effective_light_height(&self) -> u32 {
        match self.mode {
            BakeMode::Single => 1,
            BakeMode::Full | BakeMode::Isotropic => self.light_height,
        }
    }

    pub fn light_count(&self) -> u32 {
        self.effective_light_width() * self.effective_light_height()
    }

    pub fn light_direction_for_tile(&self, light_x: u32, light_y: u32) -> [f32; 3] {
        if self.mode == BakeMode::Single {
            return self.light;
        }

        hemisphere_latlong_direction(
            light_x,
            light_y,
            self.effective_light_width(),
            self.effective_light_height(),
        )
        .to_array()
    }

    pub fn config_for_tile(&self, light_x: u32, light_y: u32) -> Self {
        let mut config = self.clone();
        config.width = self.camera_tile_width();
        config.height = self.camera_tile_height();
        config.light = self.light_direction_for_tile(light_x, light_y);
        config.mode = BakeMode::Single;
        config.light_width = 1;
        config.light_height = 1;
        config
    }
}

impl BakeConfigFile {
    pub fn read(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path).map_err(|source| ConfigError::ReadConfig {
            path: path.to_path_buf(),
            source,
        })?;

        toml::from_str(&contents).map_err(|source| ConfigError::ParseConfig {
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn resolve(
        self,
        config_path: Option<&Path>,
        overrides: BakeOverrides,
    ) -> Result<ResolvedBakeConfig, ConfigError> {
        let config_dir = config_path.and_then(Path::parent);
        let material = resolve_material(self.material, &overrides)?;
        let obj = overrides
            .obj
            .or(self.obj)
            .ok_or(ConfigError::MissingObj)
            .map(|path| resolve_path(config_dir, path))?;

        let width = overrides.width.or(self.width).unwrap_or(DEFAULT_WIDTH);
        let height = overrides.height.or(self.height).unwrap_or(DEFAULT_HEIGHT);
        if width == 0 || height == 0 {
            return Err(ConfigError::InvalidResolution);
        }

        let mode = overrides.mode.or(self.mode).unwrap_or(BakeMode::Single);
        let light_width = overrides
            .light_width
            .or(self.light_width)
            .unwrap_or(DEFAULT_LIGHT_WIDTH);
        let light_height = overrides
            .light_height
            .or(self.light_height)
            .unwrap_or(DEFAULT_LIGHT_HEIGHT);
        if light_width == 0 || light_height == 0 {
            return Err(ConfigError::InvalidResolution);
        }

        let samples = overrides
            .samples
            .or(self.samples)
            .unwrap_or(DEFAULT_SAMPLES);
        if samples == 0 {
            return Err(ConfigError::InvalidSamples);
        }

        let tile_width_override = overrides.tile_width.or(self.tile_width);
        let tile_depth_override = overrides.tile_depth.or(self.tile_depth);
        if !valid_positive_override(tile_width_override)
            || !valid_positive_override(tile_depth_override)
        {
            return Err(ConfigError::InvalidTileOverride);
        }

        let light_value = overrides.light.or(self.light).unwrap_or(DEFAULT_LIGHT);
        let light = Vec3::from_array(light_value)
            .normalize()
            .filter(|light| light.y > MIN_LIGHT_COS)
            .ok_or(ConfigError::InvalidLight)?
            .to_array();

        let max_repeat_radius = overrides
            .max_repeat_radius
            .or(self.max_repeat_radius)
            .unwrap_or(DEFAULT_MAX_REPEAT_RADIUS);
        if max_repeat_radius > MAX_REPEAT_RADIUS_LIMIT {
            return Err(ConfigError::InvalidMaxRepeatRadius);
        }

        Ok(ResolvedBakeConfig {
            obj,
            width,
            height,
            mode,
            light_width,
            light_height,
            samples,
            tile_width_override,
            tile_depth_override,
            light,
            max_repeat_radius,
            sampler: overrides
                .sampler
                .or(self.sampler)
                .unwrap_or(SamplerKind::Halton),
            enable_shadows: overrides
                .enable_shadows
                .or(self.enable_shadows)
                .unwrap_or(true),
            material,
        })
    }
}

impl Manifest {
    pub fn new(config: &ResolvedBakeConfig, mesh: &crate::geometry::Mesh) -> Self {
        Self {
            tool: ToolInfo {
                name: "xbrdf-bake".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            input: InputInfo {
                obj: config.obj.clone(),
                triangle_count: mesh.triangles.len(),
                color_source: mesh.color_source.as_str().to_string(),
                height_offset_to_zero: mesh.y_offset_to_zero,
                tile_min: [mesh.tile_min_x, mesh.tile_min_z],
                tile_size: [mesh.tile_width, mesh.tile_depth],
                bounds_min: mesh.bounds.min.to_array(),
                bounds_max: mesh.bounds.max.to_array(),
                original_bounds_min: mesh.original_bounds.min.to_array(),
                original_bounds_max: mesh.original_bounds.max.to_array(),
            },
            output: OutputInfo {
                image: "xbrdf_view.exr".to_string(),
                format: "OpenEXR".to_string(),
                channels: "RGB f32".to_string(),
                width: config.atlas_width(),
                height: config.atlas_height(),
                tile_width: config.camera_tile_width(),
                tile_height: config.camera_tile_height(),
            },
            convention: ConventionInfo {
                coordinate_system: "Houdini Y-up, sample tile in XZ".to_string(),
                macro_normal: [0.0, 1.0, 0.0],
                normal_shading:
                    "faceted geometric triangle normals; OBJ vertex normals and smoothing groups are ignored"
                        .to_string(),
                pano_domain: "upper hemisphere only".to_string(),
                pano_center: [0.0, 0.0, 1.0],
                azimuth_positive_toward: [1.0, 0.0, 0.0],
                row_order: "top row is zenith, bottom row approaches horizon".to_string(),
                normalization:
                    "direct outgoing radiance divided by dot(+Y, light_direction_surface_to_light)"
                        .to_string(),
            },
            bake: BakeInfo {
                mode: config.mode,
                samples_per_direction: config.samples,
                max_repeat_radius: config.max_repeat_radius,
                camera_width: config.camera_tile_width(),
                camera_height: config.camera_tile_height(),
                light_width: config.effective_light_width(),
                light_height: config.effective_light_height(),
                light_count: config.light_count(),
                sampler: config.sampler,
                enable_shadows: config.enable_shadows,
                light_direction_surface_to_light: config.light,
                material: MaterialInfo {
                    kind: config.material.kind,
                    color: config.material.color,
                    roughness: config.material.roughness,
                    phong_exponent: config.material.phong_exponent(),
                    model: config.material.model_description(),
                },
                transport: if config.enable_shadows {
                    "direct lighting with visibility and shadow rays".to_string()
                } else {
                    "direct lighting with camera visibility only; shadow rays disabled".to_string()
                },
            },
        }
    }
}

fn resolve_path(config_dir: Option<&Path>, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else if let Some(config_dir) = config_dir {
        config_dir.join(path)
    } else {
        path
    }
}

fn valid_positive_override(value: Option<f32>) -> bool {
    value
        .map(|value| value.is_finite() && value > 0.0)
        .unwrap_or(true)
}

fn resolve_material(
    material: MaterialConfigFile,
    overrides: &BakeOverrides,
) -> Result<ResolvedMaterial, ConfigError> {
    let kind = overrides
        .material_kind
        .or(material.kind)
        .unwrap_or(MaterialKind::Lambertian);
    let color = overrides
        .material_color
        .or(material.color)
        .unwrap_or(DEFAULT_COLOR);
    if color
        .iter()
        .any(|component| !component.is_finite() || *component < 0.0)
    {
        return Err(ConfigError::InvalidMaterialColor);
    }

    let roughness = overrides.material_roughness.or(material.roughness);
    if roughness
        .map(|roughness| !roughness.is_finite() || !(0.0..=1.0).contains(&roughness))
        .unwrap_or(false)
    {
        return Err(ConfigError::InvalidMaterialRoughness);
    }

    Ok(ResolvedMaterial {
        kind,
        color,
        roughness: match kind {
            MaterialKind::Lambertian => None,
            MaterialKind::SpecularPhong => Some(roughness.unwrap_or(DEFAULT_SPECULAR_ROUGHNESS)),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_overrides_resolve() {
        let config = BakeConfigFile {
            obj: Some("asset.obj".into()),
            width: Some(8),
            height: None,
            mode: None,
            light_width: None,
            light_height: None,
            samples: Some(2),
            tile_width: None,
            tile_depth: None,
            light: None,
            max_repeat_radius: None,
            sampler: None,
            enable_shadows: None,
            material: MaterialConfigFile::default(),
        };
        let overrides = BakeOverrides {
            width: Some(16),
            light: Some([0.0, 2.0, 0.0]),
            ..BakeOverrides::default()
        };

        let resolved = config
            .resolve(Some(Path::new("assets/fixtures/flat.toml")), overrides)
            .unwrap();

        assert_eq!(resolved.obj, PathBuf::from("assets/fixtures/asset.obj"));
        assert_eq!(resolved.width, 16);
        assert_eq!(resolved.height, DEFAULT_HEIGHT);
        assert_eq!(resolved.mode, BakeMode::Single);
        assert_eq!(resolved.atlas_width(), 16);
        assert_eq!(resolved.atlas_height(), DEFAULT_HEIGHT);
        assert_eq!(resolved.samples, 2);
        assert_eq!(resolved.light, [0.0, 1.0, 0.0]);
        assert_eq!(resolved.max_repeat_radius, DEFAULT_MAX_REPEAT_RADIUS);
        assert_eq!(resolved.sampler, SamplerKind::Halton);
        assert!(resolved.enable_shadows);
        assert_eq!(resolved.material.kind, MaterialKind::Lambertian);
    }

    #[test]
    fn specular_material_resolves_with_sharp_exponent() {
        let config = BakeConfigFile {
            obj: Some("asset.obj".into()),
            material: MaterialConfigFile {
                kind: Some(MaterialKind::SpecularPhong),
                color: Some([0.8, 0.9, 1.0]),
                roughness: Some(0.0),
            },
            ..BakeConfigFile::default()
        };

        let resolved = config.resolve(None, BakeOverrides::default()).unwrap();

        assert_eq!(resolved.material.kind, MaterialKind::SpecularPhong);
        assert_eq!(resolved.material.color, [0.8, 0.9, 1.0]);
        assert_eq!(resolved.material.roughness, Some(0.0));
        assert_eq!(resolved.material.phong_exponent(), Some(MAX_PHONG_EXPONENT));
    }

    #[test]
    fn atlas_dimensions_follow_bake_mode() {
        let base = ResolvedBakeConfig {
            obj: "flat.obj".into(),
            width: 16,
            height: 8,
            mode: BakeMode::Full,
            light_width: 4,
            light_height: 3,
            samples: 1,
            tile_width_override: None,
            tile_depth_override: None,
            light: [0.0, 1.0, 0.0],
            max_repeat_radius: DEFAULT_MAX_REPEAT_RADIUS,
            sampler: SamplerKind::Halton,
            enable_shadows: true,
            material: ResolvedMaterial {
                kind: MaterialKind::Lambertian,
                color: [1.0, 1.0, 1.0],
                roughness: None,
            },
        };

        assert_eq!(base.camera_tile_width(), 16);
        assert_eq!(base.camera_tile_height(), 8);
        assert_eq!(base.atlas_width(), 64);
        assert_eq!(base.atlas_height(), 24);
        assert_eq!(base.light_count(), 12);

        let isotropic = ResolvedBakeConfig {
            mode: BakeMode::Isotropic,
            ..base
        };
        assert_eq!(isotropic.camera_tile_width(), 1);
        assert_eq!(isotropic.camera_tile_height(), 8);
        assert_eq!(isotropic.atlas_width(), 4);
        assert_eq!(isotropic.atlas_height(), 24);
    }

    #[test]
    fn manifest_round_trips_through_toml() {
        let mesh = crate::geometry::Mesh {
            triangles: Vec::new(),
            bounds: crate::geometry::Bounds {
                min: Vec3::new(-1.0, 0.0, -1.0),
                max: Vec3::new(1.0, 0.0, 1.0),
            },
            original_bounds: crate::geometry::Bounds {
                min: Vec3::new(-1.0, 0.0, -1.0),
                max: Vec3::new(1.0, 0.0, 1.0),
            },
            y_offset_to_zero: 0.0,
            tile_min_x: -1.0,
            tile_min_z: -1.0,
            tile_width: 2.0,
            tile_depth: 2.0,
            color_source: crate::geometry::ColorSource::None,
        };
        let config = ResolvedBakeConfig {
            obj: "flat.obj".into(),
            width: 4,
            height: 2,
            mode: BakeMode::Single,
            light_width: DEFAULT_LIGHT_WIDTH,
            light_height: DEFAULT_LIGHT_HEIGHT,
            samples: 1,
            tile_width_override: None,
            tile_depth_override: None,
            light: [0.0, 1.0, 0.0],
            max_repeat_radius: DEFAULT_MAX_REPEAT_RADIUS,
            sampler: SamplerKind::Halton,
            enable_shadows: true,
            material: ResolvedMaterial {
                kind: MaterialKind::Lambertian,
                color: [1.0, 1.0, 1.0],
                roughness: None,
            },
        };

        let manifest = Manifest::new(&config, &mesh);
        let encoded = toml::to_string_pretty(&manifest).unwrap();
        let decoded: Manifest = toml::from_str(&encoded).unwrap();

        assert_eq!(manifest, decoded);
    }
}
