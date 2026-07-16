use crate::{BakeMode, ResolvedBakeConfig, Vec3};
use serde::{Deserialize, Serialize};

pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct AtlasMetadata {
    pub schema_version: u32,
    pub mode: BakeMode,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub camera_width: u32,
    pub camera_height: u32,
    pub light_width: u32,
    pub light_height: u32,
    pub camera_azimuth_wraps: bool,
    pub elevation_clamps: bool,
    pub samples_are_texel_centered: bool,
    pub consumer_applies_macro_cosine: bool,
}

impl AtlasMetadata {
    pub fn from_config(config: &ResolvedBakeConfig) -> Self {
        Self {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            mode: config.mode,
            atlas_width: config.atlas_width(),
            atlas_height: config.atlas_height(),
            camera_width: config.camera_tile_width(),
            camera_height: config.camera_tile_height(),
            light_width: config.effective_light_width(),
            light_height: config.effective_light_height(),
            camera_azimuth_wraps: config.mode != BakeMode::Isotropic,
            elevation_clamps: true,
            samples_are_texel_centered: true,
            consumer_applies_macro_cosine: true,
        }
    }

    pub fn dimensions_match(self, width: u32, height: u32) -> bool {
        self.schema_version == ARTIFACT_SCHEMA_VERSION
            && self.atlas_width == width
            && self.atlas_height == height
    }

    pub fn texel_for_light_tile(
        self,
        outgoing: Vec3,
        light_x: u32,
        light_y: u32,
    ) -> Option<[f32; 2]> {
        if light_x >= self.light_width || light_y >= self.light_height {
            return None;
        }
        let camera_uv = hemisphere_uv(outgoing)?;
        let local_x = if self.mode == BakeMode::Isotropic {
            0.0
        } else {
            camera_uv[0] * self.camera_width as f32 - 0.5
        };
        let local_y = camera_uv[1] * self.camera_height as f32 - 0.5;
        Some([
            light_x as f32 * self.camera_width as f32 + local_x,
            light_y as f32 * self.camera_height as f32 + local_y,
        ])
    }

    pub fn light_grid_position(self, light: Vec3) -> Option<[f32; 2]> {
        let uv = hemisphere_uv(light)?;
        Some([
            uv[0] * self.light_width as f32 - 0.5,
            uv[1] * self.light_height as f32 - 0.5,
        ])
    }
}

pub fn hemisphere_uv(direction: Vec3) -> Option<[f32; 2]> {
    let direction = direction.normalize()?;
    if direction.y < 0.0 {
        return None;
    }
    let u = (direction.x.atan2(direction.z) / std::f32::consts::TAU + 0.5).rem_euclid(1.0);
    let elevation = direction.y.clamp(0.0, 1.0).asin();
    let v = 1.0 - elevation / std::f32::consts::FRAC_PI_2;
    Some([u, v])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::hemisphere_latlong_direction;

    #[test]
    fn direction_centers_round_trip_to_texel_centers() {
        for y in 0..4 {
            for x in 0..8 {
                let direction = hemisphere_latlong_direction(x, y, 8, 4);
                let uv = hemisphere_uv(direction).unwrap();
                assert!((uv[0] * 8.0 - 0.5 - x as f32).abs() < 1.0e-5);
                assert!((uv[1] * 4.0 - 0.5 - y as f32).abs() < 1.0e-5);
            }
        }
    }

    #[test]
    fn lower_hemisphere_is_not_addressable() {
        assert_eq!(hemisphere_uv(Vec3::new(0.0, -1.0, 0.0)), None);
    }
}
