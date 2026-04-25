use crate::math::Vec3;

pub const PI: f32 = std::f32::consts::PI;
pub const INV_PI: f32 = 1.0 / std::f32::consts::PI;

pub fn hemisphere_latlong_direction(x: u32, y: u32, width: u32, height: u32) -> Vec3 {
    let u = (x as f32 + 0.5) / width as f32;
    let v = (y as f32 + 0.5) / height as f32;
    let azimuth = (u - 0.5) * std::f32::consts::TAU;
    let elevation = (1.0 - v) * std::f32::consts::FRAC_PI_2;
    let horizontal = elevation.cos();

    Vec3::new(
        azimuth.sin() * horizontal,
        elevation.sin(),
        azimuth.cos() * horizontal,
    )
}

pub fn stratified_sample_2d(sample_index: u32, sample_count: u32) -> [f32; 2] {
    let grid = (sample_count as f32).sqrt().ceil() as u32;
    let sx = sample_index % grid;
    let sy = sample_index / grid;
    [
        (sx as f32 + 0.5) / grid as f32,
        (sy as f32 + 0.5) / grid as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latlong_center_faces_positive_z() {
        let direction = hemisphere_latlong_direction(2, 3, 5, 4);
        assert!(direction.z > 0.9, "{direction:?}");
        assert!(direction.x.abs() < 1.0e-5, "{direction:?}");
        assert!(direction.y > 0.0, "{direction:?}");
    }

    #[test]
    fn azimuth_increases_toward_positive_x() {
        let left = hemisphere_latlong_direction(1, 3, 8, 4);
        let right = hemisphere_latlong_direction(5, 3, 8, 4);

        assert!(right.x > left.x, "left={left:?} right={right:?}");
    }

    #[test]
    fn rows_run_from_zenith_to_horizon() {
        let top = hemisphere_latlong_direction(2, 0, 4, 4);
        let bottom = hemisphere_latlong_direction(2, 3, 4, 4);

        assert!(top.y > bottom.y, "top={top:?} bottom={bottom:?}");
    }
}
