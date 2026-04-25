use crate::geometry::Triangle;
use crate::math::Vec3;
use crate::sampling::INV_PI;

const EPSILON: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub t: f32,
    pub position: Vec3,
    pub normal: Vec3,
}

pub fn intersect_triangle(origin: Vec3, direction: Vec3, triangle: &Triangle) -> Option<RayHit> {
    let edge1 = triangle.v1 - triangle.v0;
    let edge2 = triangle.v2 - triangle.v0;
    let p = direction.cross(edge2);
    let det = edge1.dot(p);

    if det.abs() < EPSILON {
        return None;
    }

    let inv_det = 1.0 / det;
    let tvec = origin - triangle.v0;
    let u = tvec.dot(p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = tvec.cross(edge1);
    let v = direction.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = edge2.dot(q) * inv_det;
    if t <= EPSILON {
        return None;
    }

    Some(RayHit {
        t,
        position: origin + direction * t,
        normal: triangle.normal,
    })
}

pub fn normalized_lambertian(normal: Vec3, light: Vec3) -> f32 {
    let macro_irradiance = Vec3::Y.dot(light).max(EPSILON);
    normal.dot(light).max(0.0) * INV_PI / macro_irradiance
}

pub fn normalized_phong_specular(normal: Vec3, light: Vec3, outgoing: Vec3, exponent: f32) -> f32 {
    let macro_irradiance = Vec3::Y.dot(light).max(EPSILON);
    let n_dot_l = normal.dot(light).max(0.0);
    if n_dot_l == 0.0 {
        return 0.0;
    }

    let mirror = (normal * (2.0 * n_dot_l) - light)
        .normalize()
        .unwrap_or(Vec3::ZERO);
    let lobe = mirror.dot(outgoing).max(0.0).powf(exponent.max(1.0));
    n_dot_l * ((exponent.max(1.0) + 2.0) / (2.0 * std::f32::consts::PI)) * lobe / macro_irradiance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_triangle_hit_works() {
        let triangle = Triangle {
            v0: Vec3::new(-1.0, 0.0, -1.0),
            v1: Vec3::new(1.0, 0.0, -1.0),
            v2: Vec3::new(0.0, 0.0, 1.0),
            normal: Vec3::Y,
            color: Vec3::new(1.0, 1.0, 1.0),
        };

        let hit = intersect_triangle(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            &triangle,
        )
        .unwrap();

        assert!((hit.t - 1.0).abs() < 1.0e-5);
        assert_eq!(hit.position, Vec3::ZERO);
    }

    #[test]
    fn flat_lambertian_normalizes_to_inverse_pi() {
        let light = Vec3::new(0.0, 1.0, -1.0).normalize().unwrap();
        let value = normalized_lambertian(Vec3::Y, light);

        assert!((value - INV_PI).abs() < 1.0e-5);
    }

    #[test]
    fn phong_specular_peaks_at_mirror_direction() {
        let light = Vec3::new(0.0, 1.0, -1.0).normalize().unwrap();
        let mirror = Vec3::new(0.0, light.y, -light.z).normalize().unwrap();
        let off_mirror = Vec3::new(0.2, light.y, -light.z).normalize().unwrap();

        let peak = normalized_phong_specular(Vec3::Y, light, mirror, 500.0);
        let off_peak = normalized_phong_specular(Vec3::Y, light, off_mirror, 500.0);

        assert!(peak > 10.0, "{peak}");
        assert!(off_peak < peak * 0.1, "peak={peak} off_peak={off_peak}");
    }
}
