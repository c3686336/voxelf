use spirv_std::glam::Vec3;
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use core::f32::consts::PI;

/// Evaluates the metallic-roughness Cook–Torrance BRDF in linear RGB.
///
/// `albedo` is linear RGB reflectance, and `metallic` and perceptual
/// `roughness` are in the range 0–1. The normal, view, and light directions
/// must be normalized; view and light point away from the surface. The result
/// is the BRDF alone: callers must separately multiply it by incident radiance
/// and the cosine of the incident angle. Directions on or below the surface
/// return zero. Roughness is clamped to a small positive value to keep the
/// perfectly smooth limit finite.
pub fn brdf(
    albedo: Vec3,
    metallic: f32,
    roughness: f32,
    normal: Vec3,
    view: Vec3,
    light: Vec3,
) -> Vec3 {
    let n_dot_v = normal.dot(view).max(0.0);
    let n_dot_l = normal.dot(light).max(0.0);
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
        return Vec3::ZERO;
    }

    let half_sum = view + light;
    let half_length_squared = half_sum.dot(half_sum);
    if half_length_squared <= 1.0e-12 {
        return Vec3::ZERO;
    }
    let half_vector = half_sum / half_length_squared.sqrt();
    let n_dot_h = normal.dot(half_vector).clamp(0.0, 1.0);
    let v_dot_h = view.dot(half_vector).clamp(0.0, 1.0);
    let metallic = metallic.clamp(0.0, 1.0);
    let roughness = roughness.clamp(0.045, 1.0);
    let alpha = roughness * roughness;
    let alpha_squared = alpha * alpha;

    // GGX / Trowbridge–Reitz normal distribution. This form avoids
    // cancellation near n_dot_h = 1 for smooth surfaces.
    let n_dot_h_squared = n_dot_h * n_dot_h;
    let distribution_denominator = (1.0 - n_dot_h_squared) + alpha_squared * n_dot_h_squared;
    let distribution = alpha_squared / (PI * distribution_denominator * distribution_denominator);

    let f0 = Vec3::splat(0.04) * (1.0 - metallic) + albedo * metallic;
    let one_minus_v_dot_h = 1.0 - v_dot_h;
    let fresnel_factor = one_minus_v_dot_h
        * one_minus_v_dot_h
        * one_minus_v_dot_h
        * one_minus_v_dot_h
        * one_minus_v_dot_h;
    let fresnel = f0 + (Vec3::ONE - f0) * fresnel_factor;

    // Separable Smith GGX, with 4 NdotV NdotL cancelled algebraically
    // against the Cook–Torrance denominator to remain stable at grazing angles.
    let smith_v = n_dot_v + (alpha_squared + (1.0 - alpha_squared) * n_dot_v * n_dot_v).sqrt();
    let smith_l = n_dot_l + (alpha_squared + (1.0 - alpha_squared) * n_dot_l * n_dot_l).sqrt();
    let specular = fresnel * (distribution / (smith_v * smith_l));
    let diffuse = (Vec3::ONE - fresnel) * albedo * ((1.0 - metallic) / PI);
    diffuse + specular
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirv_std::glam::vec3;

    fn assert_close(actual: Vec3, expected: Vec3, tolerance: f32) {
        let difference = (actual - expected).abs();
        assert!(
            difference.x <= tolerance && difference.y <= tolerance && difference.z <= tolerance,
            "actual: {actual:?}, expected: {expected:?}"
        );
    }

    #[test]
    fn dielectric_normal_incidence_matches_lambert_and_ggx() {
        let albedo = vec3(0.8, 0.4, 0.2);
        let result = brdf(albedo, 0.0, 0.5, Vec3::Z, Vec3::Z, Vec3::Z);
        // D = 16/pi, F = 0.04, G = 1 at normal incidence.
        let expected = albedo * (0.96 / PI) + Vec3::splat(0.16 / PI);
        assert_close(result, expected, 1.0e-6);
    }

    #[test]
    fn metal_has_no_diffuse_and_uses_albedo_as_f0() {
        let albedo = vec3(0.8, 0.4, 0.2);
        let result = brdf(albedo, 1.0, 0.5, Vec3::Z, Vec3::Z, Vec3::Z);
        assert_close(result, albedo * (4.0 / PI), 1.0e-6);
        assert_close(
            brdf(Vec3::ZERO, 1.0, 0.5, Vec3::Z, Vec3::Z, Vec3::Z),
            Vec3::ZERO,
            0.0,
        );
    }

    #[test]
    fn directions_on_or_below_surface_return_zero() {
        let evaluate = |view, light| brdf(Vec3::ONE, 0.5, 0.5, Vec3::Z, view, light);
        assert_close(evaluate(-Vec3::Z, Vec3::Z), Vec3::ZERO, 0.0);
        assert_close(evaluate(Vec3::Z, -Vec3::Z), Vec3::ZERO, 0.0);
        assert_close(evaluate(Vec3::X, Vec3::Z), Vec3::ZERO, 0.0);
        assert_close(evaluate(Vec3::Z, Vec3::X), Vec3::ZERO, 0.0);
    }

    #[test]
    fn smooth_and_grazing_limits_are_finite_and_nonnegative() {
        let grazing = vec3(1.0, 0.0, 1.0e-6).normalize();
        for roughness in [0.0, 0.5, 1.0] {
            for direction in [Vec3::Z, grazing] {
                let value = brdf(Vec3::ONE, 1.0, roughness, Vec3::Z, direction, direction);
                assert!(value.x.is_finite() && value.y.is_finite() && value.z.is_finite());
                assert!(value.x >= 0.0 && value.y >= 0.0 && value.z >= 0.0);
            }
        }
    }

    #[test]
    fn brdf_is_reciprocal() {
        let view = vec3(0.6, 0.0, 0.8);
        let light = vec3(0.0, 0.8, 0.6);
        let albedo = vec3(0.7, 0.3, 0.1);
        assert_close(
            brdf(albedo, 0.4, 0.7, Vec3::Z, view, light),
            brdf(albedo, 0.4, 0.7, Vec3::Z, light, view),
            1.0e-6,
        );
    }
}
