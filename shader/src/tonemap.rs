//! Display transforms only; linear validation captures bypass this module.

use spirv_std::glam::{Vec3, vec3};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use crate::RendererMode;

/// Reinhard's per-channel radiance compression, before display encoding.
pub fn reinhard(radiance: Vec3) -> Vec3 {
    radiance / (radiance + Vec3::ONE)
}

/// Preserve hit albedo in the debug view; tone-map reference lighting and sky.
/// Encode sRGB only when the attachment does not perform that conversion itself.
pub fn display_color(radiance: Vec3, renderer: RendererMode, hit: bool, encode_srgb: bool) -> Vec3 {
    let mapped = if hit && renderer == RendererMode::Albedo {
        radiance
    } else {
        reinhard(radiance)
    };
    if encode_srgb {
        vec3(
            linear_to_srgb(mapped.x),
            linear_to_srgb(mapped.y),
            linear_to_srgb(mapped.z),
        )
    } else {
        mapped
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_encoding_matches_reference_values() {
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(0.003_130_8) - 0.040_449_936).abs() < 0.000_001);
        assert!((linear_to_srgb(0.215_86) - 0.501_96).abs() < 0.000_1);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 0.000_001);
    }

    #[test]
    fn debug_hits_bypass_tone_mapping_but_background_and_lighting_do_not() {
        let radiance = vec3(0.25, 0.5, 2.0);
        let mapped = vec3(0.2, 1.0 / 3.0, 2.0 / 3.0);
        assert_eq!(
            display_color(radiance, RendererMode::Albedo, true, false),
            radiance
        );
        assert_eq!(
            display_color(radiance, RendererMode::Albedo, false, false),
            mapped
        );
        assert_eq!(
            display_color(radiance, RendererMode::Reference, true, false),
            mapped
        );
    }

    #[test]
    fn display_encoding_is_applied_after_tone_mapping() {
        let actual = display_color(vec3(1.0, 3.0, 0.0), RendererMode::Reference, true, true);
        let expected = vec3(0.735_357, 0.880_825, 0.0);
        assert!((actual - expected).abs().max_element() < 0.000_001);
    }
}
