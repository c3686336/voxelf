//! Color transfer functions, independent of tone mapping and display policy.

#[allow(unused_imports)]
use spirv_std::num_traits::Float;

/// Encode one linear channel with the sRGB transfer function.
pub fn linear_to_srgb(value: f32) -> f32 {
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
}
