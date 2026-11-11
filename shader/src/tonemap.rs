//! Tone-mapping operators, independent of display policy and color encoding.

use spirv_std::glam::Vec3;

/// Reinhard's per-channel radiance compression, before display encoding.
pub fn reinhard(radiance: Vec3) -> Vec3 {
    radiance / (radiance + Vec3::ONE)
}
