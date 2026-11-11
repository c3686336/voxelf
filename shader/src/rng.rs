//! Stateless Threefry2x32-20 samples: a pixel is the key; sample index and frame
//! seed are the counter. Either component can be replayed without prior draws.

use spirv_std::glam::{UVec2, Vec2, uvec2, vec2};

/// Two uniforms in [0, 1), addressed independently of sample count or call order.
#[inline]
pub fn sample_2d(pixel: UVec2, seed: u32, sample_index: u32) -> Vec2 {
    let words = threefry2x32(uvec2(sample_index, seed), pixel);
    vec2(uniform(words.x), uniform(words.y))
}

fn uniform(word: u32) -> f32 {
    (word >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[inline]
fn mix<const ROTATION: u32>(x: &mut u32, y: &mut u32) {
    *x = x.wrapping_add(*y);
    *y = y.rotate_left(ROTATION) ^ *x;
}

// Random123's 20-round Threefry2x32 permutation. Only 32-bit add/rotate/xor:
// no Int64 capability or emulated wide multiplication is needed in WGSL.
fn threefry2x32(counter: UVec2, key: UVec2) -> UVec2 {
    let parity = 0x1bd1_1bda ^ key.x ^ key.y;
    let mut x = counter.x.wrapping_add(key.x);
    let mut y = counter.y.wrapping_add(key.y);

    mix::<13>(&mut x, &mut y);
    mix::<15>(&mut x, &mut y);
    mix::<26>(&mut x, &mut y);
    mix::<6>(&mut x, &mut y);
    x = x.wrapping_add(key.y);
    y = y.wrapping_add(parity).wrapping_add(1);

    mix::<17>(&mut x, &mut y);
    mix::<29>(&mut x, &mut y);
    mix::<16>(&mut x, &mut y);
    mix::<24>(&mut x, &mut y);
    x = x.wrapping_add(parity);
    y = y.wrapping_add(key.x).wrapping_add(2);

    mix::<13>(&mut x, &mut y);
    mix::<15>(&mut x, &mut y);
    mix::<26>(&mut x, &mut y);
    mix::<6>(&mut x, &mut y);
    x = x.wrapping_add(key.x);
    y = y.wrapping_add(key.y).wrapping_add(3);

    mix::<17>(&mut x, &mut y);
    mix::<29>(&mut x, &mut y);
    mix::<16>(&mut x, &mut y);
    mix::<24>(&mut x, &mut y);
    x = x.wrapping_add(key.y);
    y = y.wrapping_add(parity).wrapping_add(4);

    mix::<13>(&mut x, &mut y);
    mix::<15>(&mut x, &mut y);
    mix::<26>(&mut x, &mut y);
    mix::<6>(&mut x, &mut y);
    x = x.wrapping_add(parity);
    y = y.wrapping_add(key.x).wrapping_add(5);
    uvec2(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threefry_matches_random123_known_answer_vectors() {
        // https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
        for (counter, key, expected) in [
            (UVec2::ZERO, UVec2::ZERO, uvec2(0x6b20_0159, 0x99ba_4efe)),
            (
                UVec2::splat(u32::MAX),
                UVec2::splat(u32::MAX),
                uvec2(0x1cb9_96fc, 0xbb00_2be7),
            ),
            (
                uvec2(0x243f_6a88, 0x85a3_08d3),
                uvec2(0x1319_8a2e, 0x0370_7344),
                uvec2(0xc492_3a9c, 0x483d_f7a0),
            ),
        ] {
            assert_eq!(threefry2x32(counter, key), expected);
        }
    }

    #[test]
    fn sample_address_preserves_both_pixel_axes_seed_and_index() {
        let expected =
            vec2((0xc492_3a9c_u32 >> 8) as f32, (0x483d_f7a0_u32 >> 8) as f32) / 16_777_216.0;
        assert_eq!(
            sample_2d(uvec2(0x1319_8a2e, 0x0370_7344), 0x85a3_08d3, 0x243f_6a88),
            expected
        );
    }

    #[test]
    fn uniform_conversion_includes_zero_but_never_one() {
        assert_eq!(uniform(0), 0.0);
        assert_eq!(uniform(255), 0.0);
        assert_eq!(uniform(256), 1.0 / 16_777_216.0);
        assert_eq!(uniform(u32::MAX), f32::from_bits(0x3f7f_ffff));
    }
}
