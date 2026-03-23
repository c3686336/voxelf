#![no_std]

extern crate spirv_std;

use spirv_std::glam::{vec2, vec4, Vec2, Vec4};
use spirv_std::spirv;

#[spirv(vertex)]
pub fn main_vs(
    #[spirv(position)] output: &mut Vec4,
    position: Vec2,
    uv: Vec2,
    #[spirv(location = 0)] uv_output: &mut Vec2,
) {
    *output = vec4(position.x, position.y, 0.0, 1.0);
    *uv_output = uv;
}

#[spirv(fragment)]
pub fn main_fs(output: &mut Vec4, #[spirv(location = 0)] uv: Vec2) {
    let c: Vec2 = uv * 4.0 - vec2(2.0, 2.0);
    let mut z: Vec2 = Vec2::ZERO;

    for i in 0..100 {
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;

        if z.length_squared() >= 4.0 {
            *output = vec4(1.0, 0.0, 0.0, 0.0);
            return;
        }
    }

    *output = vec4(0.0, 0.0, 0.0, 1.0);
}
