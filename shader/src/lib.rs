#![no_std]

extern crate spirv_std;

use spirv_std::glam::{vec2, vec3, vec4, Mat4, Vec2, Vec3, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

const STACK_DEPTH: usize = 20;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::AnyBitPattern)]
pub struct PushConsts {
    pub max_depth: u32,
    pub root_index: SvoIndex,
    pad: u16,
}

impl PushConsts {
    pub fn new(max_depth: u32, root_index: SvoIndex) -> Self {
        Self {
            max_depth,
            root_index,
            pad: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::AnyBitPattern)]
pub struct SvodagMetaData {
    pub root_index: SvoIndex,
    pad: u16,
    pub inv_model_transform: Mat4,
}

impl SvodagMetaData {
    pub fn new(root_index: SvoIndex, inv_model_transform: Mat4) -> Self {
        Self {
            root_index,
            pad: 0,
            inv_model_transform,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, bytemuck::AnyBitPattern)]
#[repr(C)]
pub struct SvoIndex(u16);

impl SvoIndex {
    pub unsafe fn from_raw(index: u16) -> Self {
        Self(index)
    }

    pub fn to_raw(&self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SerializedSvodagView<'a> {
    data: &'a [u16],
}

impl<'a> SerializedSvodagView<'a> {
    pub fn get_childmask(&self, index: SvoIndex) -> u8 {
        (self.data[index.to_raw() as usize] >> 8) as u8
    }

    pub fn is_leaf(&self, index: SvoIndex) -> bool {
        (self.data[index.to_raw() as usize] & 0b1) != 0
    }

    pub fn get_child(&self, index: SvoIndex, child_index: u8) -> SvoIndex {
        unsafe {
            SvoIndex::from_raw(self.data[index.to_raw() as usize + (1 + child_index) as usize])
        }
    }

    pub fn get_data(&self) -> &[u16] {
        &self.data
    }

    pub fn from_slice(slice: &'a [u16]) -> Self {
        Self { data: slice }
    }
}

fn index_of_morton(svodag: &SerializedSvodagView, node: SvoIndex, morton: u8) -> u8 {
    ((svodag.get_childmask(node) >> morton).count_ones().max(1) - 1) as u8
}

pub fn raymarch(
    origin: Vec3,
    direction: Vec3,
    max_depth: u32,
    svodag: SerializedSvodagView,
    root_ref: SvoIndex,
) -> bool {
    let dir_threshold = -direction.abs().clamp(Vec3::ONE * f32::EPSILON, Vec3::MAX);
    let dir_sign = direction.cmpgt(Vec3::ZERO);
    let mask = ((dir_sign.x as u8) << 2) | ((dir_sign.y as u8) << 1) | ((dir_sign.z as u8) << 0);
    let dir_inv = dir_threshold.recip();

    let mut origin = origin;
    if dir_sign.x {
        origin.x = 3.0 - origin.x;
    }
    if dir_sign.y {
        origin.y = 3.0 - origin.y;
    }
    if dir_sign.z {
        origin.z = 3.0 - origin.z;
    }
    let origin = origin;

    let t1 = (Vec3::ONE - origin) * dir_inv;
    let t2 = (2.0 * Vec3::ONE - origin) * dir_inv;

    let mut tmin = t2.max_element(); // Because of the mirroring
    let tmax = t1.min_element(); // Because of the mirroring

    if tmax <= 0.0 || tmin > tmax {
        return false;
    }

    let mut position = Vec3::ONE;
    let mut parent = root_ref;
    if svodag.is_leaf(parent) && tmin <= tmax {
        return true;
    }
    let mut idx: u8 = 0;
    let mut scale = 0.5;

    let center_t = (1.5 * Vec3::ONE - origin) * dir_inv;
    if center_t.x > tmin {
        idx ^= 4;
        position.x = 1.5;
    }
    if center_t.y > tmin {
        idx ^= 2;
        position.y = 1.5;
    }
    if center_t.z > tmin {
        idx ^= 1;
        position.z = 1.5;
    }

    let mut stack: [SvoIndex; STACK_DEPTH] = [parent; STACK_DEPTH];
    let mut stack_ind = 0;

    loop {
        let t2_current_cube = (position - origin) * dir_inv;
        let tmax_current_cube = t2_current_cube.min_element();

        // PUSH
        if (svodag.get_childmask(parent) >> (idx ^ mask)) & 1 == 1 && tmin <= tmax_current_cube {
            // let child = svodag.deref(&parent.children[parent.index_of_morton(idx ^ mask)]);
            let child = svodag.get_child(parent, index_of_morton(&svodag, parent, idx ^ mask));

            if svodag.is_leaf(child) {
                return true;
            }

            stack[stack_ind] = parent;
            stack_ind += 1;

            parent = child;
            scale *= 0.5;
            let center_t = (position + Vec3::ONE * scale - origin) * dir_inv;
            idx = 0;
            if center_t.x > tmin {
                idx ^= 4;
                position.x += scale;
            }
            if center_t.y > tmin {
                idx ^= 2;
                position.y += scale;
            }
            if center_t.z > tmin {
                idx ^= 1;
                position.z += scale;
            }

            continue;
        }

        // ADVANCE
        let mut step_mask = 0;
        if t2_current_cube.x <= tmax_current_cube {
            step_mask ^= 4;
            position.x -= scale;
        }
        if t2_current_cube.y <= tmax_current_cube {
            step_mask ^= 2;
            position.y -= scale;
        }
        if t2_current_cube.z <= tmax_current_cube {
            step_mask ^= 1;
            position.z -= scale;
        }

        let step_mask = step_mask;
        tmin = tmax_current_cube;

        idx ^= step_mask;

        // POP
        if (idx & step_mask) != 0
        /*If any of these cause idx to increase, not decrease*/
        {
            let mut diff_bits = 0;
            if (step_mask & 4) != 0 {
                diff_bits |= position.x.to_bits() ^ (position.x + scale).to_bits();
            }
            if (step_mask & 2) != 0 {
                diff_bits |= position.y.to_bits() ^ (position.y + scale).to_bits();
            }
            if (step_mask & 1) != 0 {
                diff_bits |= position.z.to_bits() ^ (position.z + scale).to_bits();
            }

            if position.cmplt(Vec3::ONE).any() {
                return false;
            }

            let depth = (diff_bits << 9).leading_zeros() + 1;
            scale = (-(depth as f32)).exp2();

            let x = position.x.to_bits();
            let y = position.y.to_bits();
            let z = position.z.to_bits();

            idx = ((((x >> (23 - depth)) & 1) << 2)
                | (((y >> (23 - depth)) & 1) << 1)
                | (((z >> (23 - depth)) & 1) << 0)) as u8;
            parent = stack[(depth - 1) as usize];
            stack_ind = depth as usize;
            // TODO: Instead of popping an item which will be pushed back again anyway, just set a flag.

            position.x = f32::from_bits(x >> (23 - depth) << (23 - depth));
            position.y = f32::from_bits(y >> (23 - depth) << (23 - depth));
            position.z = f32::from_bits(z >> (23 - depth) << (23 - depth));
        }
    }

    false
}

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
pub fn main_fs(
    output: &mut Vec4,
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buffer: &[u16],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] metadata: &[SvodagMetaData],
    #[spirv(push_constant)] constants: &PushConsts,
) {
    let svodag = SerializedSvodagView::from_slice(buffer);

    let camera_origin = vec3(1.5, 1.5, 3.0);
    let camera_direction = vec3(0.0, 0.0, -1.0);
    let camera_up = vec3(0.0, 1.0, 0.0);
    let camera_right = camera_direction.cross(camera_up);

    let film_width = 2.0;
    let film_height = 2.0;
    let film_distance = 1.0;

    let x = uv - vec2(0.5, 0.5);
    let film_point: Vec3 = camera_direction * film_distance
        + x.x * camera_right * film_width
        + x.y * camera_up * film_height;

    let result = raymarch(
        camera_origin,
        film_point.normalize(),
        constants.max_depth,
        svodag,
        metadata[0].root_index,
    );

    if result {
        *output = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        *output = vec4(0.0, 1.0, 0.0, 1.0);
    }
}
