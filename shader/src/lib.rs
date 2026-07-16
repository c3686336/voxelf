#![no_std]

extern crate spirv_std;

#[cfg(test)]
extern crate std;

use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4, vec2, vec3, vec4};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

const STACK_DEPTH: usize = 8;

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
pub struct CameraData {
    pub origin: Vec4,
    pub direction: Vec4,
    pub up: Vec4,
    pub right: Vec4,
    pub viewport: Vec4,
}

impl CameraData {
    pub fn new(origin: [f32; 3], direction: [f32; 3], up: [f32; 3], right: [f32; 3]) -> Self {
        Self {
            origin: vec4(origin[0], origin[1], origin[2], 0.0),
            direction: vec4(direction[0], direction[1], direction[2], 0.0),
            up: vec4(up[0], up[1], up[2], 0.0),
            right: vec4(right[0], right[1], right[2], 0.0),
            viewport: vec4(1.0, 0.0, 0.0, 0.0),
        }
    }
}

fn vec4_to_vec3(value: Vec4) -> Vec3 {
    vec3(value.x, value.y, value.z)
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
    _max_depth: u32,
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
    tmin = tmin.max(0.0);

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

            if stack_ind >= STACK_DEPTH {
                return false;
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

            if depth == 0 || depth > 23 || depth as usize > STACK_DEPTH {
                return false;
            }

            idx = ((((x >> (23 - depth)) & 1) << 2)
                | (((y >> (23 - depth)) & 1) << 1)
                | (((z >> (23 - depth)) & 1) << 0)) as u8;
            let stack_index = (depth - 1) as usize;
            parent = stack[stack_index];
            stack_ind = stack_index;
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] camera: &[CameraData],
    #[spirv(push_constant)] constants: &PushConsts,
) {
    let svodag = SerializedSvodagView::from_slice(buffer);
    let camera = camera[0];

    let camera_origin = vec4_to_vec3(camera.origin);
    let camera_direction = vec4_to_vec3(camera.direction);
    let camera_up = vec4_to_vec3(camera.up);
    let camera_right = vec4_to_vec3(camera.right);

    let film_width = 2.0 * camera.viewport.x;
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
        constants.root_index,
    );

    if result {
        *output = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        *output = vec4(0.0, 1.0, 0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    fn index(raw: u16) -> SvoIndex {
        unsafe { SvoIndex::from_raw(raw) }
    }

    fn default_push_consts(max_depth: u32, root_index: SvoIndex) -> PushConsts {
        PushConsts::new(max_depth, root_index)
    }

    fn default_camera_data() -> [CameraData; 1] {
        [CameraData::new(
            [1.5, 1.5, 3.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        )]
    }

    fn single_voxel_at_origin_depth_3() -> [u16; 7] {
        [0x0100, 2, 0x0100, 4, 0x0100, 6, 1]
    }

    fn morton_at(position: [u32; 3], level: u32, max_level: u32) -> u8 {
        let shift_amount = max_level - level - 1;
        let x = (position[0] >> shift_amount) & 1;
        let y = (position[1] >> shift_amount) & 1;
        let z = (position[2] >> shift_amount) & 1;

        ((x << 2) | (y << 1) | z) as u8
    }

    fn single_voxel(position: [u32; 3], depth: u32) -> Vec<u16> {
        let mut data = Vec::new();

        for level in 0..depth {
            let child_mask = 1_u16 << morton_at(position, level, depth);
            data.push(child_mask << 8);
            data.push(data.len() as u16 + 1);
        }

        data.push(1);
        data
    }

    #[test]
    fn constructors_initialize_fields_and_padding() {
        let root_index = index(17);
        let push_consts = default_push_consts(5, root_index);

        assert_eq!(push_consts.max_depth, 5);
        assert_eq!(push_consts.root_index.to_raw(), 17);
        assert_eq!(push_consts.pad, 0);

        let camera = default_camera_data()[0];
        assert_eq!(camera.origin, vec4(1.5, 1.5, 3.0, 0.0));
        assert_eq!(camera.direction, vec4(0.0, 0.0, -1.0, 0.0));
        assert_eq!(camera.up, vec4(0.0, 1.0, 0.0, 0.0));
        assert_eq!(camera.right, vec4(1.0, 0.0, 0.0, 0.0));
        assert_eq!(camera.viewport, vec4(1.0, 0.0, 0.0, 0.0));

        let metadata = SvodagMetaData::new(root_index, Mat4::IDENTITY);
        assert_eq!(metadata.root_index.to_raw(), 17);
        assert_eq!(metadata.pad, 0);
        assert_eq!(metadata.inv_model_transform, Mat4::IDENTITY);
    }

    #[test]
    fn serialized_view_reads_node_metadata_and_children() {
        let data = [0b1010_0101_0000_0001, 3, 7, 11, 13];
        let view = SerializedSvodagView::from_slice(&data);
        let root = index(0);

        assert_eq!(view.get_data(), &data);
        assert_eq!(view.get_childmask(root), 0b1010_0101);
        assert!(view.is_leaf(root));
        assert_eq!(view.get_child(root, 0).to_raw(), 3);
        assert_eq!(view.get_child(root, 2).to_raw(), 11);
    }

    #[test]
    fn index_of_morton_uses_packed_child_mask_order() {
        let data = [0b1000_1001_0000_0000];
        let view = SerializedSvodagView::from_slice(&data);
        let root = index(0);

        assert_eq!(index_of_morton(&view, root, 0), 2);
        assert_eq!(index_of_morton(&view, root, 3), 1);
        assert_eq!(index_of_morton(&view, root, 7), 0);
    }

    #[test]
    fn raymarch_hits_serialized_single_voxel() {
        let data = single_voxel_at_origin_depth_3();

        assert!(raymarch(
            vec3(0.5, 1.05, 1.05),
            vec3(1.0, 0.0, 0.0),
            3,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_hits_far_positive_direction_voxel() {
        let data = single_voxel([7, 7, 7], 3);

        assert!(raymarch(
            vec3(0.5, 1.95, 1.95),
            vec3(1.0, 0.0, 0.0),
            3,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_hits_negative_direction_voxel() {
        let data = single_voxel([7, 7, 7], 3);

        assert!(raymarch(
            vec3(2.5, 1.95, 1.95),
            vec3(-1.0, 0.0, 0.0),
            3,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_misses_negative_direction_voxel() {
        let data = single_voxel([7, 7, 7], 3);

        assert!(!raymarch(
            vec3(2.5, 1.5, 1.5),
            vec3(-1.0, 0.0, 0.0),
            3,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_hits_diagonal_voxel() {
        let data = single_voxel([1, 1, 1], 2);

        assert!(raymarch(
            vec3(0.5, 0.5, 0.5),
            vec3(1.0, 1.0, 1.0).normalize(),
            2,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_hits_when_starting_inside_voxel() {
        let data = single_voxel_at_origin_depth_3();

        assert!(raymarch(
            vec3(1.05, 1.05, 1.05),
            vec3(1.0, 0.0, 0.0),
            3,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_hits_when_starting_inside_volume() {
        let data = single_voxel([7, 7, 7], 3);

        assert!(raymarch(
            vec3(1.5, 1.95, 1.95),
            vec3(1.0, 0.0, 0.0),
            3,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_misses_when_starting_inside_volume() {
        let data = single_voxel([7, 7, 7], 3);

        assert!(!raymarch(
            vec3(1.5, 1.5, 1.5),
            vec3(1.0, 0.0, 0.0),
            3,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_does_not_hit_mirrored_voxel_when_starting_inside_volume() {
        let data = single_voxel([0, 7, 7], 3);

        assert!(!raymarch(
            vec3(1.5, 1.95, 1.95),
            vec3(1.0, 0.0, 0.0),
            3,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_misses_serialized_single_voxel() {
        let data = single_voxel_at_origin_depth_3();

        assert!(!raymarch(
            vec3(0.5, 1.5, 1.5),
            vec3(1.0, 0.0, 0.0),
            3,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_hits_root_leaf() {
        let data = [1];

        assert!(raymarch(
            vec3(1.5, 1.5, 3.0),
            vec3(0.0, 0.0, -1.0),
            1,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_rejects_ray_outside_root_bounds() {
        let data = [1];

        assert!(!raymarch(
            vec3(3.0, 3.0, 3.0),
            vec3(1.0, 0.0, 0.0),
            1,
            SerializedSvodagView::from_slice(&data),
            index(0),
        ));
    }

    #[test]
    fn raymarch_handles_stack_capacity_boundary() {
        let supported_depth = STACK_DEPTH as u32 + 1;
        let supported_data = single_voxel([0, 0, 0], supported_depth);
        let supported_center = 1.0 + 0.5 / (1_u32 << supported_depth) as f32;

        assert!(raymarch(
            vec3(0.5, supported_center, supported_center),
            vec3(1.0, 0.0, 0.0),
            supported_depth,
            SerializedSvodagView::from_slice(&supported_data),
            index(0),
        ));

        let overflow_depth = supported_depth + 1;
        let overflow_data = single_voxel([0, 0, 0], overflow_depth);
        let overflow_center = 1.0 + 0.5 / (1_u32 << overflow_depth) as f32;

        assert!(!raymarch(
            vec3(0.5, overflow_center, overflow_center),
            vec3(1.0, 0.0, 0.0),
            overflow_depth,
            SerializedSvodagView::from_slice(&overflow_data),
            index(0),
        ));
    }

    #[test]
    fn vertex_shader_writes_position_and_uv() {
        let mut output = Vec4::ZERO;
        let mut uv_output = Vec2::ZERO;

        main_vs(
            &mut output,
            vec2(-0.25, 0.75),
            vec2(0.2, 0.8),
            &mut uv_output,
        );

        assert_eq!(output, vec4(-0.25, 0.75, 0.0, 1.0));
        assert_eq!(uv_output, vec2(0.2, 0.8));
    }
}
