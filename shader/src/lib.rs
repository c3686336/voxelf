#![no_std]

extern crate spirv_std;

#[cfg(test)]
extern crate std;

pub mod brdf;
pub mod integrator;
pub mod svodag;

use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4, vec2, vec3, vec4};
use spirv_std::image::{Image2d, SampledImage};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::spirv;

const STACK_DEPTH: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::AnyBitPattern)]
pub struct PushConsts {
    pub max_depth: u32,
    pub root_index: SvoIndex,
    pub albedo_r: f32,
    pub albedo_g: f32,
    pub albedo_b: f32,
    pub metallic: f32,
    pub roughness: f32,
    /// Nonzero when the attachment does not perform sRGB encoding.
    pub encode_srgb: u32,
    pub renderer: u32,
    pub samples_per_pixel: u32,
    pub seed: u32,
    pub linear_output: u32,
}

impl PushConsts {
    pub fn new(
        max_depth: u32,
        root_index: SvoIndex,
        albedo: [f32; 3],
        metallic: f32,
        roughness: f32,
        encode_srgb: bool,
        renderer: u32,
        samples_per_pixel: u32,
        seed: u32,
        linear_output: bool,
    ) -> Self {
        Self {
            max_depth,
            root_index,
            albedo_r: albedo[0],
            albedo_g: albedo[1],
            albedo_b: albedo[2],
            metallic,
            roughness,
            encode_srgb: encode_srgb as u32,
            renderer,
            samples_per_pixel,
            seed,
            linear_output: linear_output as u32,
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
    pad: [u32; 3],
    pub inv_model_transform: Mat4,
}

impl SvodagMetaData {
    pub fn new(root_index: SvoIndex, inv_model_transform: Mat4) -> Self {
        Self {
            root_index,
            pad: [0; 3],
            inv_model_transform,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, bytemuck::AnyBitPattern)]
#[repr(C)]
pub struct SvoIndex(u32);

impl SvoIndex {
    pub unsafe fn from_raw(index: u32) -> Self {
        Self(index)
    }

    pub fn to_raw(&self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SvodagView<'a> {
    data: &'a [u32],
}

impl<'a> SvodagView<'a> {
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

    pub fn get_data(&self) -> &[u32] {
        &self.data
    }

    pub fn from_slice(slice: &'a [u32]) -> Self {
        Self { data: slice }
    }
}

fn index_of_morton(svodag: &SvodagView, node: SvoIndex, morton: u8) -> u8 {
    ((svodag.get_childmask(node) >> morton).count_ones().max(1) - 1) as u8
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub found: bool,
    pub position: Vec3,
    pub normal: Vec3,
}

impl Hit {
    fn miss() -> Self {
        Self {
            found: false,
            position: Vec3::ZERO,
            normal: Vec3::ZERO,
        }
    }
}

fn cube_hit(origin: Vec3, direction: Vec3, min: Vec3, size: f32) -> Hit {
    let max = min + Vec3::splat(size);
    let parallel = direction.cmpeq(Vec3::ZERO);
    let inverse = Vec3::select(parallel, Vec3::ONE, direction).recip();
    let a = (min - origin) * inverse;
    let b = (max - origin) * inverse;
    let near_axes = Vec3::select(parallel, Vec3::splat(-f32::MAX), a.min(b));
    let far_axes = Vec3::select(parallel, Vec3::splat(f32::MAX), a.max(b));
    let near = near_axes.max_element();
    let far = far_axes.min_element();
    let near_normal = if near_axes.x == near {
        vec3(if direction.x > 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0)
    } else if near_axes.y == near {
        vec3(0.0, if direction.y > 0.0 { -1.0 } else { 1.0 }, 0.0)
    } else {
        vec3(0.0, 0.0, if direction.z > 0.0 { -1.0 } else { 1.0 })
    };
    let far_normal = if far_axes.x == far {
        vec3(if direction.x > 0.0 { 1.0 } else { -1.0 }, 0.0, 0.0)
    } else if far_axes.y == far {
        vec3(0.0, if direction.y > 0.0 { 1.0 } else { -1.0 }, 0.0)
    } else {
        vec3(0.0, 0.0, if direction.z > 0.0 { 1.0 } else { -1.0 })
    };
    let (t, normal) = if near >= 0.0 {
        (near, near_normal)
    } else {
        (far, far_normal)
    };
    Hit {
        found: true,
        position: origin + direction * t,
        normal,
    }
}

pub fn raymarch(
    origin: Vec3,
    direction: Vec3,
    max_depth: u32,
    svodag: SvodagView,
    root_ref: SvoIndex,
) -> bool {
    trace(origin, direction, max_depth, svodag, root_ref).found
}

pub fn trace(
    origin: Vec3,
    direction: Vec3,
    _max_depth: u32,
    svodag: SvodagView,
    root_ref: SvoIndex,
) -> Hit {
    let ray_origin = origin;
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
        return Hit::miss();
    }
    tmin = tmin.max(0.0);

    let mut position = Vec3::ONE;
    let mut parent = root_ref;
    if svodag.is_leaf(parent) && tmin <= tmax {
        return cube_hit(ray_origin, direction, Vec3::ONE, 1.0);
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
            let child = svodag.get_child(parent, index_of_morton(&svodag, parent, idx ^ mask));

            if svodag.is_leaf(child) {
                let mut min = position;
                if dir_sign.x {
                    min.x = 3.0 - position.x - scale;
                }
                if dir_sign.y {
                    min.y = 3.0 - position.y - scale;
                }
                if dir_sign.z {
                    min.z = 3.0 - position.z - scale;
                }
                return cube_hit(ray_origin, direction, min, scale);
            }

            if stack_ind >= STACK_DEPTH {
                return Hit::miss();
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
                return Hit::miss();
            }

            let depth = (diff_bits << 9).leading_zeros() + 1;
            scale = (-(depth as f32)).exp2();

            let x = position.x.to_bits();
            let y = position.y.to_bits();
            let z = position.z.to_bits();

            if depth == 0 || depth > 23 || depth as usize > STACK_DEPTH {
                return Hit::miss();
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

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[spirv(fragment)]
pub fn main_fs(
    output: &mut Vec4,
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(frag_coord)] frag_coord: Vec4,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buffer: &[u32],
    #[spirv(descriptor_set = 0, binding = 1)] environment: &SampledImage<Image2d>,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] camera: &[CameraData],
    #[spirv(push_constant)] constants: &PushConsts,
) {
    let svodag = SvodagView::from_slice(buffer);
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
    let ray_direction = film_point.normalize();

    let hit = trace(
        camera_origin,
        ray_direction,
        constants.max_depth,
        svodag,
        constants.root_index,
    );
    let radiance = if hit.found {
        integrator::shade(
            hit,
            -ray_direction,
            vec2(frag_coord.x, frag_coord.y),
            constants,
            svodag,
            environment,
        )
    } else {
        integrator::environment_radiance(environment, ray_direction)
    };
    if constants.linear_output != 0 {
        *output = vec4(radiance.x, radiance.y, radiance.z, 1.0);
        return;
    }
    let mapped = if hit.found && constants.renderer == integrator::ALBEDO {
        radiance
    } else {
        radiance / (radiance + Vec3::ONE)
    };
    let color = if constants.encode_srgb != 0 {
        vec3(
            linear_to_srgb(mapped.x),
            linear_to_srgb(mapped.y),
            linear_to_srgb(mapped.z),
        )
    } else {
        mapped
    };
    *output = vec4(color.x, color.y, color.z, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    #[test]
    fn srgb_encoding_matches_reference_values() {
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(0.003_130_8) - 0.040_449_936).abs() < 0.000_001);
        assert!((linear_to_srgb(0.215_86) - 0.501_96).abs() < 0.000_1);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 0.000_001);
    }

    fn index(raw: u32) -> SvoIndex {
        unsafe { SvoIndex::from_raw(raw) }
    }

    fn default_push_consts(max_depth: u32, root_index: SvoIndex) -> PushConsts {
        PushConsts::new(
            max_depth,
            root_index,
            [1.0, 0.0, 0.0],
            0.0,
            0.5,
            false,
            0,
            16,
            0,
            false,
        )
    }

    fn default_camera_data() -> [CameraData; 1] {
        [CameraData::new(
            [1.5, 1.5, 3.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        )]
    }

    fn single_voxel_at_origin_depth_3() -> [u32; 7] {
        [0x0100, 2, 0x0100, 4, 0x0100, 6, 1]
    }

    fn morton_at(position: [u32; 3], level: u32, max_level: u32) -> u8 {
        let shift_amount = max_level - level - 1;
        let x = (position[0] >> shift_amount) & 1;
        let y = (position[1] >> shift_amount) & 1;
        let z = (position[2] >> shift_amount) & 1;

        ((x << 2) | (y << 1) | z) as u8
    }

    fn single_voxel(position: [u32; 3], depth: u32) -> Vec<u32> {
        let mut data = Vec::new();

        for level in 0..depth {
            let child_mask = 1_u32 << morton_at(position, level, depth);
            data.push(child_mask << 8);
            data.push(data.len() as u32 + 1);
        }

        data.push(1);
        data
    }

    #[test]
    fn trace_returns_voxel_face_and_inside_exit() {
        let data = single_voxel([3, 4, 5], 3);
        let view = SvodagView::from_slice(&data);
        let hit = trace(vec3(0.0, 1.55, 1.68), Vec3::X, 3, view, index(0));
        assert!(hit.found);
        assert!((hit.position - vec3(1.375, 1.55, 1.68)).length() < 1e-5);
        assert_eq!(hit.normal, -Vec3::X);
        let hit = trace(vec3(1.4, 1.55, 1.68), Vec3::Z, 3, view, index(0));
        assert!(hit.found);
        assert!((hit.position - vec3(1.4, 1.55, 1.75)).length() < 1e-5);
        assert_eq!(hit.normal, Vec3::Z);
        let hit = trace(vec3(1.4, 2.5, 1.68), -Vec3::Y, 3, view, index(0));
        assert!(hit.found);
        assert!((hit.position.y - 1.625).abs() < 1e-5);
        assert_eq!(hit.normal, Vec3::Y);
        assert!(!trace(vec3(0.0, 1.1, 1.1), Vec3::X, 3, view, index(0)).found);
    }

    #[test]
    fn constructors_initialize_fields_and_padding() {
        let root_index = index(70_000);
        let push_consts = default_push_consts(5, root_index);

        assert_eq!(core::mem::size_of::<SvoIndex>(), 4);
        assert_eq!(core::mem::size_of::<PushConsts>(), 48);
        assert_eq!(core::mem::offset_of!(PushConsts, albedo_r), 8);
        assert_eq!(core::mem::offset_of!(PushConsts, metallic), 20);
        assert_eq!(core::mem::offset_of!(PushConsts, roughness), 24);
        assert_eq!(core::mem::offset_of!(PushConsts, encode_srgb), 28);
        assert_eq!(core::mem::size_of::<SvodagMetaData>(), 80);
        assert_eq!(push_consts.max_depth, 5);
        assert_eq!(push_consts.root_index.to_raw(), 70_000);

        let camera = default_camera_data()[0];
        assert_eq!(camera.origin, vec4(1.5, 1.5, 3.0, 0.0));
        assert_eq!(camera.direction, vec4(0.0, 0.0, -1.0, 0.0));
        assert_eq!(camera.up, vec4(0.0, 1.0, 0.0, 0.0));
        assert_eq!(camera.right, vec4(1.0, 0.0, 0.0, 0.0));
        assert_eq!(camera.viewport, vec4(1.0, 0.0, 0.0, 0.0));

        let metadata = SvodagMetaData::new(root_index, Mat4::IDENTITY);
        assert_eq!(metadata.root_index.to_raw(), 70_000);
        assert_eq!(metadata.pad, [0; 3]);
        assert_eq!(metadata.inv_model_transform, Mat4::IDENTITY);
    }

    #[test]
    fn serialized_view_reads_node_metadata_and_children() {
        let data = [0b1010_0101_0000_0001, 3, 7, 11, 13];
        let view = SvodagView::from_slice(&data);
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
        let view = SvodagView::from_slice(&data);
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&data),
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
            SvodagView::from_slice(&supported_data),
            index(0),
        ));

        let overflow_depth = supported_depth + 1;
        let overflow_data = single_voxel([0, 0, 0], overflow_depth);
        let overflow_center = 1.0 + 0.5 / (1_u32 << overflow_depth) as f32;

        assert!(!raymarch(
            vec3(0.5, overflow_center, overflow_center),
            vec3(1.0, 0.0, 0.0),
            overflow_depth,
            SvodagView::from_slice(&overflow_data),
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
