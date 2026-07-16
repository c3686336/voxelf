use crate::{
    svodag::{SvoNode, SvoNodeRef},
    Svodag,
};
use glam::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RaymarchResult {
    depth: u32,
    position: Vec3,
}

fn pos_to_morton(position: Vec3 /*position in 1~2*/, depth: usize) -> usize {
    let x = (position.x.to_bits() >> (23 - depth)) & 0b1;
    let y = (position.y.to_bits() >> (23 - depth)) & 0b1;
    let z = (position.z.to_bits() >> (23 - depth)) & 0b1;

    return ((x << 2) | (y << 1) | z) as usize;
}

pub fn raymarch(origin: Vec3, direction: Vec3, svodag: &Svodag, root_ref: &SvoNodeRef) -> bool {
    // The cube is at (1, 1, 1)

    let dir_threshold = -direction.abs().clamp(Vec3::ONE * f32::EPSILON, Vec3::MAX);

    let dir_sign = direction.cmpgt(Vec3::ZERO);
    let mask: usize =
        ((dir_sign.x as usize) << 2) | ((dir_sign.y as usize) << 1) | ((dir_sign.z as usize) << 0);

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
    let mut parent = svodag.deref(root_ref);

    if parent.leaf && tmin <= tmax {
        return true;
    }
    let mut idx = 0;
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

    let mut parent_stack = Vec::<&SvoNode>::new();

    loop {
        let t2_current_cube = (position - origin) * dir_inv;
        let tmax_current_cube = t2_current_cube.min_element();

        // PUSH
        if (parent.child_mask >> (idx ^ mask)) & 1 == 1 && tmin <= tmax_current_cube {
            let child = svodag.deref(&parent.children[parent.index_of_morton(idx ^ mask)]);

            if child.leaf {
                return true;
            }

            parent_stack.push(parent);

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
                | (((z >> (23 - depth)) & 1) << 0)) as usize;
            parent = parent_stack[(depth - 1) as usize];
            parent_stack.truncate((depth - 1) as usize);
            // TODO: Instead of popping an item which will be pushed back again anyway, just set a flag.

            position.x = f32::from_bits(x >> (23 - depth) << (23 - depth));
            position.y = f32::from_bits(y >> (23 - depth) << (23 - depth));
            position.z = f32::from_bits(z >> (23 - depth) << (23 - depth));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_voxel_hit() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(0, 0, 0), depth);

        // voxel at (0,0,0) is in [1.0, 1.125]x[1.0, 1.125]x[1.0, 1.125]
        let origin = vec3(0.5, 1.05, 1.05);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(raymarch(origin, direction, &dag, &root));
    }

    #[test]
    fn test_single_voxel_miss() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(0, 0, 0), depth);

        // ray passing above the voxel
        let origin = vec3(0.5, 1.5, 1.5);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(!raymarch(origin, direction, &dag, &root));
    }

    #[test]
    fn test_negative_direction_hit() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(7, 7, 7), depth);

        // voxel at (7,7,7) is in [1.875, 2.0]x[1.875, 2.0]x[1.875, 2.0]
        let origin = vec3(2.5, 1.95, 1.95);
        let direction = vec3(-1.0, 0.0, 0.0);

        assert!(raymarch(origin, direction, &dag, &root));
    }

    #[test]
    fn test_negative_direction_miss() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(7, 7, 7), depth);

        let origin = vec3(2.5, 1.5, 1.5);
        let direction = vec3(-1.0, 0.0, 0.0);

        assert!(!raymarch(origin, direction, &dag, &root));
    }

    #[test]
    fn test_diagonal_hit() {
        let depth = 2;
        let mut dag = Svodag::new(depth);
        // depth 2, coords 0 to 3.
        let root = dag.insert_mutate(None, uvec3(1, 1, 1), depth);

        // voxel at (1,1,1) is in [1.25, 1.5]x[1.25, 1.5]x[1.25, 1.5]
        let origin = vec3(0.5, 0.5, 0.5);
        let direction = vec3(1.0, 1.0, 1.0).normalize();

        assert!(raymarch(origin, direction, &dag, &root));
    }

    #[test]
    fn test_ray_starts_inside_hit() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(0, 0, 0), depth);

        // ray starts inside the [1.0, 1.125] box
        let origin = vec3(1.05, 1.05, 1.05);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(raymarch(origin, direction, &dag, &root));
    }

    #[test]
    fn test_ray_starts_inside_volume_hit() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(7, 7, 7), depth);

        // Starts inside [1, 2]^3, before the occupied far-x voxel slab.
        let origin = vec3(1.5, 1.95, 1.95);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(raymarch(origin, direction, &dag, &root));
    }

    #[test]
    fn test_ray_starts_inside_volume_miss() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(7, 7, 7), depth);

        // Starts inside the root volume but outside the occupied y/z slab.
        let origin = vec3(1.5, 1.5, 1.5);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(!raymarch(origin, direction, &dag, &root));
    }

    #[test]
    fn test_ray_starts_inside_volume_does_not_hit_mirrored_voxel() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(0, 7, 7), depth);

        // The occupied voxel is behind the ray along x. Starting inside the root volume
        // must not report the mirrored far-x voxel as a hit.
        let origin = vec3(1.5, 1.95, 1.95);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(!raymarch(origin, direction, &dag, &root));
    }

    #[test]
    fn test_positive_direction_hit_far() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(7, 7, 7), depth); // voxel at [1.875, 2.0]

        let origin = vec3(0.5, 1.95, 1.95);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(raymarch(origin, direction, &dag, &root));
    }

    #[test]
    fn test_single_voxel_hit_serialized() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(0, 0, 0), depth);

        let (serialized, mapping) = dag.serialize();

        // voxel at (0,0,0) is in [1.0, 1.125]x[1.0, 1.125]x[1.0, 1.125]
        let origin = vec3(0.5, 1.05, 1.05);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(shader::raymarch(
            origin,
            direction,
            dag.depth,
            shader::SerializedSvodagView::from_slice(&serialized),
            mapping[dag.to_index(&root)]
        ));
    }

    #[test]
    fn test_single_voxel_miss_serialized() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(0, 0, 0), depth);

        let (serialized, mapping) = dag.serialize();

        // ray passing above the voxel
        let origin = vec3(0.5, 1.5, 1.5);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(!shader::raymarch(
            origin,
            direction,
            dag.depth,
            shader::SerializedSvodagView::from_slice(&serialized),
            mapping[dag.to_index(&root)]
        ));
    }

    #[test]
    fn test_negative_direction_hit_serialized() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(7, 7, 7), depth);

        let (serialized, mapping) = dag.serialize();

        // voxel at (7,7,7) is in [1.875, 2.0]x[1.875, 2.0]x[1.875, 2.0]
        let origin = vec3(2.5, 1.95, 1.95);
        let direction = vec3(-1.0, 0.0, 0.0);

        assert!(shader::raymarch(
            origin,
            direction,
            dag.depth,
            shader::SerializedSvodagView::from_slice(&serialized),
            mapping[dag.to_index(&root)]
        ));
    }

    #[test]
    fn test_negative_direction_miss_serialized() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(7, 7, 7), depth);

        let (serialized, mapping) = dag.serialize();

        let origin = vec3(2.5, 1.5, 1.5);
        let direction = vec3(-1.0, 0.0, 0.0);

        assert!(!shader::raymarch(
            origin,
            direction,
            dag.depth,
            shader::SerializedSvodagView::from_slice(&serialized),
            mapping[dag.to_index(&root)]
        ));
    }

    #[test]
    fn test_diagonal_hit_serialized() {
        let depth = 2;
        let mut dag = Svodag::new(depth);
        // depth 2, coords 0 to 3.
        let root = dag.insert_mutate(None, uvec3(1, 1, 1), depth);

        let (serialized, mapping) = dag.serialize();

        // voxel at (1,1,1) is in [1.25, 1.5]x[1.25, 1.5]x[1.25, 1.5]
        let origin = vec3(0.5, 0.5, 0.5);
        let direction = vec3(1.0, 1.0, 1.0).normalize();

        assert!(shader::raymarch(
            origin,
            direction,
            dag.depth,
            shader::SerializedSvodagView::from_slice(&serialized),
            mapping[dag.to_index(&root)]
        ));
    }

    #[test]
    fn test_ray_starts_inside_hit_serialized() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(0, 0, 0), depth);

        let (serialized, mapping) = dag.serialize();

        // ray starts inside the [1.0, 1.125] box
        let origin = vec3(1.05, 1.05, 1.05);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(shader::raymarch(
            origin,
            direction,
            dag.depth,
            shader::SerializedSvodagView::from_slice(&serialized),
            mapping[dag.to_index(&root)]
        ));
    }

    #[test]
    fn test_ray_starts_inside_volume_hit_serialized() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(7, 7, 7), depth);

        let (serialized, mapping) = dag.serialize();

        let origin = vec3(1.5, 1.95, 1.95);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(shader::raymarch(
            origin,
            direction,
            dag.depth,
            shader::SerializedSvodagView::from_slice(&serialized),
            mapping[dag.to_index(&root)]
        ));
    }

    #[test]
    fn test_ray_starts_inside_volume_miss_serialized() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(7, 7, 7), depth);

        let (serialized, mapping) = dag.serialize();

        let origin = vec3(1.5, 1.5, 1.5);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(!shader::raymarch(
            origin,
            direction,
            dag.depth,
            shader::SerializedSvodagView::from_slice(&serialized),
            mapping[dag.to_index(&root)]
        ));
    }

    #[test]
    fn test_ray_starts_inside_volume_does_not_hit_mirrored_voxel_serialized() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(0, 7, 7), depth);

        let (serialized, mapping) = dag.serialize();

        let origin = vec3(1.5, 1.95, 1.95);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(!shader::raymarch(
            origin,
            direction,
            dag.depth,
            shader::SerializedSvodagView::from_slice(&serialized),
            mapping[dag.to_index(&root)]
        ));
    }

    #[test]
    fn test_positive_direction_hit_far_serialized() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let root = dag.insert_mutate(None, uvec3(7, 7, 7), depth); // voxel at [1.875, 2.0]

        let (serialized, mapping) = dag.serialize();

        let origin = vec3(0.5, 1.95, 1.95);
        let direction = vec3(1.0, 0.0, 0.0);

        assert!(shader::raymarch(
            origin,
            direction,
            dag.depth,
            shader::SerializedSvodagView::from_slice(&serialized),
            mapping[dag.to_index(&root)]
        ));
    }

    #[test]
    fn multi_branch_traversal_matches_serialized_traversal_after_reentry() {
        let depth = 3;
        let mut dag = Svodag::new(depth);
        let mut root = None;

        for position in [uvec3(6, 6, 3), uvec3(3, 7, 0), uvec3(1, 6, 0)] {
            root = Some(dag.insert_mutate(root, position, depth));
        }

        let root = root.unwrap();
        let (serialized, mapping) = dag.serialize();
        let serialized_view = shader::SerializedSvodagView::from_slice(&serialized);
        let serialized_root = mapping[dag.to_index(&root)];

        for (origin, direction, expected) in [
            (vec3(2.5, 1.8125, 1.0625), vec3(-1.0, 0.0, 0.0), true),
            (vec3(0.5, 1.8125, 1.4375), vec3(1.0, 0.0, 0.0), true),
            (vec3(0.5, 1.6875, 1.0625), vec3(1.0, 0.0, 0.0), false),
        ] {
            let cpu_result = raymarch(origin, direction, &dag, &root);
            let serialized_result =
                shader::raymarch(origin, direction, depth, serialized_view, serialized_root);

            assert_eq!(cpu_result, expected, "unexpected CPU result for {origin}");
            assert_eq!(
                serialized_result, cpu_result,
                "serialized traversal diverged for {origin}"
            );
        }

        dag.release(root);
    }

    use crate::svodag::Svodag;
    use glam::{uvec3, vec3, Vec3};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_raymarch_hits_inserted_voxel(
            depth in 1u32..4,
            x in 0u32..16,
            y in 0u32..16,
            z in 0u32..16,
            tx in 0.1f32..0.9,
            ty in 0.1f32..0.9,
            tz in 0.1f32..0.9,
            ox in -1.0f32..3.0,
            oy in -1.0f32..3.0,
            oz in -1.0f32..3.0,
        ) {
            let size = 1 << depth;
            let px = x % size;
            let py = y % size;
            let pz = z % size;

            let mut dag = Svodag::new(depth);
            let root = dag.insert_mutate(None, uvec3(px, py, pz), depth);

            let voxel_min = vec3(
                1.0 + px as f32 / size as f32,
                1.0 + py as f32 / size as f32,
                1.0 + pz as f32 / size as f32,
            );
            let voxel_size = 1.0 / size as f32;

            let target = voxel_min + vec3(tx, ty, tz) * voxel_size;

            let mut origin = vec3(ox, oy, oz);
            // force origin outside the root bounding box [1.0, 2.0]^3
            if origin.cmpge(Vec3::ONE).all() && origin.cmple(Vec3::ONE * 2.0).all() {
                 origin.x = 0.5;
            }

            let direction = (target - origin).normalize();

            println!("Origin: {}", origin);
            println!("Target: {}", target);
            println!("Target Voxel: {}", uvec3(px, py, pz));
            println!("Depth: {}", depth);
            println!("Direction: {}", direction);


            // The ray is pointing directly at a point inside the voxel from outside.
            // It should hit the voxel.
            prop_assert!(raymarch(origin, direction, &dag, &root));
        }

        #[test]
        fn prop_raymarch_misses_when_pointing_away(
            depth in 1u32..3,
            x in 0u32..4, y in 0u32..4, z in 0u32..4,
            ox in -2.0f32..0.5,
            oy in -2.0f32..0.5,
            oz in -2.0f32..0.5,
            dx in -1.0f32..-0.1,
            dy in -1.0f32..-0.1,
            dz in -1.0f32..-0.1,
        ) {
            let size = 1 << depth;
            let mut dag = Svodag::new(depth);
            let root = dag.insert_mutate(None, uvec3(x % size, y % size, z % size), depth);

            let origin = vec3(ox, oy, oz);
            let direction = vec3(dx, dy, dz).normalize();

            println!("Origin: {}", origin);
            println!("Target Voxel: {}", uvec3(x % size, y % size, z % size));
            println!("Depth: {}", depth);

            // The ray originates below/left/behind the cube and points away from it.
            // It should never hit.
            prop_assert!(!raymarch(origin, direction, &dag, &root));
        }

        #[test]
        fn prop_raymarch_hits_inserted_voxel_serialized(
            depth in 1u32..4,
            x in 0u32..16,
            y in 0u32..16,
            z in 0u32..16,
            tx in 0.1f32..0.9,
            ty in 0.1f32..0.9,
            tz in 0.1f32..0.9,
            ox in -1.0f32..3.0,
            oy in -1.0f32..3.0,
            oz in -1.0f32..3.0,
        ) {
            let size = 1 << depth;
            let px = x % size;
            let py = y % size;
            let pz = z % size;

            let mut dag = Svodag::new(depth);
            let root = dag.insert_mutate(None, uvec3(px, py, pz), depth);

            let voxel_min = vec3(
                1.0 + px as f32 / size as f32,
                1.0 + py as f32 / size as f32,
                1.0 + pz as f32 / size as f32,
            );
            let voxel_size = 1.0 / size as f32;

            let target = voxel_min + vec3(tx, ty, tz) * voxel_size;

            let mut origin = vec3(ox, oy, oz);
            // force origin outside the root bounding box [1.0, 2.0]^3
            if origin.cmpge(Vec3::ONE).all() && origin.cmple(Vec3::ONE * 2.0).all() {
                 origin.x = 0.5;
            }

            let direction = (target - origin).normalize();

            let (serialized, mapping) = dag.serialize();

            prop_assert!(shader::raymarch(
                origin,
                direction,
                dag.depth,
                shader::SerializedSvodagView::from_slice(&serialized),
                mapping[dag.to_index(&root)]
            ));
        }

        #[test]
        fn prop_raymarch_misses_when_pointing_away_serialized(
            depth in 1u32..3,
            x in 0u32..4, y in 0u32..4, z in 0u32..4,
            ox in -2.0f32..0.5,
            oy in -2.0f32..0.5,
            oz in -2.0f32..0.5,
            dx in -1.0f32..-0.1,
            dy in -1.0f32..-0.1,
            dz in -1.0f32..-0.1,
        ) {
            let size = 1 << depth;
            let mut dag = Svodag::new(depth);
            let root = dag.insert_mutate(None, uvec3(x % size, y % size, z % size), depth);

            let origin = vec3(ox, oy, oz);
            let direction = vec3(dx, dy, dz).normalize();

            let (serialized, mapping) = dag.serialize();

            prop_assert!(!shader::raymarch(
                origin,
                direction,
                dag.depth,
                shader::SerializedSvodagView::from_slice(&serialized),
                mapping[dag.to_index(&root)]
            ));
        }
    }
}
