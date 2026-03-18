use crate::{
    svodag::{SvoNode, SvoNodeRef, SvodagQueryResult},
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
    // The cube is at (1, 1, 1):w

    let max_depth = svodag.depth;
    let dimension = 1 << max_depth;

    let dir_threshold = -direction
        .abs()
        .clamp(Vec3::ONE * 1.0 / dimension as f32, Vec3::MAX);

    let dir_sign = direction.cmpgt(Vec3::ZERO);
    let mask: usize =
        ((dir_sign.x as usize) << 2) | ((dir_sign.y as usize) << 1) | ((dir_sign.z as usize) << 0);

    let dir_inv = dir_threshold.recip();

    let mut origin = origin;
    if dir_sign.x {
        origin.x = 1.5 - origin.x;
    }
    if dir_sign.y {
        origin.y = 1.5 - origin.y;
    }
    if dir_sign.z {
        origin.z = 1.5 - origin.z;
    }
    let origin = origin;

    let t1 = (Vec3::ONE - origin) * dir_inv;
    let t2 = (2.0 * Vec3::ONE - origin) * dir_inv;

    let mut tmin = t2.max_element(); // Because of the mirroring
    let tmax = t1.min_element(); // Because of the mirroring

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
    let mut idx_stack = Vec::<usize>::new();

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
            idx_stack.push(idx);

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

            println!("PUSH {}", position);

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

        println!("ADVANCE {}", position);

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

            if position.cmple(Vec3::ONE).any() {
                return false;
            }

            let depth = (diff_bits << 9).leading_zeros() + 1;
            scale = (-(depth as f32)).exp2();

            let x = position.x.to_bits();
            let y = position.y.to_bits();
            let z = position.z.to_bits();

            idx = ((((x >> (22 - depth)) & 1) << 2)
                | (((y >> (22 - depth)) & 1) << 1)
                | (((z >> (22 - depth)) & 1) << 0)) as usize;
            parent = parent_stack[(depth - 1) as usize];
            parent_stack.truncate((depth - 1) as usize);
            // TODO: Instead of popping an item which will be pushed back again anyway, just set a flag.

            position.x = f32::from_bits(x >> (23 - depth) << (23 - depth));
            position.y = f32::from_bits(y >> (23 - depth) << (23 - depth));
            position.z = f32::from_bits(z >> (23 - depth) << (23 - depth));

            println!("POP {}", position);
        }
    }
}
