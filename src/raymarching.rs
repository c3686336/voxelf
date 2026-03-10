use std::fs::DirEntry;

use crate::{
    svodag::{SvoNodeRef, SvodagQueryResult},
    Svodag,
};
use glam::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RaymarchResult {
    depth: u32,
    position: Vec3,
}

pub fn raymarch<'a>(
    origin: Vec3,
    direction: Vec3,
    svodag: &'a Svodag,
    root_ref: &SvoNodeRef,
) -> SvodagQueryResult {
    let dimension = 1 << svodag.depth;
    let depth_multiplier: f32 = dimension as f32;

    let direction = direction.normalize();
    let dir_recip = direction.recip();
    let min: Vec3 = vec3(0., 0., 0.);
    let max: Vec3 = vec3(1., 1., 1.);

    let mins = (min - origin) * dir_recip;
    let maxs = (max - origin) * dir_recip;

    let t1 = mins.min(maxs);
    let t2 = mins.max(maxs);

    let tmin = t1.x.min(t1.y.min(t1.z));

    let touch_point = (origin + direction * tmin).clamp(Vec3::ZERO, Vec3::ONE) * depth_multiplier;
    let touch_point_int = uvec3(
        touch_point.x.floor() as u32,
        touch_point.y.floor() as u32,
        touch_point.z.floor() as u32,
    )
    .min(UVec3::ONE * (dimension - 1));

    let step = direction.signum();
    let mut t = (touch_point - Vec3::select()) * dir_recip;

    loop {}

    // And from here, we can use the dir_recip .x, .y, .z for amanatides woo

    unimplemented!()
}
