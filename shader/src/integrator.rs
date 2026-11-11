use core::f32::consts::{PI, TAU};
use spirv_std::glam::{Vec2, Vec3, vec2, vec3};
use spirv_std::image::Image2d;
use spirv_std::Sampler;
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use crate::{Hit, RenderParams, RendererMode, SvodagView, brdf, trace};

fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn random(state: &mut u32) -> f32 {
    *state = hash(state.wrapping_add(0x9e37_79b9));
    ((*state >> 8) as f32) * (1.0 / 16_777_216.0)
}

/// Cosine-weighted hemisphere sampling, with density cos(theta)/pi.
fn sample_hemisphere(normal: Vec3, u: Vec2) -> (Vec3, f32) {
    let radius = u.x.sqrt();
    let phi = TAU * u.y;
    let z = (1.0 - u.x).sqrt();
    let helper = if normal.z.abs() < 0.999 {
        Vec3::Z
    } else {
        Vec3::X
    };
    let tangent = helper.cross(normal).normalize();
    let bitangent = normal.cross(tangent);
    let direction = tangent * (radius * phi.cos()) + bitangent * (radius * phi.sin()) + normal * z;
    (direction, z / PI)
}

pub fn environment_radiance(environment: &Image2d, sampler: &Sampler, direction: Vec3) -> Vec3 {
    let uv = vec2(
        0.5 + direction.z.atan2(direction.x) / TAU,
        0.5 - direction.y.clamp(-1.0, 1.0).asin() / PI,
    );
    let value: spirv_std::glam::Vec4 = environment.sample_by_lod(*sampler, uv, 0.0);
    vec3(value.x, value.y, value.z).max(Vec3::ZERO)
}

/// Select an integrator independently of intersection and display conversion.
pub fn shade(
    hit: Hit,
    view: Vec3,
    pixel: Vec2,
    constants: &RenderParams,
    svodag: SvodagView,
    environment: &Image2d,
    sampler: &Sampler,
) -> Vec3 {
    let albedo = vec3(constants.albedo_r, constants.albedo_g, constants.albedo_b);
    if constants.renderer == RendererMode::Albedo {
        return albedo;
    }
    reference(hit, view, pixel, constants, svodag, environment, sampler)
}

fn reference(
    hit: Hit,
    view: Vec3,
    pixel: Vec2,
    constants: &RenderParams,
    svodag: SvodagView,
    environment: &Image2d,
    sampler: &Sampler,
) -> Vec3 {
    let albedo = vec3(constants.albedo_r, constants.albedo_g, constants.albedo_b);
    let mut state = hash(
        constants.seed ^ hash(pixel.x as u32) ^ hash((pixel.y as u32).wrapping_add(0x68bc_21eb)),
    );
    let epsilon = (-(constants.max_depth.min(23) as f32)).exp2() * 0.001;
    let mut radiance = Vec3::ZERO;
    let samples = constants.samples_per_pixel.max(1);
    for _ in 0..samples {
        let u = vec2(random(&mut state), random(&mut state));
        let (light, pdf) = sample_hemisphere(hit.normal, u);
        let cosine = hit.normal.dot(light).max(0.0);
        if pdf > 0.0
            && trace(
                hit.position + hit.normal * epsilon,
                light,
                constants.max_depth,
                svodag,
                constants.root_index,
            )
            .found
                == false
        {
            let incoming = environment_radiance(environment, sampler, light);
            let f = brdf::brdf(
                albedo,
                constants.metallic,
                constants.roughness,
                hit.normal,
                view,
                light,
            );
            radiance += incoming * f * (cosine / pdf);
        }
    }
    radiance / samples as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_are_deterministic_and_on_hemisphere_with_correct_pdf() {
        let mut a = 17;
        let mut b = 17;
        for _ in 0..1024 {
            let u = vec2(random(&mut a), random(&mut a));
            assert_eq!(u, vec2(random(&mut b), random(&mut b)));
            let (direction, pdf) = sample_hemisphere(Vec3::Y, u);
            assert!((direction.length() - 1.0).abs() < 1e-5);
            assert!(direction.y >= 0.0);
            assert!((pdf - direction.y / PI).abs() < 1e-6);
        }
    }
}
