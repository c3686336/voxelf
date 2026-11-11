use core::f32::consts::{PI, TAU};
use spirv_std::glam::{UVec2, Vec2, Vec3, vec2, vec3};
use spirv_std::image::Image2d;
use spirv_std::Sampler;
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use crate::{Hit, RenderParams, RendererMode, SvodagView, brdf, rng, trace};

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
/// `pixel` is the zero-based framebuffer coordinate, independent of film UV.
pub fn shade(
    hit: Hit,
    view: Vec3,
    pixel: UVec2,
    constants: &RenderParams,
    svodag: SvodagView,
    environment: &Image2d,
    sampler: &Sampler,
) -> Vec3 {
    match constants.renderer {
        RendererMode::Reference => reference(hit, view, pixel, constants, svodag, environment, sampler),
        RendererMode::Albedo => vec3(constants.albedo_r, constants.albedo_g, constants.albedo_b),
    }
}

fn reference(
    hit: Hit,
    view: Vec3,
    pixel: UVec2,
    constants: &RenderParams,
    svodag: SvodagView,
    environment: &Image2d,
    sampler: &Sampler,
) -> Vec3 {
    let albedo = vec3(constants.albedo_r, constants.albedo_g, constants.albedo_b);
    let epsilon = (-(constants.max_depth.min(23) as f32)).exp2() * 0.001;
    let mut radiance = Vec3::ZERO;
    let samples = constants.samples_per_pixel.max(1);
    for sample_index in 0..samples {
        let u = rng::sample_2d(pixel, constants.seed, sample_index);
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
    fn samples_are_on_hemisphere_with_correct_pdf() {
        for sample_index in 0..1024 {
            let u = rng::sample_2d(UVec2::new(3, 7), 17, sample_index);
            let (direction, pdf) = sample_hemisphere(Vec3::Y, u);
            assert!((direction.length() - 1.0).abs() < 1e-5);
            assert!(direction.y >= 0.0);
            assert!((pdf - direction.y / PI).abs() < 1e-6);
        }
    }
}
