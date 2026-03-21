pub mod raymarching;
pub mod svodag;

use anyhow;
use glam::{uvec3, vec2, vec3, Vec2, Vec3};
use image::{Rgb, RgbImage};
use raymarching::raymarch;
use std::f32::consts::{PI, SQRT_2};
use std::time::Instant;
use svodag::{SvoNodeRef, Svodag};

fn main() -> anyhow::Result<()> {
    // println!("Hello, world!");

    // let mut dag = Svodag::new(2);

    // println!("{}", dag);
    // let mut root1 = dag.insert_mutate(None, uvec3(0, 0, 0), 2);
    // println!("{} {}", dag, root1);
    // root1 = dag.insert_mutate(Some(root1), uvec3(1, 0, 0), 2);
    // root1 = dag.insert_mutate(Some(root1), uvec3(0, 1, 0), 2);
    // root1 = dag.insert_mutate(Some(root1), uvec3(0, 0, 1), 2);
    // root1 = dag.insert_mutate(Some(root1), uvec3(1, 1, 0), 2);
    // root1 = dag.insert_mutate(Some(root1), uvec3(1, 0, 1), 2);
    // root1 = dag.insert_mutate(Some(root1), uvec3(0, 1, 1), 2);
    // root1 = dag.insert_mutate(Some(root1), uvec3(1, 1, 1), 2);
    // println!("{} {root1}", dag);

    let depth = 6;
    let width = 1 << depth;

    let mut dag = Svodag::new(depth);

    let mut root: Option<SvoNodeRef> = None;
    for i in 0..width {
        for j in 0..width {
            for k in 0..width {
                if (i | j | k) == i + j + k {
                    root = Some(dag.insert_mutate(root, uvec3(i, j, k), depth));
                }
            }
        }
    }

    println!("{}", dag.len());

    // ray starts inside the [1.0, 2] box
    let origin = vec3(1.05, 1.05, 1.05);
    let direction = vec3(1.0, 0.0, 0.0);

    let img_width = 1920;
    let img_height = 1080;
    let mut img = RgbImage::new(img_width, img_height);

    let camera_origin = vec3(1.5, 1.5, 3.);
    let camera_direction = vec3(0.0, 0.0, -1.0);
    let camera_up = vec3(0.0, 1.0, 0.0);
    let camera_right = camera_direction.cross(camera_up);
    let film_width = 2.0;
    let film_height = 2.0;
    let film_distance = 1.0;

    let now = Instant::now();
    for i in 0..img_width {
        for j in 0..img_height {
            let x = (i as f32) / (img_width as f32) * 1.0 - 0.5;
            let y = (j as f32) / (img_height as f32) * 1.0 - 0.5;

            let film_point = camera_direction * film_distance
                + x * camera_right * film_width
                + y * camera_up * film_height;

            let result = raymarch(camera_origin, film_point.normalize(), &dag, &root.unwrap());

            if result {
                img.put_pixel(i, j, Rgb([0, 255, 0]));
            } else {
                img.put_pixel(i, j, Rgb([0, 0, 255]));
            }
        }
    }

    let elapsed = now.elapsed();
    println!(
        "Elapsed: {}, Time per ray: {}",
        elapsed.as_secs_f32(),
        elapsed.as_secs_f32() / ((img_width * img_height) as f32)
    );

    img.save("asdf.png").unwrap();

    Ok(())
}
