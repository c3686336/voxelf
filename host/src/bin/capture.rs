use std::path::Path;
use voxelf::capture::capture;
use voxelf::render::RenderSettings;
use voxelf::svodag::get_example_svodag;
use voxelf::vulkan::VulkanCapture;

// Validation recipe: edit these or call the library API from a test.
const EXTENT: [u32; 2] = [640, 480];
const FRAMES: u32 = 64;
const SEED: u32 = 0;
const OUTPUT: &str = "reference.exr";

fn main() -> anyhow::Result<()> {
    let (scene, root) = get_example_svodag(7);
    let environment = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/envmap.hdr");
    let mut backend = VulkanCapture::new(
        scene.data(),
        scene.depth,
        root,
        Some(&environment),
        EXTENT,
        RenderSettings::default(),
    )?;
    let radiance = capture(&mut backend, FRAMES, SEED)?;
    radiance.save(OUTPUT)?;
    println!(
        "Saved {OUTPUT}: {}x{}, {FRAMES} averaged frames",
        EXTENT[0], EXTENT[1]
    );
    Ok(())
}
