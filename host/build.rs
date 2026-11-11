use spirv_builder::{SpirvBuilder, SpirvMetadata};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = SpirvBuilder::new("../shader", "spirv-unknown-naga-wgsl");
    builder.build_script.defaults = true;
    builder.spirv_metadata = SpirvMetadata::Full;
    let result = builder.build()?;
    println!(
        "cargo:rustc-env=VOXELF_SHADER_WGSL={}",
        result.module.unwrap_single().display()
    );

    Ok(())
}
