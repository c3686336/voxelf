use spirv_builder::{SpirvBuilder, SpirvMetadata};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = SpirvBuilder::new("../shader", "spirv-unknown-vulkan1.2");
    builder.build_script.defaults = true;
    builder.spirv_metadata = SpirvMetadata::Full;
    builder.build_script.env_shader_spv_path = Some(true);
    builder.capabilities.push(spirv_builder::Capability::Int8);
    builder.build()?;

    Ok(())
}
