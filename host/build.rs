use spirv_builder::SpirvBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Point the builder to the path of your shader crate
    SpirvBuilder::new("../shader", "spirv-unknown-vulkan1.1")
        .build()?;
        
    Ok(())
}
