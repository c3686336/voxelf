use super::context::GpuContext;
use anyhow::{Context, Result};
use half::f16;
use image::ImageReader;
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::{self, GpuFuture};

pub(super) struct EnvironmentMap {
    _image: Arc<Image>,
    pub view: Arc<ImageView>,
    pub sampler: Arc<Sampler>,
    pub extent: [u32; 2],
    pub source: String,
}

pub(super) struct SceneResources {
    pub svodag_buffer: Subbuffer<[u32]>,
    pub environment: EnvironmentMap,
    _metadata_buffer: Subbuffer<[shader::SvodagMetaData]>,
}

impl SceneResources {
    pub fn new(
        context: &GpuContext,
        serialized_svodag: &[u32],
        environment_path: Option<&Path>,
    ) -> Result<Self> {
        let svodag_buffer =
            Self::svodag_buffer(context, serialized_svodag, serialized_svodag.len().max(1))?;
        let metadata_buffer = Buffer::new_slice(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            10,
        )?;
        let environment = EnvironmentMap::load(context, environment_path)?;

        Ok(Self {
            svodag_buffer,
            environment,
            _metadata_buffer: metadata_buffer,
        })
    }

    pub fn svodag_buffer(
        context: &GpuContext,
        words: &[u32],
        capacity: usize,
    ) -> Result<Subbuffer<[u32]>> {
        let buffer = Buffer::new_slice(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            u64::try_from(capacity)?,
        )?;
        if !words.is_empty() {
            buffer.write()?[..words.len()].copy_from_slice(words);
        }
        Ok(buffer)
    }
}

impl EnvironmentMap {
    fn load(context: &GpuContext, path: Option<&Path>) -> Result<Self> {
        let (pixels, extent, source) = if let Some(path) = path {
            let image = ImageReader::open(path)
                .with_context(|| format!("failed to open environment map {}", path.display()))?
                .with_guessed_format()?
                .decode()
                .with_context(|| format!("failed to decode environment map {}", path.display()))?
                .into_rgb32f();
            let extent = [image.width(), image.height()];
            let pixels = image
                .pixels()
                .map(|pixel| [pixel[0], pixel[1], pixel[2], 1.0])
                .collect::<Vec<_>>();
            (pixels, extent, path.display().to_string())
        } else {
            let pixels = vec![
                [0.03, 0.05, 0.12, 1.0],
                [0.08, 0.14, 0.30, 1.0],
                [0.18, 0.28, 0.50, 1.0],
                [0.08, 0.12, 0.24, 1.0],
                [0.30, 0.22, 0.12, 1.0],
                [0.80, 0.55, 0.24, 1.0],
                [0.35, 0.24, 0.12, 1.0],
                [0.12, 0.09, 0.08, 1.0],
            ];
            (pixels, [4, 2], "generated fallback".to_owned())
        };

        let pixels = pixels.into_iter().map(|pixel| pixel.map(f16::from_f32));
        let staging = Buffer::from_iter(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            pixels,
        )?;
        let image = Image::new(
            context.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_SFLOAT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;
        let mut builder = AutoCommandBufferBuilder::primary(
            context.command_buffer_allocator.clone(),
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone()))?;
        let command_buffer = builder.build()?;
        sync::now(context.device.clone())
            .then_execute(context.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        let view = ImageView::new_default(image.clone())?;
        let sampler = Sampler::new(
            context.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [
                    SamplerAddressMode::Repeat,
                    SamplerAddressMode::ClampToEdge,
                    SamplerAddressMode::ClampToEdge,
                ],
                ..Default::default()
            },
        )?;

        Ok(Self {
            _image: image,
            view,
            sampler,
            extent,
            source,
        })
    }
}
