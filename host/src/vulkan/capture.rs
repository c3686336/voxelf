use super::context::GpuContext;
use super::frame::CameraBuffer;
use super::passes::RaymarchPass;
use super::resources::SceneResources;
use crate::capture::CaptureBackend;
use crate::render::RenderSettings;
use anyhow::{ensure, Context, Result};
use shader::SvoIndex;
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageUsage, SampleCount};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo};
use vulkano::sync::{self, GpuFuture};

/// A reusable offscreen renderer with fixed scene and render settings.
///
/// No surface, swapchain, window or GUI is created. Readback preserves the
/// shader's linear radiance without display encoding or tone mapping.
pub struct VulkanCapture {
    context: GpuContext,
    _scene: SceneResources,
    _camera_buffer: Subbuffer<[CameraBuffer]>,
    raymarch_pass: RaymarchPass,
    image: Arc<Image>,
    framebuffer: Arc<Framebuffer>,
    staging: Subbuffer<[f32]>,
    extent: [u32; 2],
    pixel_count: usize,
    constants: shader::PushConsts,
}

impl VulkanCapture {
    pub fn new(
        words: &[u32],
        depth: u32,
        root: SvoIndex,
        environment: Option<&Path>,
        extent: [u32; 2],
        settings: RenderSettings,
    ) -> Result<Self> {
        ensure!(
            extent[0] > 0 && extent[1] > 0,
            "Capture extent must be nonzero"
        );
        ensure!(
            (1..=1024).contains(&settings.samples_per_pixel),
            "Samples per pixel must be within 1..=1024"
        );
        let pixel_count = u64::from(extent[0]) * u64::from(extent[1]);
        let channel_count = pixel_count
            .checked_mul(4)
            .context("Capture image size overflow")?;
        let pixel_count = usize::try_from(pixel_count).context("Capture image size overflow")?;
        let context = GpuContext::new_headless()?;
        let scene = SceneResources::new(&context, words, environment)?;
        let camera_buffer = Buffer::from_iter(
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
            [CameraBuffer::from_view(
                settings.camera,
                extent[0] as f32 / extent[1] as f32,
            )],
        )?;
        let image = Image::new(
            context.memory_allocator.clone(),
            ImageCreateInfo {
                format: Format::R32G32B32A32_SFLOAT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )?;
        let render_pass = vulkano::single_pass_renderpass!(
            context.device.clone(),
            attachments: {
                color: {
                    format: Format::R32G32B32A32_SFLOAT,
                    samples: SampleCount::Sample1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: { color: [color], depth_stencil: {} },
        )?;
        let framebuffer = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![ImageView::new_default(image.clone())?],
                ..Default::default()
            },
        )?;
        let raymarch_pass = RaymarchPass::new(
            &context,
            &scene,
            std::slice::from_ref(&camera_buffer),
            render_pass,
        )?;
        let staging = Buffer::new_slice::<f32>(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter {
                    required_flags: MemoryPropertyFlags::HOST_VISIBLE,
                    preferred_flags: MemoryPropertyFlags::HOST_CACHED,
                    ..Default::default()
                },
                ..Default::default()
            },
            channel_count,
        )?;
        let constants = shader::PushConsts::new(
            depth,
            root,
            settings.material.albedo,
            settings.material.metallic,
            settings.material.roughness,
            false,
            settings.renderer.shader_value(),
            settings.samples_per_pixel,
            0,
            true,
        );

        Ok(Self {
            context,
            _scene: scene,
            _camera_buffer: camera_buffer,
            raymarch_pass,
            image,
            framebuffer,
            staging,
            extent,
            pixel_count,
            constants,
        })
    }
}

impl CaptureBackend for VulkanCapture {
    fn extent(&self) -> [u32; 2] {
        self.extent
    }

    fn render_frame(&mut self, seed: u32, pixels: &mut [[f32; 3]]) -> Result<()> {
        ensure!(
            pixels.len() == self.pixel_count,
            "Capture pixel buffer has the wrong length"
        );
        let mut builder = AutoCommandBufferBuilder::primary(
            self.context.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        let mut constants = self.constants;
        constants.seed = seed;
        self.raymarch_pass.record(
            &mut builder,
            self.framebuffer.clone(),
            Viewport {
                offset: [0.0, 0.0],
                extent: [self.extent[0] as f32, self.extent[1] as f32],
                depth_range: 0.0..=1.0,
            },
            0,
            constants,
            None,
        )?;
        builder.end_render_pass(Default::default())?;
        builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            self.image.clone(),
            self.staging.clone(),
        ))?;
        let command_buffer = builder.build()?;
        sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        // A tightly packed Vulkan copy starts at image coordinate (0, 0),
        // the top row for this renderer. Alpha is not part of the capture API.
        let mapped = self.staging.read().context("Mapping capture radiance")?;
        for (rgb, rgba) in pixels.iter_mut().zip(mapped.chunks_exact(4)) {
            rgb.copy_from_slice(&rgba[..3]);
        }
        Ok(())
    }
}
