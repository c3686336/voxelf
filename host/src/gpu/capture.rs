use super::{context::GpuContext, pass::RaymarchPass, resources::SceneResources};
use crate::{capture::CaptureBackend, render::RenderSettings};
use anyhow::{ensure, Context, Result};
use shader::SvoIndex;
use std::{path::Path, sync::mpsc};

/// A reusable, windowless renderer that reads back linear scene radiance.
pub struct WgpuCapture {
    context: GpuContext,
    _scene: SceneResources,
    raymarch_pass: RaymarchPass,
    texture: wgpu::Texture,
    target: wgpu::TextureView,
    staging: wgpu::Buffer,
    extent: [u32; 2],
    pixel_count: usize,
    padded_bytes_per_row: u32,
    map_sender: mpsc::SyncSender<Result<(), wgpu::BufferAsyncError>>,
    map_receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl WgpuCapture {
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
        let pixel_count = usize::try_from(u64::from(extent[0]) * u64::from(extent[1]))
            .context("Capture image size overflow")?;
        let bytes_per_row = extent[0]
            .checked_mul(16)
            .context("Capture row size overflow")?;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = bytes_per_row
            .checked_add(alignment - 1)
            .context("Capture row size overflow")?
            / alignment
            * alignment;
        let staging_size = u64::from(padded_bytes_per_row) * u64::from(extent[1]);
        usize::try_from(staging_size).context("Capture readback size overflow")?;

        let context = GpuContext::new(GpuContext::instance(), None)?;
        let limits = context.device.limits();
        ensure!(
            extent[0] <= limits.max_texture_dimension_2d
                && extent[1] <= limits.max_texture_dimension_2d,
            "Capture extent {extent:?} exceeds the device's texture limit {}",
            limits.max_texture_dimension_2d
        );
        ensure!(
            staging_size <= limits.max_buffer_size,
            "Capture readback exceeds the device's buffer limit"
        );
        let format = wgpu::TextureFormat::Rgba32Float;
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        ensure!(
            context
                .adapter
                .get_texture_format_features(format)
                .allowed_usages
                .contains(usage),
            "The adapter cannot render and read back RGBA32Float radiance"
        );
        let scene = SceneResources::new(&context, words, environment)?;
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Capture linear radiance"),
            size: wgpu::Extent3d {
                width: extent[0],
                height: extent[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let target = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Capture readback"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let raymarch_pass = RaymarchPass::new(&context, &scene, format)?;
        let constants = shader::RenderParams::new(
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
        raymarch_pass.write_frame(
            &context,
            settings.camera,
            extent[0] as f32 / extent[1] as f32,
            constants,
        );
        let (map_sender, map_receiver) = mpsc::sync_channel(1);
        Ok(Self {
            context,
            _scene: scene,
            raymarch_pass,
            texture,
            target,
            staging,
            extent,
            pixel_count,
            padded_bytes_per_row,
            map_sender,
            map_receiver,
        })
    }
}

impl CaptureBackend for WgpuCapture {
    fn extent(&self) -> [u32; 2] {
        self.extent
    }

    fn render_frame(&mut self, seed: u32, pixels: &mut [[f32; 3]]) -> Result<()> {
        ensure!(
            pixels.len() == self.pixel_count,
            "Capture pixel buffer has the wrong length"
        );
        self.raymarch_pass.write_seed(&self.context, seed);
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Capture frame"),
                });
        self.raymarch_pass.record(&mut encoder, &self.target, None);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.extent[1]),
                },
            },
            self.texture.size(),
        );
        self.context.queue.submit([encoder.finish()]);
        let slice = self.staging.slice(..);
        let sender = self.map_sender.clone();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // A dropped receiver means the capture itself has already been dropped.
            let _ = sender.send(result);
        });
        let _ = self.context.device.poll(wgpu::Maintain::Wait);
        self.map_receiver
            .recv()
            .context("Waiting for capture readback")?
            .context("Mapping capture radiance")?;
        {
            let mapped = slice.get_mapped_range();
            let row_bytes = self.extent[0] as usize * 16;
            for (source, target) in mapped
                .chunks_exact(self.padded_bytes_per_row as usize)
                .zip(pixels.chunks_exact_mut(self.extent[0] as usize))
            {
                // Texture row zero is the top row; discard row padding and alpha.
                let rgba: &[f32] = bytemuck::cast_slice(&source[..row_bytes]);
                for (rgb, rgba) in target.iter_mut().zip(rgba.chunks_exact(4)) {
                    rgb.copy_from_slice(&rgba[..3]);
                }
            }
        }
        self.staging.unmap();
        Ok(())
    }
}
