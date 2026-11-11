use super::context::GpuContext;
use anyhow::{ensure, Context, Result};
use half::f16;
use image::ImageReader;
use std::{ops::Range, path::Path};

pub(super) struct EnvironmentMap {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub extent: [u32; 2],
    pub source: String,
}

pub(super) struct SceneResources {
    pub svodag_buffer: wgpu::Buffer,
    pub capacity: usize,
    pub environment: EnvironmentMap,
}

impl SceneResources {
    pub fn new(context: &GpuContext, words: &[u32], environment: Option<&Path>) -> Result<Self> {
        let capacity = words.len().max(1);
        let svodag_buffer = Self::buffer(context, words, capacity)?;
        let environment = EnvironmentMap::load(context, environment)?;
        Ok(Self {
            svodag_buffer,
            capacity,
            environment,
        })
    }

    fn maximum_capacity(context: &GpuContext) -> usize {
        let limits = context.device.limits();
        usize::try_from(
            limits
                .max_buffer_size
                .min(u64::from(limits.max_storage_buffer_binding_size))
                / 4,
        )
        .unwrap_or(usize::MAX)
    }

    fn buffer(context: &GpuContext, words: &[u32], capacity: usize) -> Result<wgpu::Buffer> {
        ensure!(
            capacity > 0 && capacity <= Self::maximum_capacity(context),
            "SVODAG exceeds the device's storage buffer limit"
        );
        let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SVODAG words"),
            size: u64::try_from(capacity)?
                .checked_mul(4)
                .context("SVODAG size overflow")?,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !words.is_empty() {
            context
                .queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(words));
        }
        Ok(buffer)
    }

    /// Returns whether a replacement buffer requires the scene bind group to be rebuilt.
    ///
    /// `Queue::write_buffer` copies the words into CPU-side staging memory.
    /// The next `Queue::submit` schedules the GPU copies before its command buffers.
    pub fn sync(
        &mut self,
        context: &GpuContext,
        words: &[u32],
        dirty: &[Range<usize>],
    ) -> Result<bool> {
        for range in dirty {
            ensure!(
                range.start <= range.end && range.end <= words.len(),
                "SVODAG dirty range is outside the supplied snapshot"
            );
        }
        let maximum = Self::maximum_capacity(context);
        ensure!(
            words.len() <= maximum,
            "SVODAG exceeds the device's storage buffer limit"
        );
        if words.len() > self.capacity {
            let capacity = grown_capacity(self.capacity, words.len(), maximum)?;
            let buffer = Self::buffer(context, words, capacity)?;
            self.svodag_buffer = buffer;
            self.capacity = capacity;
            return Ok(true);
        }
        for range in dirty.iter().filter(|range| !range.is_empty()) {
            context.queue.write_buffer(
                &self.svodag_buffer,
                u64::try_from(range.start)? * 4,
                bytemuck::cast_slice(&words[range.clone()]),
            );
        }
        Ok(false)
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
                .map(|pixel| {
                    [pixel[0], pixel[1], pixel[2], 1.0].map(|value| f16::from_f32(value).to_bits())
                })
                .collect::<Vec<_>>();
            (pixels, extent, path.display().to_string())
        } else {
            let pixels = [
                [0.03, 0.05, 0.12, 1.0],
                [0.08, 0.14, 0.30, 1.0],
                [0.18, 0.28, 0.50, 1.0],
                [0.08, 0.12, 0.24, 1.0],
                [0.30, 0.22, 0.12, 1.0],
                [0.80, 0.55, 0.24, 1.0],
                [0.35, 0.24, 0.12, 1.0],
                [0.12, 0.09, 0.08, 1.0],
            ]
            .map(|pixel| pixel.map(|value| f16::from_f32(value).to_bits()))
            .to_vec();
            (pixels, [4, 2], "generated fallback".to_owned())
        };
        let limit = context.device.limits().max_texture_dimension_2d;
        ensure!(
            extent[0] > 0 && extent[1] > 0 && extent[0] <= limit && extent[1] <= limit,
            "Environment extent {extent:?} exceeds the device's texture limit {limit}"
        );
        let bytes_per_row = extent[0]
            .checked_mul(8)
            .context("Environment row size overflow")?;
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Environment radiance"),
            size: wgpu::Extent3d {
                width: extent[0],
                height: extent[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        context.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(extent[1]),
            },
            texture.size(),
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Environment sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Ok(Self {
            view,
            sampler,
            extent,
            source,
        })
    }
}

fn grown_capacity(current: usize, required: usize, max: usize) -> Result<usize> {
    ensure!(
        required <= max,
        "SVODAG exceeds the device's storage buffer limit"
    );
    Ok(current.saturating_mul(2).min(max).max(required).max(1))
}

#[cfg(test)]
mod tests {
    use super::grown_capacity;

    #[test]
    fn buffer_capacity_grows_geometrically_and_respects_limits() {
        assert_eq!(grown_capacity(4, 5, 100).unwrap(), 8);
        assert_eq!(grown_capacity(4, 20, 100).unwrap(), 20);
        assert_eq!(grown_capacity(60, 80, 100).unwrap(), 100);
        assert!(grown_capacity(60, 101, 100).is_err());
    }
}
