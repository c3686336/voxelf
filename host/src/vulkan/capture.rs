use super::context::GpuContext;
use anyhow::{bail, Context, Result};
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::format::Format;
use vulkano::image::{Image, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::memory::MemoryPropertyFlags;

pub(super) fn is_hdr(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("hdr"))
}

#[derive(Clone, Copy)]
enum PixelFormat {
    Rgba8,
    Bgra8,
    Rgba16Float,
}

impl PixelFormat {
    fn bytes_per_pixel(self) -> u64 {
        match self {
            Self::Rgba8 | Self::Bgra8 => 4,
            Self::Rgba16Float => 8,
        }
    }
}

/// A recorded swapchain readback. Keep this alive until the submission has completed.
pub(super) struct FrameCapture {
    pixels: Subbuffer<[u8]>,
    extent: [u32; 2],
    format: PixelFormat,
}

impl FrameCapture {
    /// Record after the render pass and before presentation. The builder manages the image
    /// layout transitions; the acquire future must be chained into its submission.
    pub fn record(
        context: &GpuContext,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image: Arc<Image>,
    ) -> Result<Self> {
        if !image.usage().contains(ImageUsage::TRANSFER_SRC) {
            bail!("Frame capture is unavailable: the surface does not support swapchain TRANSFER_SRC usage");
        }
        let format = match image.format() {
            Format::B8G8R8A8_SRGB | Format::B8G8R8A8_UNORM => PixelFormat::Bgra8,
            Format::R8G8B8A8_SRGB | Format::R8G8B8A8_UNORM => PixelFormat::Rgba8,
            Format::R16G16B16A16_SFLOAT => PixelFormat::Rgba16Float,
            format => bail!("Frame capture does not support swapchain format {format:?}"),
        };
        let [width, height, depth] = image.extent();
        if width == 0 || height == 0 || depth != 1 {
            bail!("Invalid swapchain image extent: {width}x{height}x{depth}");
        }
        let len = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|size| size.checked_mul(format.bytes_per_pixel()))
            .context("Frame capture image size overflow")?;
        let pixels = Buffer::new_slice::<u8>(
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
            len,
        )?;
        builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, pixels.clone()))?;
        Ok(Self {
            pixels,
            extent: [width, height],
            format,
        })
    }

    /// Save untone-mapped linear radiance from an RGBA16F offscreen image.
    pub fn save_hdr(self, path: &Path) -> Result<()> {
        if !matches!(self.format, PixelFormat::Rgba16Float) {
            bail!("Linear HDR capture requires an RGBA16F render target");
        }
        let pixels = self
            .pixels
            .read()
            .context("Mapping linear frame capture buffer")?;
        let rgb: Vec<image::Rgb<f32>> = pixels
            .chunks_exact(8)
            .map(|pixel| {
                let channel = |offset| {
                    half::f16::from_bits(u16::from_ne_bytes([pixel[offset], pixel[offset + 1]]))
                        .to_f32()
                };
                image::Rgb([channel(0), channel(2), channel(4)])
            })
            .collect();
        let file = std::fs::File::create(path)
            .with_context(|| format!("Creating HDR capture {}", path.display()))?;
        let mut writer = std::io::BufWriter::new(file);
        image::codecs::hdr::HdrEncoder::new(&mut writer)
            .encode(&rgb, self.extent[0] as usize, self.extent[1] as usize)
            .with_context(|| format!("Writing linear HDR capture {}", path.display()))?;
        std::io::Write::flush(&mut writer)
            .with_context(|| format!("Flushing linear HDR capture {}", path.display()))?;
        Ok(())
    }

    /// Write after the submission's fence has completed. Vulkan's tightly packed copy
    /// starts at image coordinate (0, 0), which is the top row for this renderer.
    /// Both sRGB attachments and the renderer's explicitly sRGB-encoded UNORM output
    /// contain display-encoded bytes; do not apply the transfer function again.
    pub fn save_png(self, path: &Path) -> Result<()> {
        let pixels = self.pixels.read().context("Mapping frame capture buffer")?;
        let mut converted = Vec::new();
        let rgba = match self.format {
            PixelFormat::Rgba8 => &pixels[..],
            PixelFormat::Bgra8 => {
                converted.extend_from_slice(&pixels);
                for pixel in converted.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
                converted.as_slice()
            }
            PixelFormat::Rgba16Float => {
                converted.reserve(pixels.len() / 2);
                for bytes in pixels.chunks_exact(2) {
                    let value =
                        half::f16::from_bits(u16::from_ne_bytes([bytes[0], bytes[1]])).to_f32();
                    converted.push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
                converted.as_slice()
            }
        };
        image::save_buffer_with_format(
            path,
            rgba,
            self.extent[0],
            self.extent[1],
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .with_context(|| format!("Writing frame capture PNG to {}", path.display()))?;
        Ok(())
    }
}
