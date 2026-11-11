//! Backend-neutral linear-radiance capture and lossless RGB32F OpenEXR storage.

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Seek, Write},
    path::Path,
};

use anyhow::{ensure, Context, Result};
use image::{
    codecs::openexr::{OpenExrDecoder, OpenExrEncoder},
    ExtendedColorType, ImageDecoder, ImageEncoder,
};

/// A renderer that writes top-to-bottom, row-major linear RGB pixels.
pub trait CaptureBackend {
    fn extent(&self) -> [u32; 2];

    /// Fill every pixel using the supplied independent frame seed.
    /// The buffer has exactly `extent()[0] * extent()[1]` pixels.
    fn render_frame(&mut self, seed: u32, pixels: &mut [[f32; 3]]) -> Result<()>;
}

/// A top-to-bottom, row-major image of linear RGB radiance.
#[derive(Debug)]
pub struct RadianceImage {
    extent: [u32; 2],
    pixels: Vec<[f32; 3]>,
}

impl RadianceImage {
    /// Validate nonzero dimensions and the matching pixel count.
    pub fn new(extent: [u32; 2], pixels: Vec<[f32; 3]>) -> Result<Self> {
        let count = pixel_count(extent)?;
        ensure!(
            pixels.len() == count,
            "radiance image extent {extent:?} requires {count} pixels, got {}",
            pixels.len()
        );
        Ok(Self { extent, pixels })
    }

    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }

    pub fn pixels(&self) -> &[[f32; 3]] {
        &self.pixels
    }

    /// Save lossless RGB32F OpenEXR, regardless of the path extension.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("creating radiance image {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        self.encode(&mut writer)
            .with_context(|| format!("encoding OpenEXR {}", path.display()))?;
        writer
            .flush()
            .with_context(|| format!("flushing OpenEXR {}", path.display()))
    }

    /// Load OpenEXR directly as linear RGB floats without tone mapping.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("opening radiance image {}", path.display()))?;
        Self::decode(BufReader::new(file))
            .with_context(|| format!("decoding OpenEXR {}", path.display()))
    }

    fn encode(&self, writer: impl Write + Seek) -> Result<()> {
        OpenExrEncoder::new(writer)
            .write_image(
                bytemuck::cast_slice(&self.pixels),
                self.extent[0],
                self.extent[1],
                ExtendedColorType::Rgb32F,
            )
            .context("encoding RGB32F radiance")
    }

    fn decode(reader: impl BufRead + Seek) -> Result<Self> {
        let decoder = OpenExrDecoder::with_alpha_preference(reader, Some(false))
            .context("reading OpenEXR metadata")?;
        let (width, height) = decoder.dimensions();
        let extent = [width, height];
        let mut pixels = vec![[0.0; 3]; pixel_count(extent)?];
        decoder
            .read_image(bytemuck::cast_slice_mut(&mut pixels))
            .context("decoding RGB32F radiance")?;
        Ok(Self { extent, pixels })
    }
}

/// Average `frames` independently seeded frames in linear space using f64 sums.
/// Seeds are `seed + frame_index`, wrapping at `u32::MAX`; zero frames is an error.
pub fn capture<B: CaptureBackend + ?Sized>(
    backend: &mut B,
    frames: u32,
    seed: u32,
) -> Result<RadianceImage> {
    ensure!(frames != 0, "capture requires at least one frame");
    let extent = backend.extent();
    let count = pixel_count(extent)?;
    let mut pixels = vec![[0.0; 3]; count];
    backend
        .render_frame(seed, &mut pixels)
        .with_context(|| format!("rendering capture frame 0 with seed {seed}"))?;
    if frames == 1 {
        return Ok(RadianceImage { extent, pixels });
    }

    let mut sums = Vec::new();
    sums.try_reserve_exact(count)
        .context("allocating capture accumulation buffer")?;
    sums.extend(pixels.iter().map(|pixel| pixel.map(f64::from)));

    for frame in 1..frames {
        let frame_seed = seed.wrapping_add(frame);
        backend
            .render_frame(frame_seed, &mut pixels)
            .with_context(|| format!("rendering capture frame {frame} with seed {frame_seed}"))?;
        for (sum, pixel) in sums.iter_mut().zip(&pixels) {
            for channel in 0..3 {
                sum[channel] += f64::from(pixel[channel]);
            }
        }
    }

    let frames = f64::from(frames);
    for (pixel, sum) in pixels.iter_mut().zip(sums) {
        for channel in 0..3 {
            pixel[channel] = (sum[channel] / frames) as f32;
        }
    }
    Ok(RadianceImage { extent, pixels })
}

fn pixel_count([width, height]: [u32; 2]) -> Result<usize> {
    ensure!(
        width != 0 && height != 0,
        "radiance image dimensions must be nonzero"
    );
    let width = usize::try_from(width).context("radiance image width overflows usize")?;
    let height = usize::try_from(height).context("radiance image height overflows usize")?;
    let count = width
        .checked_mul(height)
        .context("radiance image pixel count overflows usize")?;
    let bytes = count
        .checked_mul(std::mem::size_of::<[f32; 3]>())
        .context("radiance image byte count overflows usize")?;
    ensure!(
        bytes <= isize::MAX as usize,
        "radiance image exceeds allocation limits"
    );
    Ok(count)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    struct SeededBackend {
        seeds: Vec<u32>,
    }

    impl CaptureBackend for SeededBackend {
        fn extent(&self) -> [u32; 2] {
            [2, 1]
        }

        fn render_frame(&mut self, seed: u32, pixels: &mut [[f32; 3]]) -> Result<()> {
            self.seeds.push(seed);
            let value = match seed {
                u32::MAX => 16_777_216.0,
                0 => 1.0,
                1 => -16_777_216.0,
                _ => panic!("unexpected frame seed"),
            };
            pixels[0] = [value, value * 2.0, value * 4.0];
            pixels[1] = [value * 8.0, value * 16.0, value * 32.0];
            Ok(())
        }
    }

    #[test]
    fn averages_independent_seeds_with_f64_sums() {
        let mut backend = SeededBackend { seeds: Vec::new() };
        let image = capture(&mut backend, 3, u32::MAX).unwrap();
        assert_eq!(backend.seeds, [u32::MAX, 0, 1]);
        assert_eq!(
            image.pixels(),
            &[
                [1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0],
                [8.0 / 3.0, 16.0 / 3.0, 32.0 / 3.0],
            ]
        );
    }

    #[test]
    fn rejects_zero_frames_before_rendering() {
        let mut backend = SeededBackend { seeds: Vec::new() };
        assert!(capture(&mut backend, 0, 0).is_err());
        assert!(backend.seeds.is_empty());
    }

    #[test]
    fn exr_roundtrips_hdr_channels_and_row_order_losslessly() {
        let pixels = vec![
            [0.125, 4.0, 65_537.0],
            [-3.25, 0.333_333_34, 1.0e20],
            [f32::MIN_POSITIVE, 12_345.678, -100_000.5],
            [2.0, 0.0, 9.876_543],
            [16_777_216.0, 0.000_001, 7.5],
            [0.75, 1024.5, -0.25],
        ];
        let image = RadianceImage::new([3, 2], pixels).unwrap();
        let mut encoded = Cursor::new(Vec::new());
        image.encode(&mut encoded).unwrap();
        encoded.set_position(0);
        let decoded = RadianceImage::decode(encoded).unwrap();
        assert_eq!(decoded.extent(), [3, 2]);
        for (actual, expected) in decoded.pixels().iter().zip(image.pixels()) {
            assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }
}
