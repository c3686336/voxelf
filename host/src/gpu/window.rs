use super::context::GpuContext;
use super::pass::RaymarchPass;
use super::resources::SceneResources;
use crate::render::{CameraView, RendererMode};
use anyhow::{Context, Result};
use shader::SvoIndex;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

const GPU_TIMING_SAMPLE_COUNT: usize = 240;

struct Material {
    albedo_srgb: [u8; 3],
    metallic: f32,
    roughness: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo_srgb: [255, 0, 0],
            metallic: 0.0,
            roughness: 0.5,
        }
    }
}

impl Material {
    fn linear_albedo(&self) -> [f32; 3] {
        let [r, g, b] = self.albedo_srgb;
        let [r, g, b, _] = egui::Rgba::from(egui::Color32::from_rgb(r, g, b)).to_array();
        [r, g, b]
    }
}

struct PendingScene {
    words: Vec<u32>,
    environment: Option<PathBuf>,
}

pub struct WgpuRenderer {
    pending: Option<PendingScene>,
    state: Option<WindowState>,
    svodag_depth: u32,
    material: Material,
    renderer: RendererMode,
    samples_per_pixel: u32,
    seed: u32,
}

impl WgpuRenderer {
    pub fn new(
        _event_loop: &EventLoop<()>,
        words: &[u32],
        svodag_depth: u32,
        environment: Option<&Path>,
    ) -> Result<Self> {
        // Adapter selection must wait until winit can create a compatible surface.
        Ok(Self {
            pending: Some(PendingScene {
                words: words.to_vec(),
                environment: environment.map(Path::to_path_buf),
            }),
            state: None,
            svodag_depth,
            material: Material::default(),
            renderer: RendererMode::Reference,
            samples_per_pixel: 16,
            seed: 0,
        })
    }

    /// Uploads edits only after the previous submitted frame has completed.
    pub fn sync_svodag(&mut self, words: &[u32], dirty: &[Range<usize>]) -> Result<()> {
        for range in dirty {
            anyhow::ensure!(
                range.start <= range.end && range.end <= words.len(),
                "SVODAG dirty range is outside the supplied snapshot"
            );
        }
        if let Some(state) = &mut self.state {
            if state.scene.sync(&state.context, words, dirty)? {
                state.raymarch.rebind_scene(&state.context, &state.scene);
            }
        } else if let Some(pending) = &mut self.pending {
            if pending.words.len() != words.len() {
                pending.words.clear();
                pending.words.extend_from_slice(words);
            } else {
                for range in dirty {
                    pending.words[range.clone()].copy_from_slice(&words[range.clone()]);
                }
            }
        }
        Ok(())
    }

    pub fn resumed(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        if let Some(state) = &mut self.state {
            state.needs_configure = true;
            state.window.request_redraw();
        } else {
            let pending = self.pending.as_ref().context("Missing initial scene")?;
            let state = WindowState::new(event_loop, pending)?;
            self.state = Some(state);
            self.pending = None;
        }
        Ok(())
    }

    pub fn resized(&mut self) -> Result<()> {
        if let Some(state) = &mut self.state {
            state.needs_configure = true;
        }
        Ok(())
    }

    pub fn window_event(&mut self, event: &WindowEvent) -> bool {
        self.state.as_mut().is_some_and(|state| {
            let response = state.egui_state.on_window_event(&state.window, event);
            if response.repaint {
                state.window.request_redraw();
            }
            response.consumed
        })
    }

    pub fn request_redraw(&self) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }

    pub fn redraw(&mut self, roots: &[SvoIndex], camera: CameraView, fps: f32) -> Result<bool> {
        let Some(&root) = roots.first() else {
            return Ok(false);
        };
        let state = self.state.as_mut().context("Renderer is not resumed")?;
        if !state.configure()? {
            return Ok(false);
        }
        let frame = match state.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                state.needs_configure = true;
                return Ok(false);
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        state.needs_configure |= frame.suboptimal;
        let target = frame.texture.create_view(&Default::default());
        let gpu_timing = state.timing.as_ref().and_then(GpuTiming::summary);
        let raw_input = state.egui_state.take_egui_input(&state.window);
        let output = state.egui_context.run(raw_input, |ctx| {
            egui::Window::new("Voxelf")
                .default_pos([12.0, 12.0])
                .show(ctx, |ui| {
                    ui.heading("Voxelf");
                    ui.label(format!("SVoDAG depth: {}", self.svodag_depth));
                    ui.label(format!("Root objects: {}", roots.len()));
                    ui.label(format!(
                        "Camera: {:.2}, {:.2}, {:.2}",
                        camera.origin[0], camera.origin[1], camera.origin[2]
                    ));
                    ui.label(format!("Environment: {}", state.scene.environment.source));
                    let extent = state.scene.environment.extent;
                    ui.label(format!("Environment size: {}x{}", extent[0], extent[1]));
                    ui.label(format!("FPS: {:.0}", fps));
                    if let Some(timing) = &gpu_timing {
                        ui.label(format!("GPU draw: {:.2} ms", timing.latest_ms));
                        ui.label(format!(
                            "Mean/P50/P95/Max: {:.2}/{:.2}/{:.2}/{:.2} ms",
                            timing.mean_ms, timing.p50_ms, timing.p95_ms, timing.max_ms
                        ));
                        ui.label(format!("GPU-only capacity: {:.0} FPS", 1000.0 / timing.mean_ms));
                    } else if state.timing.is_some() {
                        ui.label("GPU draw: waiting for first sample");
                    } else {
                        ui.label("GPU draw: unavailable (timestamp queries unsupported)");
                    }
                    ui.separator();
                    ui.heading("Renderer");
                    ui.radio_value(
                        &mut self.renderer,
                        RendererMode::Reference,
                        "Reference direct illumination",
                    );
                    ui.radio_value(&mut self.renderer, RendererMode::Albedo, "Albedo debug");
                    ui.add(egui::Slider::new(&mut self.samples_per_pixel, 1..=1024).text("Samples per pixel"));
                    ui.separator();
                    ui.heading("Material");
                    ui.horizontal(|ui| {
                        ui.label("Albedo");
                        ui.color_edit_button_srgb(&mut self.material.albedo_srgb);
                    });
                    ui.add(egui::Slider::new(&mut self.material.metallic, 0.0..=1.0).text("Metallic"));
                    ui.add(egui::Slider::new(&mut self.material.roughness, 0.0..=1.0).text("Roughness"));
                    ui.separator();
                    ui.label("W/A/S/D move, Q/E rise/drop, left mouse drag looks around. Z removes / X adds voxel at view center.");
                });
        });
        state
            .egui_state
            .handle_platform_output(&state.window, output.platform_output);
        let paint_jobs = state
            .egui_context
            .tessellate(output.shapes, output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [state.config.width, state.config.height],
            pixels_per_point: output.pixels_per_point,
        };
        for (id, delta) in &output.textures_delta.set {
            state.egui_renderer.update_texture(
                &state.context.device,
                &state.context.queue,
                *id,
                delta,
            );
        }
        state.raymarch.write_frame(
            &state.context,
            camera,
            state.config.width as f32 / state.config.height as f32,
            shader::RenderParams::new(
                self.svodag_depth,
                root,
                self.material.linear_albedo(),
                self.material.metallic,
                self.material.roughness,
                !state.config.format.is_srgb(),
                self.renderer,
                self.samples_per_pixel,
                self.seed,
                false,
            ),
        );
        let mut encoder =
            state
                .context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Voxelf frame"),
                });
        let gui_commands = state.egui_renderer.update_buffers(
            &state.context.device,
            &state.context.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );
        let timestamps = state
            .timing
            .as_ref()
            .map(|timing| wgpu::RenderPassTimestampWrites {
                query_set: &timing.query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            });
        state.raymarch.record(&mut encoder, &target, timestamps);
        {
            // The overlay must load, not clear, the completed scene attachment.
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Voxelf GUI"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            state.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }
        if let Some(timing) = &state.timing {
            encoder.resolve_query_set(&timing.query_set, 0..2, &timing.resolve, 0);
            encoder.copy_buffer_to_buffer(&timing.resolve, 0, &timing.readback, 0, 16);
        }
        let submission = state.context.queue.submit(
            gui_commands
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        if let Some(timing) = &state.timing {
            let sender = timing.sender.clone();
            timing
                .readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
        }
        state.window.pre_present_notify();
        frame.present();
        // One in-flight frame keeps camera/storage writes and subsequent scene edits safe.
        state
            .context
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
        self.seed = self.seed.wrapping_add(1);
        for id in &output.textures_delta.free {
            state.egui_renderer.free_texture(id);
        }
        if let Some(timing) = &mut state.timing {
            timing.finish()?;
        }
        Ok(true)
    }
}

struct WindowState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    context: GpuContext,
    scene: SceneResources,
    raymarch: RaymarchPass,
    config: wgpu::SurfaceConfiguration,
    needs_configure: bool,
    egui_context: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    timing: Option<GpuTiming>,
}

impl WindowState {
    fn new(event_loop: &ActiveEventLoop, pending: &PendingScene) -> Result<Self> {
        let window =
            Arc::new(event_loop.create_window(Window::default_attributes().with_title("Voxelf"))?);
        let instance = GpuContext::instance();
        let surface = instance.create_surface(window.clone())?;
        let context = GpuContext::new(instance, Some(&surface))?;
        let capabilities = surface.get_capabilities(&context.adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .or_else(|| capabilities.formats.first().copied())
            .context("Surface reported no formats")?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: *capabilities
                .alpha_modes
                .first()
                .context("Surface reported no alpha modes")?,
            view_formats: vec![],
        };
        let scene = SceneResources::new(&context, &pending.words, pending.environment.as_deref())?;
        let raymarch = RaymarchPass::new(&context, &scene, format)?;
        let egui_context = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_context.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(context.device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&context.device, format, None, 1, false);
        let timing = context
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| GpuTiming::new(&context));
        let mut state = Self {
            window,
            surface,
            context,
            scene,
            raymarch,
            config,
            needs_configure: true,
            egui_context,
            egui_state,
            egui_renderer,
            timing,
        };
        state.configure()?;
        state.window.request_redraw();
        Ok(state)
    }

    fn configure(&mut self) -> Result<bool> {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(false);
        }
        if self.needs_configure
            || size.width != self.config.width
            || size.height != self.config.height
        {
            let limit = self.context.device.limits().max_texture_dimension_2d;
            anyhow::ensure!(
                size.width <= limit && size.height <= limit,
                "Window exceeds the device texture extent limit"
            );
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.context.device, &self.config);
            self.needs_configure = false;
        }
        Ok(true)
    }
}

struct GpuTiming {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    sender: mpsc::SyncSender<std::result::Result<(), wgpu::BufferAsyncError>>,
    receiver: mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
    period_ns: f64,
    samples_ms: [f64; GPU_TIMING_SAMPLE_COUNT],
    next: usize,
    count: usize,
}

struct GpuTimingSummary {
    latest_ms: f64,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}

impl GpuTiming {
    fn new(context: &GpuContext) -> Self {
        let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Voxelf draw timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxelf resolved timestamps"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Voxelf timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (sender, receiver) = mpsc::sync_channel(1);
        Self {
            query_set,
            resolve,
            readback,
            sender,
            receiver,
            period_ns: f64::from(context.queue.get_timestamp_period()),
            samples_ms: [0.0; GPU_TIMING_SAMPLE_COUNT],
            next: 0,
            count: 0,
        }
    }

    fn finish(&mut self) -> Result<()> {
        self.receiver
            .recv()
            .context("Timestamp mapping callback disconnected")??;
        let mapped = self.readback.slice(..).get_mapped_range();
        let start = u64::from_ne_bytes(mapped[0..8].try_into()?);
        let end = u64::from_ne_bytes(mapped[8..16].try_into()?);
        let elapsed_ms = end.wrapping_sub(start) as f64 * self.period_ns / 1_000_000.0;
        drop(mapped);
        self.readback.unmap();
        self.samples_ms[self.next] = elapsed_ms;
        self.next = (self.next + 1) % GPU_TIMING_SAMPLE_COUNT;
        self.count = (self.count + 1).min(GPU_TIMING_SAMPLE_COUNT);
        Ok(())
    }

    fn summary(&self) -> Option<GpuTimingSummary> {
        if self.count == 0 {
            return None;
        }
        let mut sorted = [0.0; GPU_TIMING_SAMPLE_COUNT];
        sorted[..self.count].copy_from_slice(&self.samples_ms[..self.count]);
        let sorted = &mut sorted[..self.count];
        sorted.sort_unstable_by(f64::total_cmp);
        Some(GpuTimingSummary {
            latest_ms: self.samples_ms
                [(self.next + GPU_TIMING_SAMPLE_COUNT - 1) % GPU_TIMING_SAMPLE_COUNT],
            mean_ms: sorted.iter().sum::<f64>() / self.count as f64,
            p50_ms: sorted[(self.count - 1) * 50 / 100],
            p95_ms: sorted[(self.count - 1) * 95 / 100],
            max_ms: sorted[self.count - 1],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Material;

    #[test]
    fn material_albedo_is_converted_from_srgb_to_linear() {
        let material = Material {
            albedo_srgb: [0, 128, 255],
            ..Material::default()
        };
        let linear = material.linear_albedo();
        assert_eq!(linear[0], 0.0);
        assert!((linear[1] - 0.215_86).abs() < 0.000_1);
        assert_eq!(linear[2], 1.0);
    }
}
