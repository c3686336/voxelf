mod context;
mod frame;
mod passes;
mod resources;
mod surface;

use anyhow::{anyhow, Context, Result};
use context::GpuContext;
use egui_winit_vulkano::egui;
use frame::FrameResources;
use passes::RaymarchPass;
use resources::SceneResources;
use shader::SvoIndex;
use std::path::Path;
use surface::SurfaceState;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::query::QueryResultFlags;
use vulkano::swapchain::{acquire_next_image, SwapchainPresentInfo};
use vulkano::sync::GpuFuture;
use vulkano::{Validated, VulkanError};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};

pub use frame::CameraView;

pub struct Vulkan {
    context: GpuContext,
    scene: SceneResources,
    frames: FrameResources,
    surface: Option<SurfaceState>,
    raymarch_pass: Option<RaymarchPass>,
    svodag_depth: u32,
}

impl Vulkan {
    pub fn new(
        event_loop: &EventLoop<()>,
        serialized_svodag: &[u32],
        svodag_depth: u32,
        environment_path: Option<&Path>,
    ) -> Result<Self> {
        let context = GpuContext::new(event_loop)?;
        let scene = SceneResources::new(&context, serialized_svodag, environment_path)?;
        let frames = FrameResources::new(&context)?;

        Ok(Self {
            context,
            scene,
            frames,
            surface: None,
            raymarch_pass: None,
            svodag_depth,
        })
    }

    pub fn resumed(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let surface = SurfaceState::new(&self.context, event_loop)?;
        let raymarch_pass = RaymarchPass::new(
            &self.context,
            &self.scene,
            &self.frames,
            surface.render_pass.clone(),
        )?;
        self.surface = Some(surface);
        self.raymarch_pass = Some(raymarch_pass);

        Ok(())
    }

    pub fn resized(&mut self) -> Result<()> {
        self.surface
            .as_mut()
            .context("Renderer is not resumed")?
            .mark_for_recreation();
        Ok(())
    }

    pub fn window_event(&mut self, event: &WindowEvent) -> bool {
        self.surface
            .as_mut()
            .is_some_and(|surface| surface.handle_window_event(event))
    }

    pub fn request_redraw(&self) {
        if let Some(surface) = &self.surface {
            surface.request_redraw();
        }
    }

    pub fn redraw(
        &mut self,
        root_indices: &[SvoIndex],
        camera: CameraView,
        fps: f32,
    ) -> Result<()> {
        let Some(&root_index) = root_indices.first() else {
            return Ok(());
        };
        let surface = self.surface.as_mut().context("Renderer is not resumed")?;
        let raymarch_pass = self
            .raymarch_pass
            .as_ref()
            .context("Raymarch pass is not initialized")?;
        let window_size = surface.window_size();
        if window_size.width == 0 || window_size.height == 0 {
            return Ok(());
        }

        surface.recreate_if_needed()?;
        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let Some(camera_buffer_index) = self.frames.write_camera(camera, aspect_ratio) else {
            return Ok(());
        };

        let gpu_timing = self.frames.timing_summary();
        let environment_source = self.scene.environment.source.clone();
        let environment_extent = self.scene.environment.extent;
        surface.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            egui::Window::new("Voxelf")
                .default_pos([12.0, 12.0])
                .show(&ctx, |ui| {
                    ui.heading("Voxelf");
                    ui.label(format!("SVoDAG depth: {}", self.svodag_depth));
                    ui.label(format!("Root objects: {}", root_indices.len()));
                    ui.label(format!(
                        "Camera: {:.2}, {:.2}, {:.2}",
                        camera.origin[0], camera.origin[1], camera.origin[2]
                    ));
                    ui.label(format!("Environment: {environment_source}"));
                    ui.label(format!(
                        "Environment size: {}x{}",
                        environment_extent[0], environment_extent[1]
                    ));
                    ui.label(format!("FPS: {:.0}", fps));
                    if let Some(timing) = gpu_timing {
                        ui.label(format!("GPU draw: {:.2} ms", timing.latest_ms));
                        ui.label(format!(
                            "Mean/P50/P95/Max: {:.2}/{:.2}/{:.2}/{:.2} ms",
                            timing.mean_ms, timing.p50_ms, timing.p95_ms, timing.max_ms
                        ));
                        ui.label(format!(
                            "GPU-only capacity: {:.0} FPS",
                            1000.0 / timing.mean_ms
                        ));
                    }
                    ui.separator();
                    ui.label("W/A/S/D move, Q/E rise/drop, left mouse drag looks around.");
                });
        });

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(surface.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(result) => result,
                Err(VulkanError::OutOfDate) => {
                    surface.mark_for_recreation();
                    return Ok(());
                }
                Err(error) => return Err(anyhow!(error)),
            };
        if suboptimal {
            surface.mark_for_recreation();
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.context.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        unsafe {
            builder.reset_query_pool(self.frames.timestamp_query_pool.clone(), 0..2)?;
        }
        raymarch_pass.record(
            &mut builder,
            surface.framebuffers[image_index as usize].clone(),
            surface.viewport.clone(),
            camera_buffer_index,
            root_index,
            self.svodag_depth,
            self.frames.timestamp_query_pool.clone(),
        )?;
        let gui_command_buffer = surface
            .gui
            .draw_on_subpass_image([window_size.width, window_size.height]);
        builder.execute_commands(gui_command_buffer)?;
        builder.end_render_pass(Default::default())?;
        let command_buffer = builder.build()?;

        let future = acquire_future
            .then_execute(self.context.queue.clone(), command_buffer)?
            .then_signal_semaphore()
            .then_swapchain_present(
                self.context.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(surface.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();
        let future = match future.map_err(Validated::unwrap) {
            Ok(future) => future,
            Err(VulkanError::OutOfDate) => {
                surface.mark_for_recreation();
                return Ok(());
            }
            Err(error) => return Err(anyhow!(error)),
        };
        future.wait(None)?;

        let mut timestamps = [0_u64; 2];
        let available = self.frames.timestamp_query_pool.get_results(
            0..2,
            &mut timestamps,
            QueryResultFlags::empty(),
        )?;
        if available {
            let period_ns = self
                .context
                .device
                .physical_device()
                .properties()
                .timestamp_period;
            let elapsed_ticks = timestamps[1].wrapping_sub(timestamps[0]);
            let queue_family = &self
                .context
                .device
                .physical_device()
                .queue_family_properties()[self.context.queue.queue_family_index() as usize];
            let elapsed_ticks = match queue_family.timestamp_valid_bits {
                Some(valid_bits) if valid_bits < 64 => elapsed_ticks & ((1_u64 << valid_bits) - 1),
                _ => elapsed_ticks,
            };
            self.frames
                .record_gpu_time(elapsed_ticks as f64 * period_ns as f64 / 1_000_000.0);
        }

        Ok(())
    }
}
