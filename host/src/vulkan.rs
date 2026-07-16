use anyhow::{anyhow, Context, Result};
use egui_winit_vulkano::{egui, Gui, GuiConfig};
use glam::{vec2, Vec2};
use shader::SvoIndex;
use std::collections::VecDeque;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage, SampleCount};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{DynamicState, Pipeline};
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo};
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::swapchain::acquire_next_image;
use vulkano::swapchain::SwapchainCreateInfo;
use vulkano::swapchain::SwapchainPresentInfo;
use vulkano::swapchain::{Surface, Swapchain};
use vulkano::sync::{GpuFuture, PipelineStage};
use vulkano::Validated;
use vulkano::VulkanError;
use vulkano::VulkanLibrary;
use winit::{event::WindowEvent, event_loop::EventLoop, window::Window};

const CAMERA_BUFFER_COUNT: usize = 3;
const GPU_TIMING_SAMPLE_COUNT: usize = 240;

#[derive(Default)]
struct GpuTimingStats {
    samples_ms: VecDeque<f64>,
}

#[derive(Clone, Copy)]
struct GpuTimingSummary {
    latest_ms: f64,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}

impl GpuTimingStats {
    fn record(&mut self, elapsed_ms: f64) {
        if self.samples_ms.len() == GPU_TIMING_SAMPLE_COUNT {
            self.samples_ms.pop_front();
        }
        self.samples_ms.push_back(elapsed_ms);
    }

    fn summary(&self) -> Option<GpuTimingSummary> {
        let latest_ms = *self.samples_ms.back()?;
        let mean_ms = self.samples_ms.iter().sum::<f64>() / self.samples_ms.len() as f64;
        let mut sorted = self.samples_ms.iter().copied().collect::<Vec<_>>();
        sorted.sort_by(f64::total_cmp);
        let percentile = |percent: usize| sorted[(sorted.len() - 1) * percent / 100];

        Some(GpuTimingSummary {
            latest_ms,
            mean_ms,
            p50_ms: percentile(50),
            p95_ms: percentile(95),
            max_ms: *sorted.last().unwrap(),
        })
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: Vec2,
    #[format(R32G32_SFLOAT)]
    uv: Vec2,
}

pub struct Vulkan {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    svodag_buffer: Subbuffer<[u16]>,
    camera_buffers: Vec<Subbuffer<[CameraBuffer]>>,
    svodag_depth: u32,
    svodag_metadata_buffer: Subbuffer<[shader::SvodagMetaData]>,
    rcx: Option<RenderContext>,
    timestamp_query_pool: Arc<QueryPool>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    gui: Gui,
    recreate_swapchain: bool,
    descriptor_sets: Vec<Arc<DescriptorSet>>,
    camera_buffer_index: usize,
    gpu_timing: GpuTimingStats,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraView {
    pub origin: [f32; 3],
    pub direction: [f32; 3],
    pub up: [f32; 3],
    pub right: [f32; 3],
}

impl Default for CameraView {
    fn default() -> Self {
        Self {
            origin: [1.5, 1.5, 3.0],
            direction: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
        }
    }
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct CameraBuffer {
    origin: [f32; 4],
    direction: [f32; 4],
    up: [f32; 4],
    right: [f32; 4],
    viewport: [f32; 4],
}

impl CameraBuffer {
    fn from_view(view: CameraView, aspect_ratio: f32) -> Self {
        Self {
            origin: extend_vec3(view.origin),
            direction: extend_vec3(view.direction),
            up: extend_vec3(view.up),
            right: extend_vec3(view.right),
            viewport: [aspect_ratio, 0.0, 0.0, 0.0],
        }
    }
}

fn extend_vec3(value: [f32; 3]) -> [f32; 4] {
    [value[0], value[1], value[2], 0.0]
}

fn write_camera_buffer(
    camera_buffers: &[Subbuffer<[CameraBuffer]>],
    start_index: usize,
    camera: CameraView,
    aspect_ratio: f32,
) -> Option<usize> {
    for offset in 0..camera_buffers.len() {
        let index = (start_index + offset) % camera_buffers.len();
        if let Ok(mut write) = camera_buffers[index].write() {
            write[0] = CameraBuffer::from_view(camera, aspect_ratio);
            return Some(index);
        }
    }

    None
}

impl Vulkan {
    pub fn new(
        event_loop: &EventLoop<()>,
        serialized_svodag: &[u16],
        svodag_depth: u32,
    ) -> anyhow::Result<Self> {
        let library = VulkanLibrary::new()?;
        let required_extensions = Surface::required_extensions(&event_loop)?;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_vulkan_memory_model: true,
            ..DeviceExtensions::empty()
        };

        let device_features = DeviceFeatures {
            vulkan_memory_model: true,
            shader_int16: true,
            shader_int8: true,
            storage_buffer8_bit_access: true,
            storage_buffer16_bit_access: true,
            ..DeviceFeatures::empty()
        };

        let physical_device = instance
            .enumerate_physical_devices()?
            .next()
            .context("No GPU?")?;

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|queue_family_properties| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS)
            })
            .context("No queue?")? as u32;

        println!(
            "Using device: {} (type {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .context("Failed to create device")?;

        let queue = queues.next().context("Empty queues")?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let vertices = vec![
            MyVertex {
                position: vec2(3.0, -1.0),
                uv: vec2(2.0, 0.0),
            },
            MyVertex {
                position: vec2(-1.0, 3.0),
                uv: vec2(0.0, 2.0),
            },
            MyVertex {
                position: vec2(-1.0, -1.0),
                uv: vec2(0.0, 0.0),
            },
        ];

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .context("Cannot create VBO")?;

        let svodag_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            serialized_svodag.iter().map(|x| *x),
        )?;

        let svodag_metadata_buffer = Buffer::new_slice(
            memory_allocator.clone(),
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

        let camera_buffers = (0..CAMERA_BUFFER_COUNT)
            .map(|_| {
                Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    [CameraBuffer::from_view(CameraView::default(), 1.0)],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let rcx = None;

        let timestamp_query_pool = QueryPool::new(
            device.clone(),
            QueryPoolCreateInfo {
                query_count: 2,
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )?;

        Ok(Vulkan {
            instance,
            device,
            queue,
            command_buffer_allocator,
            vertex_buffer,
            svodag_buffer,
            camera_buffers,
            svodag_depth,
            svodag_metadata_buffer,
            rcx,
            timestamp_query_pool,
        })
    }

    pub fn resumed(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> anyhow::Result<()> {
        let window = Arc::new(event_loop.create_window(Window::default_attributes())?);
        let surface = Surface::from_window(self.instance.clone(), window.clone())?;
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())?;

            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())?[0];

            Swapchain::new(
                self.device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .context("No supported composite alpha")?,
                    ..Default::default()
                },
            )?
        };

        let render_pass = vulkano::ordered_passes_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: SampleCount::Sample1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            passes: [
                { color: [color], depth_stencil: {}, input: [] },
                { color: [color], depth_stencil: {}, input: [] },
            ],
        )?;

        let framebuffers = window_size_dependent_setup(&images, &render_pass)?;

        const SHADER: &[u8] = include_bytes!(env!("shader.spv"));
        let shader_code = SHADER
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>();

        let shader_module = unsafe {
            ShaderModule::new(
                self.device.clone(),
                ShaderModuleCreateInfo::new(&shader_code),
            )
        }?;

        let pipeline = {
            let vs = shader_module
                .entry_point("main_vs")
                .context("Could not find `main_vs`")?;
            let fs = shader_module
                .entry_point("main_fs")
                .context("Could not find `main_fs`")?;

            let vertex_input_state = MyVertex::per_vertex().definition(&vs)?;

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())?,
            )?;

            let subpass =
                Subpass::from(render_pass.clone(), 0).context("Could not create subpass")?;

            GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
        }?;

        let gui = Gui::new_with_subpass(
            event_loop,
            surface,
            self.queue.clone(),
            Subpass::from(render_pass.clone(), 1).context("Could not create egui subpass")?,
            swapchain.image_format(),
            GuiConfig {
                allow_srgb_render_target: true,
                ..GuiConfig::default()
            },
        );

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            self.device.clone(),
            Default::default(),
        ));
        let pipeline_layout = pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .context("")?;

        let descriptor_sets = self
            .camera_buffers
            .iter()
            .map(|camera_buffer| {
                DescriptorSet::new(
                    descriptor_set_allocator.clone(),
                    descriptor_set_layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, self.svodag_buffer.clone()),
                        WriteDescriptorSet::buffer(1, camera_buffer.clone()),
                    ],
                    [],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let recreate_swapchain = false;

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            gui,
            recreate_swapchain,
            descriptor_sets,
            camera_buffer_index: 0,
            gpu_timing: GpuTimingStats::default(),
        });

        Ok(())
    }

    pub fn resized(&mut self) -> anyhow::Result<()> {
        self.rcx.as_mut().context("Wrong init")?.recreate_swapchain = true;

        Ok(())
    }

    pub fn window_event(&mut self, event: &WindowEvent) -> bool {
        self.rcx.as_mut().is_some_and(|rcx| rcx.gui.update(event))
    }

    pub fn request_redraw(&self) {
        if let Some(rcx) = &self.rcx {
            rcx.window.request_redraw();
        }
    }

    pub fn redraw(
        &mut self,
        root_indices: &[SvoIndex],
        camera: CameraView,
        fps: f32,
    ) -> anyhow::Result<()> {
        let rcx = self.rcx.as_mut().context("Wat?")?;

        let window_size = rcx.window.inner_size();

        if window_size.width == 0 || window_size.height == 0 {
            return Ok(());
        }

        let aspect_ratio = window_size.width as f32 / window_size.height as f32;
        let Some(descriptor_set_index) = write_camera_buffer(
            &self.camera_buffers,
            rcx.camera_buffer_index,
            camera,
            aspect_ratio,
        ) else {
            return Ok(());
        };
        rcx.camera_buffer_index = (descriptor_set_index + 1) % self.camera_buffers.len();

        if rcx.recreate_swapchain {
            let (new_swapchain, new_images) = rcx.swapchain.recreate(SwapchainCreateInfo {
                image_extent: window_size.into(),
                ..rcx.swapchain.create_info()
            })?;

            rcx.swapchain = new_swapchain;
            rcx.framebuffers = window_size_dependent_setup(&new_images, &rcx.render_pass)?;
            rcx.viewport.extent = window_size.into();
            rcx.recreate_swapchain = false;
        }

        let gpu_timing = rcx.gpu_timing.summary();
        rcx.gui.immediate_ui(|gui| {
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
                    ui.label("egui overlay boilerplate is wired into the Vulkan render pass.");
                });
        });

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    return Ok(());
                }
                Err(e) => return Err(anyhow!(e)),
            };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        unsafe {
            builder.reset_query_pool(self.timestamp_query_pool.clone(), 0..2)?;
        }

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        rcx.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())?
            .bind_pipeline_graphics(rcx.pipeline.clone())?
            .bind_vertex_buffers(0, self.vertex_buffer.clone())?
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Graphics,
                rcx.pipeline.layout().clone(),
                0,
                rcx.descriptor_sets[descriptor_set_index].clone(),
            )?
            .push_constants(
                rcx.pipeline.layout().clone(),
                0,
                shader::PushConsts::new(self.svodag_depth, root_indices[0]),
            )?;

        unsafe {
            builder.write_timestamp(
                self.timestamp_query_pool.clone(),
                0,
                PipelineStage::TopOfPipe,
            )?;
        }
        unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }?;
        unsafe {
            builder.write_timestamp(
                self.timestamp_query_pool.clone(),
                1,
                PipelineStage::BottomOfPipe,
            )?;
        }

        builder.next_subpass(
            Default::default(),
            SubpassBeginInfo {
                contents: SubpassContents::SecondaryCommandBuffers,
                ..Default::default()
            },
        )?;

        let gui_command_buffer = rcx
            .gui
            .draw_on_subpass_image([window_size.width, window_size.height]);
        builder.execute_commands(gui_command_buffer)?;

        builder.end_render_pass(Default::default())?;

        let command_buffer = builder.build()?;

        let future = acquire_future
            .then_execute(self.queue.clone(), command_buffer)?
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        let future = match future.map_err(Validated::unwrap) {
            Ok(future) => future,
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                return Ok(());
            }
            Err(e) => return Err(anyhow!(e)),
        };

        future.wait(None)?;

        let mut timestamps = [0u64; 2];
        let available = self.timestamp_query_pool.get_results(
            0..2,
            &mut timestamps,
            QueryResultFlags::empty(),
        )?;

        assert!(available);

        let period_ns = self.device.physical_device().properties().timestamp_period;
        let elapsed_ticks = timestamps[1].wrapping_sub(timestamps[0]);
        let queue_family = &self.device.physical_device().queue_family_properties()
            [self.queue.queue_family_index() as usize];
        let elapsed_ticks = match queue_family.timestamp_valid_bits {
            Some(valid_bits) if valid_bits < 64 => elapsed_ticks & ((1_u64 << valid_bits) - 1),
            _ => elapsed_ticks,
        };
        let elapsed_ms = elapsed_ticks as f64 * period_ns as f64 / 1_000_000.0;

        rcx.gpu_timing.record(elapsed_ms);

        Ok(())
    }
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> anyhow::Result<Vec<Arc<Framebuffer>>> {
    Ok(images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone())?;

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?)
}
