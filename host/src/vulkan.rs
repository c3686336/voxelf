use crate::svodag;
use anyhow::{anyhow, Context, Result};
use glam::{vec2, vec3, Mat4, Vec2, Vec3};
use shader::{SvoIndex, SvodagMetaData};
use std::sync::Arc;
use vulkano::buffer::{
    AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{
    self, physical, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue,
    QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{self, DynamicState, Pipeline};
use vulkano::pipeline::{
    GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateFlags, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo};
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::swapchain::acquire_next_image;
use vulkano::swapchain::SwapchainCreateInfo;
use vulkano::swapchain::SwapchainPresentInfo;
use vulkano::swapchain::{Surface, Swapchain};
use vulkano::sync::{self, semaphore, GpuFuture};
use vulkano::Validated;
use vulkano::VulkanError;
use vulkano::{library, render_pass, VulkanLibrary};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

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
    svodag_depth: u32,
    svodag_metadata_buffer: Subbuffer<[shader::SvodagMetaData]>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    descriptor_set: Arc<DescriptorSet>,
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

        let rcx = None;

        Ok(Vulkan {
            instance,
            device,
            queue,
            command_buffer_allocator,
            vertex_buffer,
            svodag_buffer,
            svodag_depth,
            svodag_metadata_buffer,
            rcx,
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
                surface,
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

        let render_pass = vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
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

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            descriptor_set_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.svodag_buffer.clone())],
            [],
        )?;

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let recreate_swapchain = false;

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            recreate_swapchain,
            previous_frame_end,
            descriptor_set,
        });

        Ok(())
    }

    pub fn resized(&mut self) -> anyhow::Result<()> {
        self.rcx.as_mut().context("Wrong init")?.recreate_swapchain = true;

        Ok(())
    }

    pub fn redraw(&mut self, root_indices: &[SvoIndex]) -> anyhow::Result<()> {
        let rcx = self.rcx.as_mut().context("Wat?")?;

        let window_size = rcx.window.inner_size();

        if window_size.width == 0 || window_size.height == 0 {
            return Ok(());
        }

        rcx.previous_frame_end
            .as_mut()
            .context("What?")?
            .cleanup_finished();

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
                rcx.descriptor_set.clone(),
            )?
            .push_constants(
                rcx.pipeline.layout().clone(),
                0,
                shader::PushConsts::new(self.svodag_depth, root_indices[0]),
            );

        unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }?;

        builder.end_render_pass(Default::default())?;

        let command_buffer = builder.build()?;

        let future = rcx
            .previous_frame_end
            .take()
            .context("")?
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)?
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed())
            }
            Err(e) => return Err(anyhow!(e)),
        }

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
