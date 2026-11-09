use super::context::GpuContext;
use anyhow::{Context, Result};
use egui_winit_vulkano::{Gui, GuiConfig};
use std::sync::Arc;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageUsage, SampleCount};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo};
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

pub(super) struct SurfaceState {
    pub window: Arc<Window>,
    pub _surface: Arc<Surface>,
    pub swapchain: Arc<Swapchain>,
    pub render_pass: Arc<RenderPass>,
    pub linear_image: Option<Arc<Image>>,
    pub images: Vec<Arc<Image>>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    pub viewport: Viewport,
    pub gui: Gui,
    pub recreate_swapchain: bool,
}

impl SurfaceState {
    pub fn new(
        context: &GpuContext,
        event_loop: &ActiveEventLoop,
        requested_size: Option<PhysicalSize<u32>>,
        linear_capture: bool,
    ) -> Result<Self> {
        let mut attributes = Window::default_attributes();
        if let Some(size) = requested_size {
            attributes = attributes.with_inner_size(size);
        }
        let window = Arc::new(event_loop.create_window(attributes)?);
        let surface = Surface::from_window(context.instance.clone(), window.clone())?;
        let window_size = window.inner_size();
        let physical_device = context.device.physical_device();
        let surface_capabilities =
            physical_device.surface_capabilities(&surface, Default::default())?;
        let (image_format, _) = physical_device
            .surface_formats(&surface, Default::default())?
            .into_iter()
            .next()
            .context("Surface reported no formats")?;
        let preferred_image_count = surface_capabilities.min_image_count.max(2);
        let image_count = surface_capabilities
            .max_image_count
            .map_or(preferred_image_count, |max| preferred_image_count.min(max));
        let (swapchain, images) = Swapchain::new(
            context.device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: image_count,
                image_format,
                image_extent: window_size.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT
                    | (surface_capabilities.supported_usage_flags & ImageUsage::TRANSFER_SRC),
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .context("No supported composite alpha")?,
                ..Default::default()
            },
        )?;
        let linear_image = if linear_capture {
            Some(make_linear_image(context, window_size)?)
        } else {
            None
        };
        let attachment_format = if linear_capture {
            Format::R16G16B16A16_SFLOAT
        } else {
            swapchain.image_format()
        };
        let render_pass = vulkano::ordered_passes_renderpass!(
            context.device.clone(),
            attachments: {
                color: {
                    format: attachment_format,
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
        let attachment_images = linear_image
            .as_ref()
            .map(|image| vec![image.clone(); images.len()]);
        let framebuffers = create_framebuffers(
            attachment_images.as_deref().unwrap_or(&images),
            &render_pass,
        )?;
        let gui = Gui::new_with_subpass(
            event_loop,
            surface.clone(),
            context.queue.clone(),
            Subpass::from(render_pass.clone(), 1).context("Could not create egui subpass")?,
            attachment_format,
            GuiConfig {
                allow_srgb_render_target: true,
                ..GuiConfig::default()
            },
        );
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        Ok(Self {
            window,
            _surface: surface,
            swapchain,
            render_pass,
            linear_image,
            images,
            framebuffers,
            viewport,
            gui,
            recreate_swapchain: false,
        })
    }

    pub fn window_size(&self) -> PhysicalSize<u32> {
        self.window.inner_size()
    }

    pub fn handle_window_event(&mut self, event: &WindowEvent) -> bool {
        self.gui.update(event)
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn mark_for_recreation(&mut self) {
        self.recreate_swapchain = true;
    }

    pub fn recreate_if_needed(&mut self, context: &GpuContext) -> Result<()> {
        if !self.recreate_swapchain {
            return Ok(());
        }

        let window_size = self.window_size();
        let (swapchain, images) = self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: window_size.into(),
            ..self.swapchain.create_info()
        })?;
        self.swapchain = swapchain;
        if self.linear_image.is_some() {
            self.linear_image = Some(make_linear_image(context, window_size)?);
        }
        let attachment_images = self
            .linear_image
            .as_ref()
            .map(|image| vec![image.clone(); images.len()]);
        self.framebuffers = create_framebuffers(
            attachment_images.as_deref().unwrap_or(&images),
            &self.render_pass,
        )?;
        self.images = images;
        self.viewport.extent = window_size.into();
        self.recreate_swapchain = false;

        Ok(())
    }
}

fn make_linear_image(context: &GpuContext, size: PhysicalSize<u32>) -> Result<Arc<Image>> {
    Ok(Image::new(
        context.memory_allocator.clone(),
        ImageCreateInfo {
            format: Format::R16G16B16A16_SFLOAT,
            extent: [size.width, size.height, 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )?)
}

fn create_framebuffers(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Result<Vec<Arc<Framebuffer>>> {
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
