pub mod raymarching;
pub mod svodag;
pub mod vulkan;

use anyhow::{anyhow, Context, Result};
use glam::Mat4;
use shader::SvoIndex;
use svodag::{SvoNodeRef, Svodag};
use vulkan::Vulkan;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{self, EventLoop};

const SHADER: &[u8] = include_bytes!(env!("shader.spv"));

struct Object {
    root: SvoNodeRef,
    transform: Mat4,
}

struct App {
    vulkan: Vulkan,
    svodag: Svodag,
    index_mapping: Vec<SvoIndex>,
    objects: Vec<Object>,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> anyhow::Result<Self> {
        let (svodag, root) = svodag::get_example_svodag(7);
        let (serialized, index_mapping) = svodag.serialize();

        Ok(Self {
            vulkan: Vulkan::new(event_loop, serialized.as_ref(), 7)?,
            svodag,
            index_mapping,
            objects: vec![Object {
                root,
                transform: Mat4::IDENTITY,
            }],
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        self.vulkan.resumed(event_loop).unwrap();
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let _egui_consumed = self.vulkan.window_event(&event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => self.vulkan.resized().unwrap(),
            WindowEvent::ScaleFactorChanged { .. } => self.vulkan.resized().unwrap(),
            WindowEvent::RedrawRequested => self
                .vulkan
                .redraw(&[self.index_mapping[self.svodag.to_index(&self.objects[0].root)]])
                .unwrap(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &event_loop::ActiveEventLoop) {
        self.vulkan.request_redraw();
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    // let mut app = vulkan::App::new(&event_loop)?;

    // Ok(event_loop.run_app(&mut app)?)

    let mut app = App::new(&event_loop)?;

    Ok(event_loop.run_app(&mut app)?)
}
