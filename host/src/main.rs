pub mod raymarching;
pub mod svodag;
pub mod vulkan;

use anyhow::{anyhow, Context, Result};
use svodag::Svodag;
use vulkan::Vulkan;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{self, EventLoop};

const SHADER: &[u8] = include_bytes!(env!("shader.spv"));

struct App {
    vulkan: Vulkan,
}

impl App {
    pub fn new(event_loop: &EventLoop<()>) -> anyhow::Result<Self> {
        Ok(Self {
            vulkan: Vulkan::new(event_loop)?,
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        self.vulkan.resumed(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => self.vulkan.resized().unwrap(),
            WindowEvent::RedrawRequested => self.vulkan.redraw().unwrap(),
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    // let mut app = vulkan::App::new(&event_loop)?;

    // Ok(event_loop.run_app(&mut app)?)

    let mut app = App::new(&event_loop)?;

    Ok(event_loop.run_app(&mut app)?)
}
