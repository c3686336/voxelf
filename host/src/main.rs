pub mod raymarching;
pub mod svodag;
pub mod vulkan;

use anyhow::{anyhow, Context, Result};
use svodag::Svodag;
use winit::event_loop::EventLoop;

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = vulkan::App::new(&event_loop)?;

    Ok(event_loop.run_app(&mut app)?)
}
