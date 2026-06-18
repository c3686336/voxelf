pub mod raymarching;
pub mod svodag;
pub mod vulkan;

use glam::{vec3, Mat4, Vec3};
use shader::SvoIndex;
use std::collections::HashSet;
use std::time::Instant;
use svodag::{SvoNodeRef, Svodag};
use vulkan::{CameraView, Vulkan};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{self, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};

const SHADER: &[u8] = include_bytes!(env!("shader.spv"));

struct Object {
    root: SvoNodeRef,
    transform: Mat4,
}

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    speed: f32,
    sensitivity: f32,
    pressed_keys: HashSet<KeyCode>,
    mouse_look: bool,
    last_cursor_position: Option<PhysicalPosition<f64>>,
}

impl Camera {
    fn new() -> Self {
        Self {
            position: vec3(1.5, 1.5, 3.0),
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            speed: 1.5,
            sensitivity: 0.003,
            pressed_keys: HashSet::new(),
            mouse_look: false,
            last_cursor_position: None,
        }
    }

    fn handle_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key_code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.pressed_keys.insert(key_code);
                        }
                        ElementState::Released => {
                            self.pressed_keys.remove(&key_code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_look = *state == ElementState::Pressed;
                self.last_cursor_position = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_look {
                    if let Some(last_position) = self.last_cursor_position {
                        let delta_x = position.x - last_position.x;
                        let delta_y = position.y - last_position.y;

                        self.yaw += delta_x as f32 * self.sensitivity;
                        self.pitch += delta_y as f32 * self.sensitivity;
                        self.pitch = self.pitch.clamp(-1.5, 1.5);
                    }
                    self.last_cursor_position = Some(*position);
                }
            }
            _ => {}
        }
    }

    fn update(&mut self, delta_seconds: f32) {
        let (forward, right) = self.movement_basis();
        let mut velocity = Vec3::ZERO;

        if self.pressed_keys.contains(&KeyCode::KeyW) {
            velocity += forward;
        }
        if self.pressed_keys.contains(&KeyCode::KeyS) {
            velocity -= forward;
        }
        if self.pressed_keys.contains(&KeyCode::KeyD) {
            velocity += right;
        }
        if self.pressed_keys.contains(&KeyCode::KeyA) {
            velocity -= right;
        }
        if self.pressed_keys.contains(&KeyCode::KeyQ) {
            velocity -= Vec3::Y;
        }
        if self.pressed_keys.contains(&KeyCode::KeyE) {
            velocity += Vec3::Y;
        }

        if velocity.length_squared() > 0.0 {
            self.position += velocity.normalize() * self.speed * delta_seconds;
        }
    }

    fn view(&self) -> CameraView {
        let (direction, right, up) = self.basis();

        CameraView {
            origin: self.position.to_array(),
            direction: direction.to_array(),
            up: up.to_array(),
            right: right.to_array(),
        }
    }

    fn is_mouse_look_active(&self) -> bool {
        self.mouse_look
    }

    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let direction = vec3(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();
        let right = direction.cross(Vec3::Y).normalize();
        let up = right.cross(direction).normalize();

        (direction, right, up)
    }

    fn movement_basis(&self) -> (Vec3, Vec3) {
        let forward = vec3(self.yaw.cos(), 0.0, self.yaw.sin()).normalize();
        let right = forward.cross(Vec3::Y).normalize();

        (forward, right)
    }
}

struct App {
    vulkan: Vulkan,
    svodag: Svodag,
    index_mapping: Vec<SvoIndex>,
    objects: Vec<Object>,
    camera: Camera,
    last_frame: Instant,
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
            camera: Camera::new(),
            last_frame: Instant::now(),
        })
    }

    fn update_camera(&mut self) -> f32 {
        let now = Instant::now();
        let delta_seconds = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.camera.update(delta_seconds.min(0.1));

        if delta_seconds > 0.0 {
            1.0 / delta_seconds
        } else {
            0.0
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        self.vulkan.resumed(event_loop).unwrap();
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let egui_consumed = self.vulkan.window_event(&event);
        if !egui_consumed || self.camera.is_mouse_look_active() {
            self.camera.handle_window_event(&event);
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_size) => self.vulkan.resized().unwrap(),
            WindowEvent::ScaleFactorChanged { .. } => self.vulkan.resized().unwrap(),
            WindowEvent::RedrawRequested => {
                let fps = self.update_camera();
                self.vulkan
                    .redraw(
                        &[self.index_mapping[self.svodag.to_index(&self.objects[0].root)]],
                        self.camera.view(),
                        fps,
                    )
                    .unwrap()
            }
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
