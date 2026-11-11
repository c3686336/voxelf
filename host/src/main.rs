use voxelf::render::CameraView;
use voxelf::svodag::{self, SvoNodeRef, Svodag};
use voxelf::gpu::WgpuRenderer;

use glam::{vec3, IVec3, Mat4, UVec3, Vec3};
use std::collections::HashSet;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{self, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};

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

struct FrameTiming {
    delta_seconds: f32,
    fps: f32,
}

struct FrameTimer {
    last_frame: Instant,
}

impl FrameTimer {
    fn new() -> Self {
        Self {
            last_frame: Instant::now(),
        }
    }

    fn tick(&mut self) -> FrameTiming {
        let now = Instant::now();
        let delta_seconds = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        FrameTiming {
            delta_seconds: delta_seconds.min(0.1),
            fps: if delta_seconds > 0.0 {
                1.0 / delta_seconds
            } else {
                0.0
            },
        }
    }
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
            velocity += Vec3::Y;
        }
        if self.pressed_keys.contains(&KeyCode::KeyE) {
            velocity -= Vec3::Y;
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

/// Find the first occupied voxel and the empty voxel immediately before it.
fn edit_target(
    dag: &Svodag,
    root: SvoNodeRef,
    origin: Vec3,
    direction: Vec3,
    add: bool,
) -> Option<UVec3> {
    let width = 1_i32.checked_shl(dag.depth)?;
    let mut entry = 0.0_f32;
    let mut exit = f32::INFINITY;
    for axis in 0..3 {
        if direction[axis].abs() < f32::EPSILON {
            if origin[axis] < 1.0 || origin[axis] >= 2.0 {
                return None;
            }
        } else {
            let a = (1.0 - origin[axis]) / direction[axis];
            let b = (2.0 - origin[axis]) / direction[axis];
            entry = entry.max(a.min(b));
            exit = exit.min(a.max(b));
        }
    }
    if entry > exit || exit < 0.0 {
        return None;
    }
    let point = origin + direction * (entry + 1e-5 / width as f32);
    let mut cell = ((point - Vec3::ONE) * width as f32).floor().as_ivec3();
    cell = cell.clamp(IVec3::ZERO, IVec3::splat(width - 1));
    let mut step = IVec3::ZERO;
    let mut next = Vec3::splat(f32::INFINITY);
    let mut delta = Vec3::splat(f32::INFINITY);
    for axis in 0..3 {
        if direction[axis].abs() >= f32::EPSILON {
            step[axis] = if direction[axis] > 0.0 { 1 } else { -1 };
            let boundary = 1.0 + (cell[axis] + i32::from(step[axis] > 0)) as f32 / width as f32;
            next[axis] = (boundary - origin[axis]) / direction[axis];
            delta[axis] = (1.0 / (width as f32 * direction[axis])).abs();
        }
    }
    let mut previous = None;
    while cell.cmpge(IVec3::ZERO).all() && cell.cmplt(IVec3::splat(width)).all() {
        let position = cell.as_uvec3();
        if dag.get(&root, position).is_some() {
            return if add { previous } else { Some(position) };
        }
        previous = Some(position);
        let t = next.min_element();
        if t > exit || !t.is_finite() {
            break;
        }
        for axis in 0..3 {
            if next[axis] <= t {
                cell[axis] += step[axis];
                next[axis] += delta[axis];
            }
        }
    }
    if add {
        previous
    } else {
        None
    }
}

struct App {
    renderer: WgpuRenderer,
    svodag: Svodag,
    empty_root: SvoNodeRef,
    edits_since_collect: usize,
    pending_sync: bool,
    objects: Vec<Object>,
    camera: Camera,
    render_error: Option<anyhow::Error>,
    frame_timer: FrameTimer,
}

impl App {
    pub fn new(
        event_loop: &EventLoop<()>,
        environment_path: Option<&std::path::Path>,
    ) -> anyhow::Result<Self> {
        let (mut svodag, root) = svodag::get_example_svodag(7);
        let empty_root = svodag.empty_root();
        let renderer = WgpuRenderer::new(event_loop, svodag.data(), 7, environment_path)?;
        svodag.clear_dirty();

        Ok(Self {
            renderer,
            svodag,
            empty_root,
            edits_since_collect: 0,
            pending_sync: false,
            objects: vec![Object {
                root,
                transform: Mat4::IDENTITY,
            }],
            camera: Camera::new(),
            render_error: None,
            frame_timer: FrameTimer::new(),
        })
    }

    fn update_camera(&mut self, delta_seconds: f32) {
        self.camera.update(delta_seconds);
    }

    fn edit_voxel(&mut self, add: bool) {
        let root = self.objects[0].root;
        let (direction, _, _) = self.camera.basis();
        if let Some(position) =
            edit_target(&self.svodag, root, self.camera.position, direction, add)
        {
            let next = if add {
                Some(self.svodag.insert(Some(&root), position, self.svodag.depth))
            } else {
                self.svodag.remove(Some(&root), position, self.svodag.depth)
            };
            let next = next.unwrap_or(self.empty_root);
            if next != root {
                self.objects[0].root = next;
                self.pending_sync = true;
                self.edits_since_collect += 1;
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if let Err(error) = self.renderer.resumed(event_loop) {
            self.render_error = Some(error);
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if self.render_error.is_some() {
            return;
        }
        let egui_consumed = self.renderer.window_event(&event);
        if !egui_consumed || self.camera.is_mouse_look_active() {
            self.camera.handle_window_event(&event);
        }

        match event {
            WindowEvent::KeyboardInput { event, .. }
                if !egui_consumed && event.state == ElementState::Pressed && !event.repeat =>
            {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyZ) => self.edit_voxel(false),
                    PhysicalKey::Code(KeyCode::KeyX) => self.edit_voxel(true),
                    _ => {}
                }
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_size) => self.renderer.resized().unwrap(),
            WindowEvent::ScaleFactorChanged { .. } => self.renderer.resized().unwrap(),
            WindowEvent::RedrawRequested => {
                let frame_timing = self.frame_timer.tick();
                self.update_camera(frame_timing.delta_seconds);
                if self.pending_sync {
                    let dirty = self.svodag.dirty_ranges();
                    self.renderer.sync_svodag(self.svodag.data(), &dirty).unwrap();
                    self.svodag.clear_dirty();
                    self.pending_sync = false;
                }
                if self.edits_since_collect >= 128 {
                    self.svodag
                        .collect(&[self.objects[0].root, self.empty_root]);
                    self.edits_since_collect = 0;
                }
                match self.renderer.redraw(
                    &[self.objects[0].root],
                    self.camera.view(),
                    frame_timing.fps,
                ) {
                    Ok(_) => {}
                    Err(error) => {
                        self.render_error = Some(error);
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &event_loop::ActiveEventLoop) {
        self.renderer.request_redraw();
    }
}

fn main() -> anyhow::Result<()> {
    let environment = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/envmap.hdr");
    let event_loop = EventLoop::new()?;
    let mut app = App::new(&event_loop, Some(&environment))?;
    event_loop.run_app(&mut app)?;
    if let Some(error) = app.render_error {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod edit_tests {
    use super::*;
    use glam::uvec3;

    #[test]
    fn targeting_selects_hit_and_adjacent_empty_voxel() {
        let mut dag = Svodag::new(2);
        let root = dag.insert(None, uvec3(2, 1, 1), 2);
        let origin = vec3(0.5, 1.375, 1.375);
        assert_eq!(
            edit_target(&dag, root, origin, Vec3::X, false),
            Some(uvec3(2, 1, 1))
        );
        assert_eq!(
            edit_target(&dag, root, origin, Vec3::X, true),
            Some(uvec3(1, 1, 1))
        );
        assert_eq!(edit_target(&dag, root, origin, -Vec3::X, false), None);
    }

    #[test]
    fn targeting_empty_scene_can_add_and_misses_outside_volume() {
        let mut dag = Svodag::new(2);
        let root = dag.empty_root();
        assert_eq!(
            edit_target(&dag, root, vec3(0.5, 1.375, 1.375), Vec3::X, true),
            Some(uvec3(3, 1, 1))
        );
        assert_eq!(
            edit_target(&dag, root, vec3(0.5, 2.5, 1.375), Vec3::X, true),
            None
        );
    }
}
