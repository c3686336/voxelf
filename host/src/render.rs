/// Camera basis shared by interactive and validation renderers.
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

pub use shader::RendererMode;

/// Linear scene-referred material parameters, without display encoding.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: [1.0, 0.0, 0.0],
            metallic: 0.0,
            roughness: 0.5,
        }
    }
}

/// Fixed view and lighting settings for a validation capture.
#[derive(Clone, Copy, Debug)]
pub struct RenderSettings {
    pub camera: CameraView,
    pub renderer: RendererMode,
    pub material: Material,
    pub samples_per_pixel: u32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            camera: CameraView::default(),
            renderer: RendererMode::Reference,
            material: Material::default(),
            samples_per_pixel: 16,
        }
    }
}
