use super::{context::GpuContext, resources::SceneResources};
use crate::render::CameraView;
use anyhow::{bail, Result};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

pub(super) struct RaymarchPass {
    pipeline: wgpu::RenderPipeline,
    scene_layout: wgpu::BindGroupLayout,
    scene_bind_group: wgpu::BindGroup,
    frame_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    constants_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
}

impl RaymarchPass {
    pub fn new(
        context: &GpuContext,
        scene: &SceneResources,
        format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let device = &context.device;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxel raymarch shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!(env!("VOXELF_SHADER_WGSL")).into()),
        });
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scene layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Frame layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(80),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(48),
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Raymarch pipeline layout"),
            bind_group_layouts: &[&scene_layout, &frame_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Raymarch pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("main_vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 16,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                }],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("main_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera basis"),
            size: 80,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let constants_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render constants"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Frame bindings"),
            layout: &frame_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: constants_buffer.as_entire_binding(),
                },
            ],
        });
        let scene_bind_group = Self::scene_bind_group(context, &scene_layout, scene);
        // WebGPU maps clip-space +Y to the top row. Preserve the old renderer's
        // UV origin there without changing the camera's up basis.
        let vertices = [
            [3.0_f32, 1.0, 2.0, 0.0],
            [-1.0, -3.0, 0.0, 2.0],
            [-1.0, 1.0, 0.0, 0.0],
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fullscreen triangle"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            bail!("Creating raymarch pipeline: {error}");
        }
        Ok(Self {
            pipeline,
            scene_layout,
            scene_bind_group,
            frame_bind_group,
            camera_buffer,
            constants_buffer,
            vertex_buffer,
        })
    }

    fn scene_bind_group(
        context: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        scene: &SceneResources,
    ) -> wgpu::BindGroup {
        context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Scene bindings"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: scene.svodag_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&scene.environment.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&scene.environment.sampler),
                    },
                ],
            })
    }

    pub fn rebind_scene(&mut self, context: &GpuContext, scene: &SceneResources) {
        self.scene_bind_group = Self::scene_bind_group(context, &self.scene_layout, scene);
    }

    pub fn write_frame(
        &self,
        context: &GpuContext,
        camera: CameraView,
        aspect: f32,
        constants: shader::RenderParams,
    ) {
        let extend = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
        let camera = [
            extend(camera.origin),
            extend(camera.direction),
            extend(camera.up),
            extend(camera.right),
            [aspect, 0.0, 0.0, 0.0],
        ];
        context
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&camera));
        context
            .queue
            .write_buffer(&self.constants_buffer, 0, bytemuck::bytes_of(&constants));
    }

    /// Captures have fixed camera/material settings; only their independent seed changes.
    pub fn write_seed(&self, context: &GpuContext, seed: u32) {
        context.queue.write_buffer(
            &self.constants_buffer,
            std::mem::offset_of!(shader::RenderParams, seed) as u64,
            bytemuck::bytes_of(&seed),
        );
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Raymarch"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: timestamps,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.scene_bind_group, &[]);
        pass.set_bind_group(1, &self.frame_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..3, 0..1);
    }
}
