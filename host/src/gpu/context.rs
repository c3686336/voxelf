use anyhow::{Context, Result};

pub(super) struct GpuContext {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub fn instance() -> wgpu::Instance {
        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all()),
            ..Default::default()
        })
    }

    pub fn new(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Self> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
        }))
        .context("No compatible wgpu adapter available")?;
        let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let required_limits = wgpu::Limits {
            max_push_constant_size: 0,
            ..adapter.limits()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("voxelf device"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .context("Creating wgpu device")?;
        let info = adapter.get_info();
        println!(
            "Using device: {} (type {:?}, backend {:?})",
            info.name, info.device_type, info.backend
        );

        Ok(Self {
            adapter,
            device,
            queue,
        })
    }
}
