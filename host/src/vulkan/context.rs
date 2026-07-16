use anyhow::{Context, Result};
use std::sync::Arc;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;
use winit::event_loop::EventLoop;

pub struct GpuContext {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl GpuContext {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let library = VulkanLibrary::new()?;
        let required_extensions = Surface::required_extensions(event_loop)?;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?;

        let physical_device = instance
            .enumerate_physical_devices()?
            .next()
            .context("No Vulkan physical device available")?;
        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|properties| {
                properties
                    .queue_flags
                    .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
            })
            .context("No graphics and compute queue available")?
            as u32;

        println!(
            "Using device: {} (type {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions {
                    khr_swapchain: true,
                    khr_vulkan_memory_model: true,
                    ..DeviceExtensions::empty()
                },
                enabled_features: DeviceFeatures {
                    vulkan_memory_model: true,
                    shader_int8: true,
                    storage_buffer8_bit_access: true,
                    ..DeviceFeatures::empty()
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;
        let queue = queues.next().context("Device returned no queues")?;
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        Ok(Self {
            instance,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        })
    }
}
