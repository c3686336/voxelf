use super::context::GpuContext;
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryType};

const FRAME_BUFFER_COUNT: usize = 3;
const GPU_TIMING_SAMPLE_COUNT: usize = 240;

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

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct CameraBuffer {
    origin: [f32; 4],
    direction: [f32; 4],
    up: [f32; 4],
    right: [f32; 4],
    viewport: [f32; 4],
}

impl CameraBuffer {
    fn from_view(view: CameraView, aspect_ratio: f32) -> Self {
        Self {
            origin: extend_vec3(view.origin),
            direction: extend_vec3(view.direction),
            up: extend_vec3(view.up),
            right: extend_vec3(view.right),
            viewport: [aspect_ratio, 0.0, 0.0, 0.0],
        }
    }
}

fn extend_vec3(value: [f32; 3]) -> [f32; 4] {
    [value[0], value[1], value[2], 0.0]
}

#[derive(Default)]
struct GpuTimingStats {
    samples_ms: VecDeque<f64>,
}

#[derive(Clone, Copy)]
pub(super) struct GpuTimingSummary {
    pub latest_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
}

impl GpuTimingStats {
    fn record(&mut self, elapsed_ms: f64) {
        if self.samples_ms.len() == GPU_TIMING_SAMPLE_COUNT {
            self.samples_ms.pop_front();
        }
        self.samples_ms.push_back(elapsed_ms);
    }

    fn summary(&self) -> Option<GpuTimingSummary> {
        let latest_ms = *self.samples_ms.back()?;
        let mean_ms = self.samples_ms.iter().sum::<f64>() / self.samples_ms.len() as f64;
        let mut sorted = self.samples_ms.iter().copied().collect::<Vec<_>>();
        sorted.sort_by(f64::total_cmp);
        let percentile = |percent: usize| sorted[(sorted.len() - 1) * percent / 100];

        Some(GpuTimingSummary {
            latest_ms,
            mean_ms,
            p50_ms: percentile(50),
            p95_ms: percentile(95),
            max_ms: *sorted.last().unwrap(),
        })
    }
}

pub(super) struct FrameResources {
    camera_buffers: Vec<Subbuffer<[CameraBuffer]>>,
    next_camera_buffer: usize,
    pub timestamp_query_pool: Arc<QueryPool>,
    gpu_timing: GpuTimingStats,
}

impl FrameResources {
    pub fn new(context: &GpuContext) -> Result<Self> {
        let camera_buffers = (0..FRAME_BUFFER_COUNT)
            .map(|_| {
                Buffer::from_iter(
                    context.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    [CameraBuffer::from_view(CameraView::default(), 1.0)],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let timestamp_query_pool = QueryPool::new(
            context.device.clone(),
            QueryPoolCreateInfo {
                query_count: 2,
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )?;

        Ok(Self {
            camera_buffers,
            next_camera_buffer: 0,
            timestamp_query_pool,
            gpu_timing: GpuTimingStats::default(),
        })
    }

    pub fn camera_buffers(&self) -> &[Subbuffer<[CameraBuffer]>] {
        &self.camera_buffers
    }

    pub fn write_camera(&mut self, camera: CameraView, aspect_ratio: f32) -> Option<usize> {
        for offset in 0..self.camera_buffers.len() {
            let index = (self.next_camera_buffer + offset) % self.camera_buffers.len();
            if let Ok(mut write) = self.camera_buffers[index].write() {
                write[0] = CameraBuffer::from_view(camera, aspect_ratio);
                self.next_camera_buffer = (index + 1) % self.camera_buffers.len();
                return Some(index);
            }
        }

        None
    }

    pub fn record_gpu_time(&mut self, elapsed_ms: f64) {
        self.gpu_timing.record(elapsed_ms);
    }

    pub fn timing_summary(&self) -> Option<GpuTimingSummary> {
        self.gpu_timing.summary()
    }
}
