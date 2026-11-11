use crate::vulkan::context::GpuContext;
use crate::vulkan::frame::CameraBuffer;
use crate::vulkan::resources::SceneResources;
use anyhow::{Context, Result};
use glam::{vec2, Vec2};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::query::QueryPool;
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::sync::PipelineStage;

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct FullscreenVertex {
    #[format(R32G32_SFLOAT)]
    position: Vec2,
    #[format(R32G32_SFLOAT)]
    uv: Vec2,
}

pub(crate) struct RaymarchPass {
    vertex_buffer: Subbuffer<[FullscreenVertex]>,
    pipeline: Arc<GraphicsPipeline>,
    scene_descriptor_set: Arc<DescriptorSet>,
    frame_descriptor_sets: Vec<Arc<DescriptorSet>>,
}

impl RaymarchPass {
    pub fn new(
        context: &GpuContext,
        scene: &SceneResources,
        camera_buffers: &[Subbuffer<[CameraBuffer]>],
        render_pass: Arc<RenderPass>,
    ) -> Result<Self> {
        let vertex_buffer = Buffer::from_iter(
            context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [
                FullscreenVertex {
                    position: vec2(3.0, -1.0),
                    uv: vec2(2.0, 0.0),
                },
                FullscreenVertex {
                    position: vec2(-1.0, 3.0),
                    uv: vec2(0.0, 2.0),
                },
                FullscreenVertex {
                    position: vec2(-1.0, -1.0),
                    uv: vec2(0.0, 0.0),
                },
            ],
        )?;

        const SHADER: &[u8] = include_bytes!(env!("shader.spv"));
        let shader_code = SHADER
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let shader_module = unsafe {
            ShaderModule::new(
                context.device.clone(),
                ShaderModuleCreateInfo::new(&shader_code),
            )
        }?;
        let vertex_entry = shader_module
            .entry_point("main_vs")
            .context("Could not find main_vs")?;
        let fragment_entry = shader_module
            .entry_point("main_fs")
            .context("Could not find main_fs")?;
        let vertex_input_state = FullscreenVertex::per_vertex().definition(&vertex_entry)?;
        let stages = [
            PipelineShaderStageCreateInfo::new(vertex_entry),
            PipelineShaderStageCreateInfo::new(fragment_entry),
        ];
        let layout = PipelineLayout::new(
            context.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(context.device.clone())?,
        )?;
        let subpass = Subpass::from(render_pass, 0).context("Could not create raymarch subpass")?;
        let pipeline = GraphicsPipeline::new(
            context.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?;
        let frame_descriptor_set_layout = pipeline
            .layout()
            .set_layouts()
            .get(1)
            .context("Raymarch pipeline has no frame descriptor set layout")?;
        let scene_descriptor_set =
            Self::make_scene_descriptor_set(context, &pipeline, scene, &scene.svodag_buffer)?;
        let frame_descriptor_sets = camera_buffers
            .iter()
            .map(|camera_buffer| {
                DescriptorSet::new(
                    context.descriptor_set_allocator.clone(),
                    frame_descriptor_set_layout.clone(),
                    [WriteDescriptorSet::buffer(0, camera_buffer.clone())],
                    [],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            vertex_buffer,
            pipeline,
            scene_descriptor_set,
            frame_descriptor_sets,
        })
    }

    fn make_scene_descriptor_set(
        context: &GpuContext,
        pipeline: &GraphicsPipeline,
        scene: &SceneResources,
        buffer: &Subbuffer<[u32]>,
    ) -> Result<Arc<DescriptorSet>> {
        let layout = pipeline
            .layout()
            .set_layouts()
            .first()
            .context("Raymarch pipeline has no descriptor set layout")?;
        Ok(DescriptorSet::new(
            context.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer.clone()),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    scene.environment.view.clone(),
                    scene.environment.sampler.clone(),
                ),
            ],
            [],
        )?)
    }

    pub fn scene_descriptor_set_for(
        &self,
        context: &GpuContext,
        scene: &SceneResources,
        buffer: &Subbuffer<[u32]>,
    ) -> Result<Arc<DescriptorSet>> {
        Self::make_scene_descriptor_set(context, &self.pipeline, scene, buffer)
    }

    pub fn set_scene_descriptor_set(&mut self, descriptor_set: Arc<DescriptorSet>) {
        self.scene_descriptor_set = descriptor_set;
    }

    pub fn record(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        framebuffer: Arc<Framebuffer>,
        viewport: Viewport,
        camera_buffer_index: usize,
        constants: shader::PushConsts,
        timestamp_query_pool: Option<Arc<QueryPool>>,
    ) -> Result<()> {
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .set_viewport(0, [viewport].into_iter().collect())?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_vertex_buffers(0, self.vertex_buffer.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                (
                    self.scene_descriptor_set.clone(),
                    self.frame_descriptor_sets[camera_buffer_index].clone(),
                ),
            )?
            .push_constants(self.pipeline.layout().clone(), 0, constants)?;
        unsafe {
            if let Some(query_pool) = &timestamp_query_pool {
                builder.write_timestamp(query_pool.clone(), 0, PipelineStage::TopOfPipe)?;
            }
            builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0)?;
            if let Some(query_pool) = timestamp_query_pool {
                builder.write_timestamp(query_pool, 1, PipelineStage::BottomOfPipe)?;
            }
        }

        Ok(())
    }
}
