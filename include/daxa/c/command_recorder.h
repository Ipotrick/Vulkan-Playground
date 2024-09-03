#ifndef __DAXA_COMMAND_LIST_H__
#define __DAXA_COMMAND_LIST_H__

#include <daxa/c/types.h>
#include <daxa/c/sync.h>
#include <daxa/c/gpu_resources.h>
#include <daxa/c/pipeline.h>

/// TODO: investigate software command recording
///   Might allow us some optimizations
///   Allows for easier verification
///   Allows for more predictable performance (move all vk recording ops to tight place in code)

/// WARNING:
///   Checks for command types against queue family only performed in c++ api!!

typedef struct
{
    void const * data;
    uint64_t size;
    uint32_t offset;
} daxa_PushConstantInfo;

typedef struct
{
    daxa_QueueFamily queue_family;
    daxa_SmallString name;
} daxa_CommandRecorderInfo;

static daxa_CommandRecorderInfo const DAXA_DEFAULT_COMMAND_RECORDER_INFO = DAXA_ZERO_INIT;

typedef struct
{
    daxa_ImageId src_image;
    daxa_ImageLayout src_image_layout;
    daxa_ImageId dst_image;
    daxa_ImageLayout dst_image_layout;
    daxa_ImageArraySlice src_slice;
    VkOffset3D src_offsets[2];
    daxa_ImageArraySlice dst_slice;
    VkOffset3D dst_offsets[2];
    VkFilter filter;
} daxa_ImageBlitInfo;

static daxa_ImageBlitInfo const DAXA_DEFAULT_IMAGE_BLIT_INFO = {
    .src_image = DAXA_ZERO_INIT,
    .src_image_layout = DAXA_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    .dst_image = DAXA_ZERO_INIT,
    .dst_image_layout = DAXA_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    .src_slice = DAXA_ZERO_INIT,
    .src_offsets = DAXA_ZERO_INIT,
    .dst_slice = DAXA_ZERO_INIT,
    .dst_offsets = DAXA_ZERO_INIT,
    .filter = VK_FILTER_NEAREST,
};

typedef struct
{
    daxa_BufferId src_buffer;
    daxa_BufferId dst_buffer;
    size_t src_offset;
    size_t dst_offset;
    size_t size;
} daxa_BufferCopyInfo;

static daxa_BufferCopyInfo const DAXA_DEFAULT_BUFFER_COPY_INFO = DAXA_ZERO_INIT;

typedef struct
{
    daxa_BufferId buffer;
    size_t buffer_offset;
    daxa_ImageId image;
    daxa_ImageLayout image_layout;
    daxa_ImageArraySlice image_slice;
    VkOffset3D image_offset;
    VkExtent3D image_extent;
} daxa_BufferImageCopyInfo;

static daxa_BufferImageCopyInfo const DAXA_DEFAULT_BUFFER_IMAGE_COPY_INFO = {
    .buffer = DAXA_ZERO_INIT,
    .buffer_offset = 0,
    .image = DAXA_ZERO_INIT,
    .image_layout = DAXA_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    .image_slice = DAXA_ZERO_INIT,
    .image_offset = DAXA_ZERO_INIT,
    .image_extent = DAXA_ZERO_INIT,
};

typedef struct
{
    daxa_ImageId image;
    daxa_ImageLayout image_layout;
    daxa_ImageArraySlice image_slice;
    VkOffset3D image_offset;
    VkExtent3D image_extent;
    daxa_BufferId buffer;
    size_t buffer_offset;
} daxa_ImageBufferCopyInfo;

static daxa_ImageBufferCopyInfo const DAXA_DEFAULT_IMAGE_BUFFER_COPY_INFO = {
    .image = DAXA_ZERO_INIT,
    .image_layout = DAXA_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    .image_slice = DAXA_ZERO_INIT,
    .image_offset = DAXA_ZERO_INIT,
    .image_extent = DAXA_ZERO_INIT,
    .buffer = DAXA_ZERO_INIT,
    .buffer_offset = 0,
};

typedef struct
{
    daxa_ImageId src_image;
    daxa_ImageLayout src_image_layout;
    daxa_ImageId dst_image;
    daxa_ImageLayout dst_image_layout;
    daxa_ImageArraySlice src_slice;
    VkOffset3D src_offset;
    daxa_ImageArraySlice dst_slice;
    VkOffset3D dst_offset;
    VkExtent3D extent;
} daxa_ImageCopyInfo;

static daxa_ImageCopyInfo const DAXA_DEFAULT_IMAGE_COPY_INFO = {
    .src_image = DAXA_ZERO_INIT,
    .src_image_layout = DAXA_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    .dst_image = DAXA_ZERO_INIT,
    .dst_image_layout = DAXA_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    .src_slice = DAXA_ZERO_INIT,
    .src_offset = DAXA_ZERO_INIT,
    .dst_slice = DAXA_ZERO_INIT,
    .dst_offset = DAXA_ZERO_INIT,
    .extent = DAXA_ZERO_INIT,
};

typedef struct
{
    daxa_ImageLayout image_layout;
    // Make sure this stays abi compatible with daxa::ClearValue
    daxa_Variant(VkClearValue) clear_value;
    daxa_ImageId image;
    daxa_ImageMipArraySlice dst_slice;
} daxa_ImageClearInfo;

static daxa_ImageClearInfo const DAXA_DEFAULT_IMAGE_CLEAR_INFO = {
    .image_layout = DAXA_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    .clear_value = DAXA_ZERO_INIT,
    .image = DAXA_ZERO_INIT,
    .dst_slice = DAXA_ZERO_INIT,
};

typedef struct
{
    daxa_BufferId buffer;
    size_t offset;
    size_t size;
    uint32_t clear_value;
} daxa_BufferClearInfo;

static daxa_BufferClearInfo const DAXA_DEFAULT_BUFFER_CLEAR_INFO = DAXA_ZERO_INIT;

typedef struct
{
    VkResolveModeFlagBits mode;
    daxa_ImageViewId image;
    daxa_ImageLayout layout;
} daxa_AttachmentResolveInfo;

static daxa_AttachmentResolveInfo const DAXA_DEFAULT_RENDER_ATTACHMENT_RESOLVE_INFO = {
    .mode = VK_RESOLVE_MODE_AVERAGE_BIT,
    .image = {},
    .layout = DAXA_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
};

typedef struct
{
    daxa_ImageViewId image_view;
    daxa_ImageLayout layout;
    VkAttachmentLoadOp load_op;
    VkAttachmentStoreOp store_op;
    daxa_Variant(VkClearValue) clear_value;
    daxa_Optional(daxa_AttachmentResolveInfo) resolve;
} daxa_RenderAttachmentInfo;

static daxa_RenderAttachmentInfo const DAXA_DEFAULT_RENDER_ATTACHMENT_INFO = {
    .image_view = DAXA_ZERO_INIT,
    .layout = DAXA_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
    .load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .store_op = VK_ATTACHMENT_STORE_OP_STORE,
    .clear_value = DAXA_ZERO_INIT,
    .resolve = {.has_value = 0},
};

typedef struct
{
    daxa_FixedList(daxa_RenderAttachmentInfo, 8) color_attachments;
    daxa_Optional(daxa_RenderAttachmentInfo) depth_attachment;
    daxa_Optional(daxa_RenderAttachmentInfo) stencil_attachment;
    VkRect2D render_area;
} daxa_RenderPassBeginInfo;

static daxa_RenderPassBeginInfo const DAXA_DEFAULT_RENDERPASS_BEGIN_INFO = DAXA_ZERO_INIT;

typedef struct
{
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t raygen_handle_offset;
    uint32_t miss_handle_offset;
    uint32_t hit_handle_offset;
    uint32_t callable_handle_offset;
    daxa_RayTracingShaderBindingTable shader_binding_table;
} daxa_TraceRaysInfo;

static daxa_TraceRaysInfo const DAXA_DEFAULT_TRACE_RAYS_INFO = DAXA_ZERO_INIT;

typedef struct
{
    uint64_t indirect_device_address;
    uint32_t raygen_handle_offset;
    uint32_t miss_handle_offset;
    uint32_t hit_handle_offset;
    uint32_t callable_handle_offset;
    daxa_RayTracingShaderBindingTable shader_binding_table;
} daxa_TraceRaysIndirectInfo;

static daxa_TraceRaysIndirectInfo const DAXA_DEFAULT_TRACE_RAYS_INDIRECT_INFO = DAXA_ZERO_INIT;

typedef struct
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
} daxa_DispatchInfo;

static daxa_DispatchInfo const DAXA_DEFAULT_DISPATCH_INFO = {1, 1, 1};

typedef struct
{
    daxa_BufferId indirect_buffer;
    size_t offset;
} daxa_DispatchIndirectInfo;

static daxa_DispatchIndirectInfo const DAXA_DEFAULT_DISPATCH_INDIRECT_INFO = DAXA_ZERO_INIT;

typedef struct
{
    daxa_BufferId indirect_buffer;
    size_t offset;
    uint32_t draw_count;
    uint32_t stride;
} daxa_DrawMeshTasksIndirectInfo;

static daxa_DrawMeshTasksIndirectInfo const DAXA_DEFAULT_DRAW_MESH_TASKS_INDIRECT_INFO = {
    .indirect_buffer = DAXA_ZERO_INIT,
    .offset = 0,
    .draw_count = 1,
    .stride = 12,
};

typedef struct
{
    daxa_BufferId indirect_buffer;
    size_t offset;
    daxa_BufferId count_buffer;
    size_t count_offset;
    uint32_t max_count;
    uint32_t stride;
} daxa_DrawMeshTasksIndirectCountInfo;

static daxa_DrawMeshTasksIndirectCountInfo const DAXA_DRAW_MESH_TASKS_INDIRECT_COUNT_INFO = {
    .indirect_buffer = DAXA_ZERO_INIT,
    .offset = 0,
    .count_buffer = DAXA_ZERO_INIT,
    .count_offset = 0,
    .max_count = 0,
    .stride = 12,
};

typedef struct
{
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
} daxa_DrawInfo;

static daxa_DrawInfo const DAXA_DEFAULT_DRAW_INFO = {
    .vertex_count = 0,
    .instance_count = 1,
    .first_vertex = 0,
    .first_instance = 0,
};

typedef struct
{
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t vertex_offset;
    uint32_t first_instance;
} daxa_DrawIndexedInfo;

static daxa_DrawIndexedInfo const DAXA_DEFAULT_DRAW_INDEXED_INFO = {
    .index_count = 0,
    .instance_count = 1,
    .first_index = 0,
    .vertex_offset = 0,
    .first_instance = 0,
};

typedef struct
{
    daxa_BufferId indirect_buffer;
    size_t indirect_buffer_offset;
    uint32_t draw_count;
    uint32_t draw_command_stride;
    daxa_Bool8 is_indexed;
} daxa_DrawIndirectInfo;

static daxa_DrawIndirectInfo const DAXA_DEFAULT_DRAW_INDIRECT_INFO = {
    .indirect_buffer = DAXA_ZERO_INIT,
    .indirect_buffer_offset = 0,
    .draw_count = 1,
    .draw_command_stride = 0,
    .is_indexed = 0,
};

typedef struct
{
    daxa_BufferId indirect_buffer;
    size_t indirect_buffer_offset;
    daxa_BufferId count_buffer;
    size_t count_buffer_offset;
    uint32_t max_draw_count;
    uint32_t draw_command_stride;
    daxa_Bool8 is_indexed;
} daxa_DrawIndirectCountInfo;

static daxa_DrawIndirectCountInfo const DAXA_DEFAULT_DRAW_INDIRECT_COUNT_INFO = {
    .indirect_buffer = DAXA_ZERO_INIT,
    .indirect_buffer_offset = 0,
    .count_buffer = DAXA_ZERO_INIT,
    .count_buffer_offset = 0,
    .max_draw_count = ((1 << 16) - 1),
    .draw_command_stride = 0,
    .is_indexed = 0,
};

typedef struct
{
    daxa_Event * event;
    VkPipelineStageFlags stage;
} daxa_ResetEventsInfo;

typedef struct
{
    daxa_Event * events;
    size_t event_count;
} daxa_WaitEventsInfo;

typedef struct
{
    daxa_TimelineQueryPool * query_pool;
    VkPipelineStageFlags2 pipeline_stage;
    uint32_t query_index;
} daxa_WriteTimestampInfo;

typedef struct
{
    daxa_TimelineQueryPool * query_pool;
    uint32_t start_index;
    uint32_t count;
} daxa_ResetTimestampsInfo;

typedef struct
{
    daxa_f32vec4 label_color;
    daxa_SmallString name;
} daxa_CommandLabelInfo;

typedef struct
{
    daxa_Event * barrier;
    VkPipelineStageFlags stage_masks;
} daxa_ResetEventInfo;

typedef struct
{
    float constant_factor;
    float clamp;
    float slope_factor;
} daxa_DepthBiasInfo;

static daxa_DepthBiasInfo const DAXA_DEFAULT_DEPTH_BIAS_INFO = DAXA_ZERO_INIT;

typedef struct
{
    daxa_BufferId buffer;
    size_t offset;
    VkIndexType index_type;
} daxa_SetIndexBufferInfo;

static daxa_SetIndexBufferInfo const DAXA_DEFAULT_SET_INDEX_BUFFER_INFO = {
    .buffer = DAXA_ZERO_INIT,
    .offset = 0,
    .index_type = VK_INDEX_TYPE_UINT32,
};

typedef struct
{
    daxa_TlasBuildInfo const * tlas_build_infos;
    size_t tlas_build_info_count;
    daxa_BlasBuildInfo const * blas_build_infos;
    size_t blas_build_info_count;
} daxa_BuildAccelerationStucturesInfo;

static daxa_BuildAccelerationStucturesInfo const DAXA_DEFAULT_BUILD_ACCELERATION_STRUCTURES_INFO = DAXA_ZERO_INIT;

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_set_rasterization_samples(daxa_CommandRecorder cmd_enc, VkSampleCountFlagBits samples);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_copy_buffer_to_buffer(daxa_CommandRecorder cmd_enc, daxa_BufferCopyInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_copy_buffer_to_image(daxa_CommandRecorder cmd_enc, daxa_BufferImageCopyInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_copy_image_to_buffer(daxa_CommandRecorder cmd_enc, daxa_ImageBufferCopyInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_copy_image_to_image(daxa_CommandRecorder cmd_enc, daxa_ImageCopyInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_blit_image_to_image(daxa_CommandRecorder cmd_enc, daxa_ImageBlitInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_build_acceleration_structures(daxa_CommandRecorder cmd_rec, daxa_BuildAccelerationStucturesInfo const * info);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_clear_buffer(daxa_CommandRecorder cmd_enc, daxa_BufferClearInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_clear_image(daxa_CommandRecorder cmd_enc, daxa_ImageClearInfo const * info);

/// @brief  Successive pipeline barrier calls are combined.
///         As soon as a non-pipeline barrier command is recorded, the currently recorded barriers are flushed with a vkCmdPipelineBarrier2 call.
/// @param info parameters.
DAXA_EXPORT void
daxa_cmd_pipeline_barrier(daxa_CommandRecorder cmd_enc, daxa_MemoryBarrierInfo const * info);
/// @brief  Successive pipeline barrier calls are combined.
///         As soon as a non-pipeline barrier command is recorded, the currently recorded barriers are flushed with a vkCmdPipelineBarrier2 call.
/// @param info parameters.
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_pipeline_barrier_image_transition(daxa_CommandRecorder cmd_enc, daxa_ImageMemoryBarrierInfo const * info);
DAXA_EXPORT void
daxa_cmd_signal_event(daxa_CommandRecorder cmd_enc, daxa_EventSignalInfo const * info);
DAXA_EXPORT void
daxa_cmd_wait_events(daxa_CommandRecorder cmd_enc, daxa_EventWaitInfo const * infos, size_t info_count);
DAXA_EXPORT void
daxa_cmd_wait_event(daxa_CommandRecorder cmd_enc, daxa_EventWaitInfo const * info);
DAXA_EXPORT void
daxa_cmd_reset_event(daxa_CommandRecorder cmd_enc, daxa_ResetEventInfo const * info);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_push_constant(daxa_CommandRecorder cmd_enc, daxa_PushConstantInfo const * info);
DAXA_EXPORT void
daxa_cmd_set_ray_tracing_pipeline(daxa_CommandRecorder cmd_enc, daxa_RayTracingPipeline pipeline);
DAXA_EXPORT void
daxa_cmd_set_compute_pipeline(daxa_CommandRecorder cmd_enc, daxa_ComputePipeline pipeline);
DAXA_EXPORT void
daxa_cmd_set_raster_pipeline(daxa_CommandRecorder cmd_enc, daxa_RasterPipeline pipeline);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_dispatch(daxa_CommandRecorder cmd_enc, daxa_DispatchInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_dispatch_indirect(daxa_CommandRecorder cmd_enc, daxa_DispatchIndirectInfo const * info);

/// @brief  Destroys the buffer AFTER the gpu is finished executing the command list.
///         Useful for large uploads exceeding staging memory pools.
/// @param id buffer to be destroyed after command list finishes.
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_destroy_buffer_deferred(daxa_CommandRecorder cmd_enc, daxa_BufferId id);
/// @brief  Destroys the image AFTER the gpu is finished executing the command list.
///         Useful for large uploads exceeding staging memory pools.
/// @param id image to be destroyed after command list finishes.
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_destroy_image_deferred(daxa_CommandRecorder cmd_enc, daxa_ImageId id);
/// @brief  Destroys the image view AFTER the gpu is finished executing the command list.
///         Useful for large uploads exceeding staging memory pools.
/// @param id image view to be destroyed after command list finishes.
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_destroy_image_view_deferred(daxa_CommandRecorder cmd_enc, daxa_ImageViewId id);
/// @brief  Destroys the sampler AFTER the gpu is finished executing the command list.
///         Useful for large uploads exceeding staging memory pools.
/// @param id image sampler be destroyed after command list finishes.
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_destroy_sampler_deferred(daxa_CommandRecorder cmd_enc, daxa_SamplerId id);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_trace_rays(daxa_CommandRecorder cmd_enc, daxa_TraceRaysInfo const * info);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_trace_rays_indirect(daxa_CommandRecorder cmd_enc, daxa_TraceRaysIndirectInfo const * info);

/// @brief  Starts a renderpass scope akin to the dynamic rendering feature in vulkan.
///         Between the begin and end renderpass commands, the renderpass persists and draw-calls can be recorded.
/// @param info parameters.
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_begin_renderpass(daxa_CommandRecorder cmd_enc, daxa_RenderPassBeginInfo const * info);
/// @brief  Ends a renderpass scope akin to the dynamic rendering feature in vulkan.
///         Between the begin and end renderpass commands, the renderpass persists and draw-calls can be recorded.
DAXA_EXPORT void
daxa_cmd_end_renderpass(daxa_CommandRecorder cmd_enc);
DAXA_EXPORT void
daxa_cmd_set_viewport(daxa_CommandRecorder cmd_enc, VkViewport const * info);
DAXA_EXPORT void
daxa_cmd_set_scissor(daxa_CommandRecorder cmd_enc, VkRect2D const * info);
DAXA_EXPORT void
daxa_cmd_set_depth_bias(daxa_CommandRecorder cmd_enc, daxa_DepthBiasInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_set_index_buffer(daxa_CommandRecorder cmd_enc, daxa_SetIndexBufferInfo const * info);

DAXA_EXPORT void
daxa_cmd_draw(daxa_CommandRecorder cmd_enc, daxa_DrawInfo const * info);
DAXA_EXPORT void
daxa_cmd_draw_indexed(daxa_CommandRecorder cmd_enc, daxa_DrawIndexedInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_draw_indirect(daxa_CommandRecorder cmd_enc, daxa_DrawIndirectInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_draw_indirect_count(daxa_CommandRecorder cmd_enc, daxa_DrawIndirectCountInfo const * info);
DAXA_EXPORT void
daxa_cmd_draw_mesh_tasks(daxa_CommandRecorder cmd_enc, uint32_t x, uint32_t y, uint32_t z);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_draw_mesh_tasks_indirect(daxa_CommandRecorder cmd_enc, daxa_DrawMeshTasksIndirectInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_draw_mesh_tasks_indirect_count(daxa_CommandRecorder cmd_enc, daxa_DrawMeshTasksIndirectCountInfo const * info);

DAXA_EXPORT void
daxa_cmd_write_timestamp(daxa_CommandRecorder cmd_enc, daxa_WriteTimestampInfo const * info);
DAXA_EXPORT void
daxa_cmd_reset_timestamps(daxa_CommandRecorder cmd_enc, daxa_ResetTimestampsInfo const * info);

DAXA_EXPORT void
daxa_cmd_begin_label(daxa_CommandRecorder cmd_enc, daxa_CommandLabelInfo const * info);
DAXA_EXPORT void
daxa_cmd_end_label(daxa_CommandRecorder cmd_enc);

// Is called by all other commands. Flushes internal pipeline barrier list to actual vulkan call.
DAXA_EXPORT void
daxa_cmd_flush_barriers(daxa_CommandRecorder cmd_enc);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_cmd_complete_current_commands(daxa_CommandRecorder cmd_enc, daxa_ExecutableCommandList * out_executable_cmds);
DAXA_EXPORT daxa_CommandRecorderInfo const *
daxa_cmd_info(daxa_CommandRecorder cmd_enc);
DAXA_EXPORT VkCommandBuffer
daxa_cmd_get_vk_command_buffer(daxa_CommandRecorder cmd_enc);
DAXA_EXPORT VkCommandPool
daxa_cmd_get_vk_command_pool(daxa_CommandRecorder cmd_enc);
DAXA_EXPORT void
daxa_destroy_command_recorder(daxa_CommandRecorder cmd_enc);

DAXA_EXPORT uint64_t
daxa_executable_commands_inc_refcnt(daxa_ExecutableCommandList executable_commands);
DAXA_EXPORT uint64_t
daxa_executable_commands_dec_refcnt(daxa_ExecutableCommandList executable_commands);

#endif // #ifndef __DAXA_COMMAND_LIST_H__
