#ifndef __DAXA_DEVICE_H__
#define __DAXA_DEVICE_H__

#include <daxa/c/types.h>
#include <daxa/c/sync.h>
#include <daxa/c/gpu_resources.h>
#include <daxa/c/pipeline.h>
#include <daxa/c/command_recorder.h>
#include <daxa/c/swapchain.h>
#include <daxa/c/sync.h>

#define DAXA_MAX_COMPUTE_QUEUE_COUNT 8u
#define DAXA_MAX_TRANSFER_QUEUE_COUNT 2u

typedef enum
{
    DAXA_DEVICE_TYPE_OTHER = 0,
    DAXA_DEVICE_TYPE_INTEGRATED_GPU = 1,
    DAXA_DEVICE_TYPE_DISCRETE_GPU = 2,
    DAXA_DEVICE_TYPE_VIRTUAL_GPU = 3,
    DAXA_DEVICE_TYPE_CPU = 4,
    DAXA_DEVICE_TYPE_MAX_ENUM = 0x7fffffff,
} daxa_DeviceType;

typedef struct
{
    uint32_t max_image_dimension1d;
    uint32_t max_image_dimension2d;
    uint32_t max_image_dimension3d;
    uint32_t max_image_dimension_cube;
    uint32_t max_image_array_layers;
    uint32_t max_texel_buffer_elements;
    uint32_t max_uniform_buffer_range;
    uint32_t max_storage_buffer_range;
    uint32_t max_push_constants_size;
    uint32_t max_memory_allocation_count;
    uint32_t max_sampler_allocation_count;
    uint64_t buffer_image_granularity;
    uint64_t sparse_address_space_size;
    uint32_t max_bound_descriptor_sets;
    uint32_t max_per_stage_descriptor_samplers;
    uint32_t max_per_stage_descriptor_uniform_buffers;
    uint32_t max_per_stage_descriptor_storage_buffers;
    uint32_t max_per_stage_descriptor_sampled_images;
    uint32_t max_per_stage_descriptor_storage_images;
    uint32_t max_per_stage_descriptor_input_attachments;
    uint32_t max_per_stage_resources;
    uint32_t max_descriptor_set_samplers;
    uint32_t max_descriptor_set_uniform_buffers;
    uint32_t max_descriptor_set_uniform_buffers_dynamic;
    uint32_t max_descriptor_set_storage_buffers;
    uint32_t max_descriptor_set_storage_buffers_dynamic;
    uint32_t max_descriptor_set_sampled_images;
    uint32_t max_descriptor_set_storage_images;
    uint32_t max_descriptor_set_input_attachments;
    uint32_t max_vertex_input_attributes;
    uint32_t max_vertex_input_bindings;
    uint32_t max_vertex_input_attribute_offset;
    uint32_t max_vertex_input_binding_stride;
    uint32_t max_vertex_output_components;
    uint32_t max_tessellation_generation_level;
    uint32_t max_tessellation_patch_size;
    uint32_t max_tessellation_control_per_vertex_input_components;
    uint32_t max_tessellation_control_per_vertex_output_components;
    uint32_t max_tessellation_control_per_patch_output_components;
    uint32_t max_tessellation_control_total_output_components;
    uint32_t max_tessellation_evaluation_input_components;
    uint32_t max_tessellation_evaluation_output_components;
    uint32_t max_geometry_shader_invocations;
    uint32_t max_geometry_input_components;
    uint32_t max_geometry_output_components;
    uint32_t max_geometry_output_vertices;
    uint32_t max_geometry_total_output_components;
    uint32_t max_fragment_input_components;
    uint32_t max_fragment_output_attachments;
    uint32_t max_fragment_dual_src_attachments;
    uint32_t max_fragment_combined_output_resources;
    uint32_t max_compute_shared_memory_size;
    uint32_t max_compute_work_group_count[3];
    uint32_t max_compute_work_group_invocations;
    uint32_t max_compute_work_group_size[3];
    uint32_t sub_pixel_precision_bits;
    uint32_t sub_texel_precision_bits;
    uint32_t mipmap_precision_bits;
    uint32_t max_draw_indexed_index_value;
    uint32_t max_draw_indirect_count;
    float max_sampler_lod_bias;
    float max_sampler_anisotropy;
    uint32_t max_viewports;
    uint32_t max_viewport_dimensions[2];
    float viewport_bounds_range[2];
    uint32_t viewport_sub_pixel_bits;
    size_t min_memory_map_alignment;
    uint64_t min_texel_buffer_offset_alignment;
    uint64_t min_uniform_buffer_offset_alignment;
    uint64_t min_storage_buffer_offset_alignment;
    int32_t min_texel_offset;
    uint32_t max_texel_offset;
    int32_t min_texel_gather_offset;
    uint32_t max_texel_gather_offset;
    float min_interpolation_offset;
    float max_interpolation_offset;
    uint32_t sub_pixel_interpolation_offset_bits;
    uint32_t max_framebuffer_width;
    uint32_t max_framebuffer_height;
    uint32_t max_framebuffer_layers;
    uint32_t framebuffer_color_sample_counts;
    uint32_t framebuffer_depth_sample_counts;
    uint32_t framebuffer_stencil_sample_counts;
    uint32_t framebuffer_no_attachments_sample_counts;
    uint32_t max_color_attachments;
    uint32_t sampled_image_color_sample_counts;
    uint32_t sampled_image_integer_sample_counts;
    uint32_t sampled_image_depth_sample_counts;
    uint32_t sampled_image_stencil_sample_counts;
    uint32_t storage_image_sample_counts;
    uint32_t max_sample_mask_words;
    uint32_t timestamp_compute_and_graphics;
    float timestamp_period;
    uint32_t max_clip_distances;
    uint32_t max_cull_distances;
    uint32_t max_combined_clip_and_cull_distances;
    uint32_t discrete_queue_priorities;
    float point_size_range[2];
    float line_width_range[2];
    float point_size_granularity;
    float line_width_granularity;
    uint32_t strict_lines;
    uint32_t standard_sample_locations;
    uint64_t optimal_buffer_copy_offset_alignment;
    uint64_t optimal_buffer_copy_row_pitch_alignment;
    uint64_t non_coherent_atom_size;
} daxa_DeviceLimits;

// MUST BE ABI COMPATIBLE WITH VkPhysicalDeviceRayTracingPipelinePropertiesKHR!
typedef struct
{
    uint32_t shader_group_handle_size;
    uint32_t max_ray_recursion_depth;
    uint32_t max_shader_group_stride;
    uint32_t shader_group_base_alignment;
    uint32_t shader_group_handle_capture_replay_size;
    uint32_t max_ray_dispatch_invocation_count;
    uint32_t shader_group_handle_alignment;
    uint32_t max_ray_hit_attribute_size;
} daxa_RayTracingPipelineProperties;

// MUST BE ABI COMPATIBLE WITH VkPhysicalDeviceAccelerationStructurePropertiesKHR!
typedef struct
{
    uint64_t max_geometry_count;
    uint64_t max_instance_count;
    uint64_t max_primitive_count;
    uint32_t max_per_stage_descriptor_acceleration_structures;
    uint32_t max_per_stage_descriptor_update_after_bind_acceleration_structures;
    uint32_t max_descriptor_set_acceleration_structures;
    uint32_t max_descriptor_set_update_after_bind_acceleration_structures;
    uint32_t min_acceleration_structure_scratch_offset_alignment;
} daxa_AccelerationStructureProperties;

// MUST BE ABI COMPATIBLE WITH VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV!
typedef struct
{
    uint32_t invocation_reorder_mode;
} daxa_RayTracingInvocationReorderProperties;

// Is NOT ABI Compatible with VkPhysicalDeviceMeshShaderPropertiesEXT!
typedef struct
{
    uint32_t max_task_work_group_total_count;
    uint32_t max_task_work_group_count[3];
    uint32_t max_task_work_group_invocations;
    uint32_t max_task_work_group_size[3];
    uint32_t max_task_payload_size;
    uint32_t max_task_shared_memory_size;
    uint32_t max_task_payload_and_shared_memory_size;
    uint32_t max_mesh_work_group_total_count;
    uint32_t max_mesh_work_group_count[3];
    uint32_t max_mesh_work_group_invocations;
    uint32_t max_mesh_work_group_size[3];
    uint32_t max_mesh_shared_memory_size;
    uint32_t max_mesh_payload_and_shared_memory_size;
    uint32_t max_mesh_output_memory_size;
    uint32_t max_mesh_payload_and_output_memory_size;
    uint32_t max_mesh_output_components;
    uint32_t max_mesh_output_vertices;
    uint32_t max_mesh_output_primitives;
    uint32_t max_mesh_output_layers;
    uint32_t max_mesh_multiview_view_count;
    uint32_t mesh_output_per_vertex_granularity;
    uint32_t mesh_output_per_primitive_granularity;
    uint32_t max_preferred_task_work_group_invocations;
    uint32_t max_preferred_mesh_work_group_invocations;
    daxa_Bool8 prefers_local_invocation_vertex_output;
    daxa_Bool8 prefers_local_invocation_primitive_output;
    daxa_Bool8 prefers_compact_vertex_output;
    daxa_Bool8 prefers_compact_primitive_output;
} daxa_MeshShaderProperties;

typedef enum
{
    DAXA_MISSING_REQUIRED_VK_FEATURE_NONE,
    DAXA_MISSING_REQUIRED_VK_FEATURE_IMAGE_CUBE_ARRAY,
    DAXA_MISSING_REQUIRED_VK_FEATURE_INDEPENDENT_BLEND,
    DAXA_MISSING_REQUIRED_VK_FEATURE_TESSELLATION_SHADER,
    DAXA_MISSING_REQUIRED_VK_FEATURE_MULTI_DRAW_INDIRECT,
    DAXA_MISSING_REQUIRED_VK_FEATURE_DEPTH_CLAMP,
    DAXA_MISSING_REQUIRED_VK_FEATURE_FILL_MODE_NON_SOLID,
    DAXA_MISSING_REQUIRED_VK_FEATURE_WIDE_LINES,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SAMPLER_ANISOTROPY,
    DAXA_MISSING_REQUIRED_VK_FEATURE_FRAGMENT_STORES_AND_ATOMICS,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_STORAGE_IMAGE_MULTISAMPLE,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_INT64,
    DAXA_MISSING_REQUIRED_VK_FEATURE_VARIABLE_POINTERS_STORAGE_BUFFER,
    DAXA_MISSING_REQUIRED_VK_FEATURE_VARIABLE_POINTERS,
    DAXA_MISSING_REQUIRED_VK_FEATURE_BUFFER_DEVICE_ADDRESS,
    DAXA_MISSING_REQUIRED_VK_FEATURE_BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY,
    DAXA_MISSING_REQUIRED_VK_FEATURE_BUFFER_DEVICE_ADDRESS_MULTI_DEVICE,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING,
    DAXA_MISSING_REQUIRED_VK_FEATURE_DESCRIPTOR_BINDING_SAMPLED_IMAGE_UPDATE_AFTER_BIND,
    DAXA_MISSING_REQUIRED_VK_FEATURE_DESCRIPTOR_BINDING_STORAGE_IMAGE_UPDATE_AFTER_BIND,
    DAXA_MISSING_REQUIRED_VK_FEATURE_DESCRIPTOR_BINDING_STORAGE_BUFFER_UPDATE_AFTER_BIND,
    DAXA_MISSING_REQUIRED_VK_FEATURE_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING,
    DAXA_MISSING_REQUIRED_VK_FEATURE_DESCRIPTOR_BINDING_PARTIALLY_BOUND,
    DAXA_MISSING_REQUIRED_VK_FEATURE_RUNTIME_DESCRIPTOR_ARRAY,
    DAXA_MISSING_REQUIRED_VK_FEATURE_HOST_QUERY_RESET,
    DAXA_MISSING_REQUIRED_VK_FEATURE_DYNAMIC_RENDERING,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SYNCHRONIZATION2,
    DAXA_MISSING_REQUIRED_VK_FEATURE_TIMELINE_SEMAPHORE,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SUBGROUP_SIZE_CONTROL,
    DAXA_MISSING_REQUIRED_VK_FEATURE_COMPUTE_FULL_SUBGROUPS,
    DAXA_MISSING_REQUIRED_VK_FEATURE_SCALAR_BLOCK_LAYOUT,
    DAXA_MISSING_REQUIRED_VK_FEATURE_ACCELERATION_STRUCTURE_CAPTURE_REPLAY,
    DAXA_MISSING_REQUIRED_VK_FEATURE_VULKAN_MEMORY_MODEL,
    DAXA_MISSING_REQUIRED_VK_FEATURE_ROBUST_BUFFER_ACCESS2,
    DAXA_MISSING_REQUIRED_VK_FEATURE_ROBUST_IMAGE_ACCESS2,
    DAXA_MISSING_REQUIRED_VK_FEATURE_MAX_ENUM
} daxa_MissingRequiredVkFeature;

typedef enum
{
    DAXA_EXPLICIT_FEATURE_FLAG_NONE = 0,
    DAXA_EXPLICIT_FEATURE_FLAG_BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY = 0x1 << 0,
    DAXA_EXPLICIT_FEATURE_FLAG_ACCELERATION_STRUCTURE_CAPTURE_REPLAY = 0x1 << 1,
    DAXA_EXPLICIT_FEATURE_FLAG_VK_MEMORY_MODEL = 0x1 << 2,
    DAXA_EXPLICIT_FEATURE_FLAG_ROBUSTNESS_2 = 0x1 << 3,
} daxa_DeviceExplicitFeatureFlagBits;

typedef daxa_DeviceExplicitFeatureFlagBits daxa_ExplicitFeatureFlags;

typedef enum
{
    DAXA_IMPLICIT_FEATURE_FLAG_NONE = 0,
    DAXA_IMPLICIT_FEATURE_FLAG_MESH_SHADER = 0x1 << 0,
    DAXA_IMPLICIT_FEATURE_FLAG_BASIC_RAY_TRACING = 0x1 << 1,
    DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_PIPELINE = 0x1 << 2,
    DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_INVOCATION_REORDER = 0x1 << 3,
    DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_POSITION_FETCH = 0x1 << 4,
    DAXA_IMPLICIT_FEATURE_FLAG_CONSERVATIVE_RASTERIZATION = 0x1 << 5,
    DAXA_IMPLICIT_FEATURE_FLAG_SHADER_ATOMIC_INT64 = 0x1 << 6,
    DAXA_IMPLICIT_FEATURE_FLAG_IMAGE_ATOMIC64 = 0x1 << 7,
    DAXA_IMPLICIT_FEATURE_FLAG_SHADER_FLOAT16 = 0x1 << 8,
    DAXA_IMPLICIT_FEATURE_FLAG_SHADER_INT8 = 0x1 << 9,
    DAXA_IMPLICIT_FEATURE_FLAG_DYNAMIC_STATE_3 = 0x1 << 10,
    DAXA_IMPLICIT_FEATURE_FLAG_SHADER_ATOMIC_FLOAT = 0x1 << 11,
    DAXA_IMPLICIT_FEATURE_FLAG_SWAPCHAIN = 0x1 << 12,
    DAXA_IMPLICIT_FEATURE_FLAG_SHADER_INT16 = 0x1 << 13,
} daxa_DeviceImplicitFeatureFlagBits;

typedef daxa_DeviceImplicitFeatureFlagBits daxa_ImplicitFeatureFlags;

typedef struct
{
    uint32_t vulkan_api_version;
    uint32_t driver_version;
    uint32_t vendor_id;
    uint32_t device_id;
    daxa_DeviceType device_type;
    char device_name[256U];
    char pipeline_cache_uuid[16U];
    daxa_DeviceLimits limits;
    daxa_Optional(daxa_MeshShaderProperties) mesh_shader_properties;
    daxa_Optional(daxa_RayTracingPipelineProperties) ray_tracing_pipeline_properties;
    daxa_Optional(daxa_AccelerationStructureProperties) acceleration_structure_properties;
    daxa_Optional(daxa_RayTracingInvocationReorderProperties) ray_tracing_invocation_reorder_properties;
    daxa_u32 compute_queue_count;
    daxa_u32 transfer_queue_count;
    daxa_ImplicitFeatureFlags implicit_features;
    daxa_ExplicitFeatureFlags explicit_features;
    daxa_MissingRequiredVkFeature missing_required_feature;
} daxa_DeviceProperties;

/// DEPRECATED: use daxa_instance_create_device_2 and daxa_DeviceInfo2 instead!
DAXA_EXPORT int32_t
daxa_default_device_score(daxa_DeviceProperties const * properties);

/// WARNING: DEPRECATED, use daxa_ImplicitFeatureFlags and daxa_ExplicitFeatureFlags instead!
typedef enum
{
    DAXA_DEVICE_FLAG_BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT = 0x1 << 0,
    DAXA_DEVICE_FLAG_CONSERVATIVE_RASTERIZATION = 0x1 << 1,
    DAXA_DEVICE_FLAG_MESH_SHADER_BIT = 0x1 << 2,
    DAXA_DEVICE_FLAG_SHADER_ATOMIC64 = 0x1 << 3,
    DAXA_DEVICE_FLAG_IMAGE_ATOMIC64 = 0x1 << 4,
    DAXA_DEVICE_FLAG_VK_MEMORY_MODEL = 0x1 << 5,
    DAXA_DEVICE_FLAG_RAY_TRACING = 0x1 << 6,
    DAXA_DEVICE_FLAG_SHADER_FLOAT16 = 0x1 << 7,
    DAXA_DEVICE_FLAG_SHADER_INT8 = 0x1 << 8,
    DAXA_DEVICE_FLAG_ROBUST_BUFFER_ACCESS = 0x1 << 9,
    DAXA_DEVICE_FLAG_ROBUST_IMAGE_ACCESS = 0x1 << 10,
    DAXA_DEVICE_FLAG_DYNAMIC_STATE_3 = 0x1 << 11,
    DAXA_DEVICE_FLAG_SHADER_ATOMIC_FLOAT = 0x1 << 12,
} daxa_DeviceFlagBits;

/// WARNING: DEPRECATED, use daxa_ImplicitFeatureFlags and daxa_ExplicitFeatureFlags instead!
typedef uint32_t daxa_DeviceFlags;

/// WARNING: DEPRECATED, use daxa_DeviceInfo2 instead!
typedef struct
{
    int32_t (*selector)(daxa_DeviceProperties const * properties);
    daxa_DeviceFlags flags;
    uint32_t max_allowed_images;
    uint32_t max_allowed_buffers;
    uint32_t max_allowed_samplers;
    uint32_t max_allowed_acceleration_structures;
    daxa_SmallString name;
} daxa_DeviceInfo;

/// WARNING: DEPRECATED, use daxa_DeviceInfo2 instead!
static daxa_DeviceInfo const DAXA_DEFAULT_DEVICE_INFO = {
    .selector = &daxa_default_device_score,
    .flags = DAXA_DEVICE_FLAG_BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
    .max_allowed_images = 10000,
    .max_allowed_buffers = 10000,
    .max_allowed_samplers = 400,
    .max_allowed_acceleration_structures = 10000,
    .name = DAXA_ZERO_INIT,
};

typedef struct
{
    daxa_u32 physical_device_index;              // Index into list of devices returned from daxa_instance_list_devices_properties.
    daxa_ExplicitFeatureFlags explicit_features; // Explicit features must be manually enabled.
    uint32_t max_allowed_images;
    uint32_t max_allowed_buffers;
    uint32_t max_allowed_samplers;
    uint32_t max_allowed_acceleration_structures;
    daxa_SmallString name;
} daxa_DeviceInfo2;

static daxa_DeviceInfo2 const DAXA_DEFAULT_DEVICE_INFO_2 = {
    .physical_device_index = ~0u,
    .explicit_features = DAXA_EXPLICIT_FEATURE_FLAG_BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY,
    .max_allowed_images = 10000,
    .max_allowed_buffers = 10000,
    .max_allowed_samplers = 400,
    .max_allowed_acceleration_structures = 10000,
    .name = DAXA_ZERO_INIT,
};

typedef struct
{
    daxa_QueueFamily family;
    daxa_u32 index;
} daxa_Queue;

static daxa_Queue const DAXA_QUEUE_MAIN = {DAXA_QUEUE_FAMILY_MAIN, 0};
static daxa_Queue const DAXA_QUEUE_COMPUTE_0 = {DAXA_QUEUE_FAMILY_COMPUTE, 0};
static daxa_Queue const DAXA_QUEUE_COMPUTE_1 = {DAXA_QUEUE_FAMILY_COMPUTE, 1};
static daxa_Queue const DAXA_QUEUE_COMPUTE_2 = {DAXA_QUEUE_FAMILY_COMPUTE, 2};
static daxa_Queue const DAXA_QUEUE_COMPUTE_3 = {DAXA_QUEUE_FAMILY_COMPUTE, 3};
static daxa_Queue const DAXA_QUEUE_COMPUTE_4 = {DAXA_QUEUE_FAMILY_COMPUTE, 4};
static daxa_Queue const DAXA_QUEUE_COMPUTE_5 = {DAXA_QUEUE_FAMILY_COMPUTE, 5};
static daxa_Queue const DAXA_QUEUE_COMPUTE_6 = {DAXA_QUEUE_FAMILY_COMPUTE, 6};
static daxa_Queue const DAXA_QUEUE_COMPUTE_7 = {DAXA_QUEUE_FAMILY_COMPUTE, 7};
static daxa_Queue const DAXA_QUEUE_TRANSFER_0 = {DAXA_QUEUE_FAMILY_TRANSFER, 0};
static daxa_Queue const DAXA_QUEUE_TRANSFER_1 = {DAXA_QUEUE_FAMILY_TRANSFER, 1};

typedef struct
{
    daxa_Queue queue;
    VkPipelineStageFlags wait_stages;
    daxa_ExecutableCommandList const * command_lists;
    uint64_t command_list_count;
    daxa_BinarySemaphore const * wait_binary_semaphores;
    uint64_t wait_binary_semaphore_count;
    daxa_BinarySemaphore const * signal_binary_semaphores;
    uint64_t signal_binary_semaphore_count;
    daxa_TimelinePair const * wait_timeline_semaphores;
    uint64_t wait_timeline_semaphore_count;
    daxa_TimelinePair const * signal_timeline_semaphores;
    uint64_t signal_timeline_semaphore_count;
} daxa_CommandSubmitInfo;

static daxa_CommandSubmitInfo const DAXA_DEFAULT_COMMAND_SUBMIT_INFO = DAXA_ZERO_INIT;

typedef struct
{
    daxa_BinarySemaphore const * wait_binary_semaphores;
    uint64_t wait_binary_semaphore_count;
    daxa_Swapchain swapchain;
    daxa_Queue queue;
} daxa_PresentInfo;

static daxa_PresentInfo const DAXA_DEFAULT_PRESENT_INFO = DAXA_ZERO_INIT;

typedef struct
{
    daxa_BufferInfo buffer_info;
    daxa_MemoryBlock * memory_block;
    size_t offset;
} daxa_MemoryBlockBufferInfo;

static daxa_MemoryBlockBufferInfo const DAXA_DEFAULT_MEMORY_BLOCK_BUFFER_INFO = DAXA_ZERO_INIT;

typedef struct
{
    daxa_ImageInfo image_info;
    daxa_MemoryBlock * memory_block;
    size_t offset;
} daxa_MemoryBlockImageInfo;

static daxa_MemoryBlockImageInfo const DAXA_DEFAULT_MEMORY_BLOCK_IMAGE_INFO = DAXA_ZERO_INIT;

typedef struct
{
    daxa_TlasInfo tlas_info;
    daxa_BufferId buffer_id;
    uint64_t offset;
} daxa_BufferTlasInfo;

static daxa_BufferTlasInfo const DAXA_DEFAULT_BUFFER_TLAS_INFO = DAXA_ZERO_INIT;

typedef struct
{
    daxa_BlasInfo blas_info;
    daxa_BufferId buffer_id;
    uint64_t offset;
} daxa_BufferBlasInfo;

static daxa_BufferBlasInfo const DAXA_DEFAULT_BUFFER_BLAS_INFO = DAXA_ZERO_INIT;

typedef struct
{
    uint64_t acceleration_structure_size;
    uint64_t update_scratch_size;
    uint64_t build_scratch_size;
} daxa_AccelerationStructureBuildSizesInfo;

DAXA_EXPORT VkMemoryRequirements
daxa_dvc_buffer_memory_requirements(daxa_Device device, daxa_BufferInfo const * info);
DAXA_EXPORT VkMemoryRequirements
daxa_dvc_image_memory_requirements(daxa_Device device, daxa_ImageInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_memory(daxa_Device device, daxa_MemoryBlockInfo const * info, daxa_MemoryBlock * out_memory_block);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_get_tlas_build_sizes(daxa_Device device, daxa_TlasBuildInfo const * build_info, daxa_AccelerationStructureBuildSizesInfo * out);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_get_blas_build_sizes(daxa_Device device, daxa_BlasBuildInfo const * build_info, daxa_AccelerationStructureBuildSizesInfo * out);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_buffer(daxa_Device device, daxa_BufferInfo const * info, daxa_BufferId * out_id);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_image(daxa_Device device, daxa_ImageInfo const * info, daxa_ImageId * out_id);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_buffer_from_memory_block(daxa_Device device, daxa_MemoryBlockBufferInfo const * info, daxa_BufferId * out_id);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_image_from_block(daxa_Device device, daxa_MemoryBlockImageInfo const * info, daxa_ImageId * out_id);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_image_view(daxa_Device device, daxa_ImageViewInfo const * info, daxa_ImageViewId * out_id);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_sampler(daxa_Device device, daxa_SamplerInfo const * info, daxa_SamplerId * out_id);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_tlas(daxa_Device device, daxa_TlasInfo const * info, daxa_TlasId * out_id);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_blas(daxa_Device device, daxa_BlasInfo const * info, daxa_BlasId * out_id);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_tlas_from_buffer(daxa_Device device, daxa_BufferTlasInfo const * info, daxa_TlasId * out_id);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_blas_from_buffer(daxa_Device device, daxa_BufferBlasInfo const * info, daxa_BlasId * out_id);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_destroy_buffer(daxa_Device device, daxa_BufferId buffer);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_destroy_image(daxa_Device device, daxa_ImageId image);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_destroy_image_view(daxa_Device device, daxa_ImageViewId id);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_destroy_sampler(daxa_Device device, daxa_SamplerId sampler);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_destroy_tlas(daxa_Device device, daxa_TlasId tlas);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_destroy_blas(daxa_Device device, daxa_BlasId blas);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_info_buffer(daxa_Device device, daxa_BufferId buffer, daxa_BufferInfo * out_info);
DAXA_EXPORT daxa_Result
daxa_dvc_info_image(daxa_Device device, daxa_ImageId image, daxa_ImageInfo * out_info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_info_image_view(daxa_Device device, daxa_ImageViewId id, daxa_ImageViewInfo * out_info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_info_sampler(daxa_Device device, daxa_SamplerId sampler, daxa_SamplerInfo * out_info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_info_tlas(daxa_Device device, daxa_TlasId acceleration_structure, daxa_TlasInfo * out_info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_info_blas(daxa_Device device, daxa_BlasId acceleration_structure, daxa_BlasInfo * out_info);

DAXA_EXPORT daxa_Bool8
daxa_dvc_is_buffer_valid(daxa_Device device, daxa_BufferId buffer);
DAXA_EXPORT daxa_Bool8
daxa_dvc_is_image_valid(daxa_Device device, daxa_ImageId image);
DAXA_EXPORT daxa_Bool8
daxa_dvc_is_image_view_valid(daxa_Device device, daxa_ImageViewId image_view);
DAXA_EXPORT daxa_Bool8
daxa_dvc_is_sampler_valid(daxa_Device device, daxa_SamplerId sampler);
DAXA_EXPORT daxa_Bool8
daxa_dvc_is_tlas_valid(daxa_Device device, daxa_TlasId tlas);
DAXA_EXPORT daxa_Bool8
daxa_dvc_is_blas_valid(daxa_Device device, daxa_BlasId blas);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_get_vk_buffer(daxa_Device device, daxa_BufferId buffer, VkBuffer * out_vk_handle);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_get_vk_image(daxa_Device device, daxa_ImageId image, VkImage * out_vk_handle);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_get_vk_image_view(daxa_Device device, daxa_ImageViewId id, VkImageView * out_vk_handle);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_get_vk_sampler(daxa_Device device, daxa_SamplerId sampler, VkSampler * out_vk_handle);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_get_vk_tlas(daxa_Device device, daxa_TlasId tlas, VkAccelerationStructureInstanceKHR * out_vk_handle);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_get_vk_blas(daxa_Device device, daxa_BlasId blas, VkAccelerationStructureInstanceKHR * out_vk_handle);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_buffer_device_address(daxa_Device device, daxa_BufferId buffer, daxa_DeviceAddress * out_addr);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_buffer_host_address(daxa_Device device, daxa_BufferId buffer, void ** out_addr);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_tlas_device_address(daxa_Device device, daxa_TlasId tlas, daxa_DeviceAddress * out_addr);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_blas_device_address(daxa_Device device, daxa_BlasId blas, daxa_DeviceAddress * out_addr);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_raster_pipeline(daxa_Device device, daxa_RasterPipelineInfo const * info, daxa_RasterPipeline * out_pipeline);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_compute_pipeline(daxa_Device device, daxa_ComputePipelineInfo const * info, daxa_ComputePipeline * out_pipeline);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_ray_tracing_pipeline(daxa_Device device, daxa_RayTracingPipelineInfo const * info, daxa_RayTracingPipeline * out_pipeline);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_swapchain(daxa_Device device, daxa_SwapchainInfo const * info, daxa_Swapchain * out_swapchain);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_command_recorder(daxa_Device device, daxa_CommandRecorderInfo const * info, daxa_CommandRecorder * out_command_list);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_binary_semaphore(daxa_Device device, daxa_BinarySemaphoreInfo const * info, daxa_BinarySemaphore * out_binary_semaphore);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_timeline_semaphore(daxa_Device device, daxa_TimelineSemaphoreInfo const * info, daxa_TimelineSemaphore * out_timeline_semaphore);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_event(daxa_Device device, daxa_EventInfo const * info, daxa_Event * out_event);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_create_timeline_query_pool(daxa_Device device, daxa_TimelineQueryPoolInfo const * info, daxa_TimelineQueryPool * out_timeline_query_pool);

DAXA_EXPORT VkDevice
daxa_dvc_get_vk_device(daxa_Device device);
DAXA_EXPORT VkPhysicalDevice
daxa_dvc_get_vk_physical_device(daxa_Device device);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_queue_wait_idle(daxa_Device device, daxa_Queue queue);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_queue_count(daxa_Device device, daxa_QueueFamily queue_family, daxa_u32 * out_value);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_wait_idle(daxa_Device device);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_submit(daxa_Device device, daxa_CommandSubmitInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_present(daxa_Device device, daxa_PresentInfo const * info);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_dvc_collect_garbage(daxa_Device device);

DAXA_EXPORT daxa_DeviceInfo2 const *
daxa_dvc_info(daxa_Device device);
DAXA_EXPORT daxa_DeviceProperties const *
daxa_dvc_properties(daxa_Device device);

// Returns previous ref count.
DAXA_EXPORT uint64_t
daxa_dvc_inc_refcnt(daxa_Device device);
// Returns previous ref count.
DAXA_EXPORT uint64_t
daxa_dvc_dec_refcnt(daxa_Device device);

#endif // #ifndef __DAXA_DEVICE_H__
