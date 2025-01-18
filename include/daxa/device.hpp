#pragma once

#include <daxa/core.hpp>
#include <daxa/gpu_resources.hpp>
#include <daxa/pipeline.hpp>
#include <daxa/swapchain.hpp>
#include <daxa/command_recorder.hpp>
#include <daxa/sync.hpp>

#include <bit>

namespace daxa
{
    static constexpr inline u32 MAX_COMPUTE_QUEUE_COUNT = 8u;
    static constexpr inline u32 MAX_TRANSFER_QUEUE_COUNT = 2u;

    enum struct DeviceType
    {
        OTHER = 0,
        INTEGRATED_GPU = 1,
        DISCRETE_GPU = 2,
        VIRTUAL_GPU = 3,
        CPU = 4,
        MAX_ENUM = 0x7fffffff,
    };

    struct DeviceLimits
    {
        u32 max_image_dimension1d;
        u32 max_image_dimension2d;
        u32 max_image_dimension3d;
        u32 max_image_dimension_cube;
        u32 max_image_array_layers;
        u32 max_texel_buffer_elements;
        u32 max_uniform_buffer_range;
        u32 max_storage_buffer_range;
        u32 max_push_constants_size;
        u32 max_memory_allocation_count;
        u32 max_sampler_allocation_count;
        u64 buffer_image_granularity;
        u64 sparse_address_space_size;
        u32 max_bound_descriptor_sets;
        u32 max_per_stage_descriptor_samplers;
        u32 max_per_stage_descriptor_uniform_buffers;
        u32 max_per_stage_descriptor_storage_buffers;
        u32 max_per_stage_descriptor_sampled_images;
        u32 max_per_stage_descriptor_storage_images;
        u32 max_per_stage_descriptor_input_attachments;
        u32 max_per_stage_resources;
        u32 max_descriptor_set_samplers;
        u32 max_descriptor_set_uniform_buffers;
        u32 max_descriptor_set_uniform_buffers_dynamic;
        u32 max_descriptor_set_storage_buffers;
        u32 max_descriptor_set_storage_buffers_dynamic;
        u32 max_descriptor_set_sampled_images;
        u32 max_descriptor_set_storage_images;
        u32 max_descriptor_set_input_attachments;
        u32 max_vertex_input_attributes;
        u32 max_vertex_input_bindings;
        u32 max_vertex_input_attribute_offset;
        u32 max_vertex_input_binding_stride;
        u32 max_vertex_output_components;
        u32 max_tessellation_generation_level;
        u32 max_tessellation_patch_size;
        u32 max_tessellation_control_per_vertex_input_components;
        u32 max_tessellation_control_per_vertex_output_components;
        u32 max_tessellation_control_per_patch_output_components;
        u32 max_tessellation_control_total_output_components;
        u32 max_tessellation_evaluation_input_components;
        u32 max_tessellation_evaluation_output_components;
        u32 max_geometry_shader_invocations;
        u32 max_geometry_input_components;
        u32 max_geometry_output_components;
        u32 max_geometry_output_vertices;
        u32 max_geometry_total_output_components;
        u32 max_fragment_input_components;
        u32 max_fragment_output_attachments;
        u32 max_fragment_dual_src_attachments;
        u32 max_fragment_combined_output_resources;
        u32 max_compute_shared_memory_size;
        u32 max_compute_work_group_count[3];
        u32 max_compute_work_group_invocations;
        u32 max_compute_work_group_size[3];
        u32 sub_pixel_precision_bits;
        u32 sub_texel_precision_bits;
        u32 mipmap_precision_bits;
        u32 max_draw_indexed_index_value;
        u32 max_draw_indirect_count;
        f32 max_sampler_lod_bias;
        f32 max_sampler_anisotropy;
        u32 max_viewports;
        u32 max_viewport_dimensions[2];
        f32 viewport_bounds_range[2];
        u32 viewport_sub_pixel_bits;
        usize min_memory_map_alignment;
        u64 min_texel_buffer_offset_alignment;
        u64 min_uniform_buffer_offset_alignment;
        u64 min_storage_buffer_offset_alignment;
        i32 min_texel_offset;
        u32 max_texel_offset;
        i32 min_texel_gather_offset;
        u32 max_texel_gather_offset;
        f32 min_interpolation_offset;
        f32 max_interpolation_offset;
        u32 sub_pixel_interpolation_offset_bits;
        u32 max_framebuffer_width;
        u32 max_framebuffer_height;
        u32 max_framebuffer_layers;
        u32 framebuffer_color_sample_counts;
        u32 framebuffer_depth_sample_counts;
        u32 framebuffer_stencil_sample_counts;
        u32 framebuffer_no_attachments_sample_counts;
        u32 max_color_attachments;
        u32 sampled_image_color_sample_counts;
        u32 sampled_image_integer_sample_counts;
        u32 sampled_image_depth_sample_counts;
        u32 sampled_image_stencil_sample_counts;
        u32 storage_image_sample_counts;
        u32 max_sample_mask_words;
        u32 timestamp_compute_and_graphics;
        f32 timestamp_period;
        u32 max_clip_distances;
        u32 max_cull_distances;
        u32 max_combined_clip_and_cull_distances;
        u32 discrete_queue_priorities;
        f32 point_size_range[2];
        f32 line_width_range[2];
        f32 point_size_granularity;
        f32 line_width_granularity;
        u32 strict_lines;
        u32 standard_sample_locations;
        u64 optimal_buffer_copy_offset_alignment;
        u64 optimal_buffer_copy_row_pitch_alignment;
        u64 non_coherent_atom_size;
    };

    struct MeshShaderDeviceProperties
    {
        u32 max_task_work_group_total_count = {};
        u32 max_task_work_group_count[3] = {};
        u32 max_task_work_group_invocations = {};
        u32 max_task_work_group_size[3] = {};
        u32 max_task_payload_size = {};
        u32 max_task_shared_memory_size = {};
        u32 max_task_payload_and_shared_memory_size = {};
        u32 max_mesh_work_group_total_count = {};
        u32 max_mesh_work_group_count[3] = {};
        u32 max_mesh_work_group_invocations = {};
        u32 max_mesh_work_group_size[3] = {};
        u32 max_mesh_shared_memory_size = {};
        u32 max_mesh_payload_and_shared_memory_size = {};
        u32 max_mesh_output_memory_size = {};
        u32 max_mesh_payload_and_output_memory_size = {};
        u32 max_mesh_output_components = {};
        u32 max_mesh_output_vertices = {};
        u32 max_mesh_output_primitives = {};
        u32 max_mesh_output_layers = {};
        u32 max_mesh_multiview_view_count = {};
        u32 mesh_output_per_vertex_granularity = {};
        u32 mesh_output_per_primitive_granularity = {};
        u32 max_preferred_task_work_group_invocations = {};
        u32 max_preferred_mesh_work_group_invocations = {};
        bool prefers_local_invocation_vertex_output = {};
        bool prefers_local_invocation_primitive_output = {};
        bool prefers_compact_vertex_output = {};
        bool prefers_compact_primitive_output = {};
    };

    struct RayTracingPipelineProperties
    {
        u32 shader_group_handle_size = {};
        u32 max_ray_recursion_depth = {};
        u32 max_shader_group_stride = {};
        u32 shader_group_base_alignment = {};
        u32 shader_group_handle_capture_replay_size = {};
        u32 max_ray_dispatch_invocation_count = {};
        u32 shader_group_handle_alignment = {};
        u32 max_ray_hit_attribute_size = {};
    };

    struct AccelerationStructureProperties
    {
        u64 max_geometry_count = {};
        u64 max_instance_count = {};
        u64 max_primitive_count = {};
        u32 max_per_stage_descriptor_acceleration_structures = {};
        u32 max_per_stage_descriptor_update_after_bind_acceleration_structures = {};
        u32 max_descriptor_set_acceleration_structures = {};
        u32 max_descriptor_set_update_after_bind_acceleration_structures = {};
        u32 min_acceleration_structure_scratch_offset_alignment = {};
    };

    struct InvocationReorderProperties
    {
        u32 invocation_reorder_mode = {};
    };

    struct DeviceFlagsProperties
    {
        using Data = u32;
    };
    using DeviceFlags = Flags<DeviceFlagsProperties>;
    struct [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] DeviceFlagBits
    {
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags NONE = {0x00000000};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT = {0x1 << 0};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags CONSERVATIVE_RASTERIZATION = {0x1 << 1};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags MESH_SHADER = {0x1 << 2};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags SHADER_ATOMIC64 = {0x1 << 3};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags IMAGE_ATOMIC64 = {0x1 << 4};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags VK_MEMORY_MODEL = {0x1 << 5};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags RAY_TRACING = {0x1 << 6};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags SHADER_FLOAT16 = {0x1 << 7};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags ROBUST_BUFFER_ACCESS = {0x1 << 9};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags ROBUST_IMAGE_ACCESS = {0x1 << 10};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags DYNAMIC_STATE_3 = {0x1 << 11};
        [[deprecated("Use ExplicitFeatureFlags or ImplicitFeatureFlags instead")]] static inline constexpr DeviceFlags SHADER_ATOMIC_FLOAT = {0x1 << 12};
    };

    enum struct MissingRequiredVkFeature
    {
        NONE,
        IMAGE_CUBE_ARRAY,
        INDEPENDENT_BLEND,
        TESSELLATION_SHADER,
        MULTI_DRAW_INDIRECT,
        DEPTH_CLAMP,
        FILL_MODE_NON_SOLID,
        WIDE_LINES,
        SAMPLER_ANISOTROPY,
        FRAGMENT_STORES_AND_ATOMICS,
        SHADER_STORAGE_IMAGE_MULTISAMPLE,
        SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT,
        SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT,
        SHADER_INT64,
        VARIABLE_POINTERS_STORAGE_BUFFER,
        VARIABLE_POINTERS,
        BUFFER_DEVICE_ADDRESS,
        BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY,
        BUFFER_DEVICE_ADDRESS_MULTI_DEVICE,
        SHADER_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING,
        SHADER_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        SHADER_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING,
        DESCRIPTOR_BINDING_SAMPLED_IMAGE_UPDATE_AFTER_BIND,
        DESCRIPTOR_BINDING_STORAGE_IMAGE_UPDATE_AFTER_BIND,
        DESCRIPTOR_BINDING_STORAGE_BUFFER_UPDATE_AFTER_BIND,
        DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING,
        DESCRIPTOR_BINDING_PARTIALLY_BOUND,
        RUNTIME_DESCRIPTOR_ARRAY,
        HOST_QUERY_RESET,
        DYNAMIC_RENDERING,
        SYNCHRONIZATION2,
        TIMELINE_SEMAPHORE,
        SUBGROUP_SIZE_CONTROL,
        COMPUTE_FULL_SUBGROUPS,
        SCALAR_BLOCK_LAYOUT,
        ACCELERATION_STRUCTURE_CAPTURE_REPLAY,
        VULKAN_MEMORY_MODEL,
        ROBUST_BUFFER_ACCESS2,
        ROBUST_IMAGE_ACCESS2,
        MAX_ENUM
    };

    struct ExplicitFeatureProperties
    {
        using Data = u32;
    };
    using ExplicitFeatureFlags = Flags<ExplicitFeatureProperties>;
    struct ExplicitFeatureFlagBits
    {
        static inline constexpr ExplicitFeatureFlags NONE = {0};
        static inline constexpr ExplicitFeatureFlags BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY = {0x1 << 0};
        static inline constexpr ExplicitFeatureFlags ACCELERATION_STRUCTURE_CAPTURE_REPLAY = {0x1 << 1};
        static inline constexpr ExplicitFeatureFlags VK_MEMORY_MODEL = {0x1 << 2};
        static inline constexpr ExplicitFeatureFlags ROBUSTNESS_2 = {0x1 << 3};
    };

    struct ImplicitFeatureProperties
    {
        using Data = u32;
    };
    using ImplicitFeatureFlags = Flags<ImplicitFeatureProperties>;
    struct ImplicitFeatureFlagBits
    {
        static inline constexpr ImplicitFeatureFlags NONE = {0};
        static inline constexpr ImplicitFeatureFlags MESH_SHADER = {0x1 << 0};
        static inline constexpr ImplicitFeatureFlags BASIC_RAY_TRACING = {0x1 << 1};
        static inline constexpr ImplicitFeatureFlags RAY_TRACING_PIPELINE = {0x1 << 2};
        static inline constexpr ImplicitFeatureFlags RAY_TRACING_INVOCATION_REORDER = {0x1 << 3};
        static inline constexpr ImplicitFeatureFlags RAY_TRACING_POSITION_FETCH = {0x1 << 4};
        static inline constexpr ImplicitFeatureFlags CONSERVATIVE_RASTERIZATION = {0x1 << 5};
        static inline constexpr ImplicitFeatureFlags SHADER_ATOMIC_INT64 = {0x1 << 6};
        static inline constexpr ImplicitFeatureFlags IMAGE_ATOMIC64 = {0x1 << 7};
        static inline constexpr ImplicitFeatureFlags SHADER_FLOAT16 = {0x1 << 8};
        static inline constexpr ImplicitFeatureFlags SHADER_INT8 = {0x1 << 9};
        static inline constexpr ImplicitFeatureFlags DYNAMIC_STATE_3 = {0x1 << 10};
        static inline constexpr ImplicitFeatureFlags SHADER_ATOMIC_FLOAT = {0x1 << 11};
        static inline constexpr ImplicitFeatureFlags SWAPCHAIN = {0x1 << 12};
        static inline constexpr ImplicitFeatureFlags SHADER_INT16 = {0x1 << 13};
    };

    struct DeviceProperties
    {
        u32 vulkan_api_version = {};
        u32 driver_version = {};
        u32 vendor_id = {};
        u32 device_id = {};
        DeviceType device_type = {};
        u8 device_name[256U] = {};
        u8 pipeline_cache_uuid[16U] = {};
        DeviceLimits limits = {};
        Optional<MeshShaderDeviceProperties> mesh_shading_properties = {};
        Optional<RayTracingPipelineProperties> ray_tracing_properties = {};
        Optional<AccelerationStructureProperties> acceleration_structure_properties = {};
        Optional<InvocationReorderProperties> invocation_reorder_properties = {};
        u32 compute_queue_count = {};
        u32 transfer_queue_count = {};
        ImplicitFeatureFlags implicit_features;
        ExplicitFeatureFlags explicit_features;
        MissingRequiredVkFeature missing_required_feature;
    };

    [[deprecated("Use create_device_2 and Instance::choose_device instead")]] DAXA_EXPORT_CXX auto default_device_score(DeviceProperties const & device_props) -> i32;

    struct [[deprecated("Use DeviceInfo2 instead")]] DeviceInfo
    {
        i32 (*selector)(DeviceProperties const & properties) = default_device_score;
        DeviceFlags flags =
            DeviceFlagBits::BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT |
            DeviceFlagBits::SHADER_ATOMIC64 |
            DeviceFlagBits::IMAGE_ATOMIC64 |
            DeviceFlagBits::DYNAMIC_STATE_3;
        // Make sure your device actually supports the max numbers, as device creation will fail otherwise.
        u32 max_allowed_images = 10'000;
        u32 max_allowed_buffers = 10'000;
        u32 max_allowed_samplers = 400;
        u32 max_allowed_acceleration_structures = 10'000;
        SmallString name = {};
    };

    struct DeviceInfo2
    {
        u32 physical_device_index = ~0u;
        ExplicitFeatureFlags explicit_features = {};
        // Make sure your device actually supports the max numbers, as device creation will fail otherwise.
        u32 max_allowed_images = 10'000;
        u32 max_allowed_buffers = 10'000;
        u32 max_allowed_samplers = 400;
        u32 max_allowed_acceleration_structures = 10'000;
        SmallString name = {};
    };

    struct Queue
    {
        QueueFamily family = {};
        u32 index = {};
    };

    static constexpr inline Queue QUEUE_MAIN = Queue{QueueFamily::MAIN, 0};
    static constexpr inline Queue QUEUE_COMPUTE_0 = Queue{QueueFamily::COMPUTE, 0};
    static constexpr inline Queue QUEUE_COMPUTE_1 = Queue{QueueFamily::COMPUTE, 1};
    static constexpr inline Queue QUEUE_COMPUTE_2 = Queue{QueueFamily::COMPUTE, 2};
    static constexpr inline Queue QUEUE_COMPUTE_3 = Queue{QueueFamily::COMPUTE, 3};
    static constexpr inline Queue QUEUE_COMPUTE_4 = Queue{QueueFamily::COMPUTE, 4};
    static constexpr inline Queue QUEUE_COMPUTE_5 = Queue{QueueFamily::COMPUTE, 5};
    static constexpr inline Queue QUEUE_COMPUTE_6 = Queue{QueueFamily::COMPUTE, 6};
    static constexpr inline Queue QUEUE_COMPUTE_7 = Queue{QueueFamily::COMPUTE, 7};
    static constexpr inline Queue QUEUE_TRANSFER_0 = Queue{QueueFamily::TRANSFER, 0};
    static constexpr inline Queue QUEUE_TRANSFER_1 = Queue{QueueFamily::TRANSFER, 1};

    struct CommandSubmitInfo
    {
        Queue queue = daxa::QUEUE_MAIN;
        PipelineStageFlags wait_stages = {};
        daxa::Span<ExecutableCommandList const> command_lists = {};
        daxa::Span<BinarySemaphore const> wait_binary_semaphores = {};
        daxa::Span<BinarySemaphore const> signal_binary_semaphores = {};
        daxa::Span<std::pair<TimelineSemaphore, u64> const> wait_timeline_semaphores = {};
        daxa::Span<std::pair<TimelineSemaphore, u64> const> signal_timeline_semaphores = {};
    };

    struct PresentInfo
    {
        daxa::Span<BinarySemaphore const> wait_binary_semaphores = {};
        Swapchain swapchain;
        Queue queue = QUEUE_MAIN;
    };

    struct MemoryBlockBufferInfo
    {
        BufferInfo buffer_info = {};
        MemoryBlock & memory_block;
        usize offset = {};
    };

    struct MemoryBlockImageInfo
    {
        ImageInfo image_info = {};
        MemoryBlock & memory_block;
        usize offset = {};
    };

    struct AccelerationStructureBuildSizesInfo
    {
        u64 acceleration_structure_size;
        u64 update_scratch_size;
        u64 build_scratch_size;
    };

    struct BufferTlasInfo
    {
        TlasInfo tlas_info = {};
        BufferId buffer_id = {};
        u64 offset = {};
    };

    struct BufferBlasInfo
    {
        BlasInfo blas_info = {};
        BufferId buffer_id = {};
        u64 offset = {};
    };

    /**
     * @brief   Device represents a logical device that may be a virtual or physical gpu.
     *          Device manages all general gpu operations that are not handled by other objects.
     *          All objects connected to the device are created by it.
     *
     * THREADSAFETY:
     * * is internally synchronized
     * * can be passed between different threads
     * * may be accessed by multiple threads at the same time
     */
    struct DAXA_EXPORT_CXX Device final : ManagedPtr<Device, daxa_Device>
    {
        Device() = default;

        [[nodiscard]] auto create_memory(MemoryBlockInfo const & info) -> MemoryBlock;

        [[nodiscard]] auto tlas_build_sizes(TlasBuildInfo const & info) -> AccelerationStructureBuildSizesInfo;
        [[nodiscard]] auto blas_build_sizes(BlasBuildInfo const & info) -> AccelerationStructureBuildSizesInfo;
        [[nodiscard]] auto as_build_sizes(TlasBuildInfo const & info) { return tlas_build_sizes(info); }
        [[nodiscard]] auto as_build_sizes(BlasBuildInfo const & info) { return blas_build_sizes(info); }

        [[nodiscard]] auto buffer_memory_requirements(BufferInfo const & info) const -> MemoryRequirements;
        [[nodiscard]] auto image_memory_requirements(ImageInfo const & info) const -> MemoryRequirements;
        [[nodiscard]] auto memory_requirements(BufferInfo const & info) const { return buffer_memory_requirements(info); }
        [[nodiscard]] auto memory_requirements(ImageInfo const & info) const { return image_memory_requirements(info); }

        [[nodiscard]] auto create_buffer(BufferInfo const & info) -> BufferId;
        [[nodiscard]] auto create_image(ImageInfo const & info) -> ImageId;
        [[nodiscard]] auto create_buffer_from_memory_block(MemoryBlockBufferInfo const & info) -> BufferId;
        [[nodiscard]] auto create_image_from_memory_block(MemoryBlockImageInfo const & info) -> ImageId;
        [[nodiscard]] auto create_image_view(ImageViewInfo const & info) -> ImageViewId;
        [[nodiscard]] auto create_sampler(SamplerInfo const & info) -> SamplerId;
        [[nodiscard]] auto create_tlas(TlasInfo const & info) -> TlasId;
        [[nodiscard]] auto create_blas(BlasInfo const & info) -> BlasId;
        [[nodiscard]] auto create_tlas_from_buffer(BufferTlasInfo const & info) -> TlasId;
        [[nodiscard]] auto create_blas_from_buffer(BufferBlasInfo const & info) -> BlasId;
        [[nodiscard]] auto create(BufferInfo const & info) { return create_buffer(info); }
        [[nodiscard]] auto create(ImageInfo const & info) { return create_image(info); }
        [[nodiscard]] auto create(MemoryBlockBufferInfo const & info) { return create_buffer_from_memory_block(info); }
        [[nodiscard]] auto create(MemoryBlockImageInfo const & info) { return create_image_from_memory_block(info); }
        [[nodiscard]] auto create(ImageViewInfo const & info) { return create_image_view(info); }
        [[nodiscard]] auto create(SamplerInfo const & info) { return create_sampler(info); }
        [[nodiscard]] auto create(TlasInfo const & info) { return create_tlas(info); }
        [[nodiscard]] auto create(BlasInfo const & info) { return create_blas(info); }
        [[nodiscard]] auto create(BufferTlasInfo const & info) { return create_tlas_from_buffer(info); }
        [[nodiscard]] auto create(BufferBlasInfo const & info) { return create_blas_from_buffer(info); }

        void destroy_buffer(BufferId id);
        void destroy_image(ImageId id);
        void destroy_image_view(ImageViewId id);
        void destroy_sampler(SamplerId id);
        void destroy_tlas(TlasId id);
        void destroy_blas(BlasId id);
        void destroy(BufferId id) { destroy_buffer(id); }
        void destroy(ImageId id) { destroy_image(id); }
        void destroy(ImageViewId id) { destroy_image_view(id); }
        void destroy(SamplerId id) { destroy_sampler(id); }
        void destroy(TlasId id) { destroy_tlas(id); }
        void destroy(BlasId id) { destroy_blas(id); }

        // TODO: deprecate?

        /// @brief  Daxa stores each create info and keeps it up to date if the object changes
        ///         This is also the case for gpu resources (buffer, image(view), sampler, as).
        /// @param id of the object.
        /// @return a value copy of the info. Returns nullopt when the id is invalid.
        [[nodiscard]] auto buffer_info(BufferId id) const -> Optional<BufferInfo>;
        [[nodiscard]] auto image_info(ImageId id) const -> Optional<ImageInfo>;
        [[nodiscard]] auto image_view_info(ImageViewId id) const -> Optional<ImageViewInfo>;
        [[nodiscard]] auto sampler_info(SamplerId id) const -> Optional<SamplerInfo>;
        [[nodiscard]] auto tlas_info(TlasId id) const -> Optional<TlasInfo>;
        [[nodiscard]] auto blas_info(BlasId id) const -> Optional<BlasInfo>;
        [[nodiscard]] auto info(BufferId id) const { return buffer_info(id); }
        [[nodiscard]] auto info(ImageId id) const { return image_info(id); }
        [[nodiscard]] auto info(ImageViewId id) const { return image_view_info(id); }
        [[nodiscard]] auto info(SamplerId id) const { return sampler_info(id); }
        [[nodiscard]] auto info(TlasId id) const { return tlas_info(id); }
        [[nodiscard]] auto info(BlasId id) const { return blas_info(id); }

        /// @brief  Will describe if a given id is valid.
        ///         An id is valid as long as it was created by the device and not yet destroyed.
        /// @param id or the object.
        /// @return validity of id
        [[nodiscard]] auto is_image_id_valid(ImageId id) const -> bool;
        [[nodiscard]] auto is_image_view_id_valid(ImageViewId id) const -> bool;
        [[nodiscard]] auto is_buffer_id_valid(BufferId id) const -> bool;
        [[nodiscard]] auto is_sampler_id_valid(SamplerId id) const -> bool;
        [[nodiscard]] auto is_tlas_id_valid(TlasId id) const -> bool;
        [[nodiscard]] auto is_blas_id_valid(BlasId id) const -> bool;
        [[nodiscard]] auto is_id_valid(ImageId id) const { return is_image_id_valid(id); }
        [[nodiscard]] auto is_id_valid(ImageViewId id) const { return is_image_view_id_valid(id); }
        [[nodiscard]] auto is_id_valid(BufferId id) const { return is_buffer_id_valid(id); }
        [[nodiscard]] auto is_id_valid(SamplerId id) const { return is_sampler_id_valid(id); }
        [[nodiscard]] auto is_id_valid(TlasId id) const { return is_tlas_id_valid(id); }
        [[nodiscard]] auto is_id_valid(BlasId id) const { return is_blas_id_valid(id); }

        [[nodiscard]] auto buffer_device_address(BufferId id) const -> Optional<DeviceAddress>;
        [[nodiscard]] auto blas_device_address(BlasId id) const -> Optional<DeviceAddress>;
        [[nodiscard]] auto tlas_device_address(TlasId id) const -> Optional<DeviceAddress>;
        [[nodiscard]] auto device_address(BufferId id) const { return buffer_device_address(id); }
        [[nodiscard]] auto device_address(BlasId id) const { return blas_device_address(id); }
        [[nodiscard]] auto device_address(TlasId id) const { return tlas_device_address(id); }

        [[nodiscard]] auto buffer_host_address(BufferId id) const -> Optional<std::byte *>;
        template <typename T>
        [[nodiscard]] auto buffer_host_address_as(BufferId id) const -> Optional<T *>
        {
            auto opt = buffer_host_address(id);
            if (opt.has_value())
            {
                return {reinterpret_cast<T *>(opt.value())};
            }
            return {};
        }

        [[nodiscard]] auto create_raster_pipeline(RasterPipelineInfo const & info) -> RasterPipeline;
        [[nodiscard]] auto create_compute_pipeline(ComputePipelineInfo const & info) -> ComputePipeline;
        [[nodiscard]] auto create_ray_tracing_pipeline(RayTracingPipelineInfo const & info) -> RayTracingPipeline;

        [[nodiscard]] auto create_swapchain(SwapchainInfo const & info) -> Swapchain;
        [[nodiscard]] auto create_command_recorder(CommandRecorderInfo const & info) -> CommandRecorder;
        [[nodiscard]] auto create_compute_command_recorder(CommandRecorderInfo const & info) -> ComputeCommandRecorder;
        [[nodiscard]] auto create_transfer_command_recorder(CommandRecorderInfo const & info) -> TransferCommandRecorder;
        [[nodiscard]] auto create_binary_semaphore(BinarySemaphoreInfo const & info) -> BinarySemaphore;
        [[nodiscard]] auto create_timeline_semaphore(TimelineSemaphoreInfo const & info) -> TimelineSemaphore;
        [[nodiscard]] auto create_event(EventInfo const & info) -> Event;
        [[nodiscard]] auto create_timeline_query_pool(TimelineQueryPoolInfo const & info) -> TimelineQueryPool;

        void wait_idle();

        void queue_wait_idle(Queue queue);
        auto queue_count(QueueFamily queue_count) -> u32;

        void submit_commands(CommandSubmitInfo const & submit_info);
        void present_frame(PresentInfo const & info);

        /// @brief  Actually destroys all resources that are ready to be destroyed.
        ///         When calling destroy, or removing all references to an object, it is zombified not really destroyed.
        ///         A zombie lives until the gpu catches up to the point of zombification.
        /// NOTE:
        /// * this function will block until it gains an exclusive resource lock
        /// * command lists may hold shared lifetime locks, those must all unlock before an exclusive lock can be made
        /// * look at CommandRecorder for more info on this
        /// * SoftwareCommandRecorder is exempt from this limitation,
        ///   you can freely record those in parallel with collect_garbage
        void collect_garbage();

        /// THREADSAFETY:
        /// * reference MUST NOT be read after the device is destroyed.
        /// @return reference to info of object.
        [[nodiscard]] auto info() const -> DeviceInfo2 const &;
        /// THREADSAFETY:
        /// * reference MUST NOT be read after the device is destroyed.
        /// @return reference to device properties
        [[nodiscard]] auto properties() const -> DeviceProperties const &;
        [[nodiscard]] auto get_supported_present_modes(NativeWindowHandle native_handle, NativeWindowPlatform native_platform) const -> std::vector<PresentMode>;

        /// DEPRECATED:

        [[deprecated("Use tlas_build_sizes or as_build_sizes Instead")]] [[nodiscard]] auto get_tlas_build_sizes(TlasBuildInfo const & info) { return tlas_build_sizes(info); }
        [[deprecated("Use blas_build_sizes or as_build_sizes Instead")]] [[nodiscard]] auto get_blas_build_sizes(BlasBuildInfo const & info) { return blas_build_sizes(info); }
        [[deprecated("Use buffer_memory_requirements or memory_requirements Instead")]] [[nodiscard]] auto get_memory_requirements(BufferInfo const & info) const { return buffer_memory_requirements(info); }
        [[deprecated("Use image_memory_requirements or memory_requirements Instead")]] [[nodiscard]] auto get_memory_requirements(ImageInfo const & info) const { return image_memory_requirements(info); }
        [[deprecated("Use buffer_info or info instead")]] [[nodiscard]] auto info_buffer(BufferId id) const { return buffer_info(id); }
        [[deprecated("Use image_info or info instead")]] [[nodiscard]] auto info_image(ImageId id) const { return image_info(id); }
        [[deprecated("Use image_view_info or info instead")]] [[nodiscard]] auto info_image_view(ImageViewId id) const { return image_view_info(id); }
        [[deprecated("Use sampler_info or info instead")]] [[nodiscard]] auto info_sampler(SamplerId id) const { return sampler_info(id); }
        [[deprecated("Use tlas_info or info instead")]] [[nodiscard]] auto info_tlas(TlasId id) const { return tlas_info(id); }
        [[deprecated("Use blas_info or info instead")]] [[nodiscard]] auto info_blas(BlasId id) const { return blas_info(id); }

        template <typename T>
        [[deprecated("Use buffer_host_address_as instead")]] [[nodiscard]] auto get_host_address_as(BufferId id) const
        {
            return buffer_host_address_as<T>(id);
        }

        [[deprecated("Use buffer_host_address instead")]] [[nodiscard]] auto get_host_address(BufferId id) const { return buffer_host_address(id); }
        [[deprecated("Use buffer_device_address or device_address instead")]] [[nodiscard]] auto get_device_address(BufferId id) const { return buffer_device_address(id); }
        [[deprecated("Use blas_device_address or device_address instead")]] [[nodiscard]] auto get_device_address(BlasId id) const { return blas_device_address(id); }
        [[deprecated("Use tlas_device_address or device_address instead")]] [[nodiscard]] auto get_device_address(TlasId id) const { return tlas_device_address(id); }

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };
} // namespace daxa
