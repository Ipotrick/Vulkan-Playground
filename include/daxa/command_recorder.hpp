#pragma once

#include <daxa/core.hpp>
#include <daxa/gpu_resources.hpp>
#include <daxa/pipeline.hpp>
#include <daxa/sync.hpp>

#include <span>

namespace daxa
{
    static inline constexpr usize CONSTANT_BUFFER_BINDINGS_COUNT = 8;

    struct PushConstantInfo
    {
        void const * data = {};
        u64 size = {};
        u32 offset = {};
    };

    struct CommandRecorderInfo
    {
        QueueFamily queue_family = {};
        SmallString name = {};
    };

    struct ImageBlitInfo
    {
        ImageId src_image = {};
        ImageLayout src_image_layout = ImageLayout::TRANSFER_SRC_OPTIMAL;
        ImageId dst_image = {};
        ImageLayout dst_image_layout = ImageLayout::TRANSFER_DST_OPTIMAL;
        ImageArraySlice src_slice = {};
        std::array<Offset3D, 2> src_offsets = {};
        ImageArraySlice dst_slice = {};
        std::array<Offset3D, 2> dst_offsets = {};
        Filter filter = {};
    };

    struct BufferCopyInfo
    {
        BufferId src_buffer = {};
        BufferId dst_buffer = {};
        usize src_offset = {};
        usize dst_offset = {};
        usize size = {};
    };

    struct BufferImageCopyInfo
    {
        BufferId buffer = {};
        usize buffer_offset = {};
        ImageId image = {};
        ImageLayout image_layout = ImageLayout::TRANSFER_DST_OPTIMAL;
        ImageArraySlice image_slice = {};
        Offset3D image_offset = {};
        Extent3D image_extent = {};
    };

    struct ImageBufferCopyInfo
    {
        ImageId image = {};
        ImageLayout image_layout = ImageLayout::TRANSFER_SRC_OPTIMAL;
        ImageArraySlice image_slice = {};
        Offset3D image_offset = {};
        Extent3D image_extent = {};
        BufferId buffer = {};
        usize buffer_offset = {};
    };

    struct ImageCopyInfo
    {
        ImageId src_image = {};
        ImageLayout src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL;
        ImageId dst_image = {};
        ImageLayout dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL;
        ImageArraySlice src_slice = {};
        Offset3D src_offset = {};
        ImageArraySlice dst_slice = {};
        Offset3D dst_offset = {};
        Extent3D extent = {};
    };

    struct ImageClearInfo
    {
        ImageLayout dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL;
        ClearValue clear_value = {};
        ImageId dst_image = {};
        ImageMipArraySlice dst_slice = {};
    };

    struct BufferClearInfo
    {
        BufferId buffer = {};
        usize offset = {};
        usize size = {};
        u32 clear_value = {};
    };

    enum struct ResolveMode
    {
        NONE = 0,
        SAMPLE_ZERO = 0x00000001,
        AVERAGE = 0x00000002,
        MIN = 0x00000004,
        MAX = 0x00000008,
    };

    struct AttachmentResolveInfo
    {
        ResolveMode mode = ResolveMode::AVERAGE;
        ImageViewId image = {};
        ImageLayout layout = ImageLayout::ATTACHMENT_OPTIMAL;
    };

    struct RenderAttachmentInfo
    {
        ImageViewId image_view = {};
        ImageLayout layout = ImageLayout::ATTACHMENT_OPTIMAL;
        AttachmentLoadOp load_op = AttachmentLoadOp::DONT_CARE;
        AttachmentStoreOp store_op = AttachmentStoreOp::STORE;
        ClearValue clear_value = {};
        Optional<AttachmentResolveInfo> resolve = {};
    };

    struct RenderPassBeginInfo
    {
        FixedList<RenderAttachmentInfo, 8> color_attachments = {};
        Optional<RenderAttachmentInfo> depth_attachment = {};
        Optional<RenderAttachmentInfo> stencil_attachment = {};
        Rect2D render_area = {};
    };

    struct TraceRaysInfo
    {
        uint32_t width = 1;
        uint32_t height = 1;
        uint32_t depth = 1;
        uint32_t raygen_shader_binding_table_offset = {};
        uint32_t miss_shader_binding_table_offset = {};
        uint32_t miss_shader_binding_table_stride = {};
        uint32_t hit_shader_binding_table_offset = {};
        RayTracingShaderBindingTable shader_binding_table;
    };

    struct TraceRaysIndirectInfo
    {
        DeviceAddress indirect_device_address = {};
        uint32_t raygen_shader_binding_table_offset = {};
        uint32_t miss_shader_binding_table_offset = {};
        uint32_t miss_shader_binding_table_stride = {};
        uint32_t hit_shader_binding_table_offset = {};
        RayTracingShaderBindingTable shader_binding_table;
    };

    struct DispatchInfo
    {
        u32 x = 1;
        u32 y = 1;
        u32 z = 1;
    };

    struct DispatchIndirectInfo
    {
        BufferId indirect_buffer = {};
        usize offset = {};
    };

    struct DrawMeshTasksIndirectInfo
    {
        BufferId indirect_buffer = {};
        usize offset = {};
        u32 draw_count = 1;
        u32 stride = 12;
    };

    struct DrawMeshTasksIndirectCountInfo
    {
        BufferId indirect_buffer = {};
        usize offset = {};
        BufferId count_buffer = {};
        usize count_offset = {};
        u32 max_count = {};
        u32 stride = 12;
    };

    struct DrawInfo
    {
        u32 vertex_count = {};
        u32 instance_count = 1;
        u32 first_vertex = {};
        u32 first_instance = {};
    };

    struct DrawIndexedInfo
    {
        u32 index_count = {};
        u32 instance_count = 1;
        u32 first_index = {};
        i32 vertex_offset = {};
        u32 first_instance = {};
    };

    struct DrawIndirectInfo
    {
        BufferId draw_command_buffer = {};
        usize indirect_buffer_offset = {};
        u32 draw_count = 1;
        u32 draw_command_stride = {};
        bool is_indexed = {};
    };

    struct DrawIndirectCountInfo
    {
        BufferId draw_command_buffer = {};
        usize indirect_buffer = {};
        BufferId count_buffer = {};
        usize count_buffer_offset = {};
        u32 max_draw_count = static_cast<u32>(std::numeric_limits<u16>::max());
        u32 draw_command_stride = {};
        bool is_indexed = {};
    };

    struct ResetEventInfo
    {
        Event & event;
        PipelineStageFlags stage = {};
    };

    struct WaitEventInfo
    {
        std::span<Event const> events = {};
    };

    struct WriteTimestampInfo
    {
        TimelineQueryPool & query_pool;
        PipelineStageFlags pipeline_stage = {};
        u32 query_index = {};
    };

    struct ResetTimestampsInfo
    {
        TimelineQueryPool & query_pool;
        u32 start_index = {};
        u32 count = {};
    };

    struct CommandLabelInfo
    {
        std::array<f32, 4> label_color = {0.463f, 0.333f, 0.671f, 1.0f};
        SmallString name = {};
    };

    struct SetUniformBufferInfo
    {
        // Binding slot the buffer will be bound to.
        u32 slot = {};
        BufferId buffer = {};
        usize size = {};
        usize offset = {};
    };

    struct DepthBiasInfo
    {
        f32 constant_factor = {};
        f32 clamp = {};
        f32 slope_factor = {};
    };

    struct SetIndexBufferInfo
    {
        BufferId id = {};
        usize offset = {};
        IndexType index_type = IndexType::uint32;
    };

    struct BuildAccelerationStructuresInfo
    {
        std::span<TlasBuildInfo const> tlas_build_infos = {};
        std::span<BlasBuildInfo const> blas_build_infos = {};
    };

    struct DAXA_EXPORT_CXX ExecutableCommandList : ManagedPtr<ExecutableCommandList, daxa_ExecutableCommandList>
    {
      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };

    struct CommandRecorder;

    struct DAXA_EXPORT_CXX RenderCommandRecorder
    {
      private:
        daxa_CommandRecorder internal = {};
        friend struct CommandRecorder;

      public:
        RenderCommandRecorder() = default;
        ~RenderCommandRecorder();
        RenderCommandRecorder(RenderCommandRecorder const &) = delete;
        RenderCommandRecorder & operator=(RenderCommandRecorder const &) = delete;
        RenderCommandRecorder(RenderCommandRecorder &&);
        RenderCommandRecorder & operator=(RenderCommandRecorder &&);

        /// @brief  Starts a renderpass scope akin to the dynamic rendering feature in vulkan.
        ///         Between the begin and end renderpass commands, the renderpass persists and drawcalls can be recorded.
        [[nodiscard]] auto end_renderpass() && -> CommandRecorder;

        void push_constant_vptr(PushConstantInfo const & info);
        template <typename T>
        void push_constant(T const & constant, u32 offset = 0)
        {
            push_constant_vptr({
                .data = static_cast<void const *>(&constant),
                .size = static_cast<u32>(sizeof(T)),
                .offset = offset,
            });
        }
        void set_pipeline(RasterPipeline const & pipeline);
        void set_viewport(ViewportInfo const & info);
        void set_scissor(Rect2D const & info);
        void set_rasterization_samples(RasterizationSamples info);
        void set_depth_bias(DepthBiasInfo const & info);
        void set_index_buffer(SetIndexBufferInfo const & info);

        void draw(DrawInfo const & info);
        void draw_indexed(DrawIndexedInfo const & info);
        void draw_indirect(DrawIndirectInfo const & info);
        void draw_indirect_count(DrawIndirectCountInfo const & info);
        void draw_mesh_tasks(u32 x, u32 y, u32 z);
        void draw_mesh_tasks_indirect(DrawMeshTasksIndirectInfo const & info);
        void draw_mesh_tasks_indirect_count(DrawMeshTasksIndirectCountInfo const & info);
    };

    /**
     * @brief   TransferCommandRecorder is used to encode commands into a VkCommandBuffer.
     *          In order to submit a command list one must complete it.
     *          Completing a command list does SIGNIFICANT driver cpu work,
     *          so do not always complete just before submitting.
     * 
     * TRANSFER:
     * * can only be used to record "transfer commands"
     * * can be created for any queue family
     *
     * THREADSAFETY:
     * * must be externally synchronized
     * * can be passed between different threads
     * * may only be accessed by one thread at a time
     * WARNING:
     * * creating a command list, it will LOCK resource lifetimes
     * * calling collect_garbage will BLOCK until all resource lifetime locks have been unlocked
     * * completing a command list will remove its lock on the resource lifetimes
     * * most record commands can throw exceptions on invalid inputs such as invalid ids
     * * using deferred destructions will make the completed command list not reusable,
     *   as resources can only be destroyed once
     */
    struct DAXA_EXPORT_CXX TransferCommandRecorder
    {
      protected:
        daxa_CommandRecorder internal = {};
        friend struct RenderCommandRecorder;

      public:
        TransferCommandRecorder() = default;
        ~TransferCommandRecorder();
        TransferCommandRecorder(TransferCommandRecorder const &) = delete;
        TransferCommandRecorder & operator=(TransferCommandRecorder const &) = delete;
        TransferCommandRecorder(TransferCommandRecorder &&);
        TransferCommandRecorder & operator=(TransferCommandRecorder &&);

        void copy_buffer_to_buffer(BufferCopyInfo const & info);
        void copy_buffer_to_image(BufferImageCopyInfo const & info);
        void copy_image_to_buffer(ImageBufferCopyInfo const & info);
        void copy_image_to_image(ImageCopyInfo const & info);
        void blit_image_to_image(ImageBlitInfo const & info);
        void clear_buffer(BufferClearInfo const & info);
        void clear_image(ImageClearInfo const & info);

        /// @brief  Successive pipeline barrier calls are combined.
        ///         As soon as a non-pipeline barrier command is recorded, the currently recorded barriers are flushed with a vkCmdPipelineBarrier2 call.
        /// @param info parameters.
        void pipeline_barrier(MemoryBarrierInfo const & info);
        /// @brief  Successive pipeline barrier calls are combined.
        ///         As soon as a non-pipeline barrier command is recorded, the currently recorded barriers are flushed with a vkCmdPipelineBarrier2 call.
        /// @param info parameters.
        void pipeline_barrier_image_transition(ImageMemoryBarrierInfo const & info);
        void signal_event(EventSignalInfo const & info);
        void wait_events(std::span<EventWaitInfo const> const & infos);
        void wait_event(EventWaitInfo const & info);
        void reset_event(ResetEventInfo const & info);

        /// @brief  Destroys the buffer AFTER the gpu is finished executing the command list.
        ///         Zombifies object after submitting the commands.
        ///         Useful for large uploads exceeding staging memory pools.
        /// @param id buffer to be destroyed after command list finishes.
        void destroy_buffer_deferred(BufferId id);
        /// @brief  Destroys the image AFTER the gpu is finished executing the command list.
        ///         Zombifies object after submitting the commands.
        ///         Useful for large uploads exceeding staging memory pools.
        /// @param id image to be destroyed after command list finishes.
        void destroy_image_deferred(ImageId id);
        /// @brief  Destroys the image view AFTER the gpu is finished executing the command list.
        ///         Zombifies object after submitting the commands.
        ///         Useful for large uploads exceeding staging memory pools.
        /// @param id image view to be destroyed after command list finishes.
        void destroy_image_view_deferred(ImageViewId id);
        /// @brief  Destroys the sampler AFTER the gpu is finished executing the command list.
        ///         Zombifies object after submitting the commands.
        ///         Useful for large uploads exceeding staging memory pools.
        /// @param id image sampler be destroyed after command list finishes.
        void destroy_sampler_deferred(SamplerId id);

        void write_timestamp(WriteTimestampInfo const & info);
        void reset_timestamps(ResetTimestampsInfo const & info);

        void begin_label(CommandLabelInfo const & info);
        void end_label();

        [[nodiscard]] auto complete_current_commands() -> ExecutableCommandList;

        /// THREADSAFETY:
        /// * reference MUST NOT be read after the device is destroyed.
        /// @return reference to info of object.
        [[nodiscard]] auto info() const -> CommandRecorderInfo const &;
    };

    /**
     * @brief   ComputeCommandRecorder is used to encode commands into a VkCommandBuffer.
     *          In order to submit a command list one must complete it.
     *          Completing a command list does SIGNIFICANT driver cpu work,
     *          so do not always complete just before submitting.
     * 
     * COMPUTE:
     * * can only be used to record "transfer commands" and "compute commands"
     * * can only be created for compute and main queue family
     *
     * THREADSAFETY:
     * * must be externally synchronized
     * * can be passed between different threads
     * * may only be accessed by one thread at a time
     * WARNING:
     * * creating a command list, it will LOCK resource lifetimes
     * * calling collect_garbage will BLOCK until all resource lifetime locks have been unlocked
     * * completing a command list will remove its lock on the resource lifetimes
     * * most record commands can throw exceptions on invalid inputs such as invalid ids
     * * using deferred destructions will make the completed command list not reusable,
     *   as resources can only be destroyed once
     */
    struct DAXA_EXPORT_CXX ComputeCommandRecorder : TransferCommandRecorder
    {
        void push_constant_vptr(PushConstantInfo const & info);

        template <typename T>
        void push_constant(T const & constant, u32 offset = 0)
        {
            push_constant_vptr({
                .data = &constant,
                .size = static_cast<u32>(sizeof(T)),
                .offset = offset,
            });
        }

        void build_acceleration_structures(BuildAccelerationStructuresInfo const & info);

        void set_pipeline(ComputePipeline const & pipeline);

        void dispatch(DispatchInfo const & info);

        void dispatch_indirect(DispatchIndirectInfo const & info);

        void set_pipeline(RayTracingPipeline const & pipeline);

        void trace_rays(TraceRaysInfo const & info);

        void trace_rays_indirect(TraceRaysIndirectInfo const & info);
    };

    /**
     * @brief   CommandRecorder is used to encode commands into a VkCommandBuffer.
     *          In order to submit a command list one must complete it.
     *          Completing a command list does SIGNIFICANT driver cpu work,
     *          so do not always complete just before submitting.
     * 
     * GENERAL:
     * * can only be created for the main queue.
     * * can be used to record any commands.
     *
     * THREADSAFETY:
     * * must be externally synchronized
     * * can be passed between different threads
     * * may only be accessed by one thread at a time
     * WARNING:
     * * creating a command list, it will LOCK resource lifetimes
     * * calling collect_garbage will BLOCK until all resource lifetime locks have been unlocked
     * * completing a command list will remove its lock on the resource lifetimes
     * * most record commands can throw exceptions on invalid inputs such as invalid ids
     * * using deferred destructions will make the completed command list not reusable,
     *   as resources can only be destroyed once
     */
    struct DAXA_EXPORT_CXX CommandRecorder : ComputeCommandRecorder
    {
        /// @brief  Starts a renderpass scope akin to the dynamic rendering feature in vulkan.
        ///         Between the begin and end renderpass commands, the renderpass persists and drawcalls can be recorded.
        /// @param info parameters.
        [[nodiscard]] auto begin_renderpass(RenderPassBeginInfo const & info) && -> RenderCommandRecorder;
    };
} // namespace daxa
