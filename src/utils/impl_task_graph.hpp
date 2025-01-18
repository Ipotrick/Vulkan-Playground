#pragma once

#include "../impl_core.hpp"

#include <variant>
#include <sstream>
#include <daxa/utils/task_graph.hpp>

#define DAXA_TASK_GRAPH_MAX_CONDITIONALS 31

namespace daxa
{
    struct ImplDevice;

    using TaskBatchId = usize;

    using TaskId = usize;

    struct LastConcurrentAccessSplitBarrierIndex
    {
        usize index;
    };

    struct LastConcurrentAccessBarrierIndex
    {
        usize index;
    };

    struct ResourceLifetime
    {
        // TODO(msakmary, pahrens) This will not be needed once we make batches linear and not
        // contained in submit scopes
        struct CombinedBatchIndex
        {
            usize submit_scope_index = std::numeric_limits<u32>::max();
            usize task_batch_index = std::numeric_limits<u32>::max();
        };
        CombinedBatchIndex first_use;
        CombinedBatchIndex last_use;
    };

    enum struct TaskAccessConcurrency
    {
        CONCURRENT,
        EXCLUSIVE
    };

    struct PerPermTaskBuffer
    {
        /// Every permutation always has all buffers but they are not necessarily valid in that permutation.
        /// This boolean is used to check this.
        bool valid = {};
        TaskAccessConcurrency latest_access_concurrent = TaskAccessConcurrency::EXCLUSIVE;
        Access latest_access = AccessConsts::NONE;
        usize latest_access_batch_index = {};
        usize latest_access_submit_scope_index = {};
        usize first_access_batch_index = {};
        usize first_access_submit_scope_index = {};
        Access first_access = AccessConsts::NONE;
        // When the last index was a read and an additional read is followed after,
        // we will combine all barriers into one, which is the first barrier that the first read generates.
        Variant<Monostate, LastConcurrentAccessSplitBarrierIndex, LastConcurrentAccessBarrierIndex> latest_concurrent_access_barrer_index = Monostate{};
        std::variant<BufferId, BlasId, TlasId> actual_id = BufferId{};

        ResourceLifetime lifetime = {};
        usize allocation_offset = {};
        daxa::MemoryRequirements memory_requirements = {};
    };

    struct ExtendedImageSliceState
    {
        ImageSliceState state = {};
        TaskAccessConcurrency latest_access_concurrent = TaskAccessConcurrency::EXCLUSIVE;
        usize latest_access_batch_index = {};
        usize latest_access_submit_scope_index = {};
        // When the last index was a read and an additional read is followed after,
        // we will combine all barriers into one, which is the first barrier that the first read generates.
        Variant<Monostate, LastConcurrentAccessSplitBarrierIndex, LastConcurrentAccessBarrierIndex> latest_concurrent_access_barrer_index = Monostate{};
    };

    struct PerPermTaskImage
    {
        /// Every permutation always has all buffers but they are not necessarily valid in that permutation.
        /// This boolean is used to check this.
        bool valid = {};
        bool swapchain_semaphore_waited_upon = {};
        std::vector<ExtendedImageSliceState> last_slice_states = {};
        std::vector<ExtendedImageSliceState> first_slice_states = {};
        // only for transient images
        ResourceLifetime lifetime = {};
        ImageCreateFlags create_flags = ImageCreateFlagBits::NONE;
        ImageUsageFlags usage = ImageUsageFlagBits::NONE;
        ImageId actual_image = {};
        usize allocation_offset = {};
        daxa::MemoryRequirements memory_requirements = {};
    };

    struct TaskBarrier
    {
        // when this ID is invalid, this barrier is NOT an image memory barrier but just a memory barrier.
        // So when ID invalid => memory barrier, ID valid => image memory barrier.
        TaskImageView image_id = {};
        ImageMipArraySlice slice = {};
        ImageLayout layout_before = {};
        ImageLayout layout_after = {};
        Access src_access = {};
        Access dst_access = {};
    };

    struct TaskSplitBarrier : TaskBarrier
    {
        Event split_barrier_state;
    };

    struct ImplTask
    {
        std::unique_ptr<ITask> base_task = {};
        std::vector<std::vector<ImageViewId>> image_view_cache = {};
        // Used to verify image view cache:
        std::vector<std::vector<ImageId>> runtime_images_last_execution = {};
    };

    struct ImplPresentInfo
    {
        std::vector<BinarySemaphore> binary_semaphores = {};
        std::vector<BinarySemaphore> * additional_binary_semaphores = {};
    };

    struct TaskBatch
    {
        std::vector<usize> pipeline_barrier_indices = {};
        std::vector<usize> wait_split_barrier_indices = {};
        std::vector<TaskId> tasks = {};
        std::vector<usize> signal_split_barrier_indices = {};
    };

    struct TaskBatchSubmitScope
    {
        CommandSubmitInfo submit_info = {};
        TaskSubmitInfo user_submit_info = {};
        // These barriers are inserted after all batches and their sync.
        std::vector<usize> last_minute_barrier_indices = {};
        std::vector<TaskBatch> task_batches = {};
        std::vector<u64> used_swapchain_task_images = {};
        std::optional<ImplPresentInfo> present_info = {};
    };

    auto task_image_access_to_layout_access(TaskImageAccess const & access) -> std::tuple<ImageLayout, Access, TaskAccessConcurrency>;
    auto task_buffer_access_to_access(TaskBufferAccess const & access) -> std::pair<Access, TaskAccessConcurrency>;

    struct ImplTaskGraph;

    struct TaskGraphPermutation
    {
        // record time information:
        bool active = {};
        // persistent information:
        TaskImageView swapchain_image = {};
        std::vector<PerPermTaskBuffer> buffer_infos = {};
        std::vector<PerPermTaskImage> image_infos = {};
        std::vector<TaskSplitBarrier> split_barriers = {};
        std::vector<TaskBarrier> barriers = {};
        std::vector<usize> initial_barriers = {};
        // TODO(msakmary, pahrens) - Instead of storing batch submit scopes which contain batches
        // we should make a vector of batches which and a second vector of submit scopes which are
        // just offsets into the batches vector
        std::vector<TaskBatchSubmitScope> batch_submit_scopes = {};
        usize swapchain_image_first_use_submit_scope_index = std::numeric_limits<usize>::max();
        usize swapchain_image_last_use_submit_scope_index = std::numeric_limits<usize>::max();

        void add_task(ImplTaskGraph & task_graph_impl, ImplTask & impl_task, TaskId task_id);
        void submit(TaskSubmitInfo const & info);
        void present(TaskPresentInfo const & info);
    };

    struct ImplPersistentTaskBufferBlasTlas final : ImplHandle
    {
        ImplPersistentTaskBufferBlasTlas(TaskBufferInfo a_info);
        ImplPersistentTaskBufferBlasTlas(Device & device, BufferInfo const & a_info);
        ImplPersistentTaskBufferBlasTlas(TaskBlasInfo a_info);
        ImplPersistentTaskBufferBlasTlas(TaskTlasInfo a_info);
        ~ImplPersistentTaskBufferBlasTlas();

        std::variant<
            std::vector<BufferId>,
            std::vector<BlasId>,
            std::vector<TlasId>>
            actual_ids = {};

        Access latest_access = {};

        std::variant<
            TaskBufferInfo,
            TaskBlasInfo,
            TaskTlasInfo>
            info = {};

        Optional<Device> owned_buffer_device = {};
        Optional<BufferInfo> owned_buffer_info = {};

        // Used to allocate id - because all persistent resources have unique id we need a single point
        // from which they are generated
        static inline std::atomic_uint32_t exec_unique_next_index = 1;
        u32 unique_index = std::numeric_limits<u32>::max();

        static void zero_ref_callback(ImplHandle const * handle);
    };

    struct ImplPersistentTaskImage final : ImplHandle
    {
        ImplPersistentTaskImage(TaskImageInfo const & a_info);
        ~ImplPersistentTaskImage();

        TaskImageInfo info = {};
        // One task buffer can back multiple buffers.
        std::vector<ImageId> actual_images = {};
        // We store runtime information about the previous executions final resource states.
        // This is important, as with conditional execution and temporal resources we need to store this infomation to form correct state transitions.
        std::vector<ImageSliceState> latest_slice_states = {};
        // Only for swapchain images. Runtime data.
        bool waited_on_acquire = {};

        // Used to allocate id - because all persistent resources have unique id we need a single point
        // from which they are generated
        static inline std::atomic_uint32_t exec_unique_next_index = 1;
        u32 unique_index = std::numeric_limits<u32>::max();

        static void zero_ref_callback(ImplHandle const * handle);
    };

    struct PermIndepTaskBufferInfo
    {
        struct Persistent
        {
            std::variant<TaskBuffer, TaskBlas, TaskTlas> buffer_blas_tlas = {};

            auto get() -> ImplPersistentTaskBufferBlasTlas &
            {
                ImplPersistentTaskBufferBlasTlas * ret = {};
                std::visit([&](auto & ptr)
                           { ret = ptr.get(); }, buffer_blas_tlas);
                return *ret;
            }
            auto get() const -> ImplPersistentTaskBufferBlasTlas const &
            {
                ImplPersistentTaskBufferBlasTlas const * ret = {};
                std::visit([&](auto & ptr)
                           { ret = ptr.get(); }, buffer_blas_tlas);
                return *ret;
            }
        };
        struct Transient
        {
            // TODO: Add Transient Blas and Tlas
            TaskTransientBufferInfo info = {};
        };
        Variant<Persistent, Transient> task_buffer_data;

        inline auto get_name() const -> std::string_view
        {
            if (is_persistent())
            {
                std::string_view ret = {};
                std::visit([&](auto const & info)
                           { ret = info.name; }, get_persistent().info);
                return ret;
            }
            else
            {
                return daxa::get<Transient>(task_buffer_data).info.name;
            }
        }
        inline auto get_persistent() -> ImplPersistentTaskBufferBlasTlas &
        {
            return daxa::get<Persistent>(task_buffer_data).get();
        }
        inline auto get_persistent() const -> ImplPersistentTaskBufferBlasTlas const &
        {
            return daxa::get<Persistent>(task_buffer_data).get();
        }
        inline auto is_persistent() const -> bool
        {
            return daxa::holds_alternative<Persistent>(task_buffer_data);
        }
    };

    struct PermIndepTaskImageInfo
    {
        struct Persistent
        {
            TaskImage image = {};
            auto get() -> ImplPersistentTaskImage &
            {
                return **r_cast<ImplPersistentTaskImage **>(&image);
            }
            auto get() const -> ImplPersistentTaskImage const &
            {
                return **r_cast<ImplPersistentTaskImage const * const *>(&image);
            }
        };
        struct Transient
        {
            TaskTransientImageInfo info = {};
        };
        Variant<Persistent, Transient> task_image_data;

        inline auto get_name() const -> std::string_view
        {
            if (is_persistent())
            {
                return get_persistent().info.name;
            }
            else
            {
                return daxa::get<Transient>(task_image_data).info.name;
            }
        }
        inline auto get_persistent() -> ImplPersistentTaskImage &
        {
            return daxa::get<Persistent>(task_image_data).get();
        }
        inline auto get_persistent() const -> ImplPersistentTaskImage const &
        {
            return daxa::get<Persistent>(task_image_data).get();
        }
        inline auto is_persistent() const -> bool
        {
            return daxa::holds_alternative<Persistent>(task_image_data);
        }
    };

    struct ImplTaskRuntimeInterface
    {
        // interface:
        ImplTaskGraph & task_graph;
        TaskGraphPermutation & permutation;
        CommandRecorder & recorder;
        ImplTask * current_task = {};
        types::DeviceAddress device_address = {};
        bool reuse_last_command_list = true;
        std::optional<BinarySemaphore> last_submit_semaphore = {};
    };

    struct ImplTaskGraph final : ImplHandle
    {
        ImplTaskGraph(TaskGraphInfo a_info);
        ~ImplTaskGraph();

        static inline std::atomic_uint32_t exec_unique_next_index = 1;
        u32 unique_index = {};

        TaskGraphInfo info;
        std::vector<PermIndepTaskBufferInfo> global_buffer_infos = {};
        std::vector<PermIndepTaskImageInfo> global_image_infos = {};
        std::vector<TaskGraphPermutation> permutations = {};
        std::vector<ImplTask> tasks = {};
        // TODO: replace with faster hash map.
        std::unordered_map<u32, u32> persistent_buffer_index_to_local_index;
        std::unordered_map<u32, u32> persistent_image_index_to_local_index;

        // record time information:
        u32 record_active_conditional_scopes = {};
        u32 record_conditional_states = {};
        std::vector<TaskGraphPermutation *> record_active_permutations = {};
        std::unordered_map<std::string, TaskBufferView> buffer_name_to_id = {};
        std::unordered_map<std::string, TaskBlasView> blas_name_to_id = {};
        std::unordered_map<std::string, TaskTlasView> tlas_name_to_id = {};
        std::unordered_map<std::string, TaskImageView> image_name_to_id = {};

        usize memory_block_size = {};
        u32 memory_type_bits = 0xFFFFFFFFu;
        MemoryBlock transient_data_memory_block = {};
        bool compiled = {};

        // execution time information:
        std::optional<daxa::TransferMemoryPool> staging_memory = {};
        std::array<bool, DAXA_TASK_GRAPH_MAX_CONDITIONALS> execution_time_current_conditionals = {};

        // post execution information:
        u32 chosen_permutation_last_execution = {};
        std::vector<ExecutableCommandList> left_over_command_lists = {};
        bool executed_once = {};
        u32 prev_frame_permutation_index = {};
        std::stringstream debug_string_stream = {};

        template <typename TaskIdT>
        auto get_actual_buffer_blas_tlas(TaskIdT id, TaskGraphPermutation const & perm) const -> std::span<typename TaskIdT::ID_T const>
        {
            static constexpr std::array<typename TaskIdT::ID_T, 64> NULL_ID_ARRAY = {};
            if (id.is_null())
            {
                return NULL_ID_ARRAY;
            }
            auto const & global_buffer = global_buffer_infos.at(id.index);
            if (global_buffer.is_persistent())
            {
                return std::span{
                    std::get<std::vector<typename TaskIdT::ID_T>>(global_buffer.get_persistent().actual_ids).data(),
                    std::get<std::vector<typename TaskIdT::ID_T>>(global_buffer.get_persistent().actual_ids).size(),
                };
            }
            else
            {
                auto const & perm_buffer = perm.buffer_infos.at(id.index);
                DAXA_DBG_ASSERT_TRUE_M(perm_buffer.valid, "Can not get actual buffer - buffer is not valid in this permutation");
                return std::span{&std::get<typename TaskIdT::ID_T>(perm_buffer.actual_id), 1};
            }
        }

        using GetActualIdsVariant = std::variant<
            std::span<BufferId const>,
            std::span<BlasId const>,
            std::span<TlasId const>>;
        auto get_actual_buffer_blas_tlas_generic(TaskGPUResourceView id, TaskGraphPermutation const & perm) const -> GetActualIdsVariant
        {
            static constexpr std::array<BufferId const, 64> NULL_ID_ARRAY = {};
            if (id.is_null())
            {
                return std::span{NULL_ID_ARRAY.data(), NULL_ID_ARRAY.size()};
            }
            auto const & global_buffer = global_buffer_infos.at(id.index);
            if (global_buffer.is_persistent())
            {
                GetActualIdsVariant ret = std::span{NULL_ID_ARRAY.data(), NULL_ID_ARRAY.size()};
                std::visit([&](auto const & ids)
                           { ret = std::span{ids.data(), ids.size()}; }, global_buffer.get_persistent().actual_ids);
                return ret;
            }
            else
            {
                auto const & perm_buffer = perm.buffer_infos.at(id.index);
                DAXA_DBG_ASSERT_TRUE_M(perm_buffer.valid, "Can not get actual buffer - buffer is not valid in this permutation");
                GetActualIdsVariant ret = std::span{NULL_ID_ARRAY.data(), NULL_ID_ARRAY.size()};
                std::visit([&](auto const & id)
                           { ret = std::span{&id, 1}; }, perm_buffer.actual_id);
                return ret;
            }
        }

        auto buffer_blas_tlas_str(TaskGPUResourceView id, TaskGraphPermutation const & perm) const -> std::string_view
        {
            if (id.is_null())
            {
                return "NULL";
            }
            static constexpr std::string_view names[3] = {"buffer", "blas", "tlas"};
            auto const & global_buffer = global_buffer_infos.at(id.index);
            if (global_buffer.is_persistent())
            {
                return names[global_buffer.get_persistent().actual_ids.index()];
            }
            else
            {
                auto const & perm_buffer = perm.buffer_infos.at(id.index);
                DAXA_DBG_ASSERT_TRUE_M(perm_buffer.valid, "Can not get actual buffer - buffer is not valid in this permutation");
                return names[perm_buffer.actual_id.index()];
            }
        }

        template <typename TaskIdT>
        auto buffer_blas_tlas_id_to_local_id(TaskIdT id) const -> TaskIdT
        {
            if (id.is_null())
                return id;
            DAXA_DBG_ASSERT_TRUE_M(!id.is_empty(), "Detected empty task buffer id. Please make sure to only use initialized task buffer ids.");
            if (id.is_persistent())
            {
                DAXA_DBG_ASSERT_TRUE_M(
                    persistent_buffer_index_to_local_index.contains(id.index),
                    std::format("Detected invalid access of persistent task buffer id ({}) in task graph \"{}\"; "
                                "please make sure to declare persistent resource use to each task graph that uses this buffer with the function use_persistent_buffer!",
                                id.index, info.name));
                return TaskIdT{{.task_graph_index = this->unique_index, .index = persistent_buffer_index_to_local_index.at(id.index)}};
            }
            else
            {
                DAXA_DBG_ASSERT_TRUE_M(
                    id.task_graph_index == this->unique_index,
                    std::format("Detected invalid access of transient task buffer id ({}) in task graph \"{}\"; "
                                "please make sure that you only use transient buffers within the list they are created in!",
                                id.index, info.name));
                return TaskIdT{{.task_graph_index = this->unique_index, .index = id.index}};
            }
        }

        // auto get_actual_buffers(TaskBufferView id, TaskGraphPermutation const & perm) const -> std::span<BufferId const>;
        auto get_actual_images(TaskImageView id, TaskGraphPermutation const & perm) const -> std::span<ImageId const>;
        auto id_to_local_id(TaskImageView id) const -> TaskImageView;
        void update_active_permutations();
        void update_image_view_cache(ImplTask & task, TaskGraphPermutation const & permutation);
        void execute_task(ImplTaskRuntimeInterface & impl_runtime, TaskGraphPermutation & permutation, u32 batch_index, TaskBatchId in_batch_task_index, TaskId task_id);
        void insert_pre_batch_barriers(TaskGraphPermutation & permutation);
        void create_transient_runtime_buffers(TaskGraphPermutation & permutation);
        void create_transient_runtime_images(TaskGraphPermutation & permutation);
        void allocate_transient_resources();
        void print_task_buffer_blas_tlas_to(std::string & out, std::string indent, TaskGraphPermutation const & permutation, TaskGPUResourceView local_id);
        void print_task_image_to(std::string & out, std::string indent, TaskGraphPermutation const & permutation, TaskImageView image);
        void print_task_barrier_to(std::string & out, std::string & indent, TaskGraphPermutation const & permutation, usize index, bool const split_barrier);
        void print_task_to(std::string & out, std::string & indent, TaskGraphPermutation const & permutation, TaskId task_id);
        void print_permutation_aliasing_to(std::string & out, std::string indent, TaskGraphPermutation const & permutation);
        void debug_print();

        static void zero_ref_callback(ImplHandle const * handle);
    };

} // namespace daxa
