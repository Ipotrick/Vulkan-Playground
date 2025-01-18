#pragma once

#include "impl_core.hpp"
#include "impl_sync.hpp"
#include "impl_pipeline.hpp"

#include <daxa/c/command_recorder.h>
#include <daxa/command_recorder.hpp>

using namespace daxa;

struct ImplDevice;

static inline constexpr u8 DEFERRED_DESTRUCTION_BUFFER_INDEX = 0;
static inline constexpr u8 DEFERRED_DESTRUCTION_IMAGE_INDEX = 1;
static inline constexpr u8 DEFERRED_DESTRUCTION_IMAGE_VIEW_INDEX = 2;
static inline constexpr u8 DEFERRED_DESTRUCTION_SAMPLER_INDEX = 3;
static inline constexpr u8 DEFERRED_DESTRUCTION_TIMELINE_QUERY_POOL_INDEX = 4;
// TODO: maybe reintroduce this in some fashion?
// static inline constexpr usize DEFERRED_DESTRUCTION_COUNT_MAX = 32;

static inline constexpr usize COMMAND_LIST_BARRIER_MAX_BATCH_SIZE = 16;
static inline constexpr usize COMMAND_LIST_COLOR_ATTACHMENT_MAX = 16;

struct CommandPoolPool
{
    auto get(daxa_Device device) -> VkCommandPool;

    void put_back(VkCommandPool pool_and_buffer);

    void cleanup(daxa_Device device);

    std::vector<VkCommandPool> pools_and_buffers = {};
    u32 queue_family_index = {~0u};
    std::mutex mtx = {};
};

struct CommandRecorderZombie
{
    daxa_QueueFamily queue_family = {};
    VkCommandPool vk_cmd_pool = {};
    std::vector<VkCommandBuffer> allocated_command_buffers = {};
};

struct ExecutableCommandListData
{
    VkCommandBuffer vk_cmd_buffer = {};
    std::vector<std::pair<GPUResourceId, u8>> deferred_destructions = {};
    // TODO:    These vectors seem to be fast enough. overhead is around 1-4% in cmd recording.
    //          It might be cool to have some slab allocator for these.
    // If there is demand, we could make an instance or cmd list flag to disable the submit checks.
    // TODO:    Also collect ref counted handles.
    std::vector<BufferId> used_buffers = {};
    std::vector<ImageId> used_images = {};
    std::vector<ImageViewId> used_image_views = {};
    std::vector<SamplerId> used_samplers = {};
    std::vector<TlasId> used_tlass = {};
    std::vector<BlasId> used_blass = {};
};

struct daxa_ImplCommandRecorder final : ImplHandle
{
    daxa_Device device = {};
    bool in_renderpass = {};
    daxa_CommandRecorderInfo info = {};
    VkCommandPool vk_cmd_pool = {};
    std::vector<VkCommandBuffer> allocated_command_buffers = {};
    std::array<VkMemoryBarrier2, COMMAND_LIST_BARRIER_MAX_BATCH_SIZE> memory_barrier_batch = {};
    std::array<VkImageMemoryBarrier2, COMMAND_LIST_BARRIER_MAX_BATCH_SIZE> image_barrier_batch = {};
    usize image_barrier_batch_count = {};
    usize memory_barrier_batch_count = {};
    usize split_barrier_batch_count = {};
    struct NoPipeline
    {
    };
    Variant<NoPipeline, daxa_ComputePipeline, daxa_RasterPipeline, daxa_RayTracingPipeline> current_pipeline = NoPipeline{};

    ExecutableCommandListData current_command_data = {};

    auto generate_new_current_command_data() -> daxa_Result;

    static void zero_ref_callback(ImplHandle const * handle);
};

struct daxa_ImplExecutableCommandList final : ImplHandle
{
    daxa_CommandRecorder cmd_recorder = {};
    ExecutableCommandListData data = {};

    static void zero_ref_callback(ImplHandle const * handle);
};

void executable_cmd_list_execute_deferred_destructions(daxa_Device device, ExecutableCommandListData & cmd_list);