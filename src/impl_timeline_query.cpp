#include "impl_timeline_query.hpp"

#include <utility>

#include "impl_device.hpp"

// --- Begin API Functions ---

auto daxa_dvc_create_timeline_query_pool(daxa_Device device, daxa_TimelineQueryPoolInfo const * info, daxa_TimelineQueryPool * out_tqp) -> daxa_Result
{
    auto ret = daxa_ImplTimelineQueryPool{};
    ret.device = device;
    ret.info = *reinterpret_cast<TimelineQueryPoolInfo const *>(info);
    // TODO(msakmary) Should Add a check for support of timeline queries
    //                here or earlier (during device creation/section) I'm not sure...
    VkQueryPoolCreateInfo const vk_query_pool_create_info{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = ret.info.query_count,
        .pipelineStatistics = {},
    };
    auto vk_result = vkCreateQueryPool(ret.device->vk_device, &vk_query_pool_create_info, nullptr, &ret.vk_timeline_query_pool);
    if (vk_result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(vk_result);
    }
    vkResetQueryPool(ret.device->vk_device, ret.vk_timeline_query_pool, 0, ret.info.query_count);
    if ((ret.device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE && !ret.info.name.empty())
    {
        VkDebugUtilsObjectNameInfoEXT const query_pool_name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_QUERY_POOL,
            .objectHandle = std::bit_cast<uint64_t>(ret.vk_timeline_query_pool),
            .pObjectName = ret.info_name.c_str(),
        };
        ret.device->vkSetDebugUtilsObjectNameEXT(ret.device->vk_device, &query_pool_name_info);
    }
    ret.strong_count = 1;
    device->inc_weak_refcnt();
    *out_tqp = new daxa_ImplTimelineQueryPool{};
    **out_tqp = std::move(ret);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_timeline_query_pool_info(daxa_TimelineQueryPool self) -> daxa_TimelineQueryPoolInfo const *
{
    return reinterpret_cast<daxa_TimelineQueryPoolInfo const *>(&self->info);
}

auto daxa_timeline_query_pool_query_results(daxa_TimelineQueryPool self, u32 start, u32 count, u64 * out_results) -> daxa_Result
{
    if (count == 0)
    {
        return DAXA_RESULT_SUCCESS;
    }
    if (!(start + count - 1 < self->info.query_count))
    {
        return DAXA_RESULT_RANGE_OUT_OF_BOUNDS;
    }
    auto vk_result = vkGetQueryPoolResults(
        self->device->vk_device,
        self->vk_timeline_query_pool,
        start,
        count,
        count * 2ul * sizeof(u64),
        out_results,
        2ul * sizeof(u64),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    return std::bit_cast<daxa_Result>(vk_result);
}

auto daxa_timeline_query_pool_inc_refcnt(daxa_TimelineQueryPool self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_timeline_query_pool_dec_refcnt(daxa_TimelineQueryPool self) -> u64
{
    return self->dec_refcnt(
        &daxa_ImplTimelineQueryPool::zero_ref_callback,
        self->device->instance);
}

// --- End API Functions ---

// --- Begin Internals ---

void daxa_ImplTimelineQueryPool::zero_ref_callback(ImplHandle const * handle)
{
    auto * self = rc_cast<daxa_TimelineQueryPool>(handle);
    std::unique_lock const lock{self->device->zombies_mtx};
    u64 const submit_timeline = self->device->global_submit_timeline.load(std::memory_order::relaxed);
    self->device->timeline_query_pool_zombies.emplace_back(
        submit_timeline,
        TimelineQueryPoolZombie{
            .vk_timeline_query_pool = self->vk_timeline_query_pool,
        });
    self->device->dec_weak_refcnt(
        daxa_ImplDevice::zero_ref_callback,
        self->device->instance);
    delete self;
}

// --- End Internals ---