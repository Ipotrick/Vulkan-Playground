#include "impl_sync.hpp"
#include "impl_device.hpp"

// --- Begin API Functions ---

auto daxa_dvc_create_binary_semaphore(daxa_Device device, daxa_BinarySemaphoreInfo const * info, daxa_BinarySemaphore * out_semaphore) -> daxa_Result
{
    auto ret = daxa_ImplBinarySemaphore{};
    ret.device = device;
    ret.info = *reinterpret_cast<BinarySemaphoreInfo const *>(info);
    VkSemaphoreCreateInfo const vk_semaphore_create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
    };
    auto vk_result = vkCreateSemaphore(device->vk_device, &vk_semaphore_create_info, nullptr, &ret.vk_semaphore);
    if (vk_result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(vk_result);
    }
    if ((device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE && (!ret.info.name.view().empty()))
    {
        auto c_str = ret.info.name.c_str();
        VkDebugUtilsObjectNameInfoEXT const name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_SEMAPHORE,
            .objectHandle = std::bit_cast<u64>(ret.vk_semaphore),
            .pObjectName = c_str.data(),
        };
        device->vkSetDebugUtilsObjectNameEXT(device->vk_device, &name_info);
    }
    ret.strong_count = 1;
    device->inc_weak_refcnt();
    *out_semaphore = new daxa_ImplBinarySemaphore{};
    **out_semaphore = ret;
    return DAXA_RESULT_SUCCESS;
}

auto daxa_binary_semaphore_info(daxa_BinarySemaphore self) -> daxa_BinarySemaphoreInfo const *
{
    return reinterpret_cast<daxa_BinarySemaphoreInfo const *>(&self->info);
}

auto daxa_binary_semaphore_get_vk_semaphore(daxa_BinarySemaphore self) -> VkSemaphore
{
    return self->vk_semaphore;
}

auto daxa_binary_semaphore_inc_refcnt(daxa_BinarySemaphore self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_binary_semaphore_dec_refcnt(daxa_BinarySemaphore self) -> u64
{
    return self->dec_refcnt(
        &daxa_ImplBinarySemaphore::zero_ref_callback,
        self->device->instance);
}

auto daxa_dvc_create_timeline_semaphore(daxa_Device device, daxa_TimelineSemaphoreInfo const * info, daxa_TimelineSemaphore * out_semaphore) -> daxa_Result
{
    auto ret = daxa_ImplTimelineSemaphore{};
    ret.device = device;
    ret.info = *reinterpret_cast<TimelineSemaphoreInfo const *>(info);
    VkSemaphoreTypeCreateInfo timeline_vk_semaphore{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = info->initial_value,
    };
    VkSemaphoreCreateInfo const vk_semaphore_create_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &timeline_vk_semaphore,
        .flags = {},
    };
    auto vk_result = vkCreateSemaphore(device->vk_device, &vk_semaphore_create_info, nullptr, &ret.vk_semaphore);
    if (vk_result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(vk_result);
    }
    if ((device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE && (!ret.info.name.span().empty()))
    {
        auto c_str = ret.info.name.c_str();
        VkDebugUtilsObjectNameInfoEXT const name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_SEMAPHORE,
            .objectHandle = std::bit_cast<u64>(ret.vk_semaphore),
            .pObjectName = c_str.data(),
        };
        device->vkSetDebugUtilsObjectNameEXT(device->vk_device, &name_info);
    }
    ret.strong_count = 1;
    device->inc_weak_refcnt();
    *out_semaphore = new daxa_ImplTimelineSemaphore{};
    **out_semaphore = ret;
    return DAXA_RESULT_SUCCESS;
}

auto daxa_timeline_semaphore_info(daxa_TimelineSemaphore self) -> daxa_TimelineSemaphoreInfo const *
{
    return reinterpret_cast<daxa_TimelineSemaphoreInfo const *>(&self->info);
}

auto daxa_timeline_semaphore_get_value(daxa_TimelineSemaphore self, uint64_t * out_value) -> daxa_Result
{
    return static_cast<daxa_Result>(vkGetSemaphoreCounterValue(self->device->vk_device, self->vk_semaphore, out_value));
}

auto daxa_timeline_semaphore_set_value(daxa_TimelineSemaphore self, uint64_t value) -> daxa_Result
{
    VkSemaphoreSignalInfo const vk_semaphore_signal_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .semaphore = self->vk_semaphore,
        .value = value,
    };

    return static_cast<daxa_Result>(vkSignalSemaphore(self->device->vk_device, &vk_semaphore_signal_info));
}

auto daxa_timeline_semaphore_wait_for_value(daxa_TimelineSemaphore self, uint64_t value, uint64_t timeout) -> daxa_Result
{
    VkSemaphoreWaitInfo const vk_semaphore_wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = {},
        .semaphoreCount = 1,
        .pSemaphores = &self->vk_semaphore,
        .pValues = &value,
    };

    return static_cast<daxa_Result>(vkWaitSemaphores(self->device->vk_device, &vk_semaphore_wait_info, timeout));
}

auto daxa_timeline_semaphore_get_vk_semaphore(daxa_TimelineSemaphore self) -> VkSemaphore
{
    return self->vk_semaphore;
}

auto daxa_timeline_semaphore_inc_refcnt(daxa_TimelineSemaphore self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_timeline_semaphore_dec_refcnt(daxa_TimelineSemaphore self) -> u64
{
    _DAXA_TEST_PRINT("daxa_timeline_semaphore_dec_refcnt\n");
    return self->dec_refcnt(
        &daxa_ImplTimelineSemaphore::zero_ref_callback,
        self->device->instance);
}

auto daxa_dvc_create_event(daxa_Device device, daxa_EventInfo const * info, daxa_Event * out_event) -> daxa_Result
{
    auto ret = daxa_ImplEvent{};
    ret.device = device;
    ret.info = *reinterpret_cast<EventInfo const *>(info);
    VkEventCreateInfo const vk_event_create_info{
        .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_EVENT_CREATE_DEVICE_ONLY_BIT,
    };
    VkEvent event = {};
    auto vk_result = vkCreateEvent(ret.device->vk_device, &vk_event_create_info, nullptr, &event);
    if (vk_result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(vk_result);
    }
    ret.vk_event = event;
    if ((device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE && (!ret.info.name.view().empty()))
    {
        auto c_str = ret.info.name.c_str();
        VkDebugUtilsObjectNameInfoEXT const name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_EVENT,
            .objectHandle = std::bit_cast<uint64_t>(ret.vk_event),
            .pObjectName = c_str.data(),
        };
        ret.device->vkSetDebugUtilsObjectNameEXT(ret.device->vk_device, &name_info);
    }
    ret.strong_count = 1;
    device->inc_weak_refcnt();
    *out_event = new daxa_ImplEvent{};
    **out_event = ret;
    return DAXA_RESULT_SUCCESS;
}

auto daxa_event_info(daxa_Event self) -> daxa_EventInfo const *
{
    return reinterpret_cast<daxa_EventInfo const *>(&self->info);
}

auto daxa_event_inc_refcnt(daxa_Event self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_event_dec_refcnt(daxa_Event self) -> u64
{
    return self->dec_refcnt(
        &daxa_ImplEvent::zero_ref_callback,
        self->device->instance);
}

// --- End API Functions ---

// --- Begin Internals ---

void daxa_ImplBinarySemaphore::zero_ref_callback(ImplHandle const * handle)
{
    auto * self = rc_cast<daxa_BinarySemaphore>(handle);
    std::unique_lock const lock{self->device->zombies_mtx};
    u64 const main_queue_cpu_timeline = self->device->global_submit_timeline.load(std::memory_order::relaxed);
    self->device->semaphore_zombies.emplace_back(
        main_queue_cpu_timeline,
        SemaphoreZombie{
            .vk_semaphore = self->vk_semaphore,
        });
    self->device->dec_weak_refcnt(
        daxa_ImplDevice::zero_ref_callback,
        self->device->instance);
    delete self;
}

void daxa_ImplTimelineSemaphore::zero_ref_callback(ImplHandle const * handle)
{
    _DAXA_TEST_PRINT("daxa_ImplTimelineSemaphore::zero_ref_callback\n");
    auto * self = rc_cast<daxa_TimelineSemaphore>(handle);
    std::unique_lock const lock{self->device->zombies_mtx};
    u64 const main_queue_cpu_timeline = self->device->global_submit_timeline.load(std::memory_order::relaxed);
    self->device->semaphore_zombies.emplace_back(
        main_queue_cpu_timeline,
        SemaphoreZombie{
            .vk_semaphore = self->vk_semaphore,
        });
    self->device->dec_weak_refcnt(
        daxa_ImplDevice::zero_ref_callback,
        self->device->instance);
    delete self;
}

void daxa_ImplEvent::zero_ref_callback(ImplHandle const * handle)
{
    auto * self = rc_cast<daxa_Event>(handle);
    std::unique_lock const lock{self->device->zombies_mtx};
    u64 const main_queue_cpu_timeline = self->device->global_submit_timeline.load(std::memory_order::relaxed);
    self->device->split_barrier_zombies.emplace_back(
        main_queue_cpu_timeline,
        EventZombie{
            .vk_event = self->vk_event,
        });
    self->device->dec_weak_refcnt(
        daxa_ImplDevice::zero_ref_callback,
        self->device->instance);
    delete self;
}

// --- End Internals ---