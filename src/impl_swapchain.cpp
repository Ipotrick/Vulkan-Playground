#include "impl_core.hpp"
#include "impl_swapchain.hpp"
#include "impl_device.hpp"

#include <utility>
#include <bit>

// --- Begin API Functions ---

auto daxa_default_format_selector(VkFormat format) -> i32
{
    switch (format)
    {
    case VK_FORMAT_R8G8B8A8_SRGB: return 90;
    case VK_FORMAT_R8G8B8A8_UNORM: return 80;
    case VK_FORMAT_B8G8R8A8_SRGB: return 70;
    case VK_FORMAT_B8G8R8A8_UNORM: return 60;
    default: return 0;
    }
}

auto daxa_dvc_create_swapchain(daxa_Device device, daxa_SwapchainInfo const * info, daxa_Swapchain * out_swapchain) -> daxa_Result
{
    device->inc_weak_refcnt();
    auto ret = daxa_ImplSwapchain{};
    ret.device = device;
    ret.info = *reinterpret_cast<SwapchainInfo const *>(info);
    auto result = ret.recreate_surface();
    if (result != DAXA_RESULT_SUCCESS)
    {
        ret.full_cleanup();
        return result;
    }

    if (info->queue_family >= DAXA_QUEUE_FAMILY_MAX_ENUM || device->queue_families[info->queue_family].queue_count == 0)
    {
        ret.full_cleanup();
        return DAXA_RESULT_ERROR_INVALID_QUEUE_FAMILY;
    }

    // Save supported present modes.
    u32 present_mode_count = {};
    auto vk_result = vkGetPhysicalDeviceSurfacePresentModesKHR(
        ret.device->vk_physical_device,
        ret.vk_surface,
        &present_mode_count,
        nullptr);
    if (vk_result != VK_SUCCESS)
    {
        ret.full_cleanup();
        return std::bit_cast<daxa_Result>(vk_result);
    }
    ret.supported_present_modes.resize(present_mode_count);
    vk_result = vkGetPhysicalDeviceSurfacePresentModesKHR(
        device->vk_physical_device,
        ret.vk_surface,
        &present_mode_count,
        r_cast<VkPresentModeKHR *>(ret.supported_present_modes.data()));
    if (vk_result != VK_SUCCESS)
    {
        ret.full_cleanup();
        return std::bit_cast<daxa_Result>(vk_result);
    }

    // Format Selection:
    u32 format_count = 0;
    vk_result = vkGetPhysicalDeviceSurfaceFormatsKHR(ret.device->vk_physical_device, ret.vk_surface, &format_count, nullptr);
    if (vk_result != VK_SUCCESS)
    {
        ret.full_cleanup();
        return std::bit_cast<daxa_Result>(vk_result);
    }
    std::vector<VkSurfaceFormatKHR> surface_formats;
    surface_formats.resize(format_count);
    vk_result = vkGetPhysicalDeviceSurfaceFormatsKHR(ret.device->vk_physical_device, ret.vk_surface, &format_count, surface_formats.data());
    if (vk_result != VK_SUCCESS)
    {
        ret.full_cleanup();
        return std::bit_cast<daxa_Result>(vk_result);
    }
    if (format_count == 0)
    {
        ret.full_cleanup();
        return DAXA_RESULT_NO_SUITABLE_FORMAT_FOUND;
    }
    auto format_comparator = [&](auto const & a, auto const & b) -> bool
    {
        return ret.info.surface_format_selector(std::bit_cast<Format>(a.format)) <
               ret.info.surface_format_selector(std::bit_cast<Format>(b.format));
    };
    auto best_format = std::max_element(surface_formats.begin(), surface_formats.end(), format_comparator);
    if (best_format == surface_formats.end())
    {
        ret.full_cleanup();
        return DAXA_RESULT_NO_SUITABLE_FORMAT_FOUND;
    }
    ret.vk_surface_format = *best_format;

    result = ret.recreate();
    if (result != DAXA_RESULT_SUCCESS)
    {
        ret.full_cleanup();
        return result;
    }
    // We have an acquire semaphore for each frame in flight.
    for (u32 i = 0; i < ret.info.max_allowed_frames_in_flight + 1; i++)
    {
        BinarySemaphore sema = {};
        daxa_BinarySemaphoreInfo const sema_info = {};
        result = daxa_dvc_create_binary_semaphore(device, &sema_info, reinterpret_cast<daxa_BinarySemaphore *>(&sema));
        if (result != DAXA_RESULT_SUCCESS)
        {
            ret.full_cleanup();
            return result;
        }
        ret.acquire_semaphores.push_back(std::move(sema));
    }
    // We have a present semaphore for each swapchain image.
    for (u32 i = 0; i < ret.images.size(); i++)
    {
        BinarySemaphore sema = {};
        daxa_BinarySemaphoreInfo const sema_info = {};
        result = daxa_dvc_create_binary_semaphore(device, &sema_info, reinterpret_cast<daxa_BinarySemaphore *>(&sema));
        if (result != DAXA_RESULT_SUCCESS)
        {
            ret.full_cleanup();
            return result;
        }
        ret.present_semaphores.push_back(std::move(sema));
    }

    auto timeline_sema_name = SmallString(std::string{ret.info.name.view()} + " ts");
    auto timeline_sema_info = daxa_TimelineSemaphoreInfo{
        .initial_value = 0,
        .name = std::bit_cast<daxa_SmallString>(timeline_sema_name),
    };
    result = daxa_dvc_create_timeline_semaphore(device, &timeline_sema_info, r_cast<daxa_TimelineSemaphore *>(&ret.gpu_frame_timeline));
    if (result != DAXA_RESULT_SUCCESS)
    {
        ret.full_cleanup();
        return result;
    }

    ret.strong_count = 1;
    *out_swapchain = new daxa_ImplSwapchain{};
    **out_swapchain = std::move(ret);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_swp_get_surface_extent(daxa_Swapchain self) -> VkExtent2D
{
    return self->surface_extent;
}

auto daxa_swp_get_format(daxa_Swapchain self) -> VkFormat
{
    return self->vk_surface_format.format;
}

auto daxa_swp_resize(daxa_Swapchain self) -> daxa_Result
{
    auto result = self->recreate();
    if (result != DAXA_RESULT_SUCCESS)
    {
        [[maybe_unused]] auto ignored = vkDeviceWaitIdle(self->device->vk_device);
        self->full_cleanup();
    }
    return result;
}

auto daxa_swp_set_present_mode(daxa_Swapchain self, VkPresentModeKHR present_mode) -> daxa_Result
{
    self->info.present_mode = std::bit_cast<PresentMode>(present_mode);
    auto result = self->recreate();
    if (result != DAXA_RESULT_SUCCESS)
    {
        [[maybe_unused]] auto ignored = vkDeviceWaitIdle(self->device->vk_device);
        self->full_cleanup();
    }
    return result;
}

auto daxa_swp_acquire_next_image(daxa_Swapchain self, daxa_ImageId * out_image_id) -> daxa_Result
{
    [[maybe_unused]] auto _ignored = self->gpu_frame_timeline.wait_for_value(
        static_cast<u64>(
            std::max<i64>(
                0,
                static_cast<i64>(self->cpu_frame_timeline) - static_cast<i64>(self->info.max_allowed_frames_in_flight))));
    self->acquire_semaphore_index = (self->cpu_frame_timeline + 1) % (self->info.max_allowed_frames_in_flight + 1);
    BinarySemaphore & acquire_semaphore = self->acquire_semaphores[self->acquire_semaphore_index];
    auto result = vkAcquireNextImageKHR(
        self->device->vk_device,
        self->vk_swapchain, UINT64_MAX,
        (**r_cast<daxa_BinarySemaphore *>(&acquire_semaphore)).vk_semaphore,
        nullptr,
        &self->current_image_index);

    // We only bump the cpu timeline, when the acquire succeeds.
    self->cpu_frame_timeline += 1;
    *out_image_id = static_cast<daxa_ImageId>(self->images[self->current_image_index]);
    return std::bit_cast<daxa_Result>(result);
}

auto daxa_swp_current_acquire_semaphore(daxa_Swapchain self) -> daxa_BinarySemaphore *
{
    return r_cast<daxa_BinarySemaphore *>(&self->acquire_semaphores[self->acquire_semaphore_index]);
}

auto daxa_swp_current_present_semaphore(daxa_Swapchain self) -> daxa_BinarySemaphore *
{
    return r_cast<daxa_BinarySemaphore *>(&self->present_semaphores[self->current_image_index]);
}

auto daxa_swp_gpu_timeline_semaphore(daxa_Swapchain self) -> daxa_TimelineSemaphore *
{
    return r_cast<daxa_TimelineSemaphore *>(&self->gpu_frame_timeline);
}

auto daxa_swp_current_cpu_timeline_value(daxa_Swapchain self) -> u64
{
    return self->cpu_frame_timeline;
}

auto daxa_swp_info(daxa_Swapchain self) -> daxa_SwapchainInfo const *
{
    return reinterpret_cast<daxa_SwapchainInfo const *>(&self->info);
}

auto daxa_swp_get_vk_swapchain(daxa_Swapchain self) -> VkSwapchainKHR
{
    return self->vk_swapchain;
}

auto daxa_swp_get_vk_surface(daxa_Swapchain self) -> VkSurfaceKHR
{
    return self->vk_surface;
}

auto daxa_swp_inc_refcnt(daxa_Swapchain self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_swp_dec_refcnt(daxa_Swapchain self) -> u64
{
    return self->dec_refcnt(
        &daxa_ImplSwapchain::zero_ref_callback,
        self->device->instance);
}

// --- End API Functions ---

// --- Begin Internals ---

auto daxa_ImplSwapchain::recreate() -> daxa_Result
{
    daxa_Result result = DAXA_RESULT_SUCCESS;
    // Check present mode:
    auto iter = std::find(this->supported_present_modes.begin(), this->supported_present_modes.end(), this->info.present_mode);
    if (iter == this->supported_present_modes.end())
    {
        result = DAXA_RESULT_DEVICE_SURFACE_UNSUPPORTED_PRESENT_MODE;
    }
    _DAXA_RETURN_IF_ERROR(result, result)

    VkSurfaceCapabilitiesKHR surface_capabilities;
    result = static_cast<daxa_Result>(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        this->device->vk_physical_device,
        this->vk_surface,
        &surface_capabilities));
    _DAXA_RETURN_IF_ERROR(result, result)

    surface_extent.width = surface_capabilities.currentExtent.width;
    surface_extent.height = surface_capabilities.currentExtent.height;

#if __linux__
    // TODO(grundlett): I (grundlett) am too lazy to find out why the other present modes
    // fail on Linux. This can be inspected by Linux people and they can
    // submit a PR if they find a fix.
    info.present_mode = PresentMode::IMMEDIATE;
#endif

    auto * old_swapchain = this->vk_swapchain;

    // NOTE: this is a hack that allows us to ignore issues caused
    // by things that are just underspecified in the Vulkan spec.
    result = daxa_dvc_wait_idle(this->device);
    _DAXA_RETURN_IF_ERROR(result, result)

    this->partial_cleanup();

    ImageUsageFlags const usage = std::bit_cast<ImageUsageFlags>(info.image_usage) | ImageUsageFlagBits::COLOR_ATTACHMENT;

    VkSwapchainCreateInfoKHR const swapchain_create_info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .surface = this->vk_surface,
        .minImageCount = 3,
        .imageFormat = this->vk_surface_format.format,
        .imageColorSpace = this->vk_surface_format.colorSpace,
        .imageExtent = surface_extent,
        .imageArrayLayers = 1,
        .imageUsage = usage.data,
        .imageSharingMode = VkSharingMode::VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = &this->device->get_queue(DAXA_QUEUE_MAIN).vk_queue_family_index,
        .preTransform = static_cast<VkSurfaceTransformFlagBitsKHR>(info.present_operation),
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = static_cast<VkPresentModeKHR>(info.present_mode),
        .clipped = VK_TRUE,
        .oldSwapchain = old_swapchain,
    };

    result = static_cast<daxa_Result>(vkCreateSwapchainKHR(
        this->device->vk_device,
        &swapchain_create_info,
        nullptr,
        &this->vk_swapchain));
    _DAXA_RETURN_IF_ERROR(result, result)

    defer
    {
        if (result != DAXA_RESULT_SUCCESS)
        {
            if (this->vk_swapchain)
            {
                vkDestroySwapchainKHR(this->device->vk_device, this->vk_swapchain, nullptr);
            }
            for (auto & image : this->images)
            {
                if (!image.is_empty())
                {
                    [[maybe_unused]] auto _ignore = daxa_dvc_destroy_image(this->device, static_cast<daxa_ImageId>(image));
                }
            }
        }
    };

    u32 image_count = 0;
    std::vector<VkImage> swapchain_images = {};
    result = static_cast<daxa_Result>(vkGetSwapchainImagesKHR(this->device->vk_device, vk_swapchain, &image_count, nullptr));
    _DAXA_RETURN_IF_ERROR(result, result)

    swapchain_images.resize(image_count);
    result = static_cast<daxa_Result>(vkGetSwapchainImagesKHR(this->device->vk_device, vk_swapchain, &image_count, swapchain_images.data()));
    _DAXA_RETURN_IF_ERROR(result, result)

    this->images.resize(image_count);
    for (u32 i = 0; i < images.size(); i++)
    {
        ImageInfo const image_info = {
            .format = static_cast<Format>(this->vk_surface_format.format),
            .size = {this->surface_extent.width, this->surface_extent.height, 1},
            .usage = usage,
            .name = this->info_name.c_str(),
        };
        ImageId id = {};
        auto result = this->device->new_swapchain_image(swapchain_images[i], vk_surface_format.format, i, usage, image_info, &id);
        _DAXA_RETURN_IF_ERROR(result, result)

        this->images[i] = id;
    }

    if ((this->device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE && !this->info_name.empty())
    {
        VkDebugUtilsObjectNameInfoEXT const swapchain_name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_SWAPCHAIN_KHR,
            .objectHandle = std::bit_cast<u64>(this->vk_swapchain),
            .pObjectName = this->info_name.c_str(),
        };
        this->device->vkSetDebugUtilsObjectNameEXT(this->device->vk_device, &swapchain_name_info);
    }

    if (old_swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(this->device->vk_device, old_swapchain, nullptr);
    }

    return DAXA_RESULT_SUCCESS;
}

void daxa_ImplSwapchain::partial_cleanup()
{
    for (auto & image : this->images)
    {
        // TODO:    Add special function that only swapchains can call do destroy these images.
        //          Make it fully illegal to destroy swapchain images from user side!
        [[maybe_unused]] auto _ignore = daxa_dvc_destroy_image(this->device, static_cast<daxa_ImageId>(image));
    }
    this->images.clear();
}

void daxa_ImplSwapchain::full_cleanup()
{
    this->partial_cleanup();
    if (this->vk_swapchain != VK_NULL_HANDLE)
    {
        // Due to wsi limitations we need to wait idle before destroying the swapchain.
        vkDeviceWaitIdle(this->device->vk_device);
        vkDestroySwapchainKHR(this->device->vk_device, this->vk_swapchain, nullptr);
    }
    if (this->vk_surface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(this->device->instance->vk_instance, this->vk_surface, nullptr);
    }
    if (this->device != nullptr)
    {
        this->device->dec_weak_refcnt(
            daxa_ImplDevice::zero_ref_callback,
            this->device->instance);
    }
    // Clear all other fields.
    // Other owned objects are handled by RAII.
    *this = {};
}

auto daxa_ImplSwapchain::recreate_surface() -> daxa_Result
{
    if (this->vk_surface != nullptr)
    {
        vkDestroySurfaceKHR(this->device->instance->vk_instance, this->vk_surface, nullptr);
    }
    return create_surface(
        this->device->instance,
        std::bit_cast<daxa_NativeWindowHandle>(this->info.native_window),
        std::bit_cast<daxa_NativeWindowPlatform>(this->info.native_window_platform),
        &this->vk_surface);
}

void daxa_ImplSwapchain::zero_ref_callback(ImplHandle const * handle)
{
    _DAXA_TEST_PRINT("      daxa_ImplSwapchain::zero_ref_callback\n");
    auto * self = rc_cast<daxa_Swapchain>(handle);
    if (self->device != nullptr)
    {
        self->full_cleanup();
    }
    delete self;
}

// --- End Internals ---