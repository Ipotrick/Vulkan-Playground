#ifndef __DAXA_SWAPCHAIN_H__
#define __DAXA_SWAPCHAIN_H__

#include <daxa/c/gpu_resources.h>
#include <daxa/c/sync.h>

DAXA_EXPORT int32_t
daxa_default_format_selector(VkFormat format);

/// @brief  A platform-dependent window resource.
///         On Windows, this is an `HWND`
///         On Linux X11, this is a `Window`
///         On Linux Wayland, this is a `wl_surface *`
typedef void * daxa_NativeWindowHandle;

typedef enum
{
    DAXA_NATIVE_WINDOW_PLATFORM_UNKNOWN,
    DAXA_NATIVE_WINDOW_PLATFORM_WIN32_API,
    DAXA_NATIVE_WINDOW_PLATFORM_XLIB_API,
    DAXA_NATIVE_WINDOW_PLATFORM_WAYLAND_API,
    DAXA_NATIVE_WINDOW_PLATFORM_MAX_ENUM = 0x7fffffff,
} daxa_NativeWindowPlatform;

typedef struct
{
    daxa_NativeWindowHandle native_window;
    daxa_NativeWindowPlatform native_window_platform;
    int32_t (*surface_format_selector)(VkFormat);
    VkPresentModeKHR present_mode;
    VkSurfaceTransformFlagBitsKHR present_operation;
    daxa_ImageUsageFlags image_usage;
    size_t max_allowed_frames_in_flight;
    daxa_QueueFamily queue_family;
    daxa_SmallString name;
} daxa_SwapchainInfo;

DAXA_EXPORT VkExtent2D
daxa_swp_get_surface_extent(daxa_Swapchain swapchain);
DAXA_EXPORT VkFormat
daxa_swp_get_format(daxa_Swapchain swapchain);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_swp_resize(daxa_Swapchain swapchain);
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_swp_set_present_mode(daxa_Swapchain swapchain, VkPresentModeKHR present_mode);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_swp_acquire_next_image(daxa_Swapchain swapchain, daxa_ImageId * out_image_id);
DAXA_EXPORT daxa_BinarySemaphore *
daxa_swp_current_acquire_semaphore(daxa_Swapchain swapchain);
DAXA_EXPORT daxa_BinarySemaphore *
daxa_swp_current_present_semaphore(daxa_Swapchain swapchain);
DAXA_EXPORT uint64_t
daxa_swp_current_cpu_timeline_value(daxa_Swapchain swapchain);
DAXA_EXPORT daxa_TimelineSemaphore *
daxa_swp_gpu_timeline_semaphore(daxa_Swapchain swapchain);

DAXA_EXPORT daxa_SwapchainInfo const *
daxa_swp_info(daxa_Swapchain swapchain);

DAXA_EXPORT VkSwapchainKHR
daxa_swp_get_vk_swapchain(daxa_Swapchain swapchain);
DAXA_EXPORT VkSurfaceKHR
daxa_swp_get_vk_surface(daxa_Swapchain swapchain);
DAXA_EXPORT uint64_t
daxa_swp_inc_refcnt(daxa_Swapchain swapchain);
DAXA_EXPORT uint64_t
daxa_swp_dec_refcnt(daxa_Swapchain swapchain);

#endif // #ifndef __DAXA_SWAPCHAIN_H__
