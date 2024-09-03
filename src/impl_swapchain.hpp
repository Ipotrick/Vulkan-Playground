#pragma once

#include "impl_core.hpp"

#include <daxa/swapchain.hpp>

#include <daxa/c/device.h>

/// I (pahrens) am going to document the internals here as wsi is really confusing and strange in vulkan.
/// Every frame we get a swapchain image index. This index can be non sequential in the case of mail box presentation and other modes.
/// This means we need to acquire a new index every frame to know what swapchain image to use.
///
/// IMPORTANT INFORMATION REGARDING SEMAPHORES IN WSI:
/// binary semaphore in acquire MUST be un-signaled when recording actions ON THE CPU!
/// Because of this, we need frames_in_flight+1 semaphores for the acquire
///
/// We need two binary semaphores here:
/// The acquire semaphore and the present semaphore.
/// The present semaphore is signaled in the last submission that uses the swapchain image and waited on in the present.
/// The acquire semaphore is signaled when the swapchain image is ready to be used.
/// This also means that the previous presentation of the image is finished and the semaphore used in the present is un-signaled.
/// Unfortunately there is NO other way to know when a present finishes (or the corresponding semaphore is un-signaled).
/// This means that in order to be able to reuse binary semaphores used in presentation,
/// one MUST pair them with the image they are used to present.
///
/// One can then rely on the acquire semaphore of the image beeing signaled to indicate that the present semaphore is able to be reused,
/// As a swapchain images acquire sema is signaled when it is available and its previous present is completed.
///
/// In order to reuse the the acquire semaphore we must set a limit in frames in flight and wait on the cpu to limit the frames in flight.
/// When we have this wait in place we can safely reuse the acquire semaphores with a linearly increasing index corresponding to the frame.
/// This means the acquire semaphores are not tied to the number of swapchain images like present semaphores but to the number of frames in flight!!
///
/// To limit the frames in flight we employ a timeline semaphore that must be signaled in a submission that uses or after one that uses the swapchain image.
///
/// WARNING: The swapchain only works on the main queue! It is directly tied to it.
///
/// TODO: investigate if wsi is improved enough to use zombies for swapchain.
struct daxa_ImplSwapchain final : ImplHandle
{
    daxa_Device device = {};
    SwapchainInfo info = {};
    std::string info_name = {};
    VkSwapchainKHR vk_swapchain = {};
    VkSurfaceKHR vk_surface = {};
    VkSurfaceFormatKHR vk_surface_format = {};
    VkExtent2D surface_extent = {};
    std::vector<PresentMode> supported_present_modes = {};
    // Swapchain holds strong references to these objects as it owns them.
    std::vector<ImageId> images = {};
    std::vector<BinarySemaphore> acquire_semaphores = {};
    std::vector<BinarySemaphore> present_semaphores = {};
    // Monotonically increasing frame index.
    usize cpu_frame_timeline = {};
    // cpu_frame_timeline % frames in flight. used to index the acquire semaphores.
    usize acquire_semaphore_index = {};
    // Gpu timeline semaphore used to track how far behind the gpu is.
    // Used to limit frames in flight.
    TimelineSemaphore gpu_frame_timeline = {};
    // This is the swapchain image index that acquire returns. THis is not necessarily linear.
    // This index must be used for present semaphores as they are paired to the images.
    u32 current_image_index = {};

    void partial_cleanup();
    void full_cleanup();
    auto recreate_surface() -> daxa_Result;
    auto recreate() -> daxa_Result;

    static auto create(daxa_Device device, daxa_SwapchainInfo const * info, daxa_Swapchain swapchain) -> daxa_Result;
    static void zero_ref_callback(ImplHandle const * handle);
};