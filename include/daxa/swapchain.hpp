#pragma once

#include <daxa/core.hpp>

#include <daxa/gpu_resources.hpp>
#include <daxa/sync.hpp>

namespace daxa
{
    static inline auto default_format_score(Format format) -> i32
    {
        switch (format)
        {
        case Format::B8G8R8A8_UNORM: return 90;
        case Format::R8G8B8A8_UNORM: return 80;
        case Format::B8G8R8A8_SRGB: return 70;
        case Format::R8G8B8A8_SRGB: return 60;
        default: return 0;
        }
    }

    struct SwapchainInfo
    {
        NativeWindowHandle native_window;
        NativeWindowPlatform native_window_platform;
        i32 (*surface_format_selector)(Format) = default_format_score;
        PresentMode present_mode = PresentMode::FIFO;
        PresentOp present_operation = PresentOp::IDENTITY;
        ImageUsageFlags image_usage = {};
        usize max_allowed_frames_in_flight = 2;
        QueueFamily queue_family = {};
        SmallString name = {};
    };

    /**
     * @brief   Swapchain represents the surface, swapchain and synch primitives regarding acquire and present operations.
     *          The swapchain has a cpu and gpu timeline in order to ensure proper frames in flight.
     * 
     * NOTE:
     * * functions that contain 'current' in their name might return different values between calling acquire_next_image
     *
     * THREADSAFETY:
     * * must be externally synchronized
     * * can be passed between different threads
     * * may be accessed by only one thread at the same time
     */
    struct DAXA_EXPORT_CXX Swapchain final : ManagedPtr<Swapchain, daxa_Swapchain>
    {
        Swapchain() = default;

        /// @brief The ImageId may change between calls. This must be called to obtain a new swapchain image to be used for rendering.
        /// WARNING:
        /// * ImageIds returned from the swapchain are INVALID after the swapchain is destroyed.
        /// * ImageIds returned from the swapchain are INVALID after calling either resize OR set_present_mode!
        /// @return A swapchain image, that will be ready to render to when the acquire semaphore is signaled. This may return an empty image id if the swapchain is out of date.
        [[nodiscard]] auto acquire_next_image() -> ImageId;
        /// The acquire semaphore must be waited on in the first submission that uses the last acquired image.
        /// This semaphore may change between acquires, so it needs to be re-queried after every current_acquire_semaphore call.
        /// @return The binary semaphore that is signaled when the last acquired image is ready to be used.
        [[nodiscard]] auto current_acquire_semaphore() const -> BinarySemaphore const &;
        /// @brief The present semaphore must be signaled in the last submission that uses the last acquired swapchain image.
        /// The present semaphore must be waited on in the present of the last acquired image.
        /// This semaphore may change between acquires, so it needs to be re-queried after every current_acquire_semaphore call.
        /// @return The present semaphore that needs to be signaled and waited on for present of the last acquired image.
        [[nodiscard]] auto current_present_semaphore() const -> BinarySemaphore const &;
        /// @brief The last submission that uses the swapchain image needs to signal the timeline with the cpu value.
        /// @return The cpu frame timeline value.
        [[nodiscard]] auto current_cpu_timeline_value() const -> u64;
        /// @brief The swapchain needs to know when the last use of the swapchain happens to limit the frames in flight.
        /// In the last submission that uses the swapchain image, signal this timeline semaphore with the cpu timeline value.
        /// @return the gpu timeline semaphore that needs to be signaled.
        [[nodiscard]] auto gpu_timeline_semaphore() const -> TimelineSemaphore const &;
        /// @brief  The swapchain needs to know when the last use of the swapchain happens to limit the frames in flight.
        ///         In the last submission that uses the swapchain image, signal this timeline semaphore with the cpu timeline value.
        ///         The cpu value timeline is incremented whenever acquire is called.
        ///         The gpu timeline must be manually incremented by the user via a submit.
        ///         The difference between cpu and gpu timeline describes how many frames in flight the gpu is behind the cpu.
        /// @return Returns pair of a gpu timeline and cpu timeline value.
        [[nodiscard]] auto current_timeline_pair() const -> std::pair<TimelineSemaphore, u64>;

        /// @brief  When the window size changes the swapchain is in an invalid state for new commands.
        ///         Calling resize will recreate the swapchain with the proper window size.
        /// WARNING:
        /// * Due to wsi limitations this function will WAIT IDLE THE DEVICE.
        /// * If the function throws an error, The swapchain will be invalidated and unusable!
        void resize();
        /// @brief Recreates swapchain with new present mode.
        /// WARNING:
        /// * Due to wsi limitations this function will WAIT IDLE THE DEVICE.
        /// * If the function throws an error, The swapchain will be invalidated and unusable!
        void set_present_mode(PresentMode present_mode);
        /// THREADSAFETY:
        /// * reference MUST NOT be read after the swapchain is destroyed
        /// * reference is INVALIDATED after calling either resize OR set_present_mode
        /// @return reference to the objects info
        [[nodiscard]] auto info() const -> SwapchainInfo const &;
        [[nodiscard]] auto get_surface_extent() const -> Extent2D;
        [[nodiscard]] auto get_format() const -> Format;

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };
} // namespace daxa
