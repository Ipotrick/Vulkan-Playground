#include <daxa/daxa.hpp>

#include <GLFW/glfw3.h>
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3native.h>

auto get_native_handle(GLFWwindow * glfw_window_ptr) -> daxa::NativeWindowHandle
{
#if defined(_WIN32)
    return glfwGetWin32Window(glfw_window_ptr);
#elif defined(__linux__)
    return reinterpret_cast<daxa::NativeWindowHandle>(glfwGetX11Window(glfw_window_ptr));
#endif
}

auto get_native_platform(GLFWwindow * /*unused*/) -> daxa::NativeWindowPlatform
{
#if defined(_WIN32)
    return daxa::NativeWindowPlatform::WIN32_API;
#elif defined(__linux__)
    return daxa::NativeWindowPlatform::XLIB_API;
#endif
}

struct WindowInfo
{
    daxa::u32 width{}, height{};
    bool swapchain_out_of_date = false;
};

auto main() -> int
{
    auto window_info = WindowInfo{.width = 800, .height = 600};
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto * glfw_window_ptr = glfwCreateWindow(
        static_cast<daxa::i32>(window_info.width),
        static_cast<daxa::i32>(window_info.height),
        "Daxa sample window name", nullptr, nullptr);
    glfwSetWindowUserPointer(glfw_window_ptr, &window_info);
    glfwSetWindowSizeCallback(
        glfw_window_ptr,
        [](GLFWwindow * glfw_window, int width, int height)
        {
            auto & info = *reinterpret_cast<WindowInfo *>(glfwGetWindowUserPointer(glfw_window));
            // We set this bool because it lets us know we need to resize the swapchain later!
            info.swapchain_out_of_date = true;
            info.width = static_cast<daxa::u32>(width);
            info.height = static_cast<daxa::u32>(height);
        });
    auto * native_window_handle = get_native_handle(glfw_window_ptr);
    auto native_window_platform = get_native_platform(glfw_window_ptr);

    // First thing we do is create a Daxa instance. This essentially exists
    // to initialize the Vulkan instance, and allows for the creation of multiple
    // devices.
    daxa::Instance instance = daxa::create_instance({});

    // We then just create a Daxa device by using the instance, and providing
    // the necessary parameters. This is where you'd select different features
    // and extensions that aren't necessarily present on all GPU's drivers.
    // A user's computer may have multiple devices, you need to choose a device.
    // You either use the daxa build in daxa::Instance::choose_device function, or pick one yourself.
    // In the following we show how you can pick a device yourself.

    // The device info contains explicit data.
    // These are settings of the choosen gpu, what resource limits, explicit features, name etc you want for the device.
    daxa::DeviceInfo2 device_info = {
        .explicit_features = {},
        .max_allowed_buffers = 1024,
        .name = "my device",
    };

    // The instance allows us to iterate over ALL devices vulkan can find.
    std::span<daxa::DeviceProperties const> device_properties_list = instance.list_devices_properties();
    for (daxa::u32 i = 0; i < device_properties_list.size(); ++i)
    {
        // The device properties contain all information needed to choose a device:
        // limits, feature limits, queue count, explicit features, implicit features.
        daxa::DeviceProperties const & properties = device_properties_list[i];

        // Not all devices are compatible with daxa!
        // Daxa has a set of hard required features that may miss on a gpu.
        // To be sure to pick a supported device, check the missing_required_feature enum:
        if (properties.missing_required_feature != daxa::MissingRequiredVkFeature::NONE)
        {
            continue;
        }

        // We can also check if the device has all the features we need:
        // Explicit features are ones the have to be manually enabled, otherwise the remain disabled!
        // They are manually enabled because they usually have a performance impact or other potentially undesirable effects.
        // We must check if the gpu supports the explicit features we want for the device.
        daxa::ExplicitFeatureFlags required_explicit_features = device_info.explicit_features;
        // These features pose no downsides, so they are always enabled when present.
        // We must check if the gpu supports the implicit features we want for the device.
        daxa::ImplicitFeatureFlags required_implicit_features = {};
        if (!(properties.explicit_features & required_explicit_features) || !(properties.implicit_features & required_implicit_features))
        {
            continue;
        }

        // We can also check for resource limits and queue counts:
        if (properties.limits.line_width_range == 0 || properties.compute_queue_count > 500)
        {
            continue;
        }
        
        // If a device fulfills all our requirements, we pick it:
        device_info.physical_device_index = i;
        break;
    }

    auto device = instance.create_device_2(device_info);
    // Alternatively we can let daxa choose a device for us if we only care about the explicit settings in the info and some implicit features:
    // auto device = instance.create_device_2(instance.choose_device(required_implicit_features, device_info));

    // To be able to render to a window, Daxa requires the user to create a swapchain
    // for a system window. To create a Swapchain, Daxa needs a native window handle
    // and native platform. Optionally, the user can provide a debug name, surface
    // format type selector, the additional image uses (image uses will be explained later),
    // and present mode (this controls sync)
    daxa::Swapchain swapchain = device.create_swapchain({
        // this handle is given by the windowing API
        .native_window = native_window_handle,
        // The platform would also be retrieved from the windowing API,
        // or by hard-coding it depending on the OS.
        .native_window_platform = native_window_platform,
        .surface_format_selector = [](daxa::Format format)
        {
            switch (format)
            {
            case daxa::Format::R8G8B8A8_UINT: return 100;
            default: return daxa::default_format_score(format);
            }
        },
        .present_mode = daxa::PresentMode::MAILBOX,
        .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
        .name = "my swapchain",
    });

    while (true)
    {
        glfwPollEvents();
        if (glfwWindowShouldClose(glfw_window_ptr) != 0)
        {
            break;
        }

        // Here's where we handle that variable we set in the resize callback of GLFW,
        // in order to properly handle resizing of a window!
        if (window_info.swapchain_out_of_date)
        {
            swapchain.resize();
            window_info.swapchain_out_of_date = false;
        }

        // The first thing that is done is retrieving the current swapchain image.
        // This is the image we are going to clear. If swapchain image acquisition
        // failed, then the result of this acquire function will be an "empty image",
        // and thus we want to skip this frame. We do this by saying "continue".
        daxa::ImageId const swapchain_image = swapchain.acquire_next_image();
        if (swapchain_image.is_empty())
        {
            continue;
        }

        // Directly after, we define an image slice. As images can be made up of multiple
        // layers, memory planes, and mip levels, Daxa takes slices in many calls to
        // specify a slice of an image that should be operated upon.
        // Technically the default value for slice in barriers is simply a 2d image.
        // So in this case we would not need to mention it in the barriers at all!
        // But for demonstration we show how to get the full slice and pass it to the barriers here.
        daxa::ImageMipArraySlice const swapchain_image_full_slice = device.image_view_info(swapchain_image.default_view()).value().slice;

        daxa::CommandRecorder recorder = device.create_command_recorder({.name = "my command recorder"});

        recorder.pipeline_barrier_image_transition({
            .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            .src_layout = daxa::ImageLayout::UNDEFINED,
            .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .image_slice = swapchain_image_full_slice,
            .image_id = swapchain_image,
        });

        recorder.clear_image({
            .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .clear_value = std::array<daxa::f32, 4>{1.0f, 0.0f, 1.0f, 1.0f},
            .dst_image = swapchain_image,
            .dst_slice = swapchain_image_full_slice,
        });

        recorder.pipeline_barrier_image_transition({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .dst_layout = daxa::ImageLayout::PRESENT_SRC,
            .image_slice = swapchain_image_full_slice,
            .image_id = swapchain_image,
        });

        // Here we create executable commands from the currently recorded commands from the command recorder.
        // After doing this the recorder is empty and can record and create new executable commands later.
        auto executalbe_commands = recorder.complete_current_commands();
        // But for now we only need this one executable commands blob, and we destroy the recorder.
        // Encoders CAN NOT be keept over multiple frames. They are temporary objects and need to be destroyed before calling collect_garbage!
        recorder.~CommandRecorder();

        auto const & acquire_semaphore = swapchain.current_acquire_semaphore();
        auto const & present_semaphore = swapchain.current_present_semaphore();
        device.submit_commands(daxa::CommandSubmitInfo{
            .command_lists = std::array{executalbe_commands},
            .wait_binary_semaphores = std::array{acquire_semaphore},
            .signal_binary_semaphores = std::array{present_semaphore},
            .signal_timeline_semaphores = std::array{swapchain.current_timeline_pair()},
        });
        device.present_frame({
            .wait_binary_semaphores = std::array{present_semaphore},
            .swapchain = swapchain,
        });
        device.collect_garbage();
    }

    device.wait_idle();
    device.collect_garbage();
}
