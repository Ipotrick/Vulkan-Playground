#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>

#include <iostream>
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

#include "shared.inl"

struct DrawTask : DrawTri::Task
{
    AttachmentViews views = {};
    std::shared_ptr<daxa::RasterPipeline> pipeline = {};
    void callback(daxa::TaskInterface ti)
    {
        daxa::ImageInfo color_img_info = ti.info(AT.render_target).value();
        auto const size_x = color_img_info.size.x;
        auto const size_y = color_img_info.size.y;
        auto render_recorder = std::move(ti.recorder).begin_renderpass({
            .color_attachments = std::array{
                daxa::RenderAttachmentInfo{
                    .image_view = ti.get(AT.render_target).view_ids[0],
                    .load_op = daxa::AttachmentLoadOp::CLEAR,
                    .clear_value = std::array<daxa::f32, 4>{0.1f, 0.0f, 0.5f, 1.0f},
                    .resolve = daxa::AttachmentResolveInfo{
                        .image = ti.get(AT.resolve_target).view_ids[0],
                    },
                },
            },
            .render_area = {.x = 0, .y = 0, .width = size_x, .height = size_y},
        });
        render_recorder.set_rasterization_samples(daxa::RasterizationSamples::E4);
        render_recorder.set_pipeline(*pipeline);
        render_recorder.draw_mesh_tasks(1, 1, 1);
        ti.recorder = std::move(render_recorder).end_renderpass();
    }
};

auto main() -> int
{
    auto window_info = WindowInfo{.width = 800, .height = 600};
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto * glfw_window_ptr = glfwCreateWindow(
        static_cast<daxa::i32>(window_info.width),
        static_cast<daxa::i32>(window_info.height),
        "Daxa mesh shader sample", nullptr, nullptr);
    glfwSetWindowUserPointer(glfw_window_ptr, &window_info);
    glfwSetWindowSizeCallback(
        glfw_window_ptr,
        [](GLFWwindow * glfw_window, int width, int height)
        {
            auto & window_info_ref = *reinterpret_cast<WindowInfo *>(glfwGetWindowUserPointer(glfw_window));
            window_info_ref.swapchain_out_of_date = true;
            window_info_ref.width = static_cast<daxa::u32>(width);
            window_info_ref.height = static_cast<daxa::u32>(height);
        });
    auto * native_window_handle = get_native_handle(glfw_window_ptr);
    auto native_window_platform = get_native_platform(glfw_window_ptr);

    daxa::Instance instance = daxa::create_instance({});

    daxa::Device device = instance.create_device_2(instance.choose_device(daxa::ImplicitFeatureFlagBits::MESH_SHADER, {}));

    daxa::Swapchain swapchain = device.create_swapchain({
        .native_window = native_window_handle,
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

    daxa::ImageId render_target = device.create_image({
        .format = daxa::Format::B8G8R8A8_UNORM,
        .size = {window_info.width, window_info.height, 1},
        .sample_count = 4,
        .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT | daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .name = "render target",
    });
    daxa::TaskImage trender_image = daxa::TaskImage{{
        .initial_images = {
            .images = std::array{render_target},
        },
        .name = "render target",
    }};

    auto pipeline_manager = daxa::PipelineManager({
        .device = device,
        .shader_compile_options = {
            .root_paths = {
                DAXA_SHADER_INCLUDE_DIR,
                "./tests/2_daxa_api/11_mesh_shader",
            },
            .language = daxa::ShaderLanguage::SLANG,
            .enable_debug_info = true,
        },
        .name = "my pipeline manager",
    });
    // Then just adding it to the pipeline manager
    std::shared_ptr<daxa::RasterPipeline> pipeline;
    {
        auto result = pipeline_manager.add_raster_pipeline({
            .mesh_shader_info = daxa::ShaderCompileInfo{
                .source = daxa::ShaderFile{"draw.slang"},
                .compile_options = daxa::ShaderCompileOptions{.entry_point = "entry_mesh"},
            },
            .fragment_shader_info = daxa::ShaderCompileInfo{
                .source = daxa::ShaderFile{"draw.slang"},
                .compile_options = daxa::ShaderCompileOptions{.entry_point = "entry_fragment"},
            },
            .color_attachments = {{.format = swapchain.get_format()}},
            .raster = {.static_state_sample_count = daxa::None},
            // .push_constant_size = sizeof(DrawTri::attachment_shader_blob_size()),
            .name = "my pipeline",
        });
        if (result.is_err())
        {
            std::cerr << result.message() << std::endl;
            return -1;
        }
        pipeline = result.value();
    }

    auto task_swapchain_image = daxa::TaskImage{{.swapchain_image = true, .name = "swapchain image"}};

    auto loop_task_graph = daxa::TaskGraph({
        .device = device,
        .swapchain = swapchain,
        .name = "loop",
    });

    loop_task_graph.use_persistent_image(task_swapchain_image);
    loop_task_graph.use_persistent_image(trender_image);

    // And a task to draw to the screen
    loop_task_graph.add_task(DrawTask{
        .views = std::array{
            daxa::attachment_view(DrawTri::AT.resolve_target, task_swapchain_image),
            daxa::attachment_view(DrawTri::AT.render_target, trender_image),
        },
        .pipeline = pipeline,
    });

    loop_task_graph.submit({});
    loop_task_graph.present({});
    loop_task_graph.complete({});

    while (true)
    {
        glfwPollEvents();
        if (glfwWindowShouldClose(glfw_window_ptr) != 0)
        {
            break;
        }

        if (window_info.swapchain_out_of_date)
        {
            swapchain.resize();
            window_info.swapchain_out_of_date = false;
        }

        // acquire the next image as usual,
        auto swapchain_image = swapchain.acquire_next_image();
        if (swapchain_image.is_empty())
        {
            continue;
        }
        task_swapchain_image.set_images({.images = std::array{swapchain_image}});
        loop_task_graph.execute({});
        device.collect_garbage();
    }

    device.wait_idle();
    device.destroy_image(render_target);
    device.collect_garbage();
}