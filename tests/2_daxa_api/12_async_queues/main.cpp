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

/// 
/// DAXA Multiqueue
/// 
/// In Vulkan, queues are very verbose
/// They expose a lot of api surface and implementation options that are rarely if ever used by hardware vendors.
/// Daxa multiqueue simplifies vulkans multiqueue capabilities to that of the current hardware.
/// It also simplifies some details like queue resource ownership to drastically simplify the api surface. 
///
/// There are at max 8 compute and 2 transfer queues.
///   There is only one graphics (the main) queue, as no hardware actually supports async graphics queues.
///   This constrains the complexity of internals and ensures performance of current code.
///   More then these queue counts are basically never supported nor useful anyways.
///   These queues are ensured to be in different queue families.
///
/// Daxa only supports swapchains for the main queue atm. This might change in the future.
/// 
/// Daxa does not expose ownership transfers as they are nearly never useful.
///   All buffers are multi queue shared.
///   Images are default exclusive to the main queue.
///   Optionally images are shared across all queues.
///   There is no way to transfer ownership and as the exclusive ownership is mostly useful for renderpass write compression, 
///   the main queue is the default exlusively owning queue.
///
/// CommandRecorder Types
///   * daxa has four command recorders
///   * CommandRecorder
///     * can be used only for the main queue family
///     * can record any command and can be morphed into a render command recorder
///   * RenderCommandRecorder
///     * can record commands that are only valid within a renderpass
///     * created by generic command recorder
///     * must be morphed back into a command recorder to finish a render pass
///   * ComputeCommandRecorder
///     * can only be used to record commands to the main and compute queues, NOT transfer queues.
///     * can record any command allowed on transfer and compute queues and the main queue, does not allow recording of graphics commands.
///   * TransferCommandRecorder
///     * can be used to record commands to ANY queue
///     * can only record commands that are valid on the transfer queue and all others.
///   * allows for cheap compile time checks. Without these types, daxa would have to check the queue family for every command.
///

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

#include "shared.inl"

namespace tests
{
    void basics()
    {
        daxa::Instance instance = daxa::create_instance({});
        daxa::Device device = instance.create_device_2(instance.choose_device({}, daxa::DeviceInfo2{}));
        
        daxa::u32 compute_queue_count = device.queue_count(daxa::QueueFamily::COMPUTE);
        daxa::u32 transfer_queue_count = device.queue_count(daxa::QueueFamily::TRANSFER);
        std::cout << "Device has " << compute_queue_count << " async compute and " << transfer_queue_count << " async transfer queues.\n";
        std::cout << "Daxa's maximum for async compute queues is 8 and daxa's maximum for async transfer queues is 2.\n";

        device.queue_wait_idle(daxa::QUEUE_MAIN);
        for (daxa::u32 queue = 0; queue < compute_queue_count; ++queue)
        {
            device.queue_wait_idle(daxa::Queue{daxa::QueueFamily::COMPUTE, queue});
        }     
        for (daxa::u32 queue = 0; queue < transfer_queue_count; ++queue)
        {
            device.queue_wait_idle(daxa::Queue{daxa::QueueFamily::TRANSFER, queue});
        }   

        bool found_error = false;
        try
        {
            /// NOTE: This should fail. This is an intentional error.
            device.queue_wait_idle(daxa::Queue{daxa::QueueFamily::COMPUTE, 800});
        }  
        catch(std::runtime_error e)
        {
            std::cout << "  ^^ Above assertion failure is expected!" << std::endl;
            found_error = true;
        }
        DAXA_DBG_ASSERT_TRUE_M(found_error, "device failed to detect invalid queue value!");
    }

    void simple_submit_chain()
    {
        daxa::Instance instance = daxa::create_instance({});
        daxa::Device device = instance.create_device_2(instance.choose_device({}, {}));

        daxa::ExecutableCommandList commands = {};
        {
            auto rec = device.create_command_recorder({});
            commands = rec.complete_current_commands();
        }

        auto sema0 = device.create_binary_semaphore({.name = "sema 0"});
        auto sema1 = device.create_binary_semaphore({.name = "sema 1"});

        constexpr daxa::u32 initial_value = 42u;
        // Make buffer for a u32[4] array and write some data into the first element
        auto buffer = device.create_buffer({sizeof(daxa::u32) * 4, daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, "buffer"});
        *device.buffer_host_address_as<daxa::u32>(buffer).value() = initial_value;    

        {
            // Copy from index 0 to index 1
            // Command recorders queue family MUST match the queue it is submitted to.
            // Commands for a transfer queue MUST ONLY be recorded by a transfer command recorder!
            // A generic or compute command recorder can not record commands for a transfer queue!
            // Tho transfer command recorders CAN record commands for any queue.
            auto rec = device.create_transfer_command_recorder({daxa::QueueFamily::TRANSFER});
            rec.pipeline_barrier({daxa::AccessConsts::TRANSFER_WRITE, daxa::AccessConsts::TRANSFER_READ_WRITE});
            rec.copy_buffer_to_buffer({
                .src_buffer = buffer,
                .dst_buffer = buffer,
                .src_offset = sizeof(daxa::u32) * 0,
                .dst_offset = sizeof(daxa::u32) * 1,
                .size = sizeof(daxa::u32),
            });
            rec.pipeline_barrier({daxa::AccessConsts::TRANSFER_WRITE, daxa::AccessConsts::TRANSFER_READ_WRITE});
            auto commands = rec.complete_current_commands();
            device.submit_commands({
                .queue = daxa::QUEUE_TRANSFER_0,
                .command_lists = std::array{commands},
                .signal_binary_semaphores = std::array{sema0},
            });
        }  

        {
            // Copy from index 1 to index 2
            // Command recorders queue family MUST match the queue it is submitted to.
            // Commands for a compute queue can only be recorded by a compute or a transfer command recorder!
            // A generic command recorder is not allowed to record commands for a compute queue!
            auto rec = device.create_compute_command_recorder({daxa::QueueFamily::COMPUTE});
            rec.pipeline_barrier({daxa::AccessConsts::TRANSFER_READ_WRITE, daxa::AccessConsts::TRANSFER_READ_WRITE});
            rec.copy_buffer_to_buffer({
                .src_buffer = buffer,
                .dst_buffer = buffer,
                .src_offset = sizeof(daxa::u32) * 1,
                .dst_offset = sizeof(daxa::u32) * 2,
                .size = sizeof(daxa::u32),
            });
            rec.pipeline_barrier({daxa::AccessConsts::TRANSFER_WRITE, daxa::AccessConsts::TRANSFER_READ_WRITE});
            auto commands = rec.complete_current_commands();
            device.submit_commands({
                .queue = daxa::QUEUE_COMPUTE_1,
                .command_lists = std::array{commands},
                .wait_binary_semaphores = std::array{sema0},
                .signal_binary_semaphores = std::array{sema1},
            });
        }  
        
        {
            // Copy from index 2 to index 3
            // Any command recorder type can be used to submit commands to the main queue.
            auto rec = device.create_command_recorder({
                // daxa::QueueFamily::MAIN // The default is the main queue family
                // The Queue MUST be main here as its a generic command recorder
            });
            rec.pipeline_barrier({daxa::AccessConsts::TRANSFER_READ_WRITE, daxa::AccessConsts::TRANSFER_READ_WRITE});
            rec.copy_buffer_to_buffer({
                .src_buffer = buffer,
                .dst_buffer = buffer,
                .src_offset = sizeof(daxa::u32) * 2,
                .dst_offset = sizeof(daxa::u32) * 3,
                .size = sizeof(daxa::u32),
            });
            rec.pipeline_barrier({daxa::AccessConsts::TRANSFER_READ_WRITE, daxa::AccessConsts::TRANSFER_READ_WRITE});
            auto commands = rec.complete_current_commands();
            device.submit_commands({
                // .queue = daxa::Queue::MAIN, // The default is the main queue.
                .command_lists = std::array{commands},
                .wait_binary_semaphores = std::array{sema1},
            });
        }   

        // The semaphores make sure that the queues submissions are processed in the correct order (transfer0 -> comp0 -> main).

        // As the queues are synchronized via semaphores and the main queue is the last to run,
        // we can safely assume that all work is done, as soon as the main queue is drained.
        device.queue_wait_idle(daxa::QUEUE_MAIN);

        daxa::u32 result = device.buffer_host_address_as<daxa::u32>(buffer).value()[3];
        DAXA_DBG_ASSERT_TRUE_M(result == initial_value, "operations resulted in incorrect final result!");

        device.destroy_buffer(buffer);
    }

    namespace mesh_shader_test
    {

        struct WindowInfo
        {
            daxa::u32 width{}, height{};
            bool swapchain_out_of_date = false;
        };

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

        void mesh_shader_tri()
        {
            auto window_info = WindowInfo{.width = 800, .height = 600};
            glfwInit();
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            auto * glfw_window_ptr = glfwCreateWindow(
                static_cast<daxa::i32>(window_info.width),
                static_cast<daxa::i32>(window_info.height),
                "Daxa async compute shader sample", nullptr, nullptr);
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
                        "./tests/2_daxa_api/12_async_queues",
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
                    return;
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
                    DrawTri::AT.resolve_target | task_swapchain_image,
                    DrawTri::AT.render_target | trender_image,
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
                {
                    auto rec = device.create_compute_command_recorder({daxa::QueueFamily::COMPUTE});
                    auto cmds = rec.complete_current_commands();
                    device.submit_commands({
                        .queue = daxa::QUEUE_COMPUTE_0, 
                        .command_lists = std::array{cmds},
                    });
                }
            }

            device.wait_idle();
            device.destroy_image(render_target);
            device.collect_garbage();
        }
    } // namespace mesh_shader_test
} // namespace tests

auto main() -> int
{
    tests::basics();
    tests::simple_submit_chain();
    tests::mesh_shader_test::mesh_shader_tri();
    return 0;
}