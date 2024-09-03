#include <0_common/window.hpp>
#include <thread>

namespace tests
{
    void simple_creation()
    {
        struct App : AppWindow<App>
        {
            daxa::Instance daxa_ctx = daxa::create_instance({});
            daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({}, {}));

            daxa::Swapchain swapchain = device.create_swapchain({
                .native_window = get_native_handle(),
                .native_window_platform = get_native_platform(),
                .present_mode = daxa::PresentMode::FIFO,
                .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = ("swapchain (simple_creation)"),
            });

            App() : AppWindow<App>(" (simple_creation)") {}

            void on_mouse_move(f32 /*unused*/, f32 /*unused*/) {}
            void on_mouse_button(i32 /*unused*/, i32 /*unused*/) {}
            void on_key(i32 /*unused*/, i32 /*unused*/) {}
            void on_resize(u32 /*unused*/, u32 /*unused*/) {}
        };
        App const app;
    }

    void clearcolor()
    {
        struct App : AppWindow<App>
        {
            daxa::Instance daxa_ctx = daxa::create_instance({});
            daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({},{}));

            daxa::Swapchain swapchain = device.create_swapchain({
                .native_window = get_native_handle(),
                .native_window_platform = get_native_platform(),
                .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = ("swapchain (clearcolor)"),
            });

            App() : AppWindow<App>(" (clearcolor)") {}

            auto update() -> bool
            {
                glfwPollEvents();
                if (glfwWindowShouldClose(glfw_window_ptr) != 0)
                {
                    return true;
                }

                if (!minimized)
                {
                    draw();
                }
                else
                {
                    using namespace std::literals;
                    std::this_thread::sleep_for(1ms);
                }

                return false;
            }

            void draw()
            {
                auto swapchain_image = swapchain.acquire_next_image();
                if (swapchain_image.is_empty())
                {
                    return;
                }
                auto recorder = device.create_command_recorder({
                    .name = ("recorder (clearcolor)"),
                });

                recorder.pipeline_barrier_image_transition({
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .src_layout = daxa::ImageLayout::UNDEFINED,
                    .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .image_id = swapchain_image,
                });

                recorder.clear_image({
                    .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .clear_value = {std::array<f32, 4>{1, 0, 1, 1}},
                    .dst_image = swapchain_image,
                });

                recorder.pipeline_barrier_image_transition({
                    .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .dst_layout = daxa::ImageLayout::PRESENT_SRC,
                    .image_id = swapchain_image,
                });

                auto executalbe_commands = recorder.complete_current_commands();
                recorder.~CommandRecorder();

                device.submit_commands({
                    .command_lists = std::array{executalbe_commands},
                    .wait_binary_semaphores = std::array{swapchain.current_acquire_semaphore()},
                    .signal_binary_semaphores = std::array{swapchain.current_present_semaphore()},
                    .signal_timeline_semaphores = std::array{swapchain.current_timeline_pair()},
                });

                device.present_frame({
                    .wait_binary_semaphores = std::array{swapchain.current_present_semaphore()},
                    .swapchain = swapchain,
                });

                device.collect_garbage();
            }

            void on_mouse_move(f32 /*unused*/, f32 /*unused*/) {}
            void on_mouse_button(i32 /*unused*/, i32 /*unused*/) {}
            void on_key(i32 /*unused*/, i32 /*unused*/) {}

            void on_resize(u32 sx, u32 sy)
            {
                minimized = sx == 0 || sy == 0;
                if (!minimized)
                {
                    swapchain.resize();
                    size_x = swapchain.get_surface_extent().x;
                    size_y = swapchain.get_surface_extent().y;
                    draw();
                }
            }
        };

        App app = {};
        while (true)
        {
            if (app.update())
            {
                break;
            }
        }
    }
} // namespace tests

auto main() -> int
{
    tests::simple_creation();
    tests::clearcolor();
}
