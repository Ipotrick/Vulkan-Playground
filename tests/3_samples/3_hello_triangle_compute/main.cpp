#define DAXA_SHADERLANG DAXA_SHADERLANG_GLSL
#define APPNAME "Daxa Sample: HelloTriangle Compute"
#include <0_common/base_app.hpp>

using namespace daxa::types;
#include "shaders/shared.inl"

struct App : BaseApp<App>
{
    // clang-format off
    std::shared_ptr<daxa::ComputePipeline> compute_pipeline = pipeline_manager.add_compute_pipeline({
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
        .shader_info = {.source = daxa::ShaderFile{"compute.glsl"}},
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
        .shader_info = {.source = daxa::ShaderFile{"compute.hlsl"}},
#endif
        .push_constant_size = sizeof(ComputePush),
        .name = "compute_pipeline",
    }).value();
    // clang-format on

    daxa::ImageId render_image = device.create_image(daxa::ImageInfo{
        .format = daxa::Format::R8G8B8A8_UNORM,
        .size = {size_x, size_y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
        .name = "render_image",
    });
    daxa::TaskImage task_render_image{{.initial_images = {.images = std::array{render_image}}, .name = "task_render_image"}};

    daxa::TaskGraph loop_task_graph = record_loop_task_graph();

    ~App()
    {
        device.wait_idle();
        device.collect_garbage();
        device.destroy_image(render_image);
    }

    static void ui_update()
    {
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Render();
    }
    void on_update()
    {
        auto reloaded_result = pipeline_manager.reload_all();
        if (auto reload_err = daxa::get_if<daxa::PipelineReloadError>(&reloaded_result))
            std::cout << "Failed to reload " << reload_err->message << '\n';
        if (daxa::get_if<daxa::PipelineReloadSuccess>(&reloaded_result))
            std::cout << "Successfully reloaded!\n";
        ui_update();

        auto swapchain_image = swapchain.acquire_next_image();
        task_swapchain_image.set_images({.images = std::array{swapchain_image}});
        if (swapchain_image.is_empty())
        {
            return;
        }
        loop_task_graph.execute({});
        device.collect_garbage();
    }

    void on_mouse_move(f32 /*unused*/, f32 /*unused*/) {}
    void on_mouse_button(i32 /*unused*/, i32 /*unused*/) {}
    void on_key(i32 /*unused*/, i32 /*unused*/) {}
    void on_resize(u32 sx, u32 sy)
    {
        minimized = (sx == 0 || sy == 0);
        if (!minimized)
        {
            swapchain.resize();
            size_x = swapchain.get_surface_extent().x;
            size_y = swapchain.get_surface_extent().y;
            device.destroy_image(render_image);
            render_image = device.create_image({
                .format = daxa::Format::R8G8B8A8_UNORM,
                .size = {size_x, size_y, 1},
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
            });
            task_render_image.set_images({.images = std::array{render_image}});
            base_on_update();
        }
    }

    void record_tasks(daxa::TaskGraph & new_task_graph)
    {
        new_task_graph.use_persistent_image(task_render_image);

        new_task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY, task_render_image)},
            .task = [this](daxa::TaskInterface ti)
            {
                ti.recorder.set_pipeline(*compute_pipeline);
                ti.recorder.push_constant(ComputePush{
                    .image = render_image.default_view(),
                    .frame_dim = {size_x, size_y},
                });
                ti.recorder.dispatch({(size_x + 7) / 8, (size_y + 7) / 8});
            },
            .name = APPNAME_PREFIX("Draw (Compute)"),
        });
        new_task_graph.add_task({
            .attachments = {
                {daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_READ, task_render_image)},
                {daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, task_swapchain_image)},
            },
            .task = [this](daxa::TaskInterface ti)
            {
                ti.recorder.blit_image_to_image({
                    .src_image = ti.get(task_render_image).ids[0],
                    .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    .dst_image = ti.get(task_swapchain_image).ids[0],
                    .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .src_offsets = {{{0, 0, 0}, {static_cast<i32>(size_x), static_cast<i32>(size_y), 1}}},
                    .dst_offsets = {{{0, 0, 0}, {static_cast<i32>(size_x), static_cast<i32>(size_y), 1}}},
                });
            },
            .name = APPNAME_PREFIX("Blit (render to swapchain)"),
        });
    }
};

auto main() -> int
{
    App app = {};
    while (true)
    {
        if (app.update())
        {
            break;
        }
    }
}
