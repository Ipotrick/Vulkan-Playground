#include <0_common/window.hpp>
#include <thread>
#include <iostream>
#include <cmath>

#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>

using namespace daxa::types;
using Clock = std::chrono::high_resolution_clock;

#include "shared.inl"

// Update task:

    DAXA_DECL_TASK_HEAD_BEGIN(UpdateBoids)
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, current)
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ, previous)
    DAXA_DECL_TASK_HEAD_END
    struct UpdateBoidsTask : UpdateBoids::Task
    {
        AttachmentViews views = {};
        std::shared_ptr<daxa::ComputePipeline> update_boids_pipeline = {};
        void callback(daxa::TaskInterface ti)
        {
            ti.recorder.set_pipeline(*update_boids_pipeline);
            ti.recorder.push_constant(UpdateBoidsPushConstant{
                .boids_buffer = ti.device_address(AT.current).value(),
                .old_boids_buffer = ti.device_address(AT.previous).value(),
            });
            ti.recorder.dispatch({(MAX_BOIDS + 63) / 64, 1, 1});
        }
    };

struct Test
{
    DAXA_TH_BLOB(UpdateBoids, test)
};

struct App : AppWindow<App>
{
    daxa::Instance daxa_ctx = daxa::create_instance({});
    daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({},{}));

    daxa::Swapchain swapchain = device.create_swapchain({
        .native_window = get_native_handle(),
        .native_window_platform = get_native_platform(),
        .present_mode = daxa::PresentMode::FIFO,
        .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
        .name = ("swapchain"),
    });

    daxa::PipelineManager pipeline_manager = daxa::PipelineManager({
        .device = device,
        .shader_compile_options = {
            .root_paths = {
                DAXA_SHADER_INCLUDE_DIR,
                "tests/3_samples/5_boids/shaders",
                "tests/3_samples/5_boids",
            },
            .language = daxa::ShaderLanguage::GLSL,
            .enable_debug_info = true,
        },
        .name = ("pipeline_manager"),
    });
    // clang-format off
    std::shared_ptr<daxa::RasterPipeline> draw_pipeline = pipeline_manager.add_raster_pipeline({
        .vertex_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"vert.glsl"}},
        .fragment_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"frag.glsl"}},
        .color_attachments = {{.format = swapchain.get_format()}},
        .raster = {},
        .push_constant_size = sizeof(DrawPushConstant),
        .name = ("draw_pipeline"),
    }).value();
    std::shared_ptr<daxa::ComputePipeline> update_boids_pipeline = pipeline_manager.add_compute_pipeline({
        .shader_info = {.source = daxa::ShaderFile{"update_boids.glsl"}},
        .push_constant_size = sizeof(UpdateBoidsPushConstant),
        .name = ("draw_pipeline"),
    }).value();
    // clang-format on

    u32 current_buffer_i = 1;

    f32 aspect = static_cast<f32>(size_x) / static_cast<f32>(size_y);

    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start = Clock::now();
    Clock::time_point prev_time = start;

    daxa::BufferId boid_buffer = device.create_buffer({
        .size = sizeof(Boids),
        .name = ("boids buffer a"),
    });

    daxa::BufferId old_boid_buffer = device.create_buffer({
        .size = sizeof(Boids),
        .name = ("boids buffer b"),
    });

    daxa::TaskImage task_swapchain_image{{.swapchain_image = true, .name = "swapchain image"}};
    daxa::TaskBuffer task_boids_current{{.initial_buffers = {.buffers = {&boid_buffer, 1}}, .name = "task_boids_current"}};
    daxa::TaskBuffer task_boids_old{{.initial_buffers = {.buffers = {&old_boid_buffer, 1}}, .name = "task_boids_old"}};

    daxa::CommandSubmitInfo submit_info;

    daxa::TaskGraph task_graph = record_tasks();

    App() : AppWindow<App>("boids")
    {
        auto recorder = device.create_command_recorder({.name = ("boid buffer init commands")});

        auto upload_buffer_id = device.create_buffer({
            .size = sizeof(Boids),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = ("boids buffer init staging buffer"),
        });
        recorder.destroy_buffer_deferred(upload_buffer_id);

        auto * ptr = device.buffer_host_address_as<Boids>(upload_buffer_id).value();

        for (auto & boid : ptr->boids)
        {
            boid.position.x = static_cast<f32>(rand() % ((FIELD_SIZE) * 100)) / 100.0f;
            boid.position.y = static_cast<f32>(rand() % ((FIELD_SIZE) * 100)) / 100.0f;
            f32 const angle = static_cast<f32>(rand() % 3600) * 0.1f;
            boid.speed.x = std::cos(angle);
            boid.speed.y = std::sin(angle);
        }

        recorder.copy_buffer_to_buffer({
            .src_buffer = upload_buffer_id,
            .dst_buffer = boid_buffer,
            .size = sizeof(Boids),
        });

        recorder.copy_buffer_to_buffer({
            .src_buffer = upload_buffer_id,
            .dst_buffer = old_boid_buffer,
            .size = sizeof(Boids),
        });

        recorder.pipeline_barrier({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ_WRITE | daxa::AccessConsts::VERTEX_SHADER_READ,
        });
        auto executable_commands = recorder.complete_current_commands();
        recorder.~CommandRecorder();
        device.submit_commands({
            .command_lists = std::array{executable_commands},
        });
        device.collect_garbage();
    }

    ~App()
    {
        device.destroy_buffer(boid_buffer);
        device.destroy_buffer(old_boid_buffer);
    }

    // Draw task:

    struct DrawBoidsTask : daxa::PartialTask<2, "DrawBoids">
    {
        static inline daxa::TaskBufferAttachmentIndex const boids = add_attachment(daxa::TaskBufferAttachment{
            .access = daxa::TaskBufferAccess::VERTEX_SHADER_READ,
        });
        static inline daxa::TaskImageAttachmentIndex const render_image = add_attachment(daxa::TaskImageAttachment{
            .access = daxa::TaskImageAccess::COLOR_ATTACHMENT,
        });
        AttachmentViews views = {};
        std::shared_ptr<daxa::RasterPipeline> draw_pipeline = {};
        u32 * size_x = {};
        u32 * size_y = {};
        void callback(daxa::TaskInterface ti) const
        {
            auto render_recorder = std::move(ti.recorder).begin_renderpass({
                .color_attachments = std::array{
                    daxa::RenderAttachmentInfo{
                        .image_view = ti.get(render_image).view_ids[0],
                        .layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                        .load_op = daxa::AttachmentLoadOp::CLEAR,
                        .store_op = daxa::AttachmentStoreOp::STORE,
                        .clear_value = std::array<f32, 4>{1.0f, 1.0f, 1.0f, 1.0f},
                    },
                },
                .render_area = {
                    .width = *size_x,
                    .height = *size_y,
                },
            });
            render_recorder.set_pipeline(*draw_pipeline);
            render_recorder.push_constant(DrawPushConstant{
                .boids_buffer = ti.device_address(boids).value(),
                .axis_scaling = {
                    std::min(1.0f, static_cast<f32>(*this->size_y) / static_cast<f32>(*this->size_x)),
                    std::min(1.0f, static_cast<f32>(*this->size_x) / static_cast<f32>(*this->size_y)),
                },
            });
            render_recorder.draw({.vertex_count = 3 * MAX_BOIDS});
            ti.recorder = std::move(render_recorder).end_renderpass();
        }
    };

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

    auto record_tasks() -> daxa::TaskGraph
    {
        daxa::TaskGraph new_task_graph = daxa::TaskGraph({.device = device, .swapchain = swapchain, .name = ("main task graph")});
        new_task_graph.use_persistent_image(task_swapchain_image);
        new_task_graph.use_persistent_buffer(task_boids_current);
        new_task_graph.use_persistent_buffer(task_boids_old);
        using namespace UpdateBoids;
        new_task_graph.add_task(UpdateBoidsTask{
            .views = std::array{
                daxa::attachment_view( UpdateBoids::AT.current, task_boids_current ),
                daxa::attachment_view( UpdateBoids::AT.previous, task_boids_old ),
            },
            .update_boids_pipeline = update_boids_pipeline,
        });
        new_task_graph.add_task(DrawBoidsTask{
            .views = std::array{
                daxa::attachment_view( DrawBoidsTask::boids, task_boids_current ),
                daxa::attachment_view( DrawBoidsTask::render_image, task_swapchain_image ),
            },
            .draw_pipeline = draw_pipeline,
            .size_x = &size_x,
            .size_y = &size_y,
        });
        new_task_graph.submit({});
        new_task_graph.present({});
        new_task_graph.complete({});
        return new_task_graph;
    }

    void draw()
    {
        auto now = Clock::now();
        prev_time = now;

        auto reloaded_result = pipeline_manager.reload_all();
        if (auto * reload_err = daxa::get_if<daxa::PipelineReloadError>(&reloaded_result))
        {
            std::cout << "Failed to reload " << reload_err->message << '\n';
        }
        if (daxa::get_if<daxa::PipelineReloadSuccess>(&reloaded_result) != nullptr)
        {
            std::cout << "Successfully reloaded!\n";
        }

        auto swapchain_image = swapchain.acquire_next_image();
        task_swapchain_image.set_images({.images = {&swapchain_image, 1}});
        if (swapchain_image.is_empty())
        {
            return;
        }
        task_graph.execute({});
        // Switch boids front and back buffers.
        task_boids_current.swap_buffers(task_boids_old);
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
            aspect = static_cast<f32>(size_x) / static_cast<f32>(size_y);
            draw();
        }
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
