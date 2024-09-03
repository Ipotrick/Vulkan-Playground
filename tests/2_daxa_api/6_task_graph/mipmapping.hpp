#pragma once

#include "common.hpp"

namespace tests
{
    DAXA_DECL_TASK_HEAD_BEGIN(MipMapH)
    DAXA_TH_IMAGE(TRANSFER_READ, REGULAR_2D, lower_mip)
    DAXA_TH_IMAGE(TRANSFER_WRITE, REGULAR_2D, higher_mip)
    DAXA_DECL_TASK_HEAD_END
    struct MipMapTask : MipMapH::Task
    {
        AttachmentViews views = {};
        struct Info
        {
            u32 mip = {};
            std::array<i32, 3> mip_size = {};
            std::array<i32, 3> next_mip_size = {};
        } info = {};
        void callback(daxa::TaskInterface tri)
        {
            tri.recorder.blit_image_to_image({
                .src_image = tri.get(AT.lower_mip).ids[0],
                .dst_image = tri.get(AT.higher_mip).ids[0],
                .src_slice = {
                    .mip_level = info.mip,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .src_offsets = {{{0, 0, 0}, {info.mip_size[0], info.mip_size[1], info.mip_size[2]}}},
                .dst_slice = {
                    .mip_level = info.mip + 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .dst_offsets = {{{0, 0, 0}, {info.next_mip_size[0], info.next_mip_size[1], info.next_mip_size[2]}}},
                .filter = daxa::Filter::LINEAR,
            });
        }
    };

    void mipmapping()
    {
        struct App : AppWindow<App>
        {
            daxa::Instance daxa_ctx = daxa::create_instance({});
            daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({}, {}));

            daxa::Swapchain swapchain = device.create_swapchain({
                .native_window = get_native_handle(),
                .native_window_platform = get_native_platform(),
                .present_mode = daxa::PresentMode::IMMEDIATE,
                .image_usage = daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "swapchain",
            });
            daxa::TaskImage task_swapchain_image{{.swapchain_image = true, .name = "swapchain"}};
            daxa::ImageId swapchain_image = {};

            daxa::PipelineManager pipeline_manager = daxa::PipelineManager({
                .device = device,
                .shader_compile_options = {
                    .root_paths = {
                        DAXA_SHADER_INCLUDE_DIR,
                        "tests/2_daxa_api/6_task_graph/shaders",
                    },
                    .language = daxa::ShaderLanguage::GLSL,
                },
                .name = "pipeline_manager",
            });
            daxa::ImGuiRenderer imgui_renderer = create_imgui_renderer();
            auto create_imgui_renderer() -> daxa::ImGuiRenderer
            {
                ImGui::CreateContext();
                ImGui_ImplGlfw_InitForVulkan(glfw_window_ptr, true);
                return daxa::ImGuiRenderer({
                    .device = device,
                    .format = swapchain.get_format(),
                });
            }

            // clang-format off
            std::shared_ptr<daxa::ComputePipeline> compute_pipeline = [&]() { 
                auto result = pipeline_manager.add_compute_pipeline({
                    .shader_info = {.source = daxa::ShaderFile{"mipmapping.glsl"}},
                    .push_constant_size = sizeof(MipmappingComputePushConstant),
                    .name = "compute_pipeline",
                });
                std::cout << result.to_string() << std::endl;
                return result.value();
            }();
            // clang-format on

            daxa::ImageId render_image = device.create_image(daxa::ImageInfo{
                .format = daxa::Format::R8G8B8A8_UNORM,
                .size = {128, 128, 1},
                .mip_level_count = 5,
                .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = "render_image",
            });

            daxa::CommandSubmitInfo submit_info = {};
            daxa::TaskImage task_render_image = [&]()
            {
                auto ret = daxa::TaskImage{{
                    .name = "persistent render image",
                }};
                ret.set_images({.images = {&render_image, 1}});
                return ret;
            }();

            MipmappingGpuInput gpu_input = {
                .paint_col = {1.0f, 0.0f, 0.0f},
                .mouse_x = {},
                .mouse_y = {},
                .p_mouse_x = {},
                .p_mouse_y = {},
                .paint_radius = 5.0f,
            };
            daxa::BufferId mipmapping_gpu_input_buffer = device.create_buffer({
                .size = sizeof(MipmappingGpuInput),
                .name = "mipmapping_gpu_input_buffer",
            });

            std::array<BufferId, 1> execution_buffers{mipmapping_gpu_input_buffer};
            daxa::TaskBuffer task_mipmapping_gpu_input_buffer = [&]()
            {
                auto ret = daxa::TaskBuffer{{
                    .name = "task_mipmapping_gpu_input_buffer",
                }};
                ret.set_buffers(daxa::TrackedBuffers{.buffers = std::span{execution_buffers.data(), execution_buffers.size()}});
                return ret;
            }();

            daxa::TaskBuffer task_staging_mipmapping_gpu_input_buffer;

            bool mouse_drawing = false;

            enum TASK_CONDITIONS
            {
                TASK_CONDITION_MOUSE_DRAWING = 0,
                TASK_CONDITION_COUNT = 1,
            };
            daxa::TaskGraph task_graph = record_tasks();

            App() : AppWindow<App>(APPNAME, 768 * 2, 512 * 2) {}
            ~App()
            {
                device.destroy_buffer(mipmapping_gpu_input_buffer);
                device.destroy_image(render_image);
                device.wait_idle();
                device.collect_garbage();
            }

            bool update()
            {
                glfwPollEvents();
                if (glfwWindowShouldClose(glfw_window_ptr))
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
            void ui_update()
            {
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();
                ImGui::Begin("Settings");
                ImGui::SliderFloat("Brush Radius", &gpu_input.paint_radius, 1.0f, 15.0f);
                ImGui::ColorEdit3("Brush Color", reinterpret_cast<f32 *>(&gpu_input.paint_col));
                ImGui::End();
                ImGui::Render();
            }
            void draw()
            {
                ui_update();
                pipeline_manager.reload_all();

                // non_task_graph_execute();

                swapchain_image = swapchain.acquire_next_image();
                task_swapchain_image.set_images({.images = std::span{&swapchain_image, 1}});
                if (swapchain_image.is_empty())
                {
                    return;
                }
                std::array<bool, TASK_CONDITION_COUNT> conditions = {};
                conditions[TASK_CONDITION_MOUSE_DRAWING] = mouse_drawing;
                task_graph.execute({.permutation_condition_values = {conditions.data(), conditions.size()}, .record_debug_string = true});
                device.collect_garbage();
                // std::cout << task_graph.get_debug_string() << std::endl;
            }

            void on_mouse_move(f32 x, f32 y)
            {
                auto & io = ImGui::GetIO();
                if (io.WantCaptureMouse || io.WantCaptureKeyboard)
                    return;
                gpu_input.mouse_x = x / static_cast<f32>(size_x) * (1.5f) * 128.0f;
                gpu_input.mouse_y = y / static_cast<f32>(size_y) * 128.0f;
            }
            void on_mouse_button(i32 button, i32 action)
            {
                auto & io = ImGui::GetIO();
                if (io.WantCaptureMouse || io.WantCaptureKeyboard)
                    return;
                if (button == GLFW_MOUSE_BUTTON_1)
                    mouse_drawing = action != GLFW_RELEASE;
            }
            void on_key(i32, i32) {}
            void on_resize(u32 sx, u32 sy)
            {
                minimized = (sx == 0 || sy == 0);
                if (!minimized)
                {
                    swapchain.resize();
                    size_x = swapchain.get_surface_extent().x;
                    size_y = swapchain.get_surface_extent().y;
                    draw();
                }
            }

            void update_gpu_input(daxa::CommandRecorder & recorder, daxa::BufferId input_buffer)
            {
                auto staging_buffer = device.create_buffer({
                    .size = sizeof(MipmappingGpuInput),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_mipmapping_gpu_input_buffer",
                });
                MipmappingGpuInput * buffer_ptr = device.buffer_host_address_as<MipmappingGpuInput>(staging_buffer).value();
                *buffer_ptr = this->gpu_input;
                this->gpu_input.p_mouse_x = this->gpu_input.mouse_x;
                this->gpu_input.p_mouse_y = this->gpu_input.mouse_y;
                recorder.destroy_buffer_deferred(staging_buffer);

                recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = input_buffer,
                    .size = sizeof(MipmappingGpuInput),
                });
            }
            void paint(daxa::CommandRecorder & recorder, daxa::ImageId render_target_id, daxa::BufferId input_buffer)
            {
                auto render_target_size = device.image_info(render_target_id).value().size;
                recorder.set_pipeline(*compute_pipeline);
                auto const push = MipmappingComputePushConstant{
                    .image = render_target_id.default_view(),
                    .gpu_input = device.device_address(input_buffer).value(),
                    .frame_dim = {render_target_size.x, render_target_size.y},
                };
                recorder.push_constant(push);
                recorder.dispatch({(render_target_size.x + 7) / 8, (render_target_size.y + 7) / 8});
            }
            void draw_ui(daxa::CommandRecorder & recorder, daxa::ImageId render_target_id)
            {
                auto render_size = device.image_info(render_target_id).value().size;
                imgui_renderer.record_commands(ImGui::GetDrawData(), recorder, render_target_id, render_size.x, render_size.y);
            }
            void blit_image_to_swapchain(daxa::CommandRecorder & recorder, daxa::ImageId src_image_id, daxa::ImageId dst_image_id)
            {
                auto const & src_info = device.image_info(src_image_id).value();
                auto src_size = src_info.size;
                auto dst_size = device.image_info(dst_image_id).value().size;
                recorder.blit_image_to_image({
                    .src_image = src_image_id,
                    .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    .dst_image = dst_image_id,
                    .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .src_offsets = {{{0, 0, 0}, {static_cast<i32>(src_size.x), static_cast<i32>(src_size.y), 1}}},
                    .dst_offsets = {
                        {
                            {0, 0, 0},
                            {static_cast<i32>(static_cast<f32>(dst_size.x) * (2.0f / 3.0f)), static_cast<i32>(dst_size.y), 1},
                        },
                    },
                });
                for (i32 i = 0; i < static_cast<i32>(src_info.mip_level_count - 1); ++i)
                {
                    f32 scl_1 = static_cast<f32>(1 << (i + 0));
                    f32 scl_2 = static_cast<f32>(1 << (i + 1));
                    f32 s0 = scl_1 * 0.5f;
                    f32 s1 = (scl_1 - 1.0f) / scl_1;
                    f32 s2 = (scl_2 - 1.0f) / scl_2;
                    recorder.blit_image_to_image({
                        .src_image = src_image_id,
                        .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        .dst_image = dst_image_id,
                        .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                        .src_slice = {
                            .mip_level = static_cast<u32>(i + 1),
                        },
                        .src_offsets = {{{0, 0, 0}, {static_cast<i32>(static_cast<f32>(src_size.x) / scl_2), static_cast<i32>(static_cast<f32>(src_size.y) / scl_2), 1}}},
                        .dst_offsets = {
                            {
                                {
                                    static_cast<i32>(static_cast<f32>(dst_size.x) * (2.0f / 3.0f)),
                                    static_cast<i32>(static_cast<f32>(dst_size.y) * s1),
                                    0,
                                },
                                {
                                    static_cast<i32>(static_cast<f32>(dst_size.x) * (2.0f / 3.0f + 1.0f / (s0 * 6.0f))),
                                    static_cast<i32>(static_cast<f32>(dst_size.y) * s2),
                                    1,
                                },
                            },
                        },
                    });
                }
            }

            void non_task_graph_execute()
            {
                swapchain_image = swapchain.acquire_next_image();
                if (swapchain_image.is_empty())
                {
                    return;
                }
                auto recorder = device.create_command_recorder({});

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
                recorder.pipeline_barrier({
                    .src_access = daxa::AccessConsts::NONE,
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                });
                // update_gpu_input(staging_mipmapping_gpu_input_buffer);
                recorder.pipeline_barrier({
                    .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .dst_access = daxa::AccessConsts::TRANSFER_READ,
                });
                update_gpu_input(recorder, mipmapping_gpu_input_buffer);
                recorder.pipeline_barrier_image_transition({
                    .src_access = daxa::AccessConsts::NONE,
                    .dst_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
                    .src_layout = daxa::ImageLayout::UNDEFINED,
                    .dst_layout = daxa::ImageLayout::GENERAL,
                    .image_id = render_image,
                });
                recorder.pipeline_barrier({
                    .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ,
                });
                paint(recorder, render_image, mipmapping_gpu_input_buffer);
                {
                    auto image_info = device.image_info(render_image).value();
                    std::array<i32, 3> mip_size = {static_cast<i32>(image_info.size.x), static_cast<i32>(image_info.size.y), static_cast<i32>(image_info.size.z)};

                    recorder.pipeline_barrier_image_transition({
                        .src_access = daxa::AccessConsts::NONE,
                        .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                        .src_layout = daxa::ImageLayout::UNDEFINED,
                        .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                        .image_slice = {
                            .base_mip_level = 1,
                            .level_count = 4,
                            .base_array_layer = 0,
                            .layer_count = 1,
                        },
                        .image_id = render_image,
                    });

                    for (u32 i = 0; i < image_info.mip_level_count - 1; ++i)
                    {
                        recorder.pipeline_barrier_image_transition({
                            .src_access = (i == 0 ? daxa::AccessConsts::COMPUTE_SHADER_WRITE : daxa::AccessConsts::TRANSFER_WRITE),
                            .dst_access = daxa::AccessConsts::BLIT_READ,
                            .src_layout = (i == 0 ? daxa::ImageLayout::GENERAL : daxa::ImageLayout::TRANSFER_DST_OPTIMAL),
                            .dst_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            .image_slice = {
                                .base_mip_level = i,
                                .level_count = 1,
                                .base_array_layer = 0,
                                .layer_count = 1,
                            },
                            .image_id = render_image,
                        });
                        std::array<i32, 3> next_mip_size = {std::max<i32>(1, mip_size[0] / 2), std::max<i32>(1, mip_size[1] / 2), std::max<i32>(1, mip_size[2] / 2)};
                        recorder.blit_image_to_image({
                            .src_image = render_image,
                            .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            .dst_image = render_image,
                            .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                            .src_slice = {
                                .mip_level = i,
                                .base_array_layer = 0,
                                .layer_count = 1,
                            },
                            .src_offsets = {{{0, 0, 0}, {mip_size[0], mip_size[1], mip_size[2]}}},
                            .dst_slice = {
                                .mip_level = i + 1,
                                .base_array_layer = 0,
                                .layer_count = 1,
                            },
                            .dst_offsets = {{{0, 0, 0}, {next_mip_size[0], next_mip_size[1], next_mip_size[2]}}},
                            .filter = daxa::Filter::LINEAR,
                        });
                        mip_size = next_mip_size;
                    }
                    recorder.pipeline_barrier_image_transition({
                        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                        .dst_access = daxa::AccessConsts::TRANSFER_READ,
                        .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                        .dst_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        .image_slice = {
                            .base_mip_level = 4,
                            .level_count = 1,
                            .base_array_layer = 0,
                            .layer_count = 1,
                        },
                        .image_id = render_image,
                    });
                }
                recorder.pipeline_barrier_image_transition({
                    .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .image_id = swapchain_image,
                });
                blit_image_to_swapchain(recorder, render_image, swapchain_image);
                recorder.pipeline_barrier_image_transition({
                    .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .dst_access = daxa::AccessConsts::COLOR_ATTACHMENT_OUTPUT_READ_WRITE,
                    .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                    .dst_layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .image_id = swapchain_image,
                });
                draw_ui(recorder, swapchain_image);
                recorder.pipeline_barrier_image_transition({
                    .src_access = daxa::AccessConsts::COLOR_ATTACHMENT_OUTPUT_READ_WRITE,
                    .dst_access = {.stages = daxa::PipelineStageFlagBits::BOTTOM_OF_PIPE, .type = daxa::AccessTypeFlagBits::NONE},
                    .src_layout = daxa::ImageLayout::ATTACHMENT_OPTIMAL,
                    .dst_layout = daxa::ImageLayout::PRESENT_SRC,
                    .image_id = swapchain_image,
                });
                device.submit_commands({
                    .command_lists = std::array{recorder.complete_current_commands()},
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

            auto record_tasks() -> daxa::TaskGraph
            {

                daxa::TaskGraph new_task_graph = daxa::TaskGraph({
                    .device = device,
                    .swapchain = swapchain,
                    .permutation_condition_count = TASK_CONDITION_COUNT,
                    .record_debug_information = true,
                    .name = "main task graph",
                });
                new_task_graph.use_persistent_image(task_swapchain_image);
                new_task_graph.use_persistent_buffer(task_mipmapping_gpu_input_buffer);
                new_task_graph.use_persistent_image(task_render_image);

                using namespace daxa;

                new_task_graph.add_task({
                    .attachments = {
                        daxa::inl_attachment(TaskBufferAccess::HOST_TRANSFER_WRITE,  task_mipmapping_gpu_input_buffer),
                    },
                    .task = [this](daxa::TaskInterface tri)
                    { update_gpu_input(tri.recorder, tri.get(task_mipmapping_gpu_input_buffer).ids[0]); },
                    .name = "Input Transfer",
                });
                new_task_graph.conditional({
                    .condition_index = TASK_CONDITION_MOUSE_DRAWING,
                    .when_true = [&]()
                    {
                        new_task_graph.add_task({
                            .attachments = {
                                daxa::inl_attachment(TaskBufferAccess::COMPUTE_SHADER_READ, task_mipmapping_gpu_input_buffer),
                                daxa::inl_attachment(TaskImageAccess::COMPUTE_SHADER_STORAGE_READ_WRITE, task_render_image),
                            },
                            .task = [=, this](daxa::TaskInterface tri)
                            {
                                paint(tri.recorder, tri.get(task_render_image).ids[0], tri.get(task_mipmapping_gpu_input_buffer).ids[0]);
                            },
                            .name = "mouse paint",
                        });
                        {
                            auto image_info = device.image_info(render_image).value();
                            std::array<i32, 3> mip_size = {std::max<i32>(1, static_cast<i32>(image_info.size.x)), std::max<i32>(1, static_cast<i32>(image_info.size.y)), std::max<i32>(1, static_cast<i32>(image_info.size.z))};
                            for (u32 i = 0; i < image_info.mip_level_count - 1; ++i)
                            {
                                std::array<i32, 3> next_mip_size = {std::max<i32>(1, mip_size[0] / 2), std::max<i32>(1, mip_size[1] / 2), std::max<i32>(1, mip_size[2] / 2)};
                                new_task_graph.add_task(MipMapTask{
                                    .views = std::array{
                                        daxa::attachment_view(MipMapH::AT.lower_mip, task_render_image.view().view({.base_mip_level = i})),
                                        daxa::attachment_view(MipMapH::AT.higher_mip, task_render_image.view().view({.base_mip_level = i + 1})),
                                    },
                                    .info = {
                                        .mip = i,
                                        .mip_size = mip_size,
                                        .next_mip_size = next_mip_size,
                                    },
                                });
                                mip_size = next_mip_size;
                            }
                        }
                    },
                });

                new_task_graph.add_task({
                    .attachments = {
                        daxa::inl_attachment(TaskImageAccess::TRANSFER_WRITE, task_swapchain_image),
                    },
                    .task = [=, this](daxa::TaskInterface const & tri)
                    {
                        tri.recorder.clear_image({
                            .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                            .clear_value = {std::array<f32, 4>{1, 0, 1, 1}},
                            .dst_image = tri.get(task_swapchain_image).ids[0],
                        });
                    },
                    .name = "clear swapchain",
                });
                auto render_img_view = task_render_image.view().view({.level_count = 5});
                new_task_graph.add_task({
                    .attachments = {
                        daxa::inl_attachment(TaskImageAccess::TRANSFER_READ, render_img_view),
                        daxa::inl_attachment(TaskImageAccess::TRANSFER_WRITE, task_swapchain_image),
                    },
                    .task = [=, this](daxa::TaskInterface const & ti)
                    {
                        /// NOTE: Its possible to use TaskImageView as an index into the task attachment intos:
                        daxa::ImageId render_img = ti.get(render_img_view).ids[0];
                        daxa::ImageId swapchain_img = ti.get(task_swapchain_image).ids[0];

                        this->blit_image_to_swapchain(ti.recorder, render_img, swapchain_img);
                    },
                    .name = "blit to swapchain",
                });
                new_task_graph.add_task({
                    .attachments = {daxa::inl_attachment( daxa::TaskImageAccess::COLOR_ATTACHMENT, task_swapchain_image ) },
                    .task = [=, this](daxa::TaskInterface const & ti)
                    {
                        this->draw_ui(ti.recorder, ti.get(task_swapchain_image).ids[0]);
                    },
                    .name = "Imgui",
                });
                new_task_graph.submit({});
                new_task_graph.present({});
                new_task_graph.complete({});
                return new_task_graph;
            }
        };

        App app = {};
        while (true)
        {
            if (app.update())
                break;
        }
    }
} // namespace tests
