#pragma once

#include "common.hpp"
#include "shaders/transient.inl"

namespace tests
{
    auto div_round_up(u32 value, u32 div) -> u32
    {
        return (value + div - 1) / div;
    }

    void set_initial_buffer_data(
        daxa::TaskInterface & ti,
        daxa::TaskBufferView buffer,
        u32 size,
        u32 value)
    {
        auto staging = ti.allocator->allocate(size * sizeof(u32)).value();
        for (u32 x = 0; x < size; ++x)
        {
            reinterpret_cast<u32 *>(staging.host_address)[x] = value;
        }
        ti.recorder.copy_buffer_to_buffer({
            .src_buffer = ti.allocator->buffer(),
            .dst_buffer = ti.get(buffer).ids[0],
            .src_offset = staging.buffer_offset,
            .size = size * sizeof(u32),
        });
    }

    void validate_buffer_data(
        daxa::TaskInterface & ti,
        daxa::TaskBufferView buffer,
        u32 size,
        u32 value,
        daxa::ComputePipeline & pipeline)
    {
        ti.recorder.set_pipeline(pipeline);
        ti.recorder.push_constant(TestBufferPush{
            .test_buffer = ti.device_address(buffer).value(),
            .size = size,
            .value = value,
        });
        ti.recorder.dispatch({div_round_up(size, 128)});
    }

    void set_initial_image_data(
        daxa::TaskInterface & ti,
        daxa::TaskImageView image,
        auto size,
        f32 value)
    {
        u32 const image_size = static_cast<u32>(sizeof(f32) * size.x * size.y * size.z);
        auto staging = ti.allocator->allocate(image_size).value();
        for (u32 x = 0; x < size.x; ++x)
        {
            for (u32 y = 0; y < size.y; ++y)
            {
                for (u32 z = 0; z < size.z; ++z)
                {
                    usize index = (x + y * size.x + z * size.x * size.y);
                    reinterpret_cast<f32 *>(staging.host_address)[index] = value;
                }
            }
        }
        ti.recorder.copy_buffer_to_image({
            .buffer = ti.allocator->buffer(),
            .buffer_offset = staging.buffer_offset,
            .image = ti.get(image).ids[0],
            .image_extent = {size.x, size.y, size.z},
        });
    }

    void validate_image_data(
        daxa::TaskInterface & ti,
        daxa::TaskImageView image,
        auto size,
        f32 value,
        daxa::ComputePipeline & pipeline)
    {
        ti.recorder.set_pipeline(pipeline);
        ti.recorder.push_constant(TestImagePush{
            .test_image = ti.get(image).view_ids[0],
            .size = size,
            .value = value,
        });
        ti.recorder.dispatch({
            div_round_up(size.x, 4),
            div_round_up(size.y, 4),
            div_round_up(size.z, 4),
        });
    }

    void transient_write_aliasing()
    {
        // TEST
        // Tests whether two transient images are aliased in memory correctly
        // Resources:
        //      Image A, B, C - all are transient
        // Tasks:
        //      Task 1 - writes into image A and B
        //      Task 2 - validate image A and B
        //      Task 3 - writes into image A and C
        //          - Note(msakmary) both tasks writing image A creates a dependency between task 1 so that
        //            they are not inserted into the same batch and we can actually test if aliasing occurs
        //      Task 4 - validate image A and C
        // Expected:
        //      Images B and C are aliased and the content is correct after execution of each of the tasks

        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({},{}));

        daxa::PipelineManager pipeline_manager = daxa::PipelineManager{{
            .device = device,
            .shader_compile_options = {
                .root_paths = {
                    DAXA_SHADER_INCLUDE_DIR,
                    "tests/2_daxa_api/6_task_graph/shaders",
                },
                .language = daxa::ShaderLanguage::GLSL,
            },
            .name = "pipeline manager",
        }};

        daxa::ComputePipelineCompileInfo const test_image_pipeline_info = {
            .shader_info = {
                .source = daxa::ShaderFile{"transient.glsl"},
                .compile_options = {
                    .defines = std::vector{daxa::ShaderDefine{"TEST_IMAGE", "1"}}},
            },
            .push_constant_size = sizeof(TestImagePush),
            .name = "test image pipeline",
        };

        auto test_image_pipeline = pipeline_manager.add_compute_pipeline(test_image_pipeline_info).value();

        {
            f32 const IMAGE_A_VALUE = 1.0f;
            f32 const IMAGE_B_VALUE = 2.0f;
            f32 const IMAGE_C_VALUE = 3.0f;
            daxa_u32vec3 const IMAGE_A_SIZE = {2, 2, 1};
            daxa_u32vec3 const IMAGE_B_SIZE = {2, 2, 1};
            daxa_u32vec3 const IMAGE_C_SIZE = {2, 2, 1};

            auto task_graph = daxa::TaskGraph({
                .device = device,
                .alias_transients = true,
                .record_debug_information = true,
                .staging_memory_pool_size = 4'000'000,
                .name = "task_graph",
            });

            // ========================================== Create resources ====================================================
            auto image_A = task_graph.create_transient_image({.dimensions = 3,
                                                              .format = daxa::Format::R32_SFLOAT,
                                                              .size = {IMAGE_A_SIZE.x, IMAGE_A_SIZE.y, IMAGE_A_SIZE.z},
                                                              .name = "Image A"});

            auto image_B = task_graph.create_transient_image({.dimensions = 3,
                                                              .format = daxa::Format::R32_SFLOAT,
                                                              .size = {IMAGE_B_SIZE.x, IMAGE_B_SIZE.y, IMAGE_B_SIZE.z},
                                                              .name = "Image B"});

            auto image_C = task_graph.create_transient_image({.dimensions = 3,
                                                              .format = daxa::Format::R32_SFLOAT,
                                                              .size = {IMAGE_C_SIZE.x, IMAGE_C_SIZE.y, IMAGE_C_SIZE.z},
                                                              .name = "Image C"});

            // ========================================== Record tasks =======================================================

            task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, image_A),
                    daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, image_B),
                },
                .task = [=](daxa::TaskInterface ti)
                {
                    set_initial_image_data(ti, image_A, IMAGE_A_SIZE, IMAGE_A_VALUE);
                    set_initial_image_data(ti, image_B, IMAGE_B_SIZE, IMAGE_B_VALUE);
                },
                .name = "Task 1 - write image A and B",
            });

            task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, image_A),
                    daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, image_B),
                },
                .task = [=](daxa::TaskInterface ti)
                {
                    validate_image_data(ti, image_A, IMAGE_A_SIZE, IMAGE_A_VALUE, *test_image_pipeline);
                    validate_image_data(ti, image_B, IMAGE_B_SIZE, IMAGE_B_VALUE, *test_image_pipeline);
                },
                .name = "Task 2 - test contents of image A and B",
            });

            task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, image_A),
                    daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, image_C),
                },
                .task = [=](daxa::TaskInterface ti)
                {
                    set_initial_image_data(ti, image_A, IMAGE_A_SIZE, IMAGE_C_VALUE);
                    set_initial_image_data(ti, image_C, IMAGE_C_SIZE, IMAGE_C_VALUE);
                },
                .name = "Task 3 - write image A and C",
            });

            task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, image_A),
                    daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, image_C),
                },
                .task = [=](daxa::TaskInterface ti)
                {
                    validate_image_data(ti, image_A, IMAGE_A_SIZE, IMAGE_C_VALUE, *test_image_pipeline);
                    validate_image_data(ti, image_C, IMAGE_C_SIZE, IMAGE_C_VALUE, *test_image_pipeline);
                },
                .name = "Task 4 - test contents of image A and C",
            });
            task_graph.submit({});
            task_graph.complete({});
            task_graph.execute({});

            std::cout << task_graph.get_debug_string() << std::endl;
        }
    }

    void permutation_aliasing()
    {
        // TEST
        // Tests whether transient images alias across permutations
        // Resources:
        //      Image Base - transient image shared by both permutations
        //      Image A, B - transient images for true and false permutations respectively
        // Tasks:
        //      Task 0 - Set Base image data
        //      Conditional True - Task 1 - writes into image A
        //      Conditional True - Task 2 - checks contents of image A
        //      Conditional False - Task 1 - writes into image B
        //      Conditional False - Task 2 - checks contents of image B
        //      Task 3 - Check Base image data
        // Expected:
        //      Images A and B are aliased - they both start at the same offset

        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({},{}));

        daxa::PipelineManager pipeline_manager = daxa::PipelineManager{{
            .device = device,
            .shader_compile_options = {
                .root_paths = {
                    DAXA_SHADER_INCLUDE_DIR,
                    "tests/2_daxa_api/6_task_graph/shaders",
                },
                .language = daxa::ShaderLanguage::GLSL,
            },
            .name = "pipeline manager",
        }};

        daxa::ComputePipelineCompileInfo const test_image_pipeline_info = {
            .shader_info = {
                .source = daxa::ShaderFile{"transient.glsl"},
                .compile_options = {
                    .defines = std::vector{daxa::ShaderDefine{"TEST_IMAGE", "1"}}},
            },
            .push_constant_size = sizeof(TestImagePush),
            .name = "test image pipeline",
        };

        auto test_image_pipeline = pipeline_manager.add_compute_pipeline(test_image_pipeline_info).value();

        {
            f32 const IMAGE_A_VALUE = 1.0f;
            f32 const IMAGE_B_VALUE = 2.0f;
            f32 const IMAGE_BASE_VALUE = 2.0f;
            daxa_u32vec3 const IMAGE_A_SIZE = {128, 128, 1};
            daxa_u32vec3 const IMAGE_B_SIZE = {256, 256, 1};
            daxa_u32vec3 const IMAGE_BASE_SIZE = {64, 64, 1};

            auto task_graph = daxa::TaskGraph({
                .device = device,
                .permutation_condition_count = 1,
                .record_debug_information = true,
                .staging_memory_pool_size = 4'000'000,
                .name = "task_graph",
            });

            // ========================================== Record tasks =======================================================
            auto image_base = task_graph.create_transient_image({.dimensions = 3,
                                                                 .format = daxa::Format::R32_SFLOAT,
                                                                 .size = {IMAGE_BASE_SIZE.x, IMAGE_BASE_SIZE.y, IMAGE_BASE_SIZE.z},
                                                                 .name = "Image Base"});

            task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, image_base),
                },
                .task = [=](daxa::TaskInterface ti)
                {
                    set_initial_image_data(ti, image_base, IMAGE_BASE_SIZE, IMAGE_BASE_VALUE);
                },
                .name = "Task 0 - write base image value",
            });

            task_graph.conditional({.condition_index = 0,
                                    .when_true = [&]()
                                    {
                    auto image_A = task_graph.create_transient_image({
                        .dimensions = 3,
                        .format = daxa::Format::R32_SFLOAT,
                        .size = {IMAGE_A_SIZE.x, IMAGE_A_SIZE.y, IMAGE_A_SIZE.z},
                        .name = "Image A"
                    });
                    task_graph.add_task({
                        .attachments = {
                            daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, image_A),
                        },
                        .task = [=](daxa::TaskInterface ti)
                        {
                            set_initial_image_data(ti, image_A, IMAGE_A_SIZE, IMAGE_A_VALUE);
                        },
                        .name = "Perm True - Task 1 - write image A",
                    });

                    task_graph.add_task({
                        .attachments = {
                            daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, image_A),
                        },
                        .task = [=](daxa::TaskInterface ti)
                        {
                            validate_image_data(ti, image_A, IMAGE_A_SIZE, IMAGE_A_VALUE, *test_image_pipeline);
                        },
                        .name = "Perm True - Task 2 - check image A",
                    }); 
                },
                .when_false = [&]()
                {
                    auto image_B = task_graph.create_transient_image({
                        .dimensions = 3,
                        .format = daxa::Format::R32_SFLOAT,
                        .size = {IMAGE_B_SIZE.x, IMAGE_B_SIZE.y, IMAGE_B_SIZE.z},
                        .name = "Image B"
                    });

                    task_graph.add_task({
                        .attachments = {
                            daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, image_B),
                        },
                        .task = [=](daxa::TaskInterface ti)
                        {
                            set_initial_image_data(ti, image_B, IMAGE_B_SIZE, IMAGE_B_VALUE);
                        },
                        .name = "Perm False - Task 1 - write image B",
                    });

                    task_graph.add_task({
                        .attachments = {
                            daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, image_B),
                        },
                        .task = [=](daxa::TaskInterface ti)
                        {
                            validate_image_data(ti, image_B, IMAGE_B_SIZE, IMAGE_B_VALUE, *test_image_pipeline);
                        },
                        .name = "Perm False - Task 2 - check image B",
                    }); 
                }
            });

            task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, image_base),
                },
                .task = [=](daxa::TaskInterface ti)
                {
                    validate_image_data(ti, image_base, IMAGE_BASE_SIZE, IMAGE_BASE_VALUE, *test_image_pipeline);
                },
                .name = "Task 3 - validate base image data",
            });

            task_graph.submit({});
            task_graph.complete({});
            bool perm_condition = true;
            task_graph.execute({.permutation_condition_values = {&perm_condition, 1}});
            std::cout << task_graph.get_debug_string() << std::endl;
            perm_condition = false;
            task_graph.execute({.permutation_condition_values = {&perm_condition, 1}});
            std::cout << task_graph.get_debug_string() << std::endl;
        }
    }

    void transient_resources()
    {
        // TEST:
        // 1. Create transient resources.
        // 2. Initialize them.
        // 3. Test if their content stays intact over multiple aliased operations.

        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({},{}));

        daxa::PipelineManager pipeline_manager = daxa::PipelineManager{{
            .device = device,
            .shader_compile_options = {
                .root_paths = {
                    DAXA_SHADER_INCLUDE_DIR,
                    "tests/2_daxa_api/6_task_graph/shaders",
                },
                .language = daxa::ShaderLanguage::GLSL,
            },
            .name = "pipeline manager",
        }};

        daxa::ComputePipelineCompileInfo const test_image_pipeline_info = {
            .shader_info = {
                .source = daxa::ShaderFile{"transient.glsl"},
                .compile_options = {
                    .defines = std::vector{daxa::ShaderDefine{"TEST_IMAGE", "1"}}},
            },
            .push_constant_size = sizeof(TestImagePush),
            .name = "test image",
        };
        auto test_image_pipeline = pipeline_manager.add_compute_pipeline(test_image_pipeline_info).value();

        daxa::ComputePipelineCompileInfo const test_buffer_pipeline_info = {
            .shader_info = {
                .source = daxa::ShaderFile{"transient.glsl"},
            },
            .push_constant_size = sizeof(TestImagePush),
            .name = "test buffer",
        };
        auto test_buffer_pipeline = pipeline_manager.add_compute_pipeline(test_buffer_pipeline_info).value();

        {
            auto task_graph = daxa::TaskGraph({
                .device = device,
                .reorder_tasks = false, // Disable reordering for testing purposes.
                .record_debug_information = true,
                .staging_memory_pool_size = 4'000'000,
                .name = "task_graph",
            });

            // Declare transient resources.
            auto long_life_buffer =
                task_graph.create_transient_buffer({
                    .size = LONG_LIFE_BUFFER_SIZE * sizeof(daxa::u32),
                    .name = "long life buffer",
                });

            auto medium_life_image =
                task_graph.create_transient_image({
                    .dimensions = 3,
                    .format = daxa::Format::R32_SFLOAT,
                    .size = daxa::Extent3D{MEDIUM_LIFE_IMAGE_SIZE.x, MEDIUM_LIFE_IMAGE_SIZE.y, MEDIUM_LIFE_IMAGE_SIZE.z},
                    .name = "medium life image",
                });

            auto long_life_image =
                task_graph.create_transient_image({
                    .dimensions = 3,
                    .format = daxa::Format::R32_SFLOAT,
                    .size = {LONG_LIFE_IMAGE_SIZE.x, LONG_LIFE_IMAGE_SIZE.y, LONG_LIFE_IMAGE_SIZE.z},
                    .name = "long life image",
                });

            auto short_life_image =
                task_graph.create_transient_image({
                    .dimensions = 3,
                    .format = daxa::Format::R32_SFLOAT,
                    .size = {SHORT_LIFE_IMAGE_SIZE.x, SHORT_LIFE_IMAGE_SIZE.y, SHORT_LIFE_IMAGE_SIZE.z},
                    .name = "short life large image",
                });

            auto short_life_buffer =
                task_graph.create_transient_buffer({
                    .size = SHORT_LIFE_BUFFER_SIZE * sizeof(daxa::u32),
                    .name = "tiny size short life buffer",
                });

            // Record tasks.
            task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, long_life_buffer),
                },
                .task = [=](daxa::TaskInterface ti)
                {
                    set_initial_buffer_data(ti, long_life_buffer, LONG_LIFE_BUFFER_SIZE, LONG_LIFE_BUFFER_VALUE);
                },
                .name = "populate long life buffer",
            });

            task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, medium_life_image),
                },
                .task = [=](daxa::TaskInterface ti)
                {
                    set_initial_image_data(ti, medium_life_image, MEDIUM_LIFE_IMAGE_SIZE, MEDIUM_LIFE_IMAGE_VALUE);
                },
                .name = "populate medium life image",
            });

            task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ, long_life_buffer),
                    daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, medium_life_image),
                    daxa::inl_attachment(daxa::TaskImageAccess::TRANSFER_WRITE, long_life_image),
                },
                .task = [=](daxa::TaskInterface ti)
                {
                    set_initial_image_data(ti, long_life_image, LONG_LIFE_IMAGE_SIZE, LONG_LIFE_IMAGE_VALUE);
                    validate_image_data(ti, medium_life_image, MEDIUM_LIFE_IMAGE_SIZE, MEDIUM_LIFE_IMAGE_VALUE, *test_image_pipeline);
                    validate_buffer_data(ti, long_life_buffer, LONG_LIFE_BUFFER_SIZE, LONG_LIFE_BUFFER_VALUE, *test_buffer_pipeline);
                },
                .name = "validate long life buffer, validate medium life image, populate long life image",
            });

            task_graph.add_task({
                .attachments = { daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_READ_WRITE, short_life_image) },
                .task = [=](daxa::TaskInterface) {},
                .name = "dummy use short life image",
            });

            task_graph.add_task({
                .attachments = {
                    daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE, short_life_buffer),
                    daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, long_life_image),
                },
                .task = [=](daxa::TaskInterface ti)
                {
                    validate_image_data(ti, long_life_image, LONG_LIFE_IMAGE_SIZE, LONG_LIFE_IMAGE_VALUE, *test_image_pipeline);
                },
                .name = "validate long life image, dummy access short life buffer",
            });

            task_graph.submit({});
            task_graph.complete({});
            task_graph.execute({});

            std::cout << task_graph.get_debug_string() << std::endl;
        }
        device.wait_idle();
        device.collect_garbage();
    }
} // namespace tests
