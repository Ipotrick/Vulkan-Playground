#pragma once

#include "common.hpp"
DAXA_DECL_TASK_HEAD_BEGIN(TestTaskHead)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, buffer0)
DAXA_TH_IMAGE(COMPUTE_SHADER_SAMPLED, REGULAR_2D, image0)
DAXA_TH_IMAGE(COMPUTE_SHADER_SAMPLED, REGULAR_2D, image1)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, test_buffer_no_shader)
DAXA_DECL_TASK_HEAD_END

struct TestTask : TestTaskHead::Task
{
    AttachmentViews views = {};
    void callback(daxa::TaskInterface ti)
    {
        // There are two ways to get the info for any attachment:
        {
            // daxa::TaskBufferAttachmentIndex index:
            [[maybe_unused]] daxa::TaskBufferAttachmentInfo const & buffer0_attachment0 = ti.get(AT.buffer0);
            // daxa::TaskBufferView assigned to the buffer attachment:
            [[maybe_unused]] daxa::TaskBufferAttachmentInfo const & buffer0_attachment1 = ti.get(buffer0_attachment0.view);
        }
        // The Buffer Attachment info contents:
        {
            [[maybe_unused]] daxa::BufferId id = ti.get(AT.buffer0).ids[0];
            [[maybe_unused]] char const * name = ti.get(AT.buffer0).name;
            [[maybe_unused]] daxa::TaskBufferAccess access = ti.get(AT.buffer0).access;
            [[maybe_unused]] u8 shader_array_size = ti.get(AT.buffer0).shader_array_size;
            [[maybe_unused]] bool shader_as_address = ti.get(AT.buffer0).shader_as_address;
            [[maybe_unused]] daxa::TaskBufferView view = ti.get(AT.buffer0).view;
            [[maybe_unused]] std::span<daxa::BufferId const> ids = ti.get(AT.buffer0).ids;
        }
        // The Image Attachment info contents:
        {
            [[maybe_unused]] char const * name = ti.get(AT.image0).name;
            [[maybe_unused]] daxa::TaskImageAccess access = ti.get(AT.image0).access;
            [[maybe_unused]] daxa::ImageViewType view_type = ti.get(AT.image0).view_type;
            [[maybe_unused]] u8 shader_array_size = ti.get(AT.image0).shader_array_size;
            [[maybe_unused]] daxa::TaskHeadImageArrayType shader_array_type = ti.get(AT.image0).shader_array_type;
            [[maybe_unused]] daxa::ImageLayout layout = ti.get(AT.image0).layout;
            [[maybe_unused]] daxa::TaskImageView view = ti.get(AT.image0).view;
            [[maybe_unused]] std::span<daxa::ImageId const> ids = ti.get(AT.image0).ids;
            [[maybe_unused]] std::span<daxa::ImageViewId const> view_ids = ti.get(AT.image0).view_ids;
        }
        // The attachment infos are also provided, directly via a span:
        for ([[maybe_unused]] daxa::TaskAttachmentInfo const & attach : ti.attachment_infos)
        {
        }
    }
};

void test_task_copy_and_move()
{
    TestTask t;
    [[maybe_unused]] TestTask t2 = t;
    std::optional<TestTask> t3;
    t3.emplace(t);
    t3.reset();
    [[maybe_unused]] TestTask t4 = std::move(t);
}

#include "mipmapping.hpp"
#include "shaders/shader_integration.inl"
#include "persistent_resources.hpp"
#include "transient_overlap.hpp"

namespace tests
{
    void simplest()
    {
        AppContext const app = {};
        auto d = app.device;
        struct S
        {
            daxa::Device d;
        } s = {d};
        auto task_graph = daxa::TaskGraph({
            .device = app.device,
            .name = APPNAME_PREFIX("task_graph (simplest)"),
        });
    }

    void execution()
    {
        AppContext const app = {};
        auto task_graph = daxa::TaskGraph({
            .device = app.device,
            .name = APPNAME_PREFIX("task_graph (execution)"),
        });

        // This is pointless, but done to show how the task graph executes
        task_graph.add_task({
            .attachments = {},
            .task = [&](daxa::TaskInterface const &)
            {
                std::cout << "Hello, ";
            },
            .name = APPNAME_PREFIX("task 1 (execution)"),
        });
        task_graph.add_task({
            .attachments = {},
            .task = [&](daxa::TaskInterface const &)
            {
                std::cout << "World!" << std::endl;
            },
            .name = APPNAME_PREFIX("task 2 (execution)"),
        });

        task_graph.complete({});

        task_graph.execute({});
    }

    void write_read_image()
    {
        // TEST:
        //    1) CREATE image
        //    2) WRITE image
        //    3) READ image
        AppContext app = {};
        // Need to scope the task graphs lifetime.
        // Task graph MUST die before we call wait_idle and collect_garbage.
        auto task_graph = daxa::TaskGraph({
            .device = app.device,
            .record_debug_information = true,
            .name = APPNAME_PREFIX("create-write-read image"),
        });
        // CREATE IMAGE
        auto task_image = task_graph.create_transient_image(daxa::TaskTransientImageInfo{.size = {1, 1, 1}, .name = "task graph tested image"});
        // WRITE IMAGE 1
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY, task_image)},
            .task = [](daxa::TaskInterface) {},
            .name = APPNAME_PREFIX("write image 1"),
        });
        // READ_IMAGE 1
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, task_image)},
            .task = [](daxa::TaskInterface) {},
            .name = APPNAME_PREFIX("read image 1"),
        });

        task_graph.complete({});
        task_graph.execute({});
        std::cout << task_graph.get_debug_string() << std::endl;
    }

    void write_read_image_layer()
    {
        // TEST:
        //    1) CREATE image
        //    2) WRITE into array layer 1 of the image
        //    3) READ from array layer 2 of the image
        AppContext app = {};
        auto task_graph = daxa::TaskGraph({
            .device = app.device,
            .record_debug_information = true,
            .name = APPNAME_PREFIX("create-write-read array layer"),
        });
        // CREATE IMAGE
        auto task_image = task_graph.create_transient_image({
            .size = {1, 1, 1},
            .array_layer_count = 2,
            .name = "task graph tested image",
        });
        auto timg_view_l0 = task_image.view({.base_array_layer = 0, .layer_count = 1});
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY, timg_view_l0)},
            .task = [](daxa::TaskInterface) {},
            .name = APPNAME_PREFIX("write image array layer 1"),
        });
        // READ_IMAGE 1
        auto timg_view_l1 = task_image.view({.base_array_layer = 1, .layer_count = 1});
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, timg_view_l1)},
            .task = [](daxa::TaskInterface) {},
            .name = APPNAME_PREFIX("read image array layer 1"),
        });
        task_graph.complete({});
        task_graph.execute({});
        std::cout << task_graph.get_debug_string() << std::endl;
    }

    void create_transfer_read_buffer()
    {
        // TEST:
        //    1) CREATE buffer
        //    2) TRANSFER into the buffer
        //    3) READ from the buffer
        AppContext app = {};
        auto task_graph = daxa::TaskGraph({
            .device = app.device,
            .record_debug_information = true,
            .name = APPNAME_PREFIX("create-transfer-read buffer"),
        });

        auto task_buffer = task_graph.create_transient_buffer({
            .size = sizeof(u32),
            .name = "task graph tested buffer",
        });

        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::HOST_TRANSFER_WRITE, task_buffer)},
            .task = [](daxa::TaskInterface const &) {},
            .name = APPNAME_PREFIX("host transfer buffer"),
        });

        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ, task_buffer)},
            .task = [](daxa::TaskInterface const &) {},
            .name = APPNAME_PREFIX("read buffer"),
        });

        task_graph.complete({});
        task_graph.execute({});
        std::cout << task_graph.get_debug_string() << std::endl;
    }

    void initial_layout_access()
    {
        // TEST:
        //    1) CREATE image - set the task image initial access to write from compute shader
        //    2) READ from a the subimage
        //    3) WRITE into the subimage
        AppContext app = {};
        auto image = app.device.create_image({
            .size = {1, 1, 1},
            .array_layer_count = 2,
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = APPNAME_PREFIX("tested image"),
        });

        std::array init_access = {
            daxa::ImageSliceState{
                .latest_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
                .latest_layout = daxa::ImageLayout::GENERAL,
            },
        };

        auto task_image = daxa::TaskImage(daxa::TaskImageInfo{
            .initial_images = {
                .images = {&image, 1},
                .latest_slice_states = {init_access.data(), 1}},
            .swapchain_image = false,
            .name = "task graph tested image",
        });

        // TG MUST die before image, as it holds image views to the image that must die before the image.
        {
            auto task_graph = daxa::TaskGraph({
                .device = app.device,
                .record_debug_information = true,
                .name = APPNAME_PREFIX("initial layout image"),
            });
            // CREATE IMAGE
            task_graph.use_persistent_image(task_image);
            task_graph.add_task({
                .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, task_image.view().view({.base_array_layer = 1, .layer_count = 1}))},
                .task = [](daxa::TaskInterface const &) {},
                .name = APPNAME_PREFIX("read array layer 2"),
            });
            task_graph.add_task({
                .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY, task_image.view().view({.base_array_layer = 0, .layer_count = 1}))},
                .task = [](daxa::TaskInterface const &) {},
                .name = APPNAME_PREFIX("write array layer 1"),
            });
            task_graph.complete({});
            task_graph.execute({});
            std::cout << task_graph.get_debug_string() << std::endl;
        }
        app.device.destroy_image(image);
    }

    void tracked_slice_barrier_collapsing()
    {
        // TEST:
        //    1) CREATE image - set the task image initial access to write from compute
        //                      shader for one subresouce and read for the other
        //    2) WRITE into the subsubimage
        //    3) READ from a the subsubimage
        //    4) WRITE into the entire image
        //    5) READ the entire image
        //    Expected: There should only be a single barrier between tests 4 and 5.
        AppContext app = {};
        auto image = app.device.create_image({
            .size = {1, 1, 1},
            .array_layer_count = 4,
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = APPNAME_PREFIX("tested image"),
        });

        // CREATE IMAGE
        std::array init_access = {
            daxa::ImageSliceState{
                .latest_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
                .latest_layout = daxa::ImageLayout::GENERAL,
                .slice = {.base_array_layer = 0, .layer_count = 2},
            },
            daxa::ImageSliceState{
                .latest_access = daxa::AccessConsts::COMPUTE_SHADER_READ,
                .latest_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
                .slice = {.base_array_layer = 2, .layer_count = 2},
            }};
        auto task_image = daxa::TaskImage({
            .name = "task graph tested image",
        });

        // TG MUST die before image, as it holds image views to the image that must die before the image.
        {
            auto task_graph = daxa::TaskGraph({
                .device = app.device,
                .record_debug_information = true,
                .name = APPNAME_PREFIX("tracked slice barrier collapsing"),
            });

            task_image.set_images({.images = {&image, 1}, .latest_slice_states = {init_access.begin(), init_access.end()}});

            task_graph.use_persistent_image(task_image);

            task_graph.add_task({
                .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, task_image.view().view({.base_array_layer = 1, .layer_count = 1}))},
                .task = [](daxa::TaskInterface const &) {},
                .name = APPNAME_PREFIX("read image layer 1"),
            });
            task_graph.add_task({
                .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY, task_image.view().view({.base_array_layer = 3, .layer_count = 1}))},
                .task = [](daxa::TaskInterface const &) {},
                .name = APPNAME_PREFIX("write image layer 3"),
            });
            task_graph.add_task({
                .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY, task_image.view().view({.base_array_layer = 0, .layer_count = 4}))},
                .task = [](daxa::TaskInterface const &) {},
                .name = APPNAME_PREFIX("write image layer 0 - 1"),
            });
            task_graph.add_task({
                .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, task_image.view().view({.base_array_layer = 0, .layer_count = 4}))},
                .task = [](daxa::TaskInterface const &) {},
                .name = APPNAME_PREFIX("read image layer 0 - 3"),
            });

            task_graph.complete({});
            task_graph.execute({});
            std::cout << task_graph.get_debug_string() << std::endl;
        }
        app.device.destroy_image(image);
    }

    void shader_integration_inl_use()
    {
        // TEST:
        //  1) Create resources
        //  2) Use Compute dispatch to write to image
        //  4) readback and validate
        AppContext app = {};
        auto dummy = app.device.create_image({
            .size = {16, 16, 1},
            .array_layer_count = 1,
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
            .name = "dummy",
        });
        auto image = app.device.create_image({
            .size = {16, 16, 1},
            .array_layer_count = 1,
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_SRC,
            .name = "underlying image",
        });
        auto task_image = daxa::TaskImage({
            // In this test, this image name will be "aliased", so the name must not be the same.
            .initial_images = {
                .images = {&image, 1},
            },
            .name = "image",
        });
        auto buffer = app.device.create_buffer({
            .size = 16,
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = "underlying buffer",
        });
        *app.device.get_host_address_as<float>(buffer).value() = 0.75f;
        auto task_buffer = daxa::TaskBuffer({
            .initial_buffers = {
                .buffers = {&buffer, 1},
                .latest_access = daxa::AccessConsts::HOST_WRITE,
            },
            .name = "settings", // This name MUST be identical to the name used in the shader.
        });

        daxa::PipelineManager pipeline_manager = daxa::PipelineManager({
            .device = app.device,
            .shader_compile_options = {
                .root_paths = {
                    DAXA_SHADER_INCLUDE_DIR,
                    "tests/2_daxa_api/6_task_graph/shaders",
                },
            },
            .name = "pipeline manager",
        });

        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"shader_integration.glsl"},
                .compile_options{
                    .enable_debug_info = true,
                },
            },
            .push_constant_size = sizeof(ShaderIntegrationTaskHead::AttachmentShaderBlob),
            .name = "compute_pipeline",
        });
        auto compute_pipeline = compile_result.value();

        auto task_graph = daxa::TaskGraph({
            .device = app.device,
            .record_debug_information = true,
            .name = "shader integration test - task graph",
        });
        task_graph.use_persistent_image(task_image);
        task_graph.use_persistent_buffer(task_buffer);

        struct WriteImage : ShaderIntegrationTaskHead::Task
        {
            AttachmentViews views = {};
            std::shared_ptr<daxa::ComputePipeline> pipeline = {};
            void callback(daxa::TaskInterface ti)
            {
                ti.recorder.set_pipeline(*pipeline);
                ti.recorder.push_constant_vptr({ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size()});
                ti.recorder.dispatch({1, 1, 1});
            }
        };
        using namespace ShaderIntegrationTaskHead;
        task_graph.add_task(WriteImage{
            .views = std::array{
                daxa::attachment_view(AT.settings, task_buffer),
                daxa::attachment_view(AT.image, task_image),
            },
            .pipeline = compute_pipeline,
        });
        task_graph.add_task(WriteImage{
            .views = std::array{
                daxa::attachment_view(AT.settings, task_buffer),
                daxa::attachment_view(AT.image, task_image),
            },
            .pipeline = compute_pipeline,
        });
        task_graph.submit({});

        task_graph.complete({});
        task_graph.execute({});
        std::cout << task_graph.get_debug_string() << std::endl;
        app.device.wait_idle();
        app.device.destroy_image(image);
        app.device.destroy_image(dummy);
        app.device.destroy_buffer(buffer);
        app.device.collect_garbage();
    }

    void correct_read_buffer_task_ordering()
    {
        // TEST:
        //  1) Create persistent image and persistent buffer
        //  2) Record two task graphs A
        //  3) Task graph A has three tasks inserted in listed order:
        //      Task 1) Writes image
        //      Task 2) Reads image and reads buffer
        //      Task 3) Reads buffer
        //  5) Execute task graph and check the ordering of tasks in batches
        //  Expected result:
        //      NOTE(msakmary): This does something different currently (task 3 is in batch 2)
        //                      - this is due to limitations of what task graph can do without having a proper render-graph
        //                      - will be fixed in the future by adding JIRA
        //      Batch 1:
        //          Task 1
        //          Task 3
        //      Batch 2:
        //          Task 2
        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device({
            .name = "device",
        });
        auto image = device.create_image({
            .size = {1, 1, 1},
            .array_layer_count = 1,
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = "actual image",
        });

        auto buffer = device.create_buffer({
            .size = 1,
            .name = "actual_buffer",
        });

        auto persistent_task_image = daxa::TaskImage(daxa::TaskImageInfo{
            .initial_images = {.images = {&image, 1}},
            .swapchain_image = false,
            .name = "image",
        });

        auto persistent_task_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{
            .initial_buffers = {.buffers = {&buffer, 1}},
            .name = "buffer",
        });

        auto task_graph = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task_graph",
        });

        task_graph.use_persistent_image(persistent_task_image);
        task_graph.use_persistent_buffer(persistent_task_buffer);
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::GRAPHICS_SHADER_STORAGE_WRITE_ONLY, persistent_task_image)},
            .task = [&](daxa::TaskInterface const &) {},
            .name = "write persistent image",
        });
        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::GRAPHICS_SHADER_READ, persistent_task_buffer),
                daxa::inl_attachment(daxa::TaskImageAccess::GRAPHICS_SHADER_SAMPLED, persistent_task_image),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "read persistent image, read persistent buffer",
        });
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::GRAPHICS_SHADER_READ, persistent_task_buffer)},
            .task = [&](daxa::TaskInterface const &) {},
            .name = "read persistent buffer",
        });
        task_graph.submit({});
        task_graph.complete({});

        task_graph.execute({});
        std::cout << task_graph.get_debug_string() << std::endl;

        device.wait_idle();
        device.destroy_image(image);
        device.destroy_buffer(buffer);
        device.collect_garbage();
    }

    void test_concurrent_read_write_buffer()
    {
        // TEST:
        //  1) Create persistent buffer
        //  2) Record single task graph A
        //  3) Task graph A has four tasks inserted in listed order:
        //      Task 1) Writes buffer
        //      Task 2) Concurrent read write buffer
        //      Task 3) Concurrent read write buffer
        //      Task 4) Reads buffer
        //  5) Execute task graph and check the generated barriers and batches
        //  Expected result:
        //  Batch 0:
        //      Task 1)
        //  Batch 1:
        //      [Barrier Write -> Read Write] + Task 2) and 3) 
        //  Batch 2:
        //      [Barrier Write -> Read] + Task 4) 
        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device({
            .name = "device",
        });

        auto buffer = device.create_buffer({
            .size = 1,
            .name = "actual_buffer",
        });

        auto persistent_task_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{
            .initial_buffers = {.buffers = {&buffer, 1}},
            .name = "buffer",
        });

        auto task_graph = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task_graph",
        });

        task_graph.use_persistent_buffer(persistent_task_buffer);
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::GRAPHICS_SHADER_WRITE, persistent_task_buffer)},
            .task = [&](daxa::TaskInterface const &) {},
            .name = "Task 1) write buffer",
        });
        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::GRAPHICS_SHADER_READ_WRITE_CONCURRENT, persistent_task_buffer),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "Task 2) concurrent write read buffer",
        });
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::GRAPHICS_SHADER_READ_WRITE_CONCURRENT, persistent_task_buffer)},
            .task = [&](daxa::TaskInterface const &) {},
            .name = "Task 3) concurrent write read buffer",
        });
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskBufferAccess::GRAPHICS_SHADER_READ, persistent_task_buffer)},
            .task = [&](daxa::TaskInterface const &) {},
            .name = "Task 4)  read buffer",
        });
        task_graph.submit({});
        task_graph.complete({});

        task_graph.execute({});
        std::cout << task_graph.get_debug_string() << std::endl;

        device.wait_idle();
        device.destroy_buffer(buffer);
        device.collect_garbage();
    }

    void test_concurrent_read_write_image()
    {
        // TEST:
        //  1) Create persistent buffer
        //  2) Record single task graph A
        //  3) Task graph A has four tasks inserted in listed order:
        //      Task 1) Read image
        //      Task 2) Concurrent read write image
        //      Task 3) Concurrent read write image
        //      Task 4) writes image
        //  5) Execute task graph and check the generated barriers and batches
        //  Expected result:
        //  Batch 0:
        //      Task 1)
        //  Batch 1:
        //      [Barrier Read -> Read Write] + Task 2) and 3) 
        //  Batch 2:
        //      [Barrier Read Write -> Write] + Task 4) 
        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device({
            .name = "device",
        });
        auto image = device.create_image({
            .size = {1, 1, 1},
            .array_layer_count = 1,
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = "actual image",
        });

        auto persistent_task_image = daxa::TaskImage(daxa::TaskImageInfo{
            .initial_images = {.images = {&image, 1}},
            .swapchain_image = false,
            .name = "image",
        });

        auto task_graph = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task_graph",
        });

        task_graph.use_persistent_image(persistent_task_image);
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::VERTEX_SHADER_STORAGE_READ_ONLY, persistent_task_image)},
            .task = [&](daxa::TaskInterface const &) {},
            .name = "Task 1) read image",
        });
        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskImageAccess::VERTEX_SHADER_STORAGE_READ_WRITE_CONCURRENT, persistent_task_image),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "Task 2) concurrent write read image",
        });
        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskImageAccess::VERTEX_SHADER_STORAGE_READ_WRITE_CONCURRENT, persistent_task_image),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "Task 3) concurrent write read image",
        });
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::VERTEX_SHADER_STORAGE_WRITE_ONLY, persistent_task_image)},
            .task = [&](daxa::TaskInterface const &) {},
            .name = "Task 4) read image",
        });
        task_graph.submit({});
        task_graph.complete({});

        task_graph.execute({});
        std::cout << task_graph.get_debug_string() << std::endl;

        device.wait_idle();
        device.destroy_image(image);
        device.collect_garbage();
    }

    void test_concurrent_read_write_buffer_cross_graphs()
    {
        // TEST:
        //  1) Create persistent buffer
        //  2) Record two task graphs A and B
        //  3) Task graph A has one task inserted in listed order:
        //      Task 1) Concurrent read write buffer
        //  4) Task graph B has one task inserted in listed order:
        //      Task 1) Concurrent read write buffer
        //  5) Execute task graphs in the order A -> B and check the jit generated barriers:
        //  Expected result:
        //      Oversync between task graph A and B aka one jit [ReadWrite -> ReadWrite] Barrier between
        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device({
            .name = "device",
        });

        auto buffer = device.create_buffer({
            .size = 1,
            .name = "actual_buffer",
        });

        auto persistent_task_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{
            .initial_buffers = {.buffers = {&buffer, 1}},
            .name = "buffer",
        });

        auto task_graph_A = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task graph A",
        });

        auto task_graph_B = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task graph B",
        });

        task_graph_A.use_persistent_buffer(persistent_task_buffer);
        task_graph_A.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::GRAPHICS_SHADER_READ_WRITE_CONCURRENT, persistent_task_buffer),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "Task 1) concurrent write read buffer",
        });
        task_graph_A.submit({});
        task_graph_A.complete({});

        task_graph_B.use_persistent_buffer(persistent_task_buffer);
        task_graph_B.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::GRAPHICS_SHADER_READ_WRITE_CONCURRENT, persistent_task_buffer),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "Task 1) concurrent write read buffer",
        });
        task_graph_B.submit({});
        task_graph_B.complete({});

        task_graph_A.execute({});
        std::cout << task_graph_A.get_debug_string() << std::endl;
        task_graph_B.execute({});
        std::cout << task_graph_B.get_debug_string() << std::endl;

        device.wait_idle();
        device.destroy_buffer(buffer);
        device.collect_garbage();
    }

    void read_on_readwriteconcurrent()
    {
        // TEST:
        //  1) read write concurrent buffer
        //  2) read write concurrent buffer
        //  3) read buffer
        // EXPECTED:
        // batch 0 : read write concurrent (1) (2)
        // batch 2 : read buffer (3)
        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device({
            .name = "device",
        });

        auto buffer = device.create_buffer({
            .size = 1,
            .name = "actual_buffer",
        });

        auto tbuffer = daxa::TaskBuffer(daxa::TaskBufferInfo{
            .initial_buffers = {.buffers = {&buffer, 1}},
            .name = "buffer",
        });

        auto task_graph = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task graph",
        });

        task_graph.use_persistent_buffer(tbuffer);

        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE, tbuffer),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "write 0",
        });

        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE_CONCURRENT, tbuffer),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "read write concurrent 1",
        });

        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE_CONCURRENT, tbuffer),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "read write concurrent 2",
        });

        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::GRAPHICS_SHADER_READ, tbuffer),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "read",
        });
        task_graph.submit({});
        task_graph.complete({});


        task_graph.execute({});
        std::cout << task_graph.get_debug_string() << std::endl;

        device.wait_idle();
        device.destroy_buffer(buffer);
        device.collect_garbage();
    }
    
    void concurrent_read_on_read()
    {
        // TEST:
        //  1) write buffer
        //  2) read buffer
        //  3) read buffer
        //  4) write buffer
        // EXPECTED:
        // batch 0 : write buffer (1)
        // batch 1 : read buffer (2 and 3)
        // batch 2 : wriite buffer (4)
        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device({
            .name = "device",
        });

        auto buffer = device.create_buffer({
            .size = 1,
            .name = "actual_buffer",
        });

        auto persistent_task_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{
            .initial_buffers = {.buffers = {&buffer, 1}},
            .name = "buffer",
        });

        auto buffer_b = daxa::TaskBuffer(daxa::TaskBufferInfo{
            .initial_buffers = {.buffers = {&buffer, 1}},
            .name = "buffer b",
        });

        auto task_graph = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task graph A",
        });

        task_graph.use_persistent_buffer(persistent_task_buffer);
        task_graph.use_persistent_buffer(buffer_b);

        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE, buffer_b),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "write buffer b",
        });
        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE, buffer_b),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "write buffer b",
        });

        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE, persistent_task_buffer),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "1 write buffer",
        });
        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ, persistent_task_buffer),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "2 read buffer",
        });
        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ, persistent_task_buffer),
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ, buffer_b),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "3 read buffer",
        });
        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE, persistent_task_buffer),
            },
            .task = [&](daxa::TaskInterface const &) {},
            .name = "4 write buffer",
        });
        task_graph.submit({});
        task_graph.complete({});


        task_graph.execute({});
        std::cout << task_graph.get_debug_string() << std::endl;

        device.wait_idle();
        device.destroy_buffer(buffer);
        device.collect_garbage();
    }

    void optional_attachments()
    {
        // TEST:
        //    1) CREATE image
        //    2) WRITE image
        //    3) READ image
        AppContext app = {};
        // Need to scope the task graphs lifetime.
        // Task graph MUST die before we call wait_idle and collect_garbage.
        auto task_graph = daxa::TaskGraph({
            .device = app.device,
            .record_debug_information = true,
            .name = APPNAME_PREFIX("create-write-read image"),
        });
        // CREATE IMAGE
        auto task_image = task_graph.create_transient_image(daxa::TaskTransientImageInfo{.size = {1, 1, 1}, .name = "task graph tested image"});
        // WRITE IMAGE 1
        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY, task_image),
                daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY, daxa::NullTaskImage),
                daxa::inl_attachment(daxa::TaskBufferAccess::COMPUTE_SHADER_READ, daxa::NullTaskBuffer),
            },
            .task = [](daxa::TaskInterface) {},
            .name = APPNAME_PREFIX("write image 1"),
        });
        // READ_IMAGE 1
        task_graph.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED, task_image)},
            .task = [](daxa::TaskInterface) {},
            .name = APPNAME_PREFIX("read image 1"),
        });

        task_graph.complete({});
        task_graph.execute({});
        std::cout << task_graph.get_debug_string() << std::endl;
    }
} //namespace tests

auto main() -> i32
{
    // tests::concurrent_read_on_read();
    tests::read_on_readwriteconcurrent();
    tests::simplest();
    tests::execution();
    tests::write_read_image();
    tests::write_read_image_layer();
    tests::create_transfer_read_buffer();
    tests::initial_layout_access();
    tests::tracked_slice_barrier_collapsing();
    tests::correct_read_buffer_task_ordering();
    tests::sharing_persistent_image();
    tests::sharing_persistent_buffer();
    tests::transient_write_aliasing();
    tests::transient_resources();
    tests::shader_integration_inl_use();
    tests::test_concurrent_read_write_buffer();
    tests::test_concurrent_read_write_image();
    tests::test_concurrent_read_write_buffer_cross_graphs();
    tests::mipmapping();
    tests::optional_attachments();
}
