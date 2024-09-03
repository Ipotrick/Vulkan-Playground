#pragma once

#include "common.hpp"

namespace tests
{
    void sharing_persistent_buffer()
    {
        // TEST:
        //  1) Create single persistent buffer
        //  2) Record two task graphs (A, B)
        //  3) Task graph A:
        //      writes persistent buffer
        //  4) Task graph B:
        //      reads persistent buffer
        //  6) Call execute task graphs in this order - A -> A -> B -> B
        //  5) Check persistent synch of task graphs A and B
        //      1) INIT -> A - Expected - First execution of A syncs on nothing
        //      2) A    -> A - Expected - Second execution of A syncs on SHADER_WRITE -> SHADER_WRITE
        //      3) A    -> B - Expected - First execution of B syncs on SHADER_WRITE -> SHADER_READ
        //      4) B    -> B - Expected - Second execution of B has no sync
        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({},{}));
        auto buffer = device.create_buffer({
            .size = 1,
            .name = "actual buffer",
        });

        auto persistent_task_buffer = daxa::TaskBuffer(daxa::TaskBufferInfo{
            .initial_buffers = {.buffers = {&buffer, 1}},
            .name = "persistent buffer",
        });

        auto task_graph_A = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task_graph_a",
        });

        auto task_graph_B = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task_graph_b",
        });

        task_graph_A.use_persistent_buffer(persistent_task_buffer);
        task_graph_B.use_persistent_buffer(persistent_task_buffer);
        task_graph_A.add_task({
            .attachments = { daxa::inl_attachment( daxa::TaskBufferAccess::GRAPHICS_SHADER_WRITE, persistent_task_buffer ) },
            .task = [&](daxa::TaskInterface) {},
            .name = "write persistent buffer",
        });
        task_graph_A.submit({});
        task_graph_A.complete({});

        task_graph_B.add_task({
            .attachments = { daxa::inl_attachment( daxa::TaskBufferAccess::GRAPHICS_SHADER_READ, persistent_task_buffer ) },
            .task = [&](daxa::TaskInterface) {},
            .name = "read persistent buffer",
        });
        task_graph_B.submit({});
        task_graph_B.complete({});

        task_graph_A.execute({});
        std::cout << task_graph_A.get_debug_string() << std::endl;
        task_graph_A.execute({});
        std::cout << task_graph_A.get_debug_string() << std::endl;
        task_graph_B.execute({});
        std::cout << task_graph_B.get_debug_string() << std::endl;
        task_graph_B.execute({});
        std::cout << task_graph_B.get_debug_string() << std::endl;

        device.wait_idle();
        device.destroy_buffer(buffer);
        device.collect_garbage();
    }
    void sharing_persistent_image()
    {
        // TEST:
        //  1) Create single persistent image
        //  2) Record two task graphs (A, B)
        //  3) Task graph A:
        //      writes persistent image
        //      uses persistent image as color attachment
        //  4) Task graph B:
        //      reads persistent image
        //  6) Call execute task graphs in this order - A -> A -> B -> B
        //  5) Check persistent synch of task graphs A and B
        //      1) INIT -> A - Expected - First execution of A transitions image from UNDEFINED to WRITE
        //      2) A    -> A - Expected - Second execution of  A transitions image from COLOR_ATTACHMENT to WRITE
        //      3) A    -> B - Expected - First execution of B transitions image from WRITE to READ
        //      4) B    -> B - Expected - Second execution of B has no transitions
        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({},{}));
        // We need an actual image, as task graph will try to populate its image view cache.
        // It will error out when it detects that there are no runtime images for a task image when updating the view cache.
        auto image = device.create_image({
            .size = {1, 1, 1},
            .array_layer_count = 1,
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE |
                     daxa::ImageUsageFlagBits::COLOR_ATTACHMENT |
                     daxa::ImageUsageFlagBits::SHADER_SAMPLED,
            .name = "actual image",
        });

        auto persistent_task_image = daxa::TaskImage(
            daxa::TaskImageInfo{
                .initial_images = {.images = {&image, 1}},
                .swapchain_image = false,
                .name = "image",
            });

        auto task_graph_A = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task_graph_a",
        });

        task_graph_A.use_persistent_image(persistent_task_image);
        task_graph_A.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::GRAPHICS_SHADER_STORAGE_WRITE_ONLY, persistent_task_image ) },
            .task = [&](daxa::TaskInterface) {},
            .name = "write persistent image",
        });
        task_graph_A.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::COLOR_ATTACHMENT, persistent_task_image ) },
            .task = [&](daxa::TaskInterface) {},
            .name = "persistent image - color attachment",
        });
        task_graph_A.submit({});
        task_graph_A.complete({});

        auto task_graph_B = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "task_graph_b",
        });
        task_graph_B.use_persistent_image(persistent_task_image);
        task_graph_B.add_task({
            .attachments = {daxa::inl_attachment(daxa::TaskImageAccess::GRAPHICS_SHADER_SAMPLED, persistent_task_image ) },
            .task = [&](daxa::TaskInterface) {},
            .name = "read persistent image",
        });
        task_graph_B.submit({});
        task_graph_B.complete({});

        task_graph_A.execute({});
        std::cout << task_graph_A.get_debug_string() << std::endl;
        task_graph_A.execute({});
        std::cout << task_graph_A.get_debug_string() << std::endl;
        task_graph_B.execute({});
        std::cout << task_graph_B.get_debug_string() << std::endl;
        task_graph_B.execute({});
        std::cout << task_graph_B.get_debug_string() << std::endl;

        device.wait_idle();
        device.destroy_image(image);
        device.collect_garbage();
    }
} // namespace tests
