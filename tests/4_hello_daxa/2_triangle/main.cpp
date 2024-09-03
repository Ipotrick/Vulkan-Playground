#include <daxa/daxa.hpp>

// Include the pipeline manager
#include <daxa/utils/pipeline_manager.hpp>
// We'll also include iostream, since we now use it
#include <iostream>

// We're going to use another optional feature of Daxa,
// called TaskGraph. We'll explain more below.
#include <daxa/utils/task_graph.hpp>

#include "shared.inl"

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

void upload_vertces_data_task(daxa::TaskGraph & tg, daxa::TaskBufferView vertices)
{
    tg.add_task({
        // Listing the attachments for the task.
        // Task graph needs this list to perform various convenience actions, such as
        // - dynamically creating image views
        // - initializing resources
        // - generating sync
        // - reordering tasks for better performance 
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, vertices),
        },
        // Callback of task.
        // This is called once every frame when executing the graph.
        .task = [=](daxa::TaskInterface ti)
        {
            // This is the data we'll send to the GPU
            auto data = std::array{
                MyVertex{.position = {-0.5f, +0.5f, 0.0f}, .color = {1.0f, 0.0f, 0.0f}},
                MyVertex{.position = {+0.5f, +0.5f, 0.0f}, .color = {0.0f, 1.0f, 0.0f}},
                MyVertex{.position = {+0.0f, -0.5f, 0.0f}, .color = {0.0f, 0.0f, 1.0f}},
            };

            // In order to send the data to the GPU, we can create a
            // staging buffer, which has host access, so that we can then
            // issue a command to copy from this buffer to the dedicated
            // GPU memory.
            auto staging_buffer_id = ti.device.create_buffer({
                .size = sizeof(data),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "my staging buffer",
            });
            // We can also ask the command list to destroy this temporary buffer,
            // since we don't care about it living, but we DO need it to survive
            // through its usage on the GPU (which won't happen until after these
            // commands are submitted), so we tell the command list to destroy it
            // in a deferred fashion.
            ti.recorder.destroy_buffer_deferred(staging_buffer_id);
            // Instead of doing this manually, we could use one of Daxa's
            // other useful utilities, "Mem", but it's simple enough for now.

            // We then get the memory mapped pointer of the staging buffer, and
            // write the data directly to it.
            auto * buffer_ptr = ti.device.buffer_host_address_as<std::array<MyVertex, 3>>(staging_buffer_id).value();
            *buffer_ptr = data;

            // For every attachment, the task graph generates metadata.
            // The interface lets you query infomation about the attachments via the get function.
            // The get function returns attachment info for exactly the view that is used in the attachment list.
            daxa::TaskBufferAttachmentInfo const & buffer_at_info = ti.get(vertices);

            // One of the things the interface gives you is the list of runtime buffer ids:
            // Importantly these ids should be only retrieved by the task interface!
            // DO NOT copy the daxa::BufferId for attachments into the task!
            // The runtime ids can change between every frame.
            std::span<daxa::BufferId const> runtime_ids = buffer_at_info.ids;

            // We only have one runtime id for the vertices task buffer:
            daxa::BufferId id = runtime_ids[0];

            // And finally, we can just copy the data from the staging buffer
            // to the actual buffer.
            ti.recorder.copy_buffer_to_buffer({
                .src_buffer = staging_buffer_id,
                .dst_buffer = id,
                .size = sizeof(data),
            });

            // We have shown this above way of uploading just as a showcase of the api.
            // Daxa provides convenience upload functions for task interfaces:
            //   allocate(ti, alloc info) -> Optional<Alloc>
            //   allocate_fill(ti, cpu data reference) -> Optional<Alloc>
            //   allocate_fill_copy(ti, cpu data reference, task buffer info) -> void
            // All of these use an automatically managed linear allocator to device local host mapped memory.
            // These allocations are ultra fast and their lifetime is automatically managed.
            // The only downside is their size limit which is quite low.
            // I highly recommend using these alloc functions for small data (~< 1kb).
        },
        .name = "upload vertices",
    });
}

// We declare a static task in two Parts.
// The first part is called the task "head".
// The second part is the actual task class we declare. The task struct inherits from a partial task declared by the head.
// This split might seems trange, especially that the heads are declared as makros.
// This has a good reason: we can have task heads within shader/host shared .inl files!
// The heads automatically declare a struct that is automatically filled by the task graph representing your attachments!
// This is can be very convenient.
// In order to make this properly work in c++ we had to wrap it all in preprocessor makros.
// The head for DrawToSwapchainTask is declared within the shared file.

// Check out the shader/c++ shared code in shared.inl

struct DrawToSwapchainTask : DrawToSwapchainH::Task
{
    // Must be declared in every task that inherits from a head.
    AttachmentViews views = {};
    daxa::RasterPipeline * pipeline = {};
    void callback(daxa::TaskInterface ti)
    {
        {
            // The task head declares a list of indices, one for each attachment.
            // They are inside a constant under the heads namespace:
            [[maybe_unused]] daxa::TaskImageAttachmentIndex image_attach_index = DrawToSwapchainH::AT.color_target;
        }
        {
            // For convenience, each task that derives from the heads base task
            // can also access it directly without the heads namespace:
            [[maybe_unused]] daxa::TaskImageAttachmentIndex image_attach_index = AT.color_target;
        }
        // Just like for task image/buffer views, 
        // the ti can give you all the info you need about every attachment via index:
        daxa::TaskImageAttachmentInfo const & image_attach_info = ti.get(AT.color_target);

        daxa::ImageInfo image_info = ti.device.image_info(image_attach_info.ids[0]).value();

        // When starting a render pass via a rasterization pipeline, daxa "eats" a generic command recorder
        // and turns it into a RenderCommandRecorder.
        // Only the RenderCommandRecorder can record raster commands.
        // The RenderCommandRecorder can only record commands that are valid within a render pass.
        // This way daxa ensures typesafety for command recording.
        daxa::RenderCommandRecorder render_recorder = std::move(ti.recorder).begin_renderpass({
            .color_attachments = std::array{
                daxa::RenderAttachmentInfo{
                    .image_view = image_attach_info.view_ids[0],
                    .load_op = daxa::AttachmentLoadOp::CLEAR,
                    .clear_value = std::array<daxa::f32, 4>{0.1f, 0.0f, 0.5f, 1.0f},
                },
            },
            .render_area = {.width = image_info.size.x, .height = image_info.size.y},
        });
        // Here, we'll bind the pipeline to be used in the draw call below
        render_recorder.set_pipeline(*pipeline);

        // Very importantly, task graph packs up our attachment shader data into a byte blob.
        // We need to pass this blob to our shader somehow.
        // The typical way to do this is to assign the blob to the push constant.
        render_recorder.push_constant(MyPushConstant {
            .attachments = ti.attachment_shader_blob,
        });
        // and issue the draw call with the desired number of vertices.
        render_recorder.draw({.vertex_count = 3});

        // VERY IMPORTANT! A renderpass must be ended after finishing!
        // The ending of a render pass returns back the original command recorder.
        // Assign it back to the task interfaces command recorder.
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
        "Daxa sample window name", nullptr, nullptr);
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

    // Let instance auto select a device
    daxa::Device device = instance.create_device_2(instance.choose_device({}, {}));

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

    // Create a raster pipeline
    // first by creating the pipeline manager
    auto pipeline_manager = daxa::PipelineManager({
        .device = device,
        .shader_compile_options = {
            .root_paths = {
                DAXA_SHADER_INCLUDE_DIR,
                "./tests/4_hello_daxa/2_triangle",
            },
            .language = daxa::ShaderLanguage::GLSL,
            .enable_debug_info = true,
        },
        .name = "my pipeline manager",
    });
    // Then just adding it to the pipeline manager
    std::shared_ptr<daxa::RasterPipeline> pipeline;
    {
        auto result = pipeline_manager.add_raster_pipeline({
            .vertex_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"main.glsl"}},
            .fragment_shader_info = daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"main.glsl"}},
            .color_attachments = {{.format = swapchain.get_format()}},
            .raster = {},
            .push_constant_size = sizeof(MyPushConstant),
            .name = "my pipeline",
        });
        if (result.is_err())
        {
            std::cerr << result.message() << std::endl;
            return -1;
        }
        pipeline = result.value();
    }

    // Since we want to draw a triangle, we'll want to have data
    // to use in the shaders describing it. To do this, we could hard
    // code the data into the shader, but we might as well showcase
    // how to create and upload a buffer

    // We'll start by "allocating" the memory on the GPU by creating
    // a buffer with the device.
    auto buffer_id = device.create_buffer({
        .size = sizeof(MyVertex) * 3,
        .name = "my vertex data",
    });
    // Obviously the vertex data is not yet on the GPU, and this buffer
    // is just empty. We will use conditional TaskGraph to upload it on
    // just the first frame. More on this soon!

    // While not entirely necessary, we're going to use TaskGraph, which
    // allows us to compile a list of GPU tasks and their dependencies
    // into a synchronized set of commands. This simplifies your code
    // by making different tasks completely self-contained, while also
    // generating the most optimal synchronization for the tasks you
    // describe.

    // When using TaskGraph, we must create "virtual" resources (we call
    // them task resources) whose usages are tracked, allowing for correct
    // synchronization for them.

    // The first we'll make is the swapchain image task resource
    // We could immediately give this task image a image id.
    // But in the case of the swapchain images we need to reacquire a new image every frame.
    auto task_swapchain_image = daxa::TaskImage{{.swapchain_image = true, .name = "swapchain image"}};

    // We will also create a buffer task resource, for our MyVertex buffer buffer_id
    // We do something a little special here, which is that we set the initial access
    // of the buffer to be vertex shader read, and that's because we'll create a task
    // list which will upload the buffer
    auto task_vertex_buffer = daxa::TaskBuffer({
        .initial_buffers = {.buffers = std::span{&buffer_id, 1}},
        .name = "task vertex buffer",
    });

    // TaskGraph can have permutations, which allow for runtime conditions
    // to trigger different outcomes. These are identified with indices,
    // so we'll define an enum representing all the condition indices
    // since we want to name them and make sure they're all unique.
    // This is commented out, since instead we'll use a separate TaskGraph
    // enum class TaskCondition
    // {
    //     VERTICES_UPLOAD,
    //     COUNT,
    // };
    // std::array<bool, static_cast<daxa::usize>(TaskCondition::COUNT)> task_condition_states{};

    auto loop_task_graph = daxa::TaskGraph({
        .device = device,
        .swapchain = swapchain,
        .name = "loop",
    });

    // We need to explicitly declare all uses of persistent task resources!
    // This not a technical limitation but an intentional choice.
    // Manually marking used resources makes it possible to detect errors in your graph recording.
    loop_task_graph.use_persistent_buffer(task_vertex_buffer);
    loop_task_graph.use_persistent_image(task_swapchain_image);

    // And a task to draw to the screen
    loop_task_graph.add_task(DrawToSwapchainTask{
        .views = std::array{
            daxa::attachment_view(DrawToSwapchainH::AT.color_target, task_swapchain_image),
            daxa::attachment_view(DrawToSwapchainH::AT.vertices, task_vertex_buffer),
        },
        .pipeline = pipeline.get(),
    });

    // We now need to tell the task graph that these commands will be submitted,
    // and that we have no additional information to provide. This exists in
    // order to allow more advanced Vulkan users to do much more complicated
    // things, that we don't care about, and that you can create a whole app
    // without ever touching.
    loop_task_graph.submit({});

    // And tell the task graph to do the present step.
    loop_task_graph.present({});
    // Finally, we complete the task graph, which essentially compiles the
    // dependency graph between tasks, and inserts the most optimal synchronization!
    loop_task_graph.complete({});

    {
        // Now we record a secondary task graph, that is only executed once.
        // This task graph uploads the vertex buffer.
        // Task Graph resources automatically link between graphcs at runtime,
        // so you dont need to be concerned about sync of the vertex buffer between the two graphs.
        auto upload_task_graph = daxa::TaskGraph({ .device = device, .name = "upload", });

        upload_task_graph.use_persistent_buffer(task_vertex_buffer);

        // Now we can record our tasks!

        // First thing we'll do is record the upload task.
        upload_vertces_data_task(upload_task_graph, task_vertex_buffer);

        upload_task_graph.submit({});
        upload_task_graph.complete({});
        upload_task_graph.execute({});
    }

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
        // We update the image id of the task swapchain image.
        task_swapchain_image.set_images({.images = std::span{&swapchain_image, 1}});

        // So, now all we need to do is execute our task graph!
        loop_task_graph.execute({});
        device.collect_garbage();
    }

    device.wait_idle();
    device.collect_garbage();
    device.destroy_buffer(buffer_id);
}