#include <daxa/daxa.hpp>
#include <iostream>
#include <chrono>
#include "../../0_common/shared.hpp"

struct App
{
    daxa::Instance daxa_ctx = daxa::create_instance({.flags = daxa::InstanceFlagBits::DEBUG_UTILS});
    daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({}, {}));
};

namespace tests
{
    using namespace daxa::types;

    void simplest(App & app)
    {
        auto recorder = app.device.create_command_recorder({});

        // CommandRecorder can create ExecutableCommandList from the currently recorded commands.
        // After calling complete_current_commands, the current commands are cleared.
        // After calling complete_current_commands, you may record more commands and make new ExecutableCommandList with the recorder.
        auto executable_commands = recorder.complete_current_commands();

        app.device.submit_commands({
            .command_lists = std::array{executable_commands},
        });

        /// WARNING:    ALL CommandRecorders from a device MUST be destroyed prior to calling collect_garbage or destroying the device!
        ///             This is because The device can only do the internal cleanup when no commands get recorded in parallel!
    }

    template <usize SX, usize SY, usize SZ>
    using ImageArray = std::array<std::array<std::array<std::array<f32, 4>, SX>, SY>, SZ>;

    void copy(App & app)
    {
        auto recorder = app.device.create_command_recorder({.name = "copy command list"});

        constexpr u32 SIZE_X = 3;
        constexpr u32 SIZE_Y = 3;
        constexpr u32 SIZE_Z = 3;

        auto get_printable_char_buffer = []<usize SX, usize SY, usize SZ>(ImageArray<SX, SY, SZ> const & in_data) -> std::vector<char>
        {
            std::vector<char> data;
            constexpr auto pixel = std::to_array("\033[48;2;000;000;000m  ");
            constexpr auto line_terminator = std::to_array("\033[0m ");
            constexpr auto newline_terminator = std::to_array("\033[0m\n");
            data.resize(SX * SY * SZ * (pixel.size() - 1) + SY * SZ * (line_terminator.size() - 1) + SZ * (newline_terminator.size() - 1) + 1);
            usize output_index = 0;
            for (usize zi = 0; zi < SZ; ++zi)
            {
                for (usize yi = 0; yi < SY; ++yi)
                {
                    for (usize xi = 0; xi < SX; ++xi)
                    {
                        u8 const r = static_cast<u8>(in_data[zi][yi][xi][0] * 255.0f);
                        u8 const g = static_cast<u8>(in_data[zi][yi][xi][1] * 255.0f);
                        u8 const b = static_cast<u8>(in_data[zi][yi][xi][2] * 255.0f);
                        auto next_pixel = pixel;
                        next_pixel[7 + 0 * 4 + 0] = static_cast<char>(static_cast<u8>('0') + (r / 100));
                        next_pixel[7 + 0 * 4 + 1] = static_cast<char>(static_cast<u8>('0') + (r % 100) / 10);
                        next_pixel[7 + 0 * 4 + 2] = static_cast<char>(static_cast<u8>('0') + (r % 10));
                        next_pixel[7 + 1 * 4 + 0] = static_cast<char>(static_cast<u8>('0') + (g / 100));
                        next_pixel[7 + 1 * 4 + 1] = static_cast<char>(static_cast<u8>('0') + (g % 100) / 10);
                        next_pixel[7 + 1 * 4 + 2] = static_cast<char>(static_cast<u8>('0') + (g % 10));
                        next_pixel[7 + 2 * 4 + 0] = static_cast<char>(static_cast<u8>('0') + (b / 100));
                        next_pixel[7 + 2 * 4 + 1] = static_cast<char>(static_cast<u8>('0') + (b % 100) / 10);
                        next_pixel[7 + 2 * 4 + 2] = static_cast<char>(static_cast<u8>('0') + (b % 10));
                        std::copy(next_pixel.begin(), next_pixel.end() - 1, data.data() + output_index);
                        output_index += pixel.size() - 1;
                    }
                    std::copy(line_terminator.begin(), line_terminator.end() - 1, data.data() + output_index);
                    output_index += line_terminator.size() - 1;
                }
                std::copy(newline_terminator.begin(), newline_terminator.end() - 1, data.data() + output_index);
                output_index += newline_terminator.size() - 1;
            }
            data[data.size() - 1] = '\0';
            return data;
        };

        auto data = ImageArray<SIZE_X, SIZE_Y, SIZE_Z>{};

        for (usize zi = 0; zi < SIZE_Z; ++zi)
        {
            for (usize yi = 0; yi < SIZE_Y; ++yi)
            {
                for (usize xi = 0; xi < SIZE_X; ++xi)
                {
                    data[zi][yi][xi][0] = static_cast<f32>(xi) / static_cast<f32>(SIZE_X - 1);
                    data[zi][yi][xi][1] = static_cast<f32>(yi) / static_cast<f32>(SIZE_Y - 1);
                    data[zi][yi][xi][2] = static_cast<f32>(zi) / static_cast<f32>(SIZE_Z - 1);
                    data[zi][yi][xi][3] = 1.0f;
                }
            }
        }

        daxa::BufferId const staging_upload_buffer = app.device.create_buffer({
            .size = sizeof(decltype(data)),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = "staging_upload_buffer",
        });

        daxa::BufferId const device_local_buffer = app.device.create_buffer({
            .size = sizeof(decltype(data)),
            .name = "device_local_buffer",
        });

        daxa::BufferId const staging_readback_buffer = app.device.create_buffer({
            .size = sizeof(decltype(data)),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "staging_readback_buffer",
        });

        daxa::ImageId const image_1 = app.device.create_image({
            .dimensions = 2 + static_cast<u32>(SIZE_Z > 1),
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = {SIZE_X, SIZE_Y, SIZE_Z},
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC,
            .name = "image_1",
        });

        daxa::ImageId const image_2 = app.device.create_image({
            .dimensions = 2 + static_cast<u32>(SIZE_Z > 1),
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = {SIZE_X, SIZE_Y, SIZE_Z},
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC,
            .name = "image_2",
        });

        daxa::TimelineQueryPool timeline_query_pool = app.device.create_timeline_query_pool({
            .query_count = 2,
            .name = "timeline_query",
        });

        auto & buffer_ptr = *app.device.buffer_host_address_as<ImageArray<SIZE_X, SIZE_Y, SIZE_Z>>(staging_upload_buffer).value();

        buffer_ptr = data;

        recorder.reset_timestamps({
            .query_pool = timeline_query_pool,
            .start_index = 0,
            .count = timeline_query_pool.info().query_count,
        });

        recorder.write_timestamp({
            .query_pool = timeline_query_pool,
            .pipeline_stage = daxa::PipelineStageFlagBits::BOTTOM_OF_PIPE,
            .query_index = 0,
        });

        recorder.pipeline_barrier({
            .src_access = daxa::AccessConsts::HOST_WRITE,
            .dst_access = daxa::AccessConsts::TRANSFER_READ,
        });

        recorder.copy_buffer_to_buffer({
            .src_buffer = staging_upload_buffer,
            .dst_buffer = device_local_buffer,
            .size = sizeof(decltype(data)),
        });

        // Barrier to make sure device_local_buffer is has no read after write hazard.
        recorder.pipeline_barrier({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_access = daxa::AccessConsts::TRANSFER_READ,
        });

        recorder.pipeline_barrier_image_transition({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .image_id = image_1,
        });

        recorder.copy_buffer_to_image({
            .buffer = device_local_buffer,
            .image = image_1,
            .image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .image_extent = {SIZE_X, SIZE_Y, SIZE_Z},
        });

        recorder.pipeline_barrier_image_transition({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_access = daxa::AccessConsts::TRANSFER_READ,
            .dst_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
            .image_id = image_1,
        });

        recorder.pipeline_barrier_image_transition({
            .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .image_id = image_2,
        });

        recorder.copy_image_to_image({
            .src_image = image_1,
            .src_image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
            .dst_image = image_2,
            .dst_image_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .extent = {SIZE_X, SIZE_Y, SIZE_Z},
        });

        recorder.pipeline_barrier_image_transition({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_access = daxa::AccessConsts::TRANSFER_READ,
            .dst_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
            .image_id = image_2,
        });

        // Barrier to make sure device_local_buffer is has no write after read hazard.
        recorder.pipeline_barrier({
            .src_access = daxa::AccessConsts::TRANSFER_READ,
            .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
        });

        recorder.copy_image_to_buffer({
            .image = image_2,
            .image_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
            .image_extent = {SIZE_X, SIZE_Y, SIZE_Z},
            .buffer = device_local_buffer,
        });

        // Barrier to make sure device_local_buffer is has no read after write hazard.
        recorder.pipeline_barrier({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_access = daxa::AccessConsts::TRANSFER_READ,
        });

        recorder.copy_buffer_to_buffer({
            .src_buffer = device_local_buffer,
            .dst_buffer = staging_readback_buffer,
            .size = sizeof(decltype(data)),
        });

        // Barrier to make sure staging_readback_buffer is has no read after write hazard.
        recorder.pipeline_barrier({
            .src_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_access = daxa::AccessConsts::HOST_READ,
        });

        recorder.write_timestamp({
            .query_pool = timeline_query_pool,
            .pipeline_stage = daxa::PipelineStageFlagBits::BOTTOM_OF_PIPE,
            .query_index = 1,
        });

        auto executable_commands = recorder.complete_current_commands();

        app.device.submit_commands({
            .command_lists = std::array{executable_commands},
        });

        app.device.wait_idle();

        auto query_results = timeline_query_pool.get_query_results(0, 2);
        if ((query_results[1] != 0u) && (query_results[3] != 0u))
        {
            std::cout << "gpu execution took " << static_cast<f64>(query_results[2] - query_results[0]) / 1000000.0 << " ms" << std::endl;
        }

        auto const & readback_data = *app.device.buffer_host_address_as<ImageArray<SIZE_X, SIZE_Y, SIZE_Z>>(staging_readback_buffer).value();

        std::cout << "Original data: " << std::endl;
        {
            auto printable_buffer = get_printable_char_buffer(data);
            std::cout << printable_buffer.data();
        }

        std::cout << "Readback data: " << std::endl;
        {
            auto printable_buffer = get_printable_char_buffer(readback_data);
            std::cout << printable_buffer.data();
        }

        for (usize zi = 0; zi < SIZE_Z; ++zi)
        {
            for (usize yi = 0; yi < SIZE_Y; ++yi)
            {
                for (usize xi = 0; xi < SIZE_X; ++xi)
                {
                    for (usize ci = 0; ci < 4; ++ci)
                    {
                        DAXA_DBG_ASSERT_TRUE_M(data[zi][yi][xi][ci] == readback_data[zi][yi][xi][ci], "readback data differs from upload data");
                    }
                }
            }
        }

        app.device.destroy_buffer(staging_upload_buffer);
        app.device.destroy_buffer(device_local_buffer);
        app.device.destroy_buffer(staging_readback_buffer);
        app.device.destroy_image(image_1);
        app.device.destroy_image(image_2);
    }

    void deferred_destruction(App & app)
    {
        auto recorder = app.device.create_command_recorder({.name = "deferred_destruction command list"});

        daxa::BufferId const buffer = app.device.create_buffer({.size = 4});
        daxa::ImageId const image = app.device.create_image({
            .size = {1, 1, 1},
            .usage = daxa::ImageUsageFlagBits::COLOR_ATTACHMENT,
        });
        daxa::ImageViewId const image_view = app.device.create_image_view({.image = image});
        daxa::SamplerId const sampler = app.device.create_sampler({});

        // The gpu resources are not destroyed here. Their destruction is deferred until the command list completes execution on the gpu.
        recorder.destroy_buffer_deferred(buffer);
        recorder.destroy_image_deferred(image);
        recorder.destroy_image_view_deferred(image_view);
        recorder.destroy_sampler_deferred(sampler);

        auto executable_commands = recorder.complete_current_commands();

        // Even after this call the resources will still be alive, as zombie resources are not checked to be dead in submit calls.
        app.device.submit_commands({
            .command_lists = std::array{executable_commands},
        });
    }

    void recreation(App & app)
    {
        std::chrono::time_point begin_time_point = std::chrono::high_resolution_clock::now();

        int const outer_iterations = 1000;
        int const inner_iterations = 1000;

        auto image = app.device.create_image({
            .size = {1, 1, 1},
            .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE,
            .name = "tested image",
        });

        for (int outer_i = 0; outer_i < outer_iterations; ++outer_i)
        {
            for (int inner_i = 0; inner_i < inner_iterations; ++inner_i)
            {
                auto view = app.device.create_image_view({
                    .image = image,
                    .name = "test image view long name, past short string optimization",
                });
                app.device.destroy_image_view(view);
            }
            app.device.collect_garbage();
        }
        std::chrono::time_point end_time_point = std::chrono::high_resolution_clock::now();
        auto total_iterations = outer_iterations * inner_iterations;
        app.device.destroy_image(image);
        auto time_taken_mics = std::chrono::duration_cast<std::chrono::microseconds>(end_time_point - begin_time_point);
        auto time_taken_per_recreation = static_cast<double>(time_taken_mics.count()) * 0.00'000'1 / 60.0 / 60.0 / static_cast<double>(total_iterations);
        auto time_taken_til_wrap = static_cast<double>(std::numeric_limits<uint32_t>::max()) * time_taken_per_recreation;
        auto time_taken_til_wrap_64 = static_cast<double>(std::numeric_limits<uint64_t>::max()) * time_taken_per_recreation / 24.0 / 356.0;
        std::cout
            << "Recreation test measured "
            << time_taken_mics.count()
            << " time passed for "
            << outer_iterations * inner_iterations
            << " total iterations for image view recreation. It will take "
            << time_taken_til_wrap
            << " hours for the ids to wrap.\nWith 64 bit ids, the wrap time would increase to "
            << time_taken_til_wrap_64
            << " years"
            << std::endl;
    }

    void ub_protection_perf(App & app)
    {
        for (int j = 0; j < 10; ++j)
        {
            int const iterations = 1000000;

            double time_taken_per_call_0 = 0.0;
            double time_taken_per_call_1 = 0.0;

            {
                auto recorder = app.device.create_command_recorder({});
                auto buf = app.device.create_buffer({
                    .size = sizeof(daxa::u32),
                    .name = "buf",
                });
                auto img = app.device.create_image({
                    .size = {1, 1, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE,
                    .name = "img",
                });
                std::chrono::time_point begin_time_point = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iterations; ++i)
                {
                    recorder.copy_image_to_buffer({
                        .image = img,
                        .buffer = buf,
                    });
                }
                std::chrono::time_point end_time_point = std::chrono::high_resolution_clock::now();
                auto executable_commands = recorder.complete_current_commands();
                app.device.destroy_buffer(buf);
                app.device.destroy_image(img);
                auto time_taken_mics = std::chrono::duration_cast<std::chrono::microseconds>(end_time_point - begin_time_point);
                time_taken_per_call_0 = static_cast<double>(time_taken_mics.count()) / static_cast<double>(iterations);
                std::cout
                    << "recording commands with unsafe calls took: "
                    << time_taken_mics.count()
                    << " for "
                    << iterations
                    << " iterations. That is "
                    << time_taken_per_call_0
                    << "us per call"
                    << std::endl;
            }

            {
                auto recorder = app.device.create_command_recorder({});
                auto buf = app.device.create_buffer({
                    .size = sizeof(daxa::u32),
                    .name = "buf",
                });
                auto img = app.device.create_image({
                    .size = {1, 1, 1},
                    .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE,
                    .name = "img",
                });
                std::chrono::time_point begin_time_point = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iterations; ++i)
                {
                    recorder.copy_buffer_to_image({
                        .buffer = buf,
                        .image = img,
                    });
                }
                std::chrono::time_point end_time_point = std::chrono::high_resolution_clock::now();
                app.device.destroy_buffer(buf);
                app.device.destroy_image(img);
                auto time_taken_mics = std::chrono::duration_cast<std::chrono::microseconds>(end_time_point - begin_time_point);
                time_taken_per_call_1 = static_cast<double>(time_taken_mics.count()) / static_cast<double>(iterations);
                std::cout
                    << "recording commands with safe calls took: "
                    << time_taken_mics.count()
                    << " for "
                    << iterations
                    << " iterations. That is "
                    << time_taken_per_call_1
                    << "us per call"
                    << std::endl;
            }

            std::cout
                << "ratio "
                << (time_taken_per_call_1 / time_taken_per_call_0 - 1.0) * 100.0
                << " %% slower"
                << std::endl;
        }
    }

    void multiple_ecl(App & app)
    {
        daxa::BufferId buf_a = app.device.create_buffer({.size = 4, .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE, .name = "buf_a"});
        daxa::BufferId buf_b = app.device.create_buffer({.size = 4, .name = "buf_b"});
        daxa::BufferId buf_c = app.device.create_buffer({.size = 4, .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, .name = "buf_c"});

        constexpr daxa::u32 TEST_VALUE = 0xf0abf0ab;

        *app.device.buffer_host_address_as<daxa::u32>(buf_a).value() = TEST_VALUE;

        daxa::CommandRecorder cmdr = app.device.create_command_recorder({});

        // create first executable command list
        cmdr.copy_buffer_to_buffer({
            .src_buffer = buf_a,
            .dst_buffer = buf_b,
            .size = 4,
        });
        daxa::ExecutableCommandList exc_commands_0 = cmdr.complete_current_commands();

        // create a second executable command list
        cmdr.copy_buffer_to_buffer({
            .src_buffer = buf_b,
            .dst_buffer = buf_c,
            .size = 4,
        });
        daxa::ExecutableCommandList exc_commands_1 = cmdr.complete_current_commands();

        daxa::BinarySemaphore sema = app.device.create_binary_semaphore({});

        // submit both executable command lists
        // make it multiple submits to showcase that these really are two different VkCommandBuffers from a single VkCommandPool
        app.device.submit_commands({
            .command_lists = std::array{exc_commands_0},
            .signal_binary_semaphores = std::array{sema},
        });
        app.device.submit_commands({
            .wait_stages = daxa::PipelineStageFlagBits::TRANSFER,
            .command_lists = std::array{exc_commands_1},
            .wait_binary_semaphores = std::array{sema},
        });

        app.device.wait_idle();

        [[maybe_unused]] daxa::u32 readback_value = *app.device.buffer_host_address_as<daxa::u32>(buf_c).value();

        DAXA_DBG_ASSERT_TRUE_M(readback_value == TEST_VALUE, "TEST VALUE DOES NOT MATCH READBACK VALUE");

        app.device.destroy_buffer(buf_c);
        app.device.destroy_buffer(buf_b);
        app.device.destroy_buffer(buf_a);
    }
    void build_acceleration_structure(App & app)
    {
        try
        {
            daxa::Device device;
            try
            {
                device = app.daxa_ctx.create_device_2(app.daxa_ctx.choose_device(daxa::ImplicitFeatureFlagBits::BASIC_RAY_TRACING, {}));
            }
            catch (std::runtime_error error)
            {
                std::cout << "Test skipped. No present device supports raytracing!" << std::endl;
                return;
            }
            /// Prepare mesh data:
            auto vertices = std::array{
                std::array{0.25, 0.75, 0.5},
                std::array{0.5, 0.25, 0.5},
                std::array{0.75, 0.75, 0.5},
            };
            auto vertex_buffer = device.create_buffer({
                .size = sizeof(decltype(vertices)),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "vertex buffer",
            });
            defer { device.destroy_buffer(vertex_buffer); };
            std::memcpy(device.buffer_host_address(vertex_buffer).value(), &vertices, sizeof(decltype(vertices)));
            auto indices = std::array{0, 1, 2};
            auto index_buffer = device.create_buffer({
                .size = sizeof(decltype(indices)),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "index buffer",
            });
            defer { device.destroy_buffer(index_buffer); };
            std::memcpy(device.buffer_host_address(index_buffer).value(), &indices, sizeof(decltype(indices)));
            auto transform = daxa_f32mat3x4{
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
            };
            auto transform_buffer = device.create_buffer({
                .size = sizeof(daxa_f32mat3x4),
                .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                .name = "transform buffer",
            });
            defer { device.destroy_buffer(transform_buffer); };
            std::memcpy(device.buffer_host_address(transform_buffer).value(), &transform, sizeof(daxa_f32mat3x4));
            /// Write As description data:
            auto geometries = std::array{
                daxa::BlasTriangleGeometryInfo{
                    .vertex_format = daxa::Format::R32G32B32_SFLOAT,
                    .vertex_data = {}, // Ignored in get_acceleration_structure_build_sizes.
                    .vertex_stride = sizeof(daxa_f32vec3),
                    .max_vertex = static_cast<u32>(vertices.size() - 1),
                    .index_type = daxa::IndexType::uint32,
                    .index_data = {},     // Ignored in get_acceleration_structure_build_sizes.
                    .transform_data = {}, // Ignored in get_acceleration_structure_build_sizes.
                    .count = 1,
                    .flags = {},
                }};
            auto build_info = daxa::BlasBuildInfo{
                .dst_blas = {}, // Ignored in get_acceleration_structure_build_sizes.
                .geometries = geometries,
                .scratch_data = {}, // Ignored in get_acceleration_structure_build_sizes.
            };
            /// Query As sizes:
            daxa::AccelerationStructureBuildSizesInfo build_size_info = device.blas_build_sizes(build_info);
            /// Create Scratch buffer and As:
            auto scratch_buffer = device.create_buffer({
                .size = build_size_info.build_scratch_size,
                .name = "scratch buffer",
            });
            defer { device.destroy_buffer(scratch_buffer); };
            daxa::BlasId blas = device.create_blas({
                .size = build_size_info.acceleration_structure_size,
                .name = "test blas",
            });
            defer { device.destroy_blas(blas); };
            /// Fill the remaining fields of the build info:
            auto & tri_geom = geometries[0];
            tri_geom.vertex_data = device.device_address(vertex_buffer).value();
            tri_geom.index_data = device.device_address(index_buffer).value();
            tri_geom.transform_data = device.device_address(transform_buffer).value();
            build_info.dst_blas = blas;
            build_info.scratch_data = device.device_address(scratch_buffer).value();
            /// Record build commands:
            auto exec_cmds = [&]()
            {
                auto recorder = device.create_command_recorder({});
                recorder.build_acceleration_structures({
                    .blas_build_infos = std::array{build_info},
                });
                return recorder.complete_current_commands();
            }();
            device.submit_commands({.command_lists = std::array{exec_cmds}});
            device.wait_idle();
        }
        catch (std::runtime_error error)
        {
            std::cout << "failed test \"acceleration_structure_creation\": " << error.what() << std::endl;
            exit(-1);
        }
    }
} // namespace tests

auto main() -> int
{
    {
        App app = {};
        tests::simplest(app);
    }
    {
        App app = {};
        tests::copy(app);
    }
    {
        App app = {};
        tests::deferred_destruction(app);
    }
    {
        App app = {};
        tests::multiple_ecl(app);
    }
    {
        App app = {};
        tests::build_acceleration_structure(app);
    }
    // Tests how long the version in ids can last for a single index.
    // {
    //     App app = {};
    //     tests::recreation(app);
    // }
    // For this test you must go into impl_command_list and comment out the id checks in one of the functions.
    // {
    //     App app = {};
    //     tests::ub_protection_perf(app);
    // }
}
