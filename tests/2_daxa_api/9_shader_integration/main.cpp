#include <iostream>

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>

#include <0_common/window.hpp>

#include "shaders/shared.inl"

namespace tests
{
    void aligned_types_templates()
    {
        daxa_f32vec3 shader_aligned_vec3{0.0f, 1.0f, 4.0f};
        shader_aligned_vec3 = {0.0f, 1.0f, 4.0f};
        shader_aligned_vec3.x = 0.0f;
        shader_aligned_vec3.y = 1.0f;
        shader_aligned_vec3.z = 4.0f;
        shader_aligned_vec3 = daxa_f32vec3{0.0f, 1.0f, 4.0f};
        shader_aligned_vec3 = daxa_f32vec3{0.0f, 1.0f, 4.0f};
        std::cout << "content of f32vec3: (" << shader_aligned_vec3.x << "," << shader_aligned_vec3.y << "," << shader_aligned_vec3.z << ")" << std::endl;
    }

    void alignment()
    {
        auto print_Testu64Alignment = [](TestU64Alignment const & v)
        {
            std::cout << "i0: " << v.i0 << std::endl;
            std::cout << "i1: " << v.i1 << std::endl;
            std::cout << "i2: " << v.i2 << std::endl;
            std::cout << "i3: " << v.i3 << std::endl;
            std::cout << "i4: " << v.i4 << std::endl;
            for (u32 i = 0; i < 3; ++i)
            {
                std::cout << "i5[" << i << "]: " << v.i5[i] << std::endl;
            }
            for (u32 i = 0; i < 3; ++i)
            {
                std::cout << "i6[" << i << "]: " << v.i5[i] << std::endl;
            }
            std::cout << "i7: " << v.i7 << std::endl;
        };
        [[maybe_unused]] auto are_same_Testu6Alignment = [](TestU64Alignment const & a, TestU64Alignment const & b)
        {
            bool same = true;
            same = same && a.i0 == b.i0;
            same = same && a.i1 == b.i1;
            same = same && a.i2 == b.i2;
            same = same && a.i3 == b.i3;
            same = same && a.i4 == b.i4;
            same = same && a.i5[0] == b.i5[0];
            same = same && a.i5[1] == b.i5[1];
            same = same && a.i5[2] == b.i5[2];
            same = same && a.i6[0] == b.i6[0];
            same = same && a.i6[1] == b.i6[1];
            same = same && a.i6[2] == b.i6[2];
            same = same && a.i7 == b.i7;
            return same;
        };

        // TEST:
        //  1) Create resources
        //  2) Use Compute dispatch read and write
        //  4) readback and validate
        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({},{}));
        auto src_buffer = device.create_buffer({
            .size = sizeof(TestU64Alignment),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = "align_test_src",
        });
        auto dst_buffer = device.create_buffer({
            .size = sizeof(TestU64Alignment),
            .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "align_test_dst",
        });

        TestU64Alignment const test_values{
            .i0 = 0,
            .i1 = 1,
            .i2 = 2,
            .i3 = 0xF000000000000003ull,
            .i4 = 4,
            .i5 = {0xF000000000000051ull, 0x0F00000000000052ull, 0x00F0000000000053ull},
            .i6 = {61, 62, 63},
            .i7 = 7,
        };
        *device.buffer_host_address_as<TestU64Alignment>(src_buffer).value() = test_values;

        daxa::PipelineManager pipeline_manager = daxa::PipelineManager({
            .device = device,
            .shader_compile_options = {
                .root_paths = {
                    DAXA_SHADER_INCLUDE_DIR,
                    "tests/2_daxa_api/9_shader_integration/shaders",
                },
            },
            .name = "pipeline manager",
        });

        auto compile_result = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"alignment_test.glsl"},
                .compile_options{
                    .enable_debug_info = true,
                },
            },
            .push_constant_size = sizeof(TestShaderTaskHead::AttachmentShaderBlob),
            .name = "compute_pipeline",
        });
        auto compute_pipeline = compile_result.value();

        auto task_graph = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "shader integration test - alignment",
        });

        auto src = daxa::TaskBuffer{{
            .initial_buffers = {.buffers = {&src_buffer, 1}},
            .name = "align_test_src",
        }};
        task_graph.use_persistent_buffer(src);
        auto dst = daxa::TaskBuffer{{
            .initial_buffers = {.buffers = {&dst_buffer, 1}},
            .name = "align_test_dst",
        }};
        task_graph.use_persistent_buffer(dst);

        struct TestTask : TestShaderTaskHead::Task
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
        task_graph.add_task(TestTask{
            .views = std::array{ 
                TestShaderTaskHead::AT.align_test_src | src,
                TestShaderTaskHead::AT.align_test_dst | dst,
            },
            .pipeline = compute_pipeline,
        });
        task_graph.submit({});
        task_graph.complete({});

        task_graph.execute({});

        device.wait_idle();

        [[maybe_unused]] TestU64Alignment readback_data = *device.buffer_host_address_as<TestU64Alignment>(dst_buffer).value();

        std::cout << "test values before: \n";
        print_Testu64Alignment(test_values);
        std::cout << "readback values after: \n";
        print_Testu64Alignment(readback_data);
        DAXA_DBG_ASSERT_TRUE_M(are_same_Testu6Alignment(test_values, readback_data), "values differ");

        device.destroy_buffer(src_buffer);
        device.destroy_buffer(dst_buffer);
    }

    void bindless_handles()
    {
        // TEST:
        //  1) create resources.
        //  2) pass handles to compute shader.
        //  3) perform showcase of bindless handle use in the shader.
        //  4) write handles to buffer.
        //  5) read and use handles in following compute shader.
        daxa::Instance daxa_ctx = daxa::create_instance({});
        daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({},{}));
        auto sampler = device.create_sampler({.name = "sampler"});

        daxa::PipelineManager pipeline_manager = daxa::PipelineManager({
            .device = device,
            .shader_compile_options = {
                .root_paths = {
                    DAXA_SHADER_INCLUDE_DIR,
                    "tests/2_daxa_api/9_shader_integration/shaders",
                },
            },
            .name = "pipeline manager",
        });

        auto compile_result0 = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"bindless_access.glsl"},
                .compile_options{
                    .enable_debug_info = true,
                },
            },
            .push_constant_size = sizeof(BindlessTestPush),
            .name = "bindless_access",
        });
        auto bindless_access = compile_result0.value();
        auto compile_result1 = pipeline_manager.add_compute_pipeline({
            .shader_info = {
                .source = daxa::ShaderFile{"bindless_access_followup.glsl"},
                .compile_options{
                    .enable_debug_info = true,
                },
            },
            .push_constant_size = sizeof(BindlessTestFollowPush),
            .name = "bindless_access_followup",
        });
        auto bindless_access_followup = compile_result1.value();

        auto task_graph = daxa::TaskGraph({
            .device = device,
            .record_debug_information = true,
            .name = "shader integration test - alignment",
        });

        auto handles_buffer = task_graph.create_transient_buffer({
            .size = sizeof(Handles),
            .name = "handles buffer",
        });
        auto f32_image = task_graph.create_transient_image({
            .format = daxa::Format::R32_SFLOAT,
            .size = {1, 1, 1},
            .name = "f32 image",
        });
        auto u32_image = task_graph.create_transient_image({
            .format = daxa::Format::R32_UINT,
            .size = {1, 1, 1},
            .name = "u32 image",
        });
        auto f32_buffer = task_graph.create_transient_buffer({
            .size = sizeof(f32),
            .name = "f32 buffer",
        });

        using BA = daxa::TaskBufferAccess;
        using IA = daxa::TaskImageAccess;

        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(BA::COMPUTE_SHADER_WRITE, handles_buffer),
                daxa::inl_attachment(BA::COMPUTE_SHADER_WRITE, f32_buffer),
                daxa::inl_attachment(IA::COMPUTE_SHADER_STORAGE_WRITE_ONLY, f32_image),
                daxa::inl_attachment(IA::COMPUTE_SHADER_STORAGE_READ_ONLY, u32_image),
            },
            .task = [=](daxa::TaskInterface ti)
            {
                ti.recorder.set_pipeline(*bindless_access);
                ti.recorder.push_constant(BindlessTestPush{
                    .handles = {
                        .my_buffer = ti.device_address(f32_buffer).value(),
                        .my_float_image = ti.get(f32_image).view_ids[0],
                        .my_uint_image = ti.get(u32_image).view_ids[0],
                        .my_sampler = sampler,
                    },
                    .next_shader_input = ti.device_address(handles_buffer).value(),
                });
                ti.recorder.dispatch({1, 1, 1});
            },
            .name = std::string_view{"bindless access"},
        });
        task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(BA::COMPUTE_SHADER_READ, handles_buffer),
                daxa::inl_attachment(BA::COMPUTE_SHADER_READ, f32_buffer),
                daxa::inl_attachment(IA::COMPUTE_SHADER_SAMPLED, f32_image),
            },
            .task = [=](daxa::TaskInterface ti)
            {
                ti.recorder.set_pipeline(*bindless_access_followup);
                ti.recorder.push_constant(BindlessTestFollowPush{
                    .shader_input = ti.device_address(handles_buffer).value(),
                });
                ti.recorder.dispatch({1, 1, 1});
            },
            .name = "bindless access",
        });
        task_graph.submit({});
        task_graph.complete({});
        task_graph.execute({});
        device.destroy_sampler(sampler);
    }
} // namespace tests

auto main() -> int
{
    tests::aligned_types_templates();
    tests::alignment();
    tests::bindless_handles();
}
