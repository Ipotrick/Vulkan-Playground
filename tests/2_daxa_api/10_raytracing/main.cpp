#include <iostream>
#include <thread>
#include <string>

#include <daxa/daxa.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>

#include <0_common/window.hpp>
#include <0_common/shared.hpp>

#include "shaders/shared.inl"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace tests
{
    void ray_query_triangle()
    {
        struct Camera
        {
            glm::vec3 position;
            glm::vec3 center;
            glm::vec3 up;
            float fov;
            unsigned int width;
            unsigned int height;
            float _near;
            float _far;
        };

        struct App : AppWindow<App>
        {
            daxa::Instance daxa_ctx = {};
            daxa::Device device = {};
            daxa::Swapchain swapchain = {};
            daxa::PipelineManager pipeline_manager = {};
            // std::shared_ptr<daxa::ComputePipeline> comp_pipeline = {};
            std::shared_ptr<daxa::RayTracingPipeline> rt_pipeline = {};
            daxa::TlasId tlas = {};
            daxa::BlasId blas = {};
            daxa::BlasId proc_blas = {};
            daxa::BufferId aabb_buffer = {};
            
            daxa::TaskBuffer blas_scratch_buffer = {};
            daxa_u64 blas_scratch_buffer_size = 1024 * 1024 * 10;
            daxa_u64 blas_scratch_buffer_offset = 0;
            daxa_u32 acceleration_structure_scratch_offset_alignment = 0;
            daxa::BufferId blas_buffer = {};
            daxa_u64 blas_buffer_size = 1024 * 1024 * 10;
            daxa_u64 blas_buffer_offset = 0;
            const daxa_u32 ACCELERATION_STRUCTURE_BUILD_OFFSET_ALIGMENT = 256; // NOTE: Requested by the spec
            bool atomic_float = false;

            daxa_u32 frame = 0;
            daxa_u32 raygen_shader_binding_table_offset = 0;

            Camera my_camera = {
                .position = {0.0f, 0.0f, -1.0f},
                .center = {0.0f, 0.0f, 0.0f},
                .up = {0.0f, 1.0f, 0.0f},
                .fov = 45.0f,
                .width = 800,
                .height = 600,
                ._near = 0.1f,
                ._far = 100.0f,
            };
            // BUFFERS
            daxa::BufferId cam_buffer = {};
            u32 cam_buffer_size = sizeof(CameraView);

            App() : AppWindow<App>("ray query test") {}

            ~App()
            {
                if (device.is_valid())
                {
                    device.destroy_tlas(tlas);
                    device.destroy_blas(blas);
                    device.destroy_blas(proc_blas);
                    device.destroy_buffer(cam_buffer);
                    device.destroy_buffer(aabb_buffer);
                    device.destroy_buffer(blas_buffer);
                }
            }

            void initialize()
            {
                daxa_ctx = daxa::create_instance({});
#if ACTIVATE_ATOMIC_FLOAT
                atomic_float = true;
#endif            
                device = [&]()
                {
                    daxa::DeviceInfo2 info = {};
                    daxa::ImplicitFeatureFlags required_implicit_features = daxa::ImplicitFeatureFlagBits::BASIC_RAY_TRACING | (atomic_float ? daxa::ImplicitFeatureFlagBits::SHADER_ATOMIC_FLOAT : daxa::ImplicitFeatureFlagBits::NONE);
                    info = daxa_ctx.choose_device(required_implicit_features, info);
                    return daxa_ctx.create_device_2(info);
                }();

                bool ray_tracing_supported = device.properties().ray_tracing_properties.has_value();
                auto invocation_reorder_mode = device.properties().invocation_reorder_properties.has_value() ? device.properties().invocation_reorder_properties.value().invocation_reorder_mode : 0;
                std::string ray_tracing_supported_str = ray_tracing_supported ? "available" : "not available";

                std::cout << "Choosen Device: " << device.properties().device_name <<
                            ", Ray Tracing: " <<  ray_tracing_supported_str <<
                            ", Invocation Reordering mode: " << invocation_reorder_mode  << std::endl;

                swapchain = device.create_swapchain({
                    .native_window = get_native_handle(),
                    .native_window_platform = get_native_platform(),
                    .surface_format_selector = [](daxa::Format format) -> i32
                    {
                        if (format == daxa::Format::B8G8R8A8_UNORM)
                        {
                            return 1000;
                        }
                        return 1;
                    },
                    .image_usage = daxa::ImageUsageFlagBits::SHADER_STORAGE,
                });

                acceleration_structure_scratch_offset_alignment = device.properties().acceleration_structure_properties.value().min_acceleration_structure_scratch_offset_alignment;


                /// Create Camera Buffer
                cam_buffer = device.create_buffer({
                    .size = cam_buffer_size,
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = ("cam_buffer"),
                });

                blas_scratch_buffer = daxa::TaskBuffer{device,{
                    .size = blas_scratch_buffer_size,
                    .name = ("blas_scratch_buffer"),
                }};

                blas_buffer = device.create_buffer({
                    .size = blas_buffer_size,
                    .name = ("blas_buffer"),
                });


                /// Alignments:
                auto get_aligned = [&](u64 operand, u64 granularity) -> u64
                {
                    return ((operand + (granularity - 1)) & ~(granularity - 1));
                };

                /// Vertices:
                auto vertices = std::array{
                    std::array{-0.25f, -0.25f, 0.5f},
                    std::array{0.0f, 0.25f, 0.5f},
                    std::array{0.25f, -0.25f, 0.5f},
                };
                auto vertex_buffer = device.create_buffer({
                    .size = sizeof(decltype(vertices)),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "vertex buffer",
                });
                defer { device.destroy_buffer(vertex_buffer); };
                // *device.buffer_host_address_as<decltype(vertices)>(vertex_buffer).value() = vertices;
                std::memcpy(device.buffer_host_address_as<daxa_f32vec3>(vertex_buffer).value(), vertices.data(), sizeof(daxa_f32vec3) * vertices.size());

                /// Indices:
                auto indices = std::array{0, 1, 2};
                auto index_buffer = device.create_buffer({
                    .size = sizeof(daxa_u32) * indices.size(),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "index buffer",
                });
                defer { device.destroy_buffer(index_buffer); };
                // *device.buffer_host_address_as<decltype(indices)>(index_buffer).value() = indices;
                std::memcpy(device.buffer_host_address_as<daxa_u32>(index_buffer).value(), indices.data(), sizeof(daxa_u32) * indices.size());
                /// Transforms:
                auto transform_buffer = device.create_buffer({
                    .size = sizeof(daxa_f32mat3x4),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "transform buffer",
                });
                defer { device.destroy_buffer(transform_buffer); };
                *device.buffer_host_address_as<daxa_f32mat3x4>(transform_buffer).value() = daxa_f32mat3x4{
                    {1, 0, 0, 0},
                    {0, 1, 0, 0},
                    {0, 0, 1, 0},
                };

                // TODO: The fact that we delayed some parameters to the create info and afterwards, we add them for building the acceleration structure cause a crash.
                // TODO: Could we improve the whole process?
                /// Triangle Geometry Info:
                auto geometries = std::array{
                    daxa::BlasTriangleGeometryInfo{
                        .vertex_format = daxa::Format::R32G32B32_SFLOAT, // Is also default
                        .vertex_data = device.device_address(vertex_buffer).value(),                               // Ignored in get_acceleration_structure_build_sizes.    // Is also default
                        .vertex_stride = sizeof(daxa_f32vec3),           // Is also default
                        .max_vertex = static_cast<u32>(vertices.size() - 1),
                        .index_type = daxa::IndexType::uint32, // Is also default
                        .index_data = device.device_address(index_buffer).value(),                      // Ignored in get_acceleration_structure_build_sizes. // Is also default
                        .transform_data = device.device_address(transform_buffer).value(),                  // Ignored in get_acceleration_structure_build_sizes. // Is also default
                        .count = static_cast<daxa_u32>(indices.size() / 3),
                        .flags = daxa::GeometryFlagBits::OPAQUE, // Is also default
                    }};
                /// Create Triangle Blas:
                auto blas_build_info = daxa::BlasBuildInfo{
                    .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE | daxa::AccelerationStructureBuildFlagBits::ALLOW_DATA_ACCESS,
                    .dst_blas = {}, // Ignored in get_acceleration_structure_build_sizes.       // Is also default
                    .geometries = geometries,
                    .scratch_data = {}, // Ignored in get_acceleration_structure_build_sizes.   // Is also default
                };
                daxa::AccelerationStructureBuildSizesInfo build_size_info = device.blas_build_sizes(blas_build_info);

                daxa_u64 scratch_alignment_size = get_aligned(build_size_info.build_scratch_size, acceleration_structure_scratch_offset_alignment);

                if ((blas_scratch_buffer_offset + scratch_alignment_size) > blas_scratch_buffer_size)
                {
                    // TODO: Try to resize buffer
                    std::cout << "blas_scratch_buffer_offset > blas_scratch_buffer_size" << std::endl;
                    abort();
                }
                blas_build_info.scratch_data = device.device_address(blas_scratch_buffer.get_state().buffers[0]).value() + blas_scratch_buffer_offset;
                blas_scratch_buffer_offset += scratch_alignment_size;

                daxa_u64 build_aligment_size = get_aligned(build_size_info.acceleration_structure_size, ACCELERATION_STRUCTURE_BUILD_OFFSET_ALIGMENT);

                if ((blas_buffer_offset + build_aligment_size) > blas_buffer_size)
                {
                    // TODO: Try to resize buffer
                    std::cout << "blas_buffer_offset > blas_buffer_size" << std::endl;
                    abort();
                }

                this->blas = device.create_blas_from_buffer({
                    .blas_info = {   
                        .size = build_size_info.acceleration_structure_size,
                        .name = "test blas",
                    },
                    .buffer_id = blas_buffer,
                    .offset = blas_buffer_offset,
                });
                blas_build_info.dst_blas = blas;
                blas_buffer_offset += build_aligment_size;

                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03792
                // GeometryType of each element of pGeometries must be the same

                // OPAQUE VS NO_DUPLICATE_ANY_HIT_INVOCATION DIFFERENT GEOMETRIES

                /// aabb data:
                auto min_max = std::array{
                    std::array{-0.25f, -0.25f, -0.25f},
                    std::array{-0.10f, -0.10f, -0.10f},
                    std::array{0.0f, 0.0f, 0.0f},
                    std::array{0.15f, 0.15f, 0.15f}};
                aabb_buffer = device.create_buffer({
                    .size = sizeof(decltype(min_max)),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "aabb buffer",
                });
                *device.buffer_host_address_as<decltype(min_max)>(aabb_buffer).value() = min_max;
                /// Procedural Geometry Info:
                auto proc_geometries = std::array{
                    daxa::BlasAabbGeometryInfo{
                        .data = device.device_address(aabb_buffer).value(),
                        .stride = sizeof(daxa_f32mat3x2),
                        .count = 1,
                        .flags = daxa::GeometryFlagBits::OPAQUE, // Is also default
                    },
                    daxa::BlasAabbGeometryInfo{
                        .data = device.device_address(aabb_buffer).value() + sizeof(daxa_f32mat3x2),
                        .stride = sizeof(daxa_f32mat3x2),
                        .count = 1,
                        .flags = daxa::GeometryFlagBits::NO_DUPLICATE_ANY_HIT_INVOCATION, // Is also default
                    }};
                /// Create Procedural Blas:
                auto proc_blas_build_info = daxa::BlasBuildInfo{
                    .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE, // Is also default
                    .dst_blas = {},                                                       // Ignored in get_acceleration_structure_build_sizes.       // Is also default
                    .geometries = proc_geometries,
                    .scratch_data = {}, // Ignored in get_acceleration_structure_build_sizes.   // Is also default
                };
                daxa::AccelerationStructureBuildSizesInfo proc_build_size_info = device.blas_build_sizes(proc_blas_build_info);
                
                scratch_alignment_size = get_aligned(proc_build_size_info.build_scratch_size, acceleration_structure_scratch_offset_alignment);
                if ((blas_scratch_buffer_offset + scratch_alignment_size) > blas_scratch_buffer_size)
                {
                    std::cout << "blas_scratch_buffer_offset > blas_scratch_buffer_size" << std::endl;
                    abort();
                }
                proc_blas_build_info.scratch_data = device.device_address(blas_scratch_buffer.get_state().buffers[0]).value() + blas_scratch_buffer_offset;
                blas_scratch_buffer_offset += scratch_alignment_size;

                
                build_aligment_size = get_aligned(proc_build_size_info.acceleration_structure_size, ACCELERATION_STRUCTURE_BUILD_OFFSET_ALIGMENT);

                if ((blas_buffer_offset + build_aligment_size) > blas_buffer_size)
                {
                    // TODO: Try to resize buffer
                    std::cout << "blas_buffer_offset > blas_buffer_size" << std::endl;
                    abort();
                }

                this->proc_blas = device.create_blas_from_buffer({
                    .blas_info = {   .size = build_size_info.acceleration_structure_size,
                        .name = "test procedural blas",
                    },
                    .buffer_id = blas_buffer,
                    .offset = blas_buffer_offset,
                });
                proc_blas_build_info.dst_blas = proc_blas;
                blas_buffer_offset += build_aligment_size;




                /// create blas instances for tlas:
                auto blas_instances_buffer = device.create_buffer({
                    .size = sizeof(daxa_BlasInstanceData) * 2,
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "blas instances array buffer",
                });
                defer { device.destroy_buffer(blas_instances_buffer); };

                auto blas_instance_array = std::array{
                    daxa_BlasInstanceData{
                        .transform = {
                            {1, 0, 0, 0},
                            {0, 1, 0, 0.0f},
                            {0, 0, 1, 0.5f},
                        },
                        .instance_custom_index = 0, // Is also default
                        .mask = 0xFF,
                        .instance_shader_binding_table_record_offset = 0,
                        .flags = {}, // Is also default
                        .blas_device_address = device.device_address(proc_blas).value(),
                    },
                    daxa_BlasInstanceData{
                        .transform = {
                            {1, 0, 0, 0.25f},
                            {0, 1, 0, -0.25f},
                            {0, 0, 1, 0.25f},
                        },
                        .instance_custom_index = 1, // Is also default
                        // .instance_custom_index = 2, // Is also default
                        .mask = 0xFF,
                        .instance_shader_binding_table_record_offset = 1,
                        .flags = DAXA_GEOMETRY_INSTANCE_FORCE_OPAQUE, // Is also default
                        .blas_device_address = device.device_address(blas).value(),
                    }};


                std::memcpy(device.buffer_host_address_as<daxa_BlasInstanceData>(blas_instances_buffer).value(),
                            blas_instance_array.data(),
                            blas_instance_array.size() * sizeof(daxa_BlasInstanceData));

                auto blas_instances = std::array{
                    daxa::TlasInstanceInfo{
                        .data = {}, // Ignored in get_acceleration_structure_build_sizes.   // Is also default
                        .count = 2,
                        .is_data_array_of_pointers = false,      // Buffer contains flat array of instances, not an array of pointers to instances.
                        .flags = daxa::GeometryFlagBits::OPAQUE, // Is also default
                    },
                };
                auto tlas_build_info = daxa::TlasBuildInfo{
                    .flags = daxa::AccelerationStructureBuildFlagBits::PREFER_FAST_TRACE,
                    .dst_tlas = {}, // Ignored in get_acceleration_structure_build_sizes.
                    .instances = blas_instances,
                    .scratch_data = {}, // Ignored in get_acceleration_structure_build_sizes.
                };
                daxa::AccelerationStructureBuildSizesInfo tlas_build_sizes = device.tlas_build_sizes(tlas_build_info);
                /// Create Tlas:
                this->tlas = device.create_tlas({
                    .size = tlas_build_sizes.acceleration_structure_size,
                    .name = "tlas",
                });
                /// Create Build Scratch buffer
                auto tlas_scratch_buffer = device.create_buffer({
                    .size = tlas_build_sizes.build_scratch_size,
                    .name = "tlas build scratch buffer",
                });
                defer { device.destroy_buffer(tlas_scratch_buffer); };
                /// Update build info:
                tlas_build_info.dst_tlas = tlas;
                tlas_build_info.scratch_data = device.device_address(tlas_scratch_buffer).value();
                blas_instances[0].data = device.device_address(blas_instances_buffer).value();
                
                /// Record build commands:
                auto exec_cmds = [&]()
                {
                    auto recorder = device.create_command_recorder({});
                    recorder.build_acceleration_structures({
                        .blas_build_infos = std::array{blas_build_info, proc_blas_build_info},
                    });
                    recorder.pipeline_barrier({
                        .src_access = daxa::AccessConsts::ACCELERATION_STRUCTURE_BUILD_WRITE,
                        .dst_access = daxa::AccessConsts::ACCELERATION_STRUCTURE_BUILD_READ_WRITE,
                    });
                    recorder.build_acceleration_structures({
                        .tlas_build_infos = std::array{tlas_build_info},
                    });
                    recorder.pipeline_barrier({
                        .src_access = daxa::AccessConsts::ACCELERATION_STRUCTURE_BUILD_WRITE,
                        .dst_access = daxa::AccessConsts::READ_WRITE,
                    });
                    return recorder.complete_current_commands();
                }();
                device.submit_commands({.command_lists = std::array{exec_cmds}});
                /// NOTE:
                /// No need to wait idle here.
                /// Daxa will defer all the destructions of the buffers until the submitted as build commands are complete.

                pipeline_manager = daxa::PipelineManager{daxa::PipelineManagerInfo{
                    .device = device,
                    .shader_compile_options = {
                        .root_paths = {
                            DAXA_SHADER_INCLUDE_DIR,
                            "tests/2_daxa_api/10_raytracing/shaders",
                        },
                    },
                }};
                // auto const compute_pipe_info = daxa::ComputePipelineCompileInfo{
                //     .shader_info = daxa::ShaderCompileInfo{
                //         .source = daxa::ShaderFile{"shaders.glsl"},
                //     },
                //     .push_constant_size = sizeof(PushConstant),
                //     .name = "ray query comp shader",
                // };
                // comp_pipeline = pipeline_manager.add_compute_pipeline(compute_pipe_info).value();


                daxa::ShaderCompileInfo prim_ray_gen_compile_info;
                daxa::ShaderCompileInfo ray_gen_compile_info;
                if(invocation_reorder_mode == static_cast<daxa_u32>(daxa::InvocationReorderMode::ALLOW_REORDER)) {
                    prim_ray_gen_compile_info = daxa::ShaderCompileInfo{
                        .source = daxa::ShaderFile{"raytracing.glsl"},
                        .compile_options = {
                            .defines = std::vector{daxa::ShaderDefine{"PRIMARY_RAYS", "1"}, daxa::ShaderDefine{"SER_ON", "1"}, atomic_float ?
                                daxa::ShaderDefine{"ATOMIC_FLOAT", "1"} : daxa::ShaderDefine{"ATOMIC_FLOAT", "0"
                            }},
                        },
                    };
                    ray_gen_compile_info = daxa::ShaderCompileInfo{
                        .source = daxa::ShaderFile{"raytracing.glsl"},
                        .compile_options = {
                            .defines = std::vector{daxa::ShaderDefine{"SER_ON", "1"}, atomic_float ? daxa::ShaderDefine{"ATOMIC_FLOAT", "1"} : daxa::ShaderDefine{"ATOMIC_FLOAT", "0"}}},
                    };
                } else {
                    prim_ray_gen_compile_info = daxa::ShaderCompileInfo{
                        .source = daxa::ShaderFile{"raytracing.glsl"},
                        .compile_options = {
                            .defines = std::vector{daxa::ShaderDefine{"PRIMARY_RAYS", "1"}, atomic_float ?
                                daxa::ShaderDefine{"ATOMIC_FLOAT", "1"} : daxa::ShaderDefine{"ATOMIC_FLOAT", "0"
                            }},
                        }
                    };
                    ray_gen_compile_info = daxa::ShaderCompileInfo{
                        .source = daxa::ShaderFile{"raytracing.glsl"},
                        .compile_options = {.defines = std::vector{daxa::ShaderDefine{"HIT_TRIANGLE", "1"}, atomic_float ?
                                daxa::ShaderDefine{"ATOMIC_FLOAT", "1"} : daxa::ShaderDefine{"ATOMIC_FLOAT", "0"
                            }}
                        },
                    };
                }

                
                auto const ray_tracing_pipe_info = daxa::RayTracingPipelineCompileInfo{
                    .ray_gen_infos = {
                        prim_ray_gen_compile_info,
                        ray_gen_compile_info
                    },
                    .intersection_infos = {daxa::ShaderCompileInfo{
                        .source = daxa::ShaderFile{"raytracing.glsl"},
                    }},
                    .any_hit_infos = {daxa::ShaderCompileInfo{
                        .source = daxa::ShaderFile{"raytracing.glsl"},
                    }},
                    .callable_infos = {
                        daxa::ShaderCompileInfo{
                            .source = daxa::ShaderFile{"raytracing.glsl"},
                            .compile_options = {
                                .defines = std::vector{daxa::ShaderDefine{"SPOT_LIGHT", "1"}}},
                        },
                        daxa::ShaderCompileInfo{
                            .source = daxa::ShaderFile{"raytracing.glsl"},
                        },
                    },
                    .closest_hit_infos = {
                        daxa::ShaderCompileInfo{
                            .source = daxa::ShaderFile{"raytracing.glsl"},
                        },
                        daxa::ShaderCompileInfo{
                            .source = daxa::ShaderFile{"raytracing.glsl"},
                            .compile_options = {.defines = std::vector{daxa::ShaderDefine{"HIT_TRIANGLE", "1"}, atomic_float ?
                                daxa::ShaderDefine{"ATOMIC_FLOAT", "1"} : daxa::ShaderDefine{"ATOMIC_FLOAT", "0"
                            }}
                        }},
                    },
                    .miss_hit_infos = {
                        daxa::ShaderCompileInfo{.source = daxa::ShaderFile{"raytracing.glsl"}},
                        daxa::ShaderCompileInfo{
                            .source = daxa::ShaderFile{"raytracing.glsl"},
                            .compile_options = {.defines = std::vector{daxa::ShaderDefine{"MISS_SHADOW", "1"}, atomic_float ?
                                daxa::ShaderDefine{"ATOMIC_FLOAT", "1"} : daxa::ShaderDefine{"ATOMIC_FLOAT", "0"
                            }}
                        }},
                    },
                    // Groups are in order of their shader indices.
                    // NOTE: The order of the groups is important! raygen, miss, hit, callable
                    .shader_groups_infos = {
                        daxa::RayTracingShaderGroupInfo{
                            .type = daxa::ShaderGroup::GENERAL,
                            .general_shader_index = 0,
                        },
                        daxa::RayTracingShaderGroupInfo{
                            .type = daxa::ShaderGroup::GENERAL,
                            .general_shader_index = 1,
                        },
                        daxa::RayTracingShaderGroupInfo{
                            .type = daxa::ShaderGroup::GENERAL,
                            .general_shader_index = 8,
                        },
                        daxa::RayTracingShaderGroupInfo{
                            .type = daxa::ShaderGroup::GENERAL,
                            .general_shader_index = 9,
                        },
                        daxa::RayTracingShaderGroupInfo{
                            .type = daxa::ShaderGroup::PROCEDURAL_HIT_GROUP,
                            .closest_hit_shader_index = 6,
                            .any_hit_shader_index = 3,
                            .intersection_shader_index = 2,
                        },
                        daxa::RayTracingShaderGroupInfo{
                            .type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP,
                            .closest_hit_shader_index = 7,
                        },
                        daxa::RayTracingShaderGroupInfo{
                            .type = daxa::ShaderGroup::GENERAL,
                            .general_shader_index = 4,
                        },
                        daxa::RayTracingShaderGroupInfo{
                            .type = daxa::ShaderGroup::GENERAL,
                            .general_shader_index = 5,
                        },
                    },
                    .max_ray_recursion_depth = 2,
                    .push_constant_size = sizeof(PushConstant),
                    .name = "basic ray tracing pipeline",
                };
                rt_pipeline = pipeline_manager.add_ray_tracing_pipeline(ray_tracing_pipe_info).value();
            }

            auto update() -> bool
            {
                auto reload_result = pipeline_manager.reload_all();

                if (auto * reload_err = daxa::get_if<daxa::PipelineReloadError>(&reload_result))
                    std::cout << reload_err->message << std::endl;
                else if (daxa::get_if<daxa::PipelineReloadSuccess>(&reload_result))
                    std::cout << "reload success" << std::endl;
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

            auto glm_mat4_to_daxa_f32mat4x4(glm::mat4 const & mat) -> daxa_f32mat4x4
            {
                return daxa_f32mat4x4{
                    {mat[0][0], mat[0][1], mat[0][2], mat[0][3]},
                    {mat[1][0], mat[1][1], mat[1][2], mat[1][3]},
                    {mat[2][0], mat[2][1], mat[2][2], mat[2][3]},
                    {mat[3][0], mat[3][1], mat[3][2], mat[3][3]},
                };
            }

            auto get_inverse_view_matrix(Camera const & cam) -> glm::mat4
            {
                glm::mat4 view = glm::lookAt(cam.position, cam.center, cam.up);
                return glm::inverse(view);
            }

            auto get_inverse_projection_matrix(Camera const & cam) -> glm::mat4
            {
                return glm::inverse(glm::perspective(glm::radians(cam.fov), static_cast<float>(cam.width) / static_cast<float>(cam.height), cam._near, cam._far));
            }

            void draw()
            {
                auto swapchain_image = swapchain.acquire_next_image();
                if (swapchain_image.is_empty())
                {
                    return;
                }
                daxa::u32 width = device.image_info(swapchain_image).value().size.x;
                daxa::u32 height = device.image_info(swapchain_image).value().size.y;

                my_camera.width = width;
                my_camera.height = height;

                // Update camera buffer
                CameraView camera_view = {
                    .inv_view = glm_mat4_to_daxa_f32mat4x4(get_inverse_view_matrix(my_camera)),
                    .inv_proj = glm_mat4_to_daxa_f32mat4x4(get_inverse_projection_matrix(my_camera))
#if ACTIVATE_ATOMIC_FLOAT 
                    ,.hit_count = 0.0f,
#endif
                    };

                // NOTE: Vulkan has inverted y axis in NDC
                camera_view.inv_proj.y.y *= -1;

                auto cam_staging_buffer = device.create_buffer({
                    .size = cam_buffer_size,
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = ("cam_staging_buffer"),
                });
                defer { device.destroy_buffer(cam_staging_buffer); };

                auto * buffer_ptr = device.buffer_host_address_as<uint8_t>(cam_staging_buffer).value();
                std::memcpy(buffer_ptr, &camera_view, cam_buffer_size);

                auto recorder = device.create_command_recorder({
                    .name = ("recorder (clearcolor)"),
                });

                recorder.pipeline_barrier({
                    .src_access = daxa::AccessConsts::HOST_WRITE,
                    .dst_access = daxa::AccessConsts::TRANSFER_READ,
                });

                recorder.copy_buffer_to_buffer({
                    .src_buffer = cam_staging_buffer,
                    .dst_buffer = cam_buffer,
                    .size = cam_buffer_size,
                });

                recorder.pipeline_barrier_image_transition({
                    .dst_access = daxa::AccessConsts::RAY_TRACING_SHADER_WRITE,
                    .src_layout = daxa::ImageLayout::UNDEFINED,
                    .dst_layout = daxa::ImageLayout::GENERAL,
                    .image_id = swapchain_image,
                });

                recorder.set_pipeline(*rt_pipeline);

                recorder.push_constant(PushConstant{
                    .frame = frame++,
                    .size = {width, height},
                    .tlas = tlas,
                    .swapchain = swapchain_image.default_view(),
                    .camera_buffer = this->device.device_address(cam_buffer).value(),
                    .aabb_buffer = this->device.device_address(aabb_buffer).value(),
                });

                recorder.trace_rays({
                    .width = width,
                    .height = height,
                    .depth = 1,
                    .raygen_shader_binding_table_offset = raygen_shader_binding_table_offset,
                });

#if ACTIVATE_ATOMIC_FLOAT == 1      
                recorder.pipeline_barrier({
                    .src_access = daxa::AccessConsts::RAY_TRACING_SHADER_WRITE,
                    .dst_access = daxa::AccessConsts::TRANSFER_READ,
                });

                recorder.copy_buffer_to_buffer({
                    .src_buffer = cam_buffer,
                    .dst_buffer = cam_staging_buffer,
                    .size = cam_buffer_size,
                });

                recorder.pipeline_barrier({
                    .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                    .dst_access = daxa::AccessConsts::HOST_READ,
                });
#endif // ACTIVATE_ATOMIC_FLOAT

                recorder.pipeline_barrier_image_transition({
                    .src_access = daxa::AccessConsts::RAY_TRACING_SHADER_WRITE,
                    .src_layout = daxa::ImageLayout::GENERAL,
                    .dst_layout = daxa::ImageLayout::PRESENT_SRC,
                    .image_id = swapchain_image,
                });

                // recorder.pipeline_barrier({
                //     .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                //     .dst_access = daxa::AccessConsts::RAY_TRACING_SHADER_READ,
                // });

                auto executable_commands = recorder.complete_current_commands();
                /// NOTE:
                /// Must destroy the command recorder here as we call collect_garbage later in this scope!
                recorder.~CommandRecorder();

                device.submit_commands({
                    .command_lists = std::array{executable_commands},
                    .wait_binary_semaphores = std::array{swapchain.current_acquire_semaphore()},
                    .signal_binary_semaphores = std::array{swapchain.current_present_semaphore()},
                    .signal_timeline_semaphores = std::array{swapchain.current_timeline_pair()},
                });

                device.present_frame({
                    .wait_binary_semaphores = std::array{swapchain.current_present_semaphore()},
                    .swapchain = swapchain,
                });

#if ACTIVATE_ATOMIC_FLOAT == 1
                device.wait_idle();
                float hit_count = 0.0f;
                std::memcpy(&hit_count, buffer_ptr + sizeof(CameraView) - sizeof(float), sizeof(float));
                std::cout << "hit count: " << hit_count << " while ray tracing" << std::endl;
#endif // ACTIVATE_ATOMIC_FLOAT

                device.collect_garbage();
            }

            void on_mouse_move(f32 /*unused*/, f32 /*unused*/) {}
            void on_mouse_button(i32 /*unused*/, i32 /*unused*/) {}
            void on_key(i32 key, i32 action) {
                if(key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
                    raygen_shader_binding_table_offset = (raygen_shader_binding_table_offset + 1) % 2;
                }
            }

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

        App app;
        try
        {
            app.initialize();
        }
        catch (std::runtime_error err)
        {
            std::cout << "Failed initialization: \"" << err.what() << "\"" << std::endl;
            return;
        }
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
    // TODO(Raytracing): Add acceleration structure updates.
    tests::ray_query_triangle();
    return 0;
}
