#include "impl_device.hpp"
#include "impl_pipeline.hpp"

// --- Begin API Functions ---

auto daxa_dvc_create_raster_pipeline(daxa_Device device, daxa_RasterPipelineInfo const * info, daxa_RasterPipeline * out_pipeline) -> daxa_Result
{
    _DAXA_TEST_PRINT("daxa_dvc_create_raster_pipeline\n");
    daxa_ImplRasterPipeline ret = {};
    ret.device = device;
    ret.info = *reinterpret_cast<RasterPipelineInfo const *>(info);
    std::vector<VkShaderModule> vk_shader_modules = {};
    // NOTE: Temporarily holds 0 terminated strings, incoming strings are data + size, not null terminated!
    std::vector<std::unique_ptr<std::string>> entry_point_names = {};
    std::vector<VkPipelineShaderStageCreateInfo> vk_pipeline_shader_stage_create_infos = {};

    std::vector<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo> require_subgroup_size_vkstructs = {};
    // Necessary to prevent re-allocation
    auto const MAXIMUM_GRAPHICS_STAGES = 6;
    require_subgroup_size_vkstructs.reserve(MAXIMUM_GRAPHICS_STAGES);

    auto create_shader_module = [&](ShaderInfo const & shader_info, VkShaderStageFlagBits shader_stage) -> VkResult
    {
        VkShaderModule vk_shader_module = nullptr;
        VkShaderModuleCreateInfo const vk_shader_module_create_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = {},
            .codeSize = static_cast<u32>(shader_info.byte_code_size * sizeof(u32)),
            .pCode = shader_info.byte_code,
        };
        auto result = vkCreateShaderModule(ret.device->vk_device, &vk_shader_module_create_info, nullptr, &vk_shader_module);
        if (result != VK_SUCCESS)
        {
            return result;
        }
        vk_shader_modules.push_back(vk_shader_module);
        entry_point_names.push_back(std::make_unique<std::string>(shader_info.entry_point.view().begin(), shader_info.entry_point.view().end()));
        require_subgroup_size_vkstructs.push_back({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
            .pNext = nullptr,
            .requiredSubgroupSize = shader_info.required_subgroup_size.value_or(0),
        });
        VkPipelineShaderStageCreateInfo const vk_pipeline_shader_stage_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = shader_info.required_subgroup_size.has_value() ? &require_subgroup_size_vkstructs.back() : nullptr,
            .flags = std::bit_cast<VkPipelineShaderStageCreateFlags>(shader_info.create_flags),
            .stage = shader_stage,
            .module = vk_shader_module,
            .pName = entry_point_names.back()->c_str(),
            .pSpecializationInfo = nullptr,
        };
        vk_pipeline_shader_stage_create_infos.push_back(vk_pipeline_shader_stage_create_info);
        return result;
    };

#define DAXA_DECL_TRY_CREATE_MODULE(name, NAME)                                                                                 \
    if (ret.info.name##_shader_info.has_value())                                                                                \
    {                                                                                                                           \
        auto result = create_shader_module(ret.info.name##_shader_info.value(), VkShaderStageFlagBits::VK_SHADER_STAGE_##NAME); \
        if (result != VK_SUCCESS)                                                                                               \
        {                                                                                                                       \
            for (auto module : vk_shader_modules)                                                                               \
            {                                                                                                                   \
                vkDestroyShaderModule(ret.device->vk_device, module, nullptr);                                                  \
            }                                                                                                                   \
            return std::bit_cast<daxa_Result>(result);                                                                          \
        }                                                                                                                       \
    }
    DAXA_DECL_TRY_CREATE_MODULE(vertex, VERTEX_BIT)
    DAXA_DECL_TRY_CREATE_MODULE(tesselation_control, TESSELLATION_CONTROL_BIT)
    DAXA_DECL_TRY_CREATE_MODULE(tesselation_evaluation, TESSELLATION_EVALUATION_BIT)
    DAXA_DECL_TRY_CREATE_MODULE(fragment, FRAGMENT_BIT)
    if ((ret.device->properties.implicit_features & ImplicitFeatureFlagBits::MESH_SHADER) != DeviceFlagBits::NONE)
    {
        DAXA_DECL_TRY_CREATE_MODULE(task, TASK_BIT_EXT)
        DAXA_DECL_TRY_CREATE_MODULE(mesh, MESH_BIT_EXT)
    }
    else
    {
        if (ret.info.mesh_shader_info.has_value() || ret.info.task_shader_info.has_value())
        {
            for (auto module : vk_shader_modules)
            {
                vkDestroyShaderModule(ret.device->vk_device, module, nullptr);
            }
            return DAXA_RESULT_MESH_SHADER_NOT_DEVICE_ENABLED;
        }
    }

    ret.vk_pipeline_layout = ret.device->gpu_sro_table.pipeline_layouts.at((ret.info.push_constant_size + 3) / 4);
    constexpr VkPipelineVertexInputStateCreateInfo vk_vertex_input_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };
    VkPipelineInputAssemblyStateCreateInfo const vk_input_assembly_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .topology = *reinterpret_cast<VkPrimitiveTopology const *>(&ret.info.raster.primitive_topology),
        .primitiveRestartEnable = static_cast<VkBool32>(ret.info.raster.primitive_restart_enable),
    };
    auto no_tess = TesselationInfo{};
    VkPipelineTessellationDomainOriginStateCreateInfo const vk_tesselation_domain_origin_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO,
        .pNext = nullptr,
        .domainOrigin = *reinterpret_cast<VkTessellationDomainOrigin const *>(&ret.info.tesselation.value_or(no_tess).origin),
    };
    VkPipelineTessellationStateCreateInfo const vk_tesselation_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .pNext = reinterpret_cast<void const *>(&vk_tesselation_domain_origin_state),
        .flags = {},
        .patchControlPoints = ret.info.tesselation.value_or(no_tess).control_points,
    };
    VkPipelineMultisampleStateCreateInfo const vk_multisample_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .rasterizationSamples =
            info->raster.static_state_sample_count.has_value
                ? info->raster.static_state_sample_count.value
                : VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .pSampleMask = {},
        .alphaToCoverageEnable = {},
        .alphaToOneEnable = {},
    };
    VkPipelineRasterizationStateCreateInfo vk_raster_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .depthClampEnable = static_cast<VkBool32>(ret.info.raster.depth_clamp_enable),
        .rasterizerDiscardEnable = static_cast<VkBool32>(ret.info.raster.rasterizer_discard_enable),
        .polygonMode = *reinterpret_cast<VkPolygonMode const *>(&ret.info.raster.polygon_mode),
        .cullMode = *reinterpret_cast<VkCullModeFlags const *>(&ret.info.raster.face_culling),
        .frontFace = *reinterpret_cast<VkFrontFace const *>(&ret.info.raster.front_face_winding),
        .depthBiasEnable = static_cast<VkBool32>(ret.info.raster.depth_bias_enable),
        .depthBiasConstantFactor = ret.info.raster.depth_bias_constant_factor,
        .depthBiasClamp = ret.info.raster.depth_bias_clamp,
        .depthBiasSlopeFactor = ret.info.raster.depth_bias_slope_factor,
        .lineWidth = ret.info.raster.line_width,
    };
    auto vk_conservative_raster_state = VkPipelineRasterizationConservativeStateCreateInfoEXT{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = {},
        .conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT,
        .extraPrimitiveOverestimationSize = 0.0f,
    };
    if (
        ret.info.raster.conservative_raster_info.has_value() &&
        (device->properties.implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_CONSERVATIVE_RASTERIZATION))
    {
        // TODO(grundlett): Ask Patrick why this doesn't work
        // auto vk_instance = ret.device->instance->vk_instance;
        // PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR =
        //     reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(vkGetInstanceProcAddr(vk_instance, "vkGetPhysicalDeviceProperties2KHR"));
        // DAXA_DBG_ASSERT_TRUE_M(vkGetPhysicalDeviceProperties2KHR != nullptr, "Failed to load this extension function function");
        // VkPhysicalDeviceProperties2KHR device_props2{};
        // VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservative_raster_props{};
        // conservative_raster_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT;
        // device_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
        // device_props2.pNext = &conservative_raster_props;
        // vkGetPhysicalDeviceProperties2KHR(ret.device->vk_physical_device, &device_props2);
        auto const & conservative_raster_info = ret.info.raster.conservative_raster_info.value();
        vk_conservative_raster_state.conservativeRasterizationMode = static_cast<VkConservativeRasterizationModeEXT>(conservative_raster_info.mode);
        vk_conservative_raster_state.extraPrimitiveOverestimationSize = conservative_raster_info.size;
        vk_raster_state.pNext = &vk_conservative_raster_state;
    }
    DepthTestInfo const no_depth = {};
    VkPipelineDepthStencilStateCreateInfo const vk_depth_stencil_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .depthTestEnable = static_cast<VkBool32>(ret.info.depth_test.has_value()),
        .depthWriteEnable = static_cast<VkBool32>(ret.info.depth_test.value_or(no_depth).enable_depth_write),
        .depthCompareOp = static_cast<VkCompareOp>(ret.info.depth_test.value_or(no_depth).depth_test_compare_op),
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {},
        .back = {},
        .minDepthBounds = ret.info.depth_test.value_or(no_depth).min_depth_bounds,
        .maxDepthBounds = ret.info.depth_test.value_or(no_depth).max_depth_bounds,
    };
    // TODO(capi): DO NOT THROW IN C FUNCTION
    // DAXA_DBG_ASSERT_TRUE_M(ret.info.color_attachments.size() < pipeline_manager_MAX_ATTACHMENTS, "too many color attachments, make pull request to bump max");
    std::array<VkPipelineColorBlendAttachmentState, pipeline_manager_MAX_ATTACHMENTS> vk_pipeline_color_blend_attachment_blend_states = {};
    auto no_blend = BlendInfo{};
    for (FixedListSizeT i = 0; i < ret.info.color_attachments.size(); ++i)
    {
        vk_pipeline_color_blend_attachment_blend_states.at(i) = VkPipelineColorBlendAttachmentState{
            .blendEnable = static_cast<VkBool32>(ret.info.color_attachments.at(i).blend.has_value()),
            .srcColorBlendFactor = static_cast<VkBlendFactor>(ret.info.color_attachments.at(i).blend.value_or(no_blend).src_color_blend_factor),
            .dstColorBlendFactor = static_cast<VkBlendFactor>(ret.info.color_attachments.at(i).blend.value_or(no_blend).dst_color_blend_factor),
            .colorBlendOp = static_cast<VkBlendOp>(ret.info.color_attachments.at(i).blend.value_or(no_blend).color_blend_op),
            .srcAlphaBlendFactor = static_cast<VkBlendFactor>(ret.info.color_attachments.at(i).blend.value_or(no_blend).src_alpha_blend_factor),
            .dstAlphaBlendFactor = static_cast<VkBlendFactor>(ret.info.color_attachments.at(i).blend.value_or(no_blend).dst_alpha_blend_factor),
            .alphaBlendOp = static_cast<VkBlendOp>(ret.info.color_attachments.at(i).blend.value_or(no_blend).alpha_blend_op),
            .colorWriteMask = std::bit_cast<VkColorComponentFlags>(ret.info.color_attachments.at(i).blend.value_or(no_blend).color_write_mask),
        };
    }
    std::array<VkFormat, pipeline_manager_MAX_ATTACHMENTS> vk_pipeline_color_attachment_formats = {};
    for (FixedListSizeT i = 0; i < ret.info.color_attachments.size(); ++i)
    {
        vk_pipeline_color_attachment_formats.at(i) = std::bit_cast<VkFormat>(ret.info.color_attachments.at(i).format);
    }
    VkPipelineColorBlendStateCreateInfo const vk_color_blend_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .logicOpEnable = VK_FALSE,
        .logicOp = {},
        .attachmentCount = static_cast<u32>(ret.info.color_attachments.size()),
        .pAttachments = vk_pipeline_color_blend_attachment_blend_states.data(),
        .blendConstants = {1.0f, 1.0f, 1.0f, 1.0f},
    };
    constexpr VkViewport DEFAULT_VIEWPORT{.x = 0, .y = 0, .width = 1, .height = 1, .minDepth = 0, .maxDepth = 0};
    constexpr VkRect2D DEFAULT_SCISSOR{.offset = {0, 0}, .extent = {1, 1}};
    VkPipelineViewportStateCreateInfo const vk_viewport_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .viewportCount = 1,
        .pViewports = &DEFAULT_VIEWPORT,
        .scissorCount = 1,
        .pScissors = &DEFAULT_SCISSOR,
    };
    auto dynamic_state = std::vector{
        VkDynamicState::VK_DYNAMIC_STATE_VIEWPORT,
        VkDynamicState::VK_DYNAMIC_STATE_SCISSOR,
        VkDynamicState::VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    if ((device->properties.implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_DYNAMIC_STATE_3) &&
        !info->raster.static_state_sample_count.has_value)
    {
        dynamic_state.push_back(VK_DYNAMIC_STATE_RASTERIZATION_SAMPLES_EXT);
    }
    VkPipelineDynamicStateCreateInfo const vk_dynamic_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .dynamicStateCount = static_cast<u32>(dynamic_state.size()),
        .pDynamicStates = dynamic_state.data(),
    };
    VkPipelineRenderingCreateInfo vk_pipeline_rendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .pNext = nullptr,
        .viewMask = {},
        .colorAttachmentCount = static_cast<u32>(ret.info.color_attachments.size()),
        .pColorAttachmentFormats = vk_pipeline_color_attachment_formats.data(),
        .depthAttachmentFormat = static_cast<VkFormat>(ret.info.depth_test.value_or(no_depth).depth_attachment_format),
        .stencilAttachmentFormat = {},
    };
    VkGraphicsPipelineCreateInfo const vk_graphics_pipeline_create_info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &vk_pipeline_rendering,
        .flags = {},
        .stageCount = static_cast<u32>(vk_pipeline_shader_stage_create_infos.size()),
        .pStages = vk_pipeline_shader_stage_create_infos.data(),
        .pVertexInputState = &vk_vertex_input_state,
        .pInputAssemblyState = &vk_input_assembly_state,
        .pTessellationState = &vk_tesselation_state,
        .pViewportState = &vk_viewport_state,
        .pRasterizationState = &vk_raster_state,
        .pMultisampleState = &vk_multisample_state,
        .pDepthStencilState = &vk_depth_stencil_state,
        .pColorBlendState = &vk_color_blend_state,
        .pDynamicState = &vk_dynamic_state,
        .layout = ret.vk_pipeline_layout,
        .renderPass = nullptr,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };
    auto result = vkCreateGraphicsPipelines(
        ret.device->vk_device,
        VK_NULL_HANDLE,
        1u,
        &vk_graphics_pipeline_create_info,
        nullptr,
        &ret.vk_pipeline);
    for (auto & vk_shader_module : vk_shader_modules)
    {
        vkDestroyShaderModule(ret.device->vk_device, vk_shader_module, nullptr);
    }
    if (result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(result);
    }
    if ((ret.device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE && !ret.info.name.empty())
    {
        auto name_cstr = ret.info.name.c_str();
        VkDebugUtilsObjectNameInfoEXT const name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_PIPELINE,
            .objectHandle = std::bit_cast<u64>(ret.vk_pipeline),
            .pObjectName = name_cstr.data(),
        };
        ret.device->vkSetDebugUtilsObjectNameEXT(ret.device->vk_device, &name_info);
    }
    ret.strong_count = 1;
    device->inc_weak_refcnt();
    *out_pipeline = new daxa_ImplRasterPipeline{};
    **out_pipeline = ret;
    return DAXA_RESULT_SUCCESS;
}

auto daxa_raster_pipeline_info(daxa_RasterPipeline self) -> daxa_RasterPipelineInfo const *
{
    return reinterpret_cast<daxa_RasterPipelineInfo const *>(&self->info);
}

auto daxa_raster_pipeline_inc_refcnt(daxa_RasterPipeline self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_raster_pipeline_dec_refcnt(daxa_RasterPipeline self) -> u64
{
    return self->dec_refcnt(
        &ImplPipeline::zero_ref_callback,
        self->device->instance);
}

auto daxa_dvc_create_compute_pipeline(daxa_Device device, daxa_ComputePipelineInfo const * info, daxa_ComputePipeline * out_pipeline) -> daxa_Result
{
    _DAXA_TEST_PRINT("daxa_dvc_create_compute_pipeline\n");
    daxa_ImplComputePipeline ret = {};
    ret.device = device;
    ret.info = *reinterpret_cast<ComputePipelineInfo const *>(info);
    VkShaderModule vk_shader_module = {};
    VkShaderModuleCreateInfo const shader_module_ci{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .codeSize = ret.info.shader_info.byte_code_size * static_cast<u32>(sizeof(u32)),
        .pCode = ret.info.shader_info.byte_code,
    };
    auto module_result = vkCreateShaderModule(ret.device->vk_device, &shader_module_ci, nullptr, &vk_shader_module);
    if (module_result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(module_result);
    }
    ret.vk_pipeline_layout = ret.device->gpu_sro_table.pipeline_layouts.at((ret.info.push_constant_size + 3) / 4);
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo require_subgroup_size_vkstruct{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = nullptr,
        .requiredSubgroupSize = ret.info.shader_info.required_subgroup_size.value_or(0),
    };
    VkComputePipelineCreateInfo const vk_compute_pipeline_create_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .stage = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = ret.info.shader_info.required_subgroup_size.has_value() ? &require_subgroup_size_vkstruct : nullptr,
            .flags = std::bit_cast<VkPipelineShaderStageCreateFlags>(ret.info.shader_info.create_flags),
            .stage = VkShaderStageFlagBits::VK_SHADER_STAGE_COMPUTE_BIT,
            .module = vk_shader_module,
            .pName = ret.info.shader_info.entry_point.data(),
            .pSpecializationInfo = nullptr,
        },
        .layout = ret.vk_pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };
    auto pipeline_result = vkCreateComputePipelines(
        ret.device->vk_device,
        VK_NULL_HANDLE,
        1u,
        &vk_compute_pipeline_create_info,
        nullptr,
        &ret.vk_pipeline);
    vkDestroyShaderModule(ret.device->vk_device, vk_shader_module, nullptr);
    if (pipeline_result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(pipeline_result);
    }
    if ((ret.device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE && !ret.info.name.view().empty())
    {
        auto name_cstr = ret.info.name.c_str();
        VkDebugUtilsObjectNameInfoEXT const name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_PIPELINE,
            .objectHandle = std::bit_cast<uint64_t>(ret.vk_pipeline),
            .pObjectName = name_cstr.data(),
        };
        ret.device->vkSetDebugUtilsObjectNameEXT(ret.device->vk_device, &name_info);
    }
    ret.strong_count = 1;
    device->inc_weak_refcnt();
    *out_pipeline = new daxa_ImplComputePipeline{};
    **out_pipeline = ret;
    return DAXA_RESULT_SUCCESS;
}

auto daxa_compute_pipeline_info(daxa_ComputePipeline self) -> daxa_ComputePipelineInfo const *
{
    return reinterpret_cast<daxa_ComputePipelineInfo const *>(&self->info);
}

auto daxa_compute_pipeline_inc_refcnt(daxa_ComputePipeline self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_compute_pipeline_dec_refcnt(daxa_ComputePipeline self) -> u64
{
    return self->dec_refcnt(
        &ImplPipeline::zero_ref_callback,
        self->device->instance);
}

auto daxa_dvc_create_ray_tracing_pipeline(daxa_Device device, daxa_RayTracingPipelineInfo const * info, daxa_RayTracingPipeline * out_pipeline) -> daxa_Result
{
    _DAXA_TEST_PRINT("daxa_dvc_create_ray_tracing_pipeline\n");
    daxa_ImplRayTracingPipeline ret = {};
    ret.device = device;
    ret.info = *reinterpret_cast<RayTracingPipelineInfo const *>(info);

    ret.shader_groups.resize(ret.info.shader_groups.size());
    for (int i = 0; i < ret.shader_groups.size(); ++i)
    {
        ret.shader_groups[i] = ret.info.shader_groups[i];
    }

    // Check if ray tracing is supported
    if ((device->properties.implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_BASIC_RAY_TRACING) == 0)
    {
        return DAXA_RESULT_INVALID_WITHOUT_ENABLING_RAY_TRACING;
    }

    // Stages are the shader modules
    std::vector<VkPipelineShaderStageCreateInfo> stages = {};
    // Shader groups are made of stages targets: [general | triangles | procedural], types: [general | closest hit + any hit + intersection]
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups = {};

    std::vector<VkShaderModule> vk_shader_modules = {};
    // NOTE: Temporarily holds 0 terminated strings, incoming strings are data + size, not null terminated!
    std::vector<std::unique_ptr<std::string>> entry_point_names = {};

    defer
    {
        for (auto & vk_shader_module : vk_shader_modules)
        {
            vkDestroyShaderModule(ret.device->vk_device, vk_shader_module, nullptr);
        }
    };

    u32 const raygen_count = static_cast<u32>(ret.info.ray_gen_shaders.size());
    u32 const intersection_count = static_cast<u32>(ret.info.intersection_shaders.size());
    u32 const any_hit_count = static_cast<u32>(ret.info.any_hit_shaders.size());
    u32 const callable_count = static_cast<u32>(ret.info.callable_shaders.size());
    u32 const closest_hit_count = static_cast<u32>(ret.info.closest_hit_shaders.size());
    u32 const miss_hit_count = static_cast<u32>(ret.info.miss_hit_shaders.size());
    u32 const all_stages_count = raygen_count + intersection_count + any_hit_count + callable_count + closest_hit_count + miss_hit_count;
    u32 const first_callable_index = raygen_count + intersection_count + any_hit_count;
    u32 const last_callable_index = raygen_count + intersection_count + any_hit_count + callable_count;
    u32 const first_miss_index = raygen_count + intersection_count + any_hit_count + callable_count + closest_hit_count;

    std::vector<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo> require_subgroup_size_vkstructs = {};
    // Necessary to prevent re-allocation
    require_subgroup_size_vkstructs.reserve(all_stages_count);

    auto create_shader_module = [&](ShaderInfo const & shader_info, VkShaderStageFlagBits shader_stage) -> VkResult
    {
        VkShaderModule vk_shader_module = nullptr;
        VkShaderModuleCreateInfo const vk_shader_module_create_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = {},
            .codeSize = static_cast<u32>(shader_info.byte_code_size * sizeof(u32)),
            .pCode = shader_info.byte_code,
        };
        auto result = vkCreateShaderModule(ret.device->vk_device, &vk_shader_module_create_info, nullptr, &vk_shader_module);
        if (result != VK_SUCCESS)
        {
            return result;
        }
        vk_shader_modules.push_back(vk_shader_module);
        entry_point_names.push_back(std::make_unique<std::string>(shader_info.entry_point.view().begin(), shader_info.entry_point.view().end()));
        require_subgroup_size_vkstructs.push_back({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
            .pNext = nullptr,
            .requiredSubgroupSize = shader_info.required_subgroup_size.value_or(0),
        });
        VkPipelineShaderStageCreateInfo const vk_pipeline_shader_stage_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = shader_info.required_subgroup_size.has_value() ? &require_subgroup_size_vkstructs.back() : nullptr,
            .flags = std::bit_cast<VkPipelineShaderStageCreateFlags>(shader_info.create_flags),
            .stage = shader_stage,
            .module = vk_shader_module,
            .pName = entry_point_names.back()->c_str(),
            .pSpecializationInfo = nullptr,
        };
        stages.push_back(vk_pipeline_shader_stage_create_info);
        return result;
    };

#define DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage, NAME)                                            \
    auto result = create_shader_module(stage, VkShaderStageFlagBits::VK_SHADER_STAGE_##NAME##_BIT_KHR); \
    if (result != VK_SUCCESS)                                                                           \
    {                                                                                                   \
        return std::bit_cast<daxa_Result>(result);                                                      \
    }

    for (FixedListSizeT i = 0; i < raygen_count; ++i)
    {
        auto stage = ret.info.ray_gen_shaders.at(i);
        DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage, RAYGEN)
    }

    for (FixedListSizeT i = 0; i < intersection_count; ++i)
    {
        auto stage = ret.info.intersection_shaders.at(i);
        DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage, INTERSECTION)
    }

    for (FixedListSizeT i = 0; i < any_hit_count; ++i)
    {
        auto stage = ret.info.any_hit_shaders.at(i);
        DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage, ANY_HIT)
    }

    for (FixedListSizeT i = 0; i < callable_count; ++i)
    {
        auto stage = ret.info.callable_shaders.at(i);
        DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage, CALLABLE)
    }

    for (FixedListSizeT i = 0; i < closest_hit_count; ++i)
    {
        auto stage = ret.info.closest_hit_shaders.at(i);
        DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage, CLOSEST_HIT)
    }

    for (FixedListSizeT i = 0; i < miss_hit_count; ++i)
    {
        auto stage = ret.info.miss_hit_shaders.at(i);
        DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage, MISS)
    }

    // Shader groups
    for (u32 i = 0; i < ret.info.shader_groups.size(); ++i)
    {
        auto shader_group = ret.info.shader_groups.at(i);
        auto const group = VkRayTracingShaderGroupCreateInfoKHR{
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .pNext = nullptr,
            .type = static_cast<VkRayTracingShaderGroupTypeKHR>(shader_group.type),
            .generalShader = shader_group.general_shader_index,
            .closestHitShader = shader_group.closest_hit_shader_index,
            .anyHitShader = shader_group.any_hit_shader_index,
            .intersectionShader = shader_group.intersection_shader_index,
        };
        groups.push_back(group);
    }

    u32 const group_count = static_cast<u32>(groups.size());
    u32 const stages_count = static_cast<u32>(stages.size());

    ret.vk_pipeline_layout = ret.device->gpu_sro_table.pipeline_layouts.at((ret.info.push_constant_size + 3) / 4);
    VkRayTracingPipelineCreateInfoKHR const vk_ray_tracing_pipeline_create_info{
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = {},
        .stageCount = stages_count,
        .pStages = stages.data(),
        .groupCount = group_count,
        .pGroups = groups.data(),
        .maxPipelineRayRecursionDepth = std::min(ret.info.max_ray_recursion_depth, ret.device->properties.ray_tracing_pipeline_properties.value.max_ray_recursion_depth),
        .pLibraryInfo = nullptr,
        .pLibraryInterface = nullptr,
        .pDynamicState = nullptr,
        .layout = ret.vk_pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };
    auto pipeline_result = ret.device->vkCreateRayTracingPipelinesKHR(
        ret.device->vk_device,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        1u,
        &vk_ray_tracing_pipeline_create_info,
        nullptr,
        &ret.vk_pipeline);

    if (pipeline_result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(pipeline_result);
    }
    if ((ret.device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE && !ret.info.name.view().empty())
    {
        auto name_cstr = ret.info.name.c_str();
        VkDebugUtilsObjectNameInfoEXT const name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_PIPELINE,
            .objectHandle = std::bit_cast<uint64_t>(ret.vk_pipeline),
            .pObjectName = name_cstr.data(),
        };
        ret.device->vkSetDebugUtilsObjectNameEXT(ret.device->vk_device, &name_info);
    }
    ret.strong_count = 1;
    device->inc_weak_refcnt();
    *out_pipeline = new daxa_ImplRayTracingPipeline{};
    **out_pipeline = ret;
    return DAXA_RESULT_SUCCESS;
}

inline auto get_aligned(u64 operand, u64 granularity) -> u64
{
    return ((operand + (granularity - 1)) & ~(granularity - 1));
}

auto daxa_ray_tracing_pipeline_create_default_sbt(daxa_RayTracingPipeline pipeline, daxa_RayTracingShaderBindingTable * out_sbt, daxa_BufferId * out_buffer) -> daxa_Result
{
    auto & info = pipeline->info;
    auto * device = pipeline->device;

    u32 const raygen_count = static_cast<u32>(info.ray_gen_shaders.size());
    u32 const intersection_count = static_cast<u32>(info.intersection_shaders.size());
    u32 const any_hit_count = static_cast<u32>(info.any_hit_shaders.size());
    u32 const callable_count = static_cast<u32>(info.callable_shaders.size());
    u32 const closest_hit_count = static_cast<u32>(info.closest_hit_shaders.size());
    u32 const miss_hit_count = static_cast<u32>(info.miss_hit_shaders.size());

    // Because the shaders are provided in order we can calculate the start and end range of shader indices that fall into each group;
    u32 const all_stages_count = raygen_count + intersection_count + any_hit_count + callable_count + closest_hit_count + miss_hit_count;
    u32 const first_callable_index = raygen_count + intersection_count + any_hit_count;
    u32 const last_callable_index = raygen_count + intersection_count + any_hit_count + callable_count;
    u32 const first_miss_index = raygen_count + intersection_count + any_hit_count + callable_count + closest_hit_count;

    u32 ray_gen_group_count = 0;
    u32 miss_group_count = 0;
    u32 hit_group_count = 0;
    u32 callable_group_count = 0;

    // We need to know into which type a shader group belongs. That is what the following piece of code is for.
    // We iterate over all the shaader groups provided by the user and get their type based on the index shader
    // range. We need to group them based on types because that is how the default generated SBT is layed out.
    for (u32 i = 0; i < pipeline->shader_groups.size(); ++i)
    {
        auto shader_group = pipeline->shader_groups.at(i);

        // Hit groups are the only ones that need the type explicitly specified, thus we can immediately 
        // deduce the group type from this.
        if (shader_group.type == ShaderGroup::TRIANGLES_HIT_GROUP || shader_group.type == ShaderGroup::PROCEDURAL_HIT_GROUP)
        {
            hit_group_count++;
        }
        else
        { 
            // Group indexes raygen shader -> it is a raygen group
            if (shader_group.general_shader_index != VK_SHADER_UNUSED_KHR &&
                shader_group.general_shader_index < raygen_count)
            {
                ray_gen_group_count++;
            }
            // Group indexes miss shader -> it is a miss group
            else if (shader_group.general_shader_index >= first_miss_index &&
                     shader_group.general_shader_index < all_stages_count)
            {
                miss_group_count++;
            }
            // Group indexes callable shader -> it is callable group
            else if (shader_group.general_shader_index >= first_callable_index &&
                     shader_group.general_shader_index < last_callable_index)
            {
                callable_group_count++;
            }
            // Group indexes invalid shader
            else
            {
                return DAXA_RESULT_ERROR_INVALID_SHADER_NV;
            }
        }
    }
    // Those will be dynamic
    u32 const ray_count_number = ray_gen_group_count;
    u32 const miss_count_number = miss_group_count;
    u32 const hit_count_number = hit_group_count;
    u32 const callable_count_number = callable_group_count;
    u32 const handle_count = ray_count_number + miss_count_number + hit_count_number + callable_count_number;

    u32 const group_handle_size = device->properties.ray_tracing_pipeline_properties.value.shader_group_handle_size;
    u32 const group_handle_alignment = device->properties.ray_tracing_pipeline_properties.value.shader_group_handle_alignment;
    u32 const group_base_alignment = device->properties.ray_tracing_pipeline_properties.value.shader_group_base_alignment;

    // The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
    u64 const handle_size_aligned = get_aligned(group_handle_size, group_handle_alignment);

    auto & raygen_region = out_sbt->raygen_region;
    auto & miss_region = out_sbt->miss_region;
    auto & hit_region = out_sbt->hit_region;
    auto & callable_region = out_sbt->callable_region;

    raygen_region.stride = get_aligned(handle_size_aligned, group_base_alignment);
    raygen_region.size = raygen_region.stride; // The size member of pRayGenShaderBindingTable must be equal to its stride member
    miss_region.stride = handle_size_aligned;
    miss_region.size = get_aligned(miss_count_number * handle_size_aligned, group_base_alignment);
    hit_region.stride = handle_size_aligned;
    hit_region.size = get_aligned(hit_count_number * handle_size_aligned, group_base_alignment);
    callable_region.stride = handle_size_aligned;
    callable_region.size = get_aligned(callable_count_number * handle_size_aligned, group_base_alignment);

    // Get the shader group handles
    u32 const data_size = handle_count * group_handle_size;
    std::vector<uint8_t> shader_handle_storage(data_size);
    // Allocate a buffer for storing the SBT.
    VkDeviceSize const sbt_size = raygen_region.size + miss_region.size + hit_region.size + callable_region.size;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetRayTracingShaderGroupHandlesNV.html
    auto const get_group_handles_result = static_cast<daxa_Result>(device->vkGetRayTracingShaderGroupHandlesKHR(
        device->vk_device,
        pipeline->vk_pipeline,
        0,
        handle_count,
        data_size,
        shader_handle_storage.data()));

    _DAXA_RETURN_IF_ERROR(get_group_handles_result, get_group_handles_result);

    auto name_cstr = info.name.c_str();
    // Allocate a buffer for storing the SBT.
    auto sbt_info = daxa_BufferInfo{
        .size = sbt_size,
        .allocate_info = DAXA_MEMORY_FLAG_HOST_ACCESS_SEQUENTIAL_WRITE,
        .name = std::bit_cast<daxa_SmallString>(name_cstr),
    };
    // TODO: We need to store the buffer id somewhere, so we can destroy after the pipeline is destroyed
    auto & sbt_buffer_id = *out_buffer;
    auto const create_buffer_result = daxa_dvc_create_buffer(device, &sbt_info, r_cast<daxa_BufferId *>(&sbt_buffer_id));
    _DAXA_RETURN_IF_ERROR(create_buffer_result, create_buffer_result);

    u8 * sbt_buffer_ptr = nullptr;
    auto const get_host_address_result = daxa_dvc_buffer_host_address(device, sbt_buffer_id, reinterpret_cast<void **>(&sbt_buffer_ptr));
    if (get_host_address_result != DAXA_RESULT_SUCCESS)
    {
        auto const destroy_buffer_result = daxa_dvc_destroy_buffer(device, sbt_buffer_id);
        _DAXA_RETURN_IF_ERROR(destroy_buffer_result, destroy_buffer_result);
        return get_host_address_result;
    }

    // Find the SBT addresses of each group
    VkDeviceAddress sbt_address = 0;
    auto const device_address_result = daxa_dvc_buffer_device_address(device, sbt_buffer_id, reinterpret_cast<daxa_DeviceAddress *>(&sbt_address));
    if (device_address_result != DAXA_RESULT_SUCCESS)
    {
        auto const destroy_buffer_result = daxa_dvc_destroy_buffer(device, sbt_buffer_id);
        _DAXA_RETURN_IF_ERROR(destroy_buffer_result, destroy_buffer_result);
        return device_address_result;
    }
    raygen_region.deviceAddress = sbt_address;
    miss_region.deviceAddress = sbt_address + raygen_region.size;
    hit_region.deviceAddress = sbt_address + raygen_region.size + miss_region.size;
    callable_region.deviceAddress = sbt_address + raygen_region.size + miss_region.size + hit_region.size;

    u64 offset = 0;
    // Iterator through the shader handles and store them in the SBT
    u8 * sbt_ptr_iterator = sbt_buffer_ptr;
    // Raygen shaders load data
    for (u32 c = 0; c < ray_count_number; c++)
    {
        std::memcpy(sbt_ptr_iterator, shader_handle_storage.data() + offset, group_handle_size);
        sbt_ptr_iterator += raygen_region.stride;
        offset += group_handle_size;
    }

    // Miss shaders (base ptr + raygen size)
    sbt_ptr_iterator = sbt_buffer_ptr + raygen_region.size;
    // Miss shaders load data
    for (u32 c = 0; c < miss_count_number; c++)
    {
        std::memcpy(sbt_ptr_iterator, shader_handle_storage.data() + offset, group_handle_size);
        sbt_ptr_iterator += miss_region.stride;
        offset += group_handle_size;
    }

    // Hit shaders (base ptr + raygen size + miss size)
    sbt_ptr_iterator = sbt_buffer_ptr + raygen_region.size + miss_region.size;
    // Closest-hit + any-hit + intersection shaders load data
    for (u32 c = 0; c < hit_count_number; c++)
    {

        std::memcpy(sbt_ptr_iterator, shader_handle_storage.data() + offset, group_handle_size);
        sbt_ptr_iterator += hit_region.stride;
        offset += group_handle_size;
    }

    // Callable shaders (base ptr + raygen size + miss size + hit size)
    sbt_ptr_iterator = sbt_buffer_ptr + raygen_region.size + miss_region.size + hit_region.size;
    // Callable shaders load data
    for (u32 c = 0; c < callable_count_number; c++)
    {
        std::memcpy(sbt_ptr_iterator, shader_handle_storage.data() + offset, group_handle_size);
        sbt_ptr_iterator += callable_region.stride;
        offset += group_handle_size;
    }

    return DAXA_RESULT_SUCCESS;
}

auto daxa_ray_tracing_pipeline_get_shader_group_handles(daxa_RayTracingPipeline pipeline, void * out_blob) -> daxa_Result
{
    auto * device = pipeline->device;
    auto & info = pipeline->info;

    u32 const handle_count = pipeline->shader_groups.size();
    u32 const handle_size = device->properties.ray_tracing_pipeline_properties.value.shader_group_handle_size;

    u32 const data_size = handle_count * handle_size;

    // Get the shader group handles
    auto vk_result = device->vkGetRayTracingShaderGroupHandlesKHR(
        device->vk_device,
        pipeline->vk_pipeline,
        0,
        handle_count,
        data_size,
        out_blob);

    return static_cast<daxa_Result>(vk_result);
}

auto daxa_ray_tracing_pipeline_info(daxa_RayTracingPipeline self) -> daxa_RayTracingPipelineInfo const *
{
    return reinterpret_cast<daxa_RayTracingPipelineInfo const *>(&self->info);
}

auto daxa_ray_tracing_pipeline_inc_refcnt(daxa_RayTracingPipeline self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_ray_tracing_pipeline_dec_refcnt(daxa_RayTracingPipeline self) -> u64
{
    return self->dec_refcnt(
        &ImplPipeline::zero_ref_callback,
        self->device->instance);
}

// --- End API Functions ---

// --- Begin Internals ---

void ImplPipeline::zero_ref_callback(ImplHandle const * handle)
{
    _DAXA_TEST_PRINT("ImplPipeline::zero_ref_callback\n");
    auto * self = rc_cast<ImplPipeline *>(handle);
    std::unique_lock const lock{self->device->zombies_mtx};
    u64 const submit_timeline_value = self->device->global_submit_timeline.load(std::memory_order::relaxed);
    self->device->pipeline_zombies.emplace_front(
        submit_timeline_value,
        PipelineZombie{
            .vk_pipeline = self->vk_pipeline,
        });
    self->device->dec_weak_refcnt(
        daxa_ImplDevice::zero_ref_callback,
        self->device->instance);
    delete self;
}

// --- End Internals ---