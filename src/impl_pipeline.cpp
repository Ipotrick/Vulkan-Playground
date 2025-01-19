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
            _DAXA_DEBUG_BREAK
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
            _DAXA_DEBUG_BREAK                                                                                                   \
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
            _DAXA_DEBUG_BREAK
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
        _DAXA_DEBUG_BREAK
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
        _DAXA_DEBUG_BREAK
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
        _DAXA_DEBUG_BREAK
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

    // Store shader stages
    ret.stages.resize(ret.info.stages.size());
    for (auto i = 0u; i < static_cast<u32>(ret.info.stages.size()); ++i)
    {
        auto const & stage = ret.info.stages.at(i);
        ret.stages[i] = stage;
    }
    // Store shader groups
    ret.groups.resize(ret.info.groups.size());
    for (auto i = 0u; i < static_cast<u32>(ret.info.groups.size()); ++i)
    {
        auto const & group = ret.info.groups.at(i);
        ret.groups[i] = group;
    }
    // Check if ray tracing is supported
    if ((device->properties.implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_BASIC_RAY_TRACING) == 0)
    {
        _DAXA_DEBUG_BREAK
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

    u32 const all_stages_count = static_cast<u32>(ret.info.stages.size());

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
            _DAXA_DEBUG_BREAK
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


    // compile all stages
    for(u32 i = 0; i < all_stages_count; ++i)
    {
        auto stage = ret.info.stages.at(i);
        if(stage.type == RayTracingShaderType::RAYGEN)
        {
            DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage.info, RAYGEN);
        } else if(stage.type == RayTracingShaderType::MISS)
        {
            DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage.info, MISS);
        } else if(stage.type == RayTracingShaderType::CLOSEST_HIT)
        {
            DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage.info, CLOSEST_HIT);
        } else if(stage.type == RayTracingShaderType::ANY_HIT)
        {
            DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage.info, ANY_HIT);
        } else if(stage.type == RayTracingShaderType::INTERSECTION)
        {
            DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage.info, INTERSECTION);
        } else if(stage.type == RayTracingShaderType::CALLABLE)
        {
            DAXA_DECL_TRY_CREATE_RAY_TRACING_MODULE(stage.info, CALLABLE);
        } else {
            return DAXA_RESULT_ERROR_INVALID_SHADER_NV;
        }
    };

    u32 const all_group_count = static_cast<u32>(ret.info.groups.size());

    std::vector<RayTracingShaderGroupInfo> shader_groups(all_group_count);

    // check if the stage is valid
    auto check_stage = [&](RayTracingShaderGroupInfo const & group, std::vector<RayTracingShaderInfo> const & stages) -> bool
    {

        auto check_is_valid_general_shader = [&](RayTracingShaderGroupInfo const & general_shader) -> bool
        {
            // check if the rest of  the indices are invalid
            if(general_shader.closest_hit_shader_index != VK_SHADER_UNUSED_KHR || general_shader.any_hit_shader_index != VK_SHADER_UNUSED_KHR || general_shader.intersection_shader_index != VK_SHADER_UNUSED_KHR)
            {
                return false;
            }

            // check if general shader is valid
            if (general_shader.general_shader_index == VK_SHADER_UNUSED_KHR || general_shader.general_shader_index > all_stages_count)
            {
                return false;
            }

            return true;
        };

        // check if raygen shader is valid
        if(group.type == ExtendedShaderGroupType::RAYGEN)
        {
            if(!check_is_valid_general_shader(group))
            {
                return false;
            }

            auto const & general_shader = stages[group.general_shader_index];

            // check if general shader is valid for raygen, miss and callable
            if(general_shader.type != RayTracingShaderType::RAYGEN)
            {
                return false;
            }

        } 
        // check if miss shader is valid
        else if(group.type == ExtendedShaderGroupType::MISS)
        {
            if(!check_is_valid_general_shader(group))
            {
                return false;
            }

            auto const & general_shader = stages[group.general_shader_index];

            // check if general shader is valid for raygen, miss and callable
            if(general_shader.type != RayTracingShaderType::MISS)
            {
                return false;
            }
        }
        // check if hit group is valid
        else if(group.type == ExtendedShaderGroupType::TRIANGLES_HIT_GROUP || group.type == ExtendedShaderGroupType::PROCEDURAL_HIT_GROUP)
        {
            // check if hit group closest hit shader is valid
            if (group.closest_hit_shader_index != VK_SHADER_UNUSED_KHR && group.closest_hit_shader_index > all_stages_count)
            {
                return false;
            } else if(group.closest_hit_shader_index != VK_SHADER_UNUSED_KHR) {
                if(stages[group.closest_hit_shader_index].type != RayTracingShaderType::CLOSEST_HIT)
                {
                    return false;
                }
            }
            
            // check if hit group any hit shader is valid
            if (group.any_hit_shader_index != VK_SHADER_UNUSED_KHR && group.any_hit_shader_index > all_stages_count)
            {
                return false;
            } else if(group.any_hit_shader_index != VK_SHADER_UNUSED_KHR) {
                if(stages[group.any_hit_shader_index].type != RayTracingShaderType::ANY_HIT)
                {
                    return false;
                }
            }

            // check if hit group intersection shader is valid
            if (group.intersection_shader_index != VK_SHADER_UNUSED_KHR && group.intersection_shader_index > all_stages_count)
            {
                return false;
            } else if(group.intersection_shader_index != VK_SHADER_UNUSED_KHR) {
                if(stages[group.intersection_shader_index].type != RayTracingShaderType::INTERSECTION)
                {
                    return false;
                }
            }
        }
        // check if miss shader is valid
        else if(group.type == ExtendedShaderGroupType::CALLABLE)
        {
            if(!check_is_valid_general_shader(group))
            {
                return false;
            }

            auto const & general_shader = stages[group.general_shader_index];

            // check if general shader is valid for raygen, miss and callable
            if(general_shader.type != RayTracingShaderType::CALLABLE)
            {
                return false;
            }
        } else {
            return false;
        }

        return true;
    };

    auto group_index = 0u;
    for(auto & group : ret.groups) {
        if(check_stage(group, ret.stages)) {
            shader_groups[group_index++] = group;
        } else {
            return DAXA_RESULT_ERROR_INVALID_SHADER_NV;
        }
    }

    // translate shader groups to vulkan shader groups
    auto translate_shader_group = [&](ExtendedShaderGroupType const & type) -> VkRayTracingShaderGroupTypeKHR
    {
        switch (type)
        {
        case ExtendedShaderGroupType::RAYGEN:
            return VkRayTracingShaderGroupTypeKHR::VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        case ExtendedShaderGroupType::MISS:
            return VkRayTracingShaderGroupTypeKHR::VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        case ExtendedShaderGroupType::TRIANGLES_HIT_GROUP:
            return VkRayTracingShaderGroupTypeKHR::VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        case ExtendedShaderGroupType::PROCEDURAL_HIT_GROUP:
            return VkRayTracingShaderGroupTypeKHR::VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        case ExtendedShaderGroupType::CALLABLE:
            return VkRayTracingShaderGroupTypeKHR::VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        default:
            return VkRayTracingShaderGroupTypeKHR::VK_RAY_TRACING_SHADER_GROUP_TYPE_MAX_ENUM_KHR;
        }
    };

    // Shader groups to vulkan shader groups
    for (u32 i = 0; i < shader_groups.size(); ++i)
    {
        auto& shader_group = shader_groups.at(i);
        auto const group = VkRayTracingShaderGroupCreateInfoKHR{
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .pNext = nullptr,
            .type = translate_shader_group(shader_group.type),
            .generalShader = shader_group.general_shader_index,
            .closestHitShader = shader_group.closest_hit_shader_index,
            .anyHitShader = shader_group.any_hit_shader_index,
            .intersectionShader = shader_group.intersection_shader_index,
            .pShaderGroupCaptureReplayHandle = nullptr,
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
        _DAXA_DEBUG_BREAK
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




struct InputGroupRegion {
    u32 index;
    RayTracingShaderGroupInfo* group;
};

struct LinkedGroupRegionInfo {
    u32 index;
    ExtendedShaderGroupType type;
};

inline auto daxa_ray_tracing_pipeline_fill_sbt_buffer(
    u8 * sbt_buffer_ptr, u32 group_handle_size, 
    std::vector<uint8_t> const & shader_handle_storage, 
    std::vector<InputGroupRegion> const & groups,
    std::vector<LinkedGroupRegionInfo> const & regions,
    std::vector<StridedDeviceAddressRegion> const & raygen_regions,
    std::vector<StridedDeviceAddressRegion> const & miss_regions, 
    std::vector<StridedDeviceAddressRegion> const & hit_regions, 
    std::vector<StridedDeviceAddressRegion> const & callable_regions) -> void
{

    auto get_region = [&](LinkedGroupRegionInfo const & link) -> StridedDeviceAddressRegion const &
    {
        switch (link.type)
        {
        case ExtendedShaderGroupType::RAYGEN:
            return raygen_regions.at(link.index);
        case ExtendedShaderGroupType::MISS:
            return miss_regions.at(link.index);
        case ExtendedShaderGroupType::TRIANGLES_HIT_GROUP:
        case ExtendedShaderGroupType::PROCEDURAL_HIT_GROUP:
            return hit_regions.at(link.index);
        case ExtendedShaderGroupType::CALLABLE:
            return callable_regions.at(link.index);
        default:
            return raygen_regions.at(link.index);
        }
    };

    // Iterator through the shader handles and store them in the SBT
    u8 * sbt_ptr_iterator = sbt_buffer_ptr;
    u32 offset = 0;
    u32 index = 0;
    // Iterate through the groups and fill the SBT buffer
    for(auto const & region : regions)
    {
        auto const & region_info = get_region(region);

        for(u32 i = 0; i < region_info.size; i += region_info.stride)
        {
            auto const & group = groups.at(index++);
            std::memcpy(sbt_ptr_iterator + offset, shader_handle_storage.data() + (group.index * group_handle_size), group_handle_size);
            offset += region_info.stride;
        }
        
    }
}

inline auto daxa_ray_tracing_pipeline_build_sbt(
    daxa_RayTracingPipeline pipeline,
    u32* region_count,
    daxa_GroupRegionInfo * out_regions,
    usize* out_buffer_size,
    daxa_BufferId * out_buffer,
    u32 group_count, u32 const * group_indices) -> daxa_Result
{
    if(!pipeline || !region_count || (!out_buffer && *region_count > 0))
    {
        return DAXA_RESULT_ERROR_INVALID_ARGUMENT;
    }
    
    auto & info = pipeline->info;
    auto * device = pipeline->device;

    // total number of groups in the pipeline
    u32 const handle_count = static_cast<u32>(pipeline->groups.size());

    auto requested_groups = std::vector<InputGroupRegion>{};

    // Check if the group indices are valid
    for(auto i = 0u; i < group_count; ++i) {
        if(group_indices[i] >= handle_count) {
            return DAXA_RESULT_ERROR_INVALID_SHADER_NV;
        }
        requested_groups.push_back({group_indices[i], &pipeline->groups[group_indices[i]]});
    }

    u32 const group_handle_size = daxa_ray_tracing_pipeline_get_shader_group_handle_size(pipeline);
    u32 const group_handle_alignment = device->properties.ray_tracing_pipeline_properties.value.shader_group_handle_alignment;
    u32 const group_base_alignment = device->properties.ray_tracing_pipeline_properties.value.shader_group_base_alignment;

    // The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
    u64 const handle_size_aligned = get_aligned(group_handle_size, group_handle_alignment);

    // Get the shader group handles
    u32 const data_size = handle_count * group_handle_size;
    std::vector<uint8_t> shader_handle_storage(data_size);
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetRayTracingShaderGroupHandlesNV.html
    auto const get_group_handles_result = static_cast<daxa_Result>(device->vkGetRayTracingShaderGroupHandlesKHR(
        device->vk_device,
        pipeline->vk_pipeline,
        0,
        handle_count,
        data_size,
        shader_handle_storage.data()));

    _DAXA_RETURN_IF_ERROR(get_group_handles_result, get_group_handles_result);

    // Add a new group region
    auto add_new_group_region = [&](std::initializer_list<ExtendedShaderGroupType> group_flags, ExtendedShaderGroupType group_type, ExtendedShaderGroupType & current_group_type, u32 & current_offset, StridedDeviceAddressRegion *& current_region, std::vector<StridedDeviceAddressRegion> & specific_regions,
                                    std::vector<LinkedGroupRegionInfo> & regions) -> void
    {
        // if there is no current region, create one
        if(!current_region)
        {
            // create a new region
            u32 index = static_cast<u32>(specific_regions.size());
            current_region = &specific_regions.emplace_back();
            regions.push_back({index, group_type});
        }
        else
        {
            // if there's a current region, check if the current region matches type with the current group
            bool match = (std::find(group_flags.begin(), group_flags.end(), current_group_type) != group_flags.end());
            if(!match)
            {
                // if the current region is not a requested region, create a new region
                // check if the current region is aligned with the group base alignment
                auto const current_size = current_region->size;
                current_region->size = get_aligned(current_region->size, group_base_alignment);
                // add the offset difference to the current offset
                auto const offset_diff = current_region->size - current_size;
                current_offset += offset_diff;
                // create a new region
                u32 index = static_cast<u32>(specific_regions.size());
                current_region = &specific_regions.emplace_back();
                regions.push_back({index, group_type});
            }
            else
            {
                // if the current region is a miss region and the current group is a miss group, increment the size
                current_region->size += handle_size_aligned;
                // increment the offset
                current_offset += handle_size_aligned;
                // continue to the next group
                return;
            }
        }
        // set the region values
        current_region->address = current_offset;
        current_region->stride = handle_size_aligned;
        current_region->size = current_region->stride;

        // increment the offset
        current_offset += handle_size_aligned;

        // set the current group type
        current_group_type = group_type;
    };

    // Get SBT entries based on requested groups
    auto raygen_regions = std::vector<StridedDeviceAddressRegion>{};
    auto miss_regions = std::vector<StridedDeviceAddressRegion>{};
    auto hit_regions = std::vector<StridedDeviceAddressRegion>{};
    auto callable_regions = std::vector<StridedDeviceAddressRegion>{};

    auto regions = std::vector<LinkedGroupRegionInfo>{};

    u32 current_offset = 0;
    StridedDeviceAddressRegion* current_region = nullptr;
    ExtendedShaderGroupType current_group_type = ExtendedShaderGroupType::MAX_ENUM;

    for(auto & group_region : requested_groups) {
        auto & group = group_region.group;
        // Check current region
        if(group->type == ExtendedShaderGroupType::RAYGEN) {
            // if there is no current region, create one
            if(!current_region) {
                // create a new region
                u32 index = static_cast<u32>(raygen_regions.size());
                current_region = &raygen_regions.emplace_back();
                regions.push_back({index, group->type});
            } else {
                // if there's a current region, check if the current region is aligned with the group base alignment
                auto const current_size = current_region->size;
                current_region->size = get_aligned(current_region->size, group_base_alignment);
                // add the offset difference to the current offset
                auto const offset_diff = current_region->size - current_size;
                current_offset += offset_diff;
                // create a new region
                u32 index = static_cast<u32>(raygen_regions.size());
                current_region = &raygen_regions.emplace_back();
                regions.push_back({index, group->type});
            }
            // set the region values
            current_region->address = current_offset;
            // NOTE: the stride and size must be the same for raygen shaders
            current_region->stride = get_aligned(handle_size_aligned, group_base_alignment);
            current_region->size = current_region->stride;
            // increment the offset
            current_offset += current_region->size;
            // reset the current region, since raygen regions need to be aligned
            current_region = nullptr;
            // set the current group type
            current_group_type = ExtendedShaderGroupType::RAYGEN;
        } else if(group->type == ExtendedShaderGroupType::MISS) {
            add_new_group_region({group->type}, group->type, current_group_type, current_offset, current_region, miss_regions, regions);
        } else if(group->type == ExtendedShaderGroupType::TRIANGLES_HIT_GROUP || group->type == ExtendedShaderGroupType::PROCEDURAL_HIT_GROUP) {
            auto requested_group_type = {ExtendedShaderGroupType::TRIANGLES_HIT_GROUP,ExtendedShaderGroupType::PROCEDURAL_HIT_GROUP};
            add_new_group_region(requested_group_type, group->type, current_group_type, current_offset, current_region, hit_regions, regions);
        } else if(group->type == ExtendedShaderGroupType::CALLABLE) {
            add_new_group_region({group->type}, group->type, current_group_type, current_offset, current_region, callable_regions, regions);
        }
    }

    auto &region_count_ref = *region_count;
    auto const region_size = static_cast<u32>(regions.size());

    if(region_count_ref == 0) {
        region_count_ref = region_size;
        if(out_buffer_size) {
            *out_buffer_size = current_offset;
        }
        return DAXA_RESULT_SUCCESS;
    }
    else if(region_size > region_count_ref) {
        // region_count_ref = region_size;
        return DAXA_RESULT_ERROR_INVALID_ARGUMENT;
    }

    // TODO: same logic as get ray tracing shader group handles

    auto name_cstr = info.name.c_str();
    // NOTE: it is responsibility of the user to destroy the buffer
    auto & sbt_buffer_id = *out_buffer;
    if(sbt_buffer_id.value == 0) {
        // Allocate a buffer for storing the SBT.
        auto sbt_info = daxa_BufferInfo{
            .size = current_offset,
            .allocate_info = DAXA_MEMORY_FLAG_HOST_ACCESS_SEQUENTIAL_WRITE,
            .name = std::bit_cast<daxa_SmallString>(info.name),
        };
        auto const create_buffer_result = daxa_dvc_create_buffer(device, &sbt_info, r_cast<daxa_BufferId *>(&sbt_buffer_id));
        _DAXA_RETURN_IF_ERROR(create_buffer_result, create_buffer_result);
    } else {
        daxa_BufferInfo buffer_info = {};
        auto result = daxa_dvc_info_buffer(device, sbt_buffer_id, &buffer_info);
        if(result != DAXA_RESULT_SUCCESS) {
            return result;
        }
        if(buffer_info.size < current_offset) {
            return DAXA_RESULT_ERROR_INVALID_ARGUMENT;
        }
    }

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
    

    for(auto & region : regions) {
        auto & region_data = region.type == ExtendedShaderGroupType::RAYGEN ? raygen_regions[region.index] : region.type == ExtendedShaderGroupType::MISS ? miss_regions[region.index] : region.type == ExtendedShaderGroupType::TRIANGLES_HIT_GROUP || region.type == ExtendedShaderGroupType::PROCEDURAL_HIT_GROUP ? hit_regions[region.index] : callable_regions[region.index];
        auto const device_address = static_cast<DeviceAddress>(sbt_address + region_data.address);
        // set the device address
        region_data.address = device_address;
    }

    daxa_ray_tracing_pipeline_fill_sbt_buffer(
        sbt_buffer_ptr,
        group_handle_size,
        shader_handle_storage,
        requested_groups,
        regions,
        raygen_regions,
        miss_regions,
        hit_regions,
        callable_regions);

    auto create_strided_device_address_region_array = [&](std::vector<LinkedGroupRegionInfo> const & regions, 
    std::vector<StridedDeviceAddressRegion> const & raygen_regions,
    std::vector<StridedDeviceAddressRegion> const & miss_regions,
    std::vector<StridedDeviceAddressRegion> const & hit_regions,
    std::vector<StridedDeviceAddressRegion> const & callable_regions,
    GroupRegionInfo * strided_device_addr_regions) -> void
    {
        auto extended_group_shader_to_group_shader = [&](ExtendedShaderGroupType const & type) -> ShaderGroupType
        {
            switch (type)
            {
            case ExtendedShaderGroupType::RAYGEN:
                return ShaderGroupType::RAYGEN;
            case ExtendedShaderGroupType::MISS:
                return ShaderGroupType::MISS;
            case ExtendedShaderGroupType::TRIANGLES_HIT_GROUP:
            case ExtendedShaderGroupType::PROCEDURAL_HIT_GROUP:
                return ShaderGroupType::HIT;
            case ExtendedShaderGroupType::CALLABLE:
                return ShaderGroupType::CALLABLE;
            default:
                return ShaderGroupType::MAX_ENUM;
            }
        };

        for(u32 i = 0; i < regions.size(); ++i) {
            auto const & region = regions[i];
            auto const & region_data = region.type == ExtendedShaderGroupType::RAYGEN ? raygen_regions[region.index] : region.type == ExtendedShaderGroupType::MISS ? miss_regions[region.index] : region.type == ExtendedShaderGroupType::TRIANGLES_HIT_GROUP || region.type == ExtendedShaderGroupType::PROCEDURAL_HIT_GROUP ? hit_regions[region.index] : callable_regions[region.index];
            auto const type = extended_group_shader_to_group_shader(region.type);
            auto & strided_device_addr_region = strided_device_addr_regions[i];
            strided_device_addr_region.type = type;
            strided_device_addr_region.region = region_data;
        }

        
    };

    create_strided_device_address_region_array(regions, raygen_regions, miss_regions, hit_regions, callable_regions, reinterpret_cast<GroupRegionInfo *>(out_regions));

    return DAXA_RESULT_SUCCESS;
}

auto daxa_ray_tracing_pipeline_create_sbt(
    daxa_RayTracingPipeline pipeline,
    u32* region_count,
    daxa_GroupRegionInfo * out_regions,
    usize* out_buffer_size,
    daxa_BufferId * out_buffer,
    daxa_BuildShaderBindingTableInfo const * info) -> daxa_Result
{

    auto group_indices = info->group_indices.data;
    auto group_count = static_cast<u32>(info->group_indices.size);

    return daxa_ray_tracing_pipeline_build_sbt(pipeline, region_count, out_regions, out_buffer_size, out_buffer, group_count, group_indices);
}

auto daxa_ray_tracing_pipeline_get_shader_group_count(daxa_RayTracingPipeline pipeline) -> u32
{
    return static_cast<u32>(pipeline->groups.size());
}

auto daxa_ray_tracing_pipeline_get_shader_group_handle_size(daxa_RayTracingPipeline pipeline) -> u32
{
    return pipeline->device->properties.ray_tracing_pipeline_properties.value.shader_group_handle_size;
}

auto daxa_ray_tracing_pipeline_get_shader_group_handles(daxa_RayTracingPipeline pipeline, void * out_blob, usize * buf_size) -> daxa_Result
{
    if(!buf_size) {
        return DAXA_RESULT_ERROR_INVALID_ARGUMENT;
    }

    auto handle_count = daxa_ray_tracing_pipeline_get_shader_group_count(pipeline);
    auto handle_size = daxa_ray_tracing_pipeline_get_shader_group_handle_size(pipeline);

    auto & device = pipeline->device;

    auto data_size = handle_count * handle_size;
    auto & size = *buf_size;

    if(size == 0) {
        size = data_size;
        return DAXA_RESULT_SUCCESS;
    } else if(size < data_size) {
        return DAXA_RESULT_ERROR_INVALID_ARGUMENT;
    } else if(!out_blob) {
        return DAXA_RESULT_ERROR_INVALID_ARGUMENT;
    }

    // Get the shader group handles
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetRayTracingShaderGroupHandlesNV.html
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