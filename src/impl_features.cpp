#include "impl_features.hpp"

namespace daxa
{
    auto PhysicalDeviceExtensionsStruct::initialize(VkPhysicalDevice physical_device) -> daxa_Result
    {
        std::vector<VkExtensionProperties> device_extensions = {};
        uint32_t device_extension_count = {};
        auto result = static_cast<daxa_Result>(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_count, nullptr));
        _DAXA_RETURN_IF_ERROR(result, result);

        device_extensions.resize(device_extension_count);
        result = static_cast<daxa_Result>(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_count, device_extensions.data()));
        _DAXA_RETURN_IF_ERROR(result, result);

        for (u32 i = 0; i < device_extensions.size(); ++i)
        {
            for (u32 j = 0; j < COUNT; ++j)
            {
                if (std::strcmp(extension_names[j], device_extensions[i].extensionName) == 0)
                {
                    extensions_present[j] = true;
                    extension_name_list[extension_name_list_size++] = extension_names[j];
                }
            }
        }

        return DAXA_RESULT_SUCCESS;
    }

    void PhysicalDeviceFeaturesStruct::initialize(PhysicalDeviceExtensionsStruct const & extensions)
    {
        void * chain = {};

        physical_device_buffer_device_address_features.pNext = chain;
        physical_device_buffer_device_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        chain = static_cast<void *>(&physical_device_buffer_device_address_features);

        physical_device_descriptor_indexing_features.pNext = chain;
        physical_device_descriptor_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
        chain = static_cast<void *>(&physical_device_descriptor_indexing_features);

        physical_device_host_query_reset_features.pNext = chain;
        physical_device_host_query_reset_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES;
        chain = static_cast<void *>(&physical_device_host_query_reset_features);

        physical_device_dynamic_rendering_features.pNext = chain;
        physical_device_dynamic_rendering_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
        chain = static_cast<void *>(&physical_device_dynamic_rendering_features);

        physical_device_synchronization2_features.pNext = chain;
        physical_device_synchronization2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
        chain = static_cast<void *>(&physical_device_synchronization2_features);

        physical_device_timeline_semaphore_features.pNext = chain;
        physical_device_timeline_semaphore_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        chain = static_cast<void *>(&physical_device_timeline_semaphore_features);

        physical_device_subgroup_size_control_features.pNext = chain;
        physical_device_subgroup_size_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
        chain = static_cast<void *>(&physical_device_subgroup_size_control_features);

        physical_device_scalar_block_layout_features.pNext = chain;
        physical_device_scalar_block_layout_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES;
        chain = static_cast<void *>(&physical_device_scalar_block_layout_features);

        physical_device_variable_pointer_features.pNext = chain;
        physical_device_variable_pointer_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES;
        chain = static_cast<void *>(&physical_device_variable_pointer_features);

        if (extensions.extensions_present[extensions.physical_device_extended_dynamic_state_3_ext])
        {
            physical_device_extended_dynamic_state3_features_ext.pNext = chain;
            physical_device_extended_dynamic_state3_features_ext.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
            chain = static_cast<void *>(&physical_device_extended_dynamic_state3_features_ext);
        }

        physical_device_shader_float16_int8_features.pNext = chain;
        physical_device_shader_float16_int8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        chain = static_cast<void *>(&physical_device_shader_float16_int8_features);

        physical_device_8bit_storage_features.pNext = chain;
        physical_device_8bit_storage_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
        chain = static_cast<void *>(&physical_device_8bit_storage_features);

        physical_device_16bit_storage_features.pNext = chain;
        physical_device_16bit_storage_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        chain = static_cast<void *>(&physical_device_16bit_storage_features);

        if (extensions.extensions_present[extensions.physical_device_robustness2_ext])
        {
            physical_device_robustness2_features_ext.pNext = chain;
            physical_device_robustness2_features_ext.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT;
            chain = static_cast<void *>(&physical_device_robustness2_features_ext);
        }

        physical_device_vulkan_memory_model_features.pNext = chain;
        physical_device_vulkan_memory_model_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
        chain = static_cast<void *>(&physical_device_vulkan_memory_model_features);

        physical_device_shader_atomic_int64_features.pNext = chain;
        physical_device_shader_atomic_int64_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES;
        chain = static_cast<void *>(&physical_device_shader_atomic_int64_features);

        if (extensions.extensions_present[extensions.physical_device_shader_image_atomic_int64_ext])
        {
            physical_device_shader_image_atomic_int64_features_ext.pNext = chain;
            physical_device_shader_image_atomic_int64_features_ext.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT;
            chain = static_cast<void *>(&physical_device_shader_image_atomic_int64_features_ext);
        }

        if (extensions.extensions_present[extensions.physical_device_mesh_shader_ext])
        {
            physical_device_mesh_shader_features_ext.pNext = chain;
            physical_device_mesh_shader_features_ext.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
            chain = static_cast<void *>(&physical_device_mesh_shader_features_ext);
        }

        if (extensions.extensions_present[extensions.physical_device_acceleration_structure_khr])
        {
            physical_device_acceleration_structure_features_khr.pNext = chain;
            physical_device_acceleration_structure_features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            chain = static_cast<void *>(&physical_device_acceleration_structure_features_khr);
        }

        if (extensions.extensions_present[extensions.physical_device_ray_tracing_pipeline_khr])
        {
            physical_device_ray_tracing_pipeline_features_khr.pNext = chain;
            physical_device_ray_tracing_pipeline_features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
            chain = static_cast<void *>(&physical_device_ray_tracing_pipeline_features_khr);
        }

        if (extensions.extensions_present[extensions.physical_device_ray_query_khr])
        {
            physical_device_ray_query_features_khr.pNext = chain;
            physical_device_ray_query_features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
            chain = static_cast<void *>(&physical_device_ray_query_features_khr);
        }

        if (extensions.extensions_present[extensions.physical_device_ray_tracing_position_fetch_khr])
        {
            physical_device_ray_tracing_position_fetch_features_khr.pNext = chain;
            physical_device_ray_tracing_position_fetch_features_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR;
            chain = static_cast<void *>(&physical_device_ray_tracing_position_fetch_features_khr);
        }

        if (extensions.extensions_present[extensions.physical_device_ray_tracing_invocation_reorder_nv])
        {
            physical_device_ray_tracing_invocation_reorder_features_nv.pNext = chain;
            physical_device_ray_tracing_invocation_reorder_features_nv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;
            chain = static_cast<void *>(&physical_device_ray_tracing_invocation_reorder_features_nv);
        }

        if (extensions.extensions_present[extensions.physical_device_shader_atomic_float_ext])
        {
            physical_device_shader_atomic_float_features_ext.pNext = chain;
            physical_device_shader_atomic_float_features_ext.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
            chain = static_cast<void *>(&physical_device_shader_atomic_float_features_ext);
        }

        conservative_rasterization = extensions.extensions_present[extensions.physical_device_conservative_rasterization_ext];
        swapchain = extensions.extensions_present[extensions.physical_device_swapchain_khr];

        physical_device_features_2.pNext = chain;
        physical_device_features_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    }

    // === Required Features ===

    struct RequiredFeature
    {
        u64 offset = {};
        daxa_MissingRequiredVkFeature problem = {};
    };

    constexpr static std::array REQUIRED_FEATURES = std::array{
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.shaderStorageImageMultisample), DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_STORAGE_IMAGE_MULTISAMPLE},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.imageCubeArray), DAXA_MISSING_REQUIRED_VK_FEATURE_IMAGE_CUBE_ARRAY},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.independentBlend), DAXA_MISSING_REQUIRED_VK_FEATURE_INDEPENDENT_BLEND},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.tessellationShader), DAXA_MISSING_REQUIRED_VK_FEATURE_TESSELLATION_SHADER},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.multiDrawIndirect), DAXA_MISSING_REQUIRED_VK_FEATURE_MULTI_DRAW_INDIRECT},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.depthClamp), DAXA_MISSING_REQUIRED_VK_FEATURE_DEPTH_CLAMP},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.fillModeNonSolid), DAXA_MISSING_REQUIRED_VK_FEATURE_FILL_MODE_NON_SOLID},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.wideLines), DAXA_MISSING_REQUIRED_VK_FEATURE_WIDE_LINES},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.samplerAnisotropy), DAXA_MISSING_REQUIRED_VK_FEATURE_SAMPLER_ANISOTROPY},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.fragmentStoresAndAtomics), DAXA_MISSING_REQUIRED_VK_FEATURE_FRAGMENT_STORES_AND_ATOMICS},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.shaderStorageImageReadWithoutFormat), DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.shaderStorageImageWriteWithoutFormat), DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.shaderInt64), DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_INT64},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_variable_pointer_features.variablePointersStorageBuffer), DAXA_MISSING_REQUIRED_VK_FEATURE_VARIABLE_POINTERS_STORAGE_BUFFER},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_variable_pointer_features.variablePointers), DAXA_MISSING_REQUIRED_VK_FEATURE_VARIABLE_POINTERS},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_buffer_device_address_features.bufferDeviceAddress), DAXA_MISSING_REQUIRED_VK_FEATURE_BUFFER_DEVICE_ADDRESS},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_descriptor_indexing_features.shaderSampledImageArrayNonUniformIndexing), DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_descriptor_indexing_features.shaderStorageBufferArrayNonUniformIndexing), DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_descriptor_indexing_features.shaderStorageImageArrayNonUniformIndexing), DAXA_MISSING_REQUIRED_VK_FEATURE_SHADER_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_descriptor_indexing_features.descriptorBindingSampledImageUpdateAfterBind), DAXA_MISSING_REQUIRED_VK_FEATURE_DESCRIPTOR_BINDING_SAMPLED_IMAGE_UPDATE_AFTER_BIND},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_descriptor_indexing_features.descriptorBindingStorageImageUpdateAfterBind), DAXA_MISSING_REQUIRED_VK_FEATURE_DESCRIPTOR_BINDING_STORAGE_IMAGE_UPDATE_AFTER_BIND},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_descriptor_indexing_features.descriptorBindingStorageBufferUpdateAfterBind), DAXA_MISSING_REQUIRED_VK_FEATURE_DESCRIPTOR_BINDING_STORAGE_BUFFER_UPDATE_AFTER_BIND},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_descriptor_indexing_features.descriptorBindingUpdateUnusedWhilePending), DAXA_MISSING_REQUIRED_VK_FEATURE_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_descriptor_indexing_features.descriptorBindingPartiallyBound), DAXA_MISSING_REQUIRED_VK_FEATURE_DESCRIPTOR_BINDING_PARTIALLY_BOUND},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_descriptor_indexing_features.runtimeDescriptorArray), DAXA_MISSING_REQUIRED_VK_FEATURE_RUNTIME_DESCRIPTOR_ARRAY},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_host_query_reset_features.hostQueryReset), DAXA_MISSING_REQUIRED_VK_FEATURE_HOST_QUERY_RESET},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_dynamic_rendering_features.dynamicRendering), DAXA_MISSING_REQUIRED_VK_FEATURE_DYNAMIC_RENDERING},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_synchronization2_features.synchronization2), DAXA_MISSING_REQUIRED_VK_FEATURE_SYNCHRONIZATION2},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_timeline_semaphore_features.timelineSemaphore), DAXA_MISSING_REQUIRED_VK_FEATURE_TIMELINE_SEMAPHORE},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_subgroup_size_control_features.subgroupSizeControl), DAXA_MISSING_REQUIRED_VK_FEATURE_SUBGROUP_SIZE_CONTROL},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_subgroup_size_control_features.computeFullSubgroups), DAXA_MISSING_REQUIRED_VK_FEATURE_COMPUTE_FULL_SUBGROUPS},
        RequiredFeature{offsetof(PhysicalDeviceFeaturesStruct, physical_device_scalar_block_layout_features.scalarBlockLayout), DAXA_MISSING_REQUIRED_VK_FEATURE_SCALAR_BLOCK_LAYOUT},
    };

    // === Implicit Features ===

    struct ImplicitFeature
    {
        // List ALL REQUIRED feature offset for this daxa feature to work.
        // It is ok if feature booleans are listed multiple times by different daxa features.
        std::span<u64 const> vk_feature_offsets = {};
        daxa_DeviceImplicitFeatureFlagBits flag = {};
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_MESH_SHADER_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_mesh_shader_features_ext.meshShader),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_mesh_shader_features_ext.taskShader),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_BASIC_RAY_TRACING_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.accelerationStructure),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.descriptorBindingAccelerationStructureUpdateAfterBind),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_ray_query_features_khr.rayQuery),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_PIPELINE_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.accelerationStructure),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.descriptorBindingAccelerationStructureUpdateAfterBind),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_ray_query_features_khr.rayQuery),

        offsetof(PhysicalDeviceFeaturesStruct, physical_device_ray_tracing_pipeline_features_khr.rayTracingPipeline),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_ray_tracing_pipeline_features_khr.rayTracingPipelineTraceRaysIndirect),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_ray_tracing_pipeline_features_khr.rayTraversalPrimitiveCulling),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_INVOCATION_REORDER_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.accelerationStructure),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.descriptorBindingAccelerationStructureUpdateAfterBind),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_ray_query_features_khr.rayQuery),

        offsetof(PhysicalDeviceFeaturesStruct, physical_device_ray_tracing_invocation_reorder_features_nv.rayTracingInvocationReorder),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_POSITION_FETCH_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.accelerationStructure),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.descriptorBindingAccelerationStructureUpdateAfterBind),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_ray_query_features_khr.rayQuery),

        offsetof(PhysicalDeviceFeaturesStruct, physical_device_ray_tracing_position_fetch_features_khr.rayTracingPositionFetch),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_CONSERVATIVE_RASTERIZATION_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, conservative_rasterization),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_SHADER_ATOMIC_INT64_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_atomic_int64_features.shaderBufferInt64Atomics),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_atomic_int64_features.shaderSharedInt64Atomics),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_IMAGE_ATOMIC64_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_image_atomic_int64_features_ext.shaderImageInt64Atomics),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_SHADER_FLOAT16_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_float16_int8_features.shaderFloat16),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_SHADER_INT8_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_float16_int8_features.shaderInt8),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_8bit_storage_features.storageBuffer8BitAccess),
        // offsetof(PhysicalDeviceFeaturesStruct, physical_device_8bit_storage_features.uniformAndStorageBuffer8BitAccess),
        // offsetof(PhysicalDeviceFeaturesStruct, physical_device_8bit_storage_features.storagePushConstant8),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_SHADER_INT16_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.shaderInt16),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_16bit_storage_features.storageBuffer16BitAccess),
        // offsetof(PhysicalDeviceFeaturesStruct, physical_device_16bit_storage_features.uniformAndStorageBuffer16BitAccess),
        // offsetof(PhysicalDeviceFeaturesStruct, physical_device_16bit_storage_features.storagePushConstant16),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_DYNAMIC_STATE_3_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_extended_dynamic_state3_features_ext.extendedDynamicState3RasterizationSamples),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_SWAPCHAIN_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, swapchain),
    };

    constexpr static std::array DAXA_IMPLICIT_FEATURE_FLAG_SHADER_ATOMIC_FLOAT_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_atomic_float_features_ext.shaderBufferFloat32Atomics),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_atomic_float_features_ext.shaderBufferFloat32AtomicAdd),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_atomic_float_features_ext.shaderSharedFloat32Atomics),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_atomic_float_features_ext.shaderSharedFloat32AtomicAdd),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_atomic_float_features_ext.shaderImageFloat32Atomics),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_shader_atomic_float_features_ext.shaderImageFloat32AtomicAdd),
    };

    constexpr static std::array IMPLICIT_FEATURES = std::array{
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_MESH_SHADER_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_MESH_SHADER},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_BASIC_RAY_TRACING_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_BASIC_RAY_TRACING},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_PIPELINE_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_PIPELINE},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_INVOCATION_REORDER_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_INVOCATION_REORDER},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_POSITION_FETCH_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_POSITION_FETCH},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_CONSERVATIVE_RASTERIZATION_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_CONSERVATIVE_RASTERIZATION},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_SHADER_ATOMIC_INT64_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_SHADER_ATOMIC_INT64},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_IMAGE_ATOMIC64_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_IMAGE_ATOMIC64},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_SHADER_FLOAT16_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_SHADER_FLOAT16},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_SHADER_INT8_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_SHADER_INT8},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_SHADER_INT16_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_SHADER_INT16},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_DYNAMIC_STATE_3_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_DYNAMIC_STATE_3},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_SHADER_ATOMIC_FLOAT_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_SHADER_ATOMIC_FLOAT},
        ImplicitFeature{DAXA_IMPLICIT_FEATURE_FLAG_SWAPCHAIN_VK_FEATURES, DAXA_IMPLICIT_FEATURE_FLAG_SWAPCHAIN},
    };

    // === Explicit Features ===

    struct ExplicitFeature
    {
        // List ALL REQUIRED feature offset for this daxa feature to work.
        // It is ok if feature booleans are listed multiple times by different daxa features.
        std::span<u64 const> vk_feature_offsets = {};
        daxa_DeviceExplicitFeatureFlagBits flag = {};
    };

    constexpr static std::array PHYSICAL_DEVICE_ROBUSTNESS_2_EXT_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_robustness2_features_ext.robustBufferAccess2),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_robustness2_features_ext.robustImageAccess2),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_features_2.features.robustBufferAccess),
    };

    constexpr static std::array PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_buffer_device_address_features.bufferDeviceAddressCaptureReplay),
    };

    constexpr static std::array PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_CAPTURE_REPLAY_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.accelerationStructureCaptureReplay),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.accelerationStructure),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_acceleration_structure_features_khr.descriptorBindingAccelerationStructureUpdateAfterBind),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_ray_query_features_khr.rayQuery),
    };

    constexpr static std::array PHYSICAL_DEVICE_VK_MEMORY_MODEL_VK_FEATURES = std::array{
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_vulkan_memory_model_features.vulkanMemoryModel),
        offsetof(PhysicalDeviceFeaturesStruct, physical_device_vulkan_memory_model_features.vulkanMemoryModelDeviceScope),
    };

    constexpr static std::array EXPLICIT_FEATURES = std::array{
        ExplicitFeature{PHYSICAL_DEVICE_ROBUSTNESS_2_EXT_VK_FEATURES, DAXA_EXPLICIT_FEATURE_FLAG_ROBUSTNESS_2},
        ExplicitFeature{PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY_VK_FEATURES, DAXA_EXPLICIT_FEATURE_FLAG_BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY},
        ExplicitFeature{PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_CAPTURE_REPLAY_VK_FEATURES, DAXA_EXPLICIT_FEATURE_FLAG_ACCELERATION_STRUCTURE_CAPTURE_REPLAY},
        ExplicitFeature{PHYSICAL_DEVICE_VK_MEMORY_MODEL_VK_FEATURES, DAXA_EXPLICIT_FEATURE_FLAG_VK_MEMORY_MODEL},
    };

    // === Feature Processing ===

    auto create_feature_flags(PhysicalDeviceFeaturesStruct const & physical_device_features) -> std::pair<daxa_ImplicitFeatureFlags, daxa_ExplicitFeatureFlags>
    {
        daxa_ImplicitFeatureFlags implicit_flags = {};
        daxa_DeviceExplicitFeatureFlagBits explicit_flags = {};
        std::byte const * address = reinterpret_cast<std::byte const *>(&physical_device_features);
        for (auto const & feature : IMPLICIT_FEATURES)
        {
            bool all_set = true;
            for (auto const & offset : feature.vk_feature_offsets)
            {
                VkBool32 value = *reinterpret_cast<VkBool32 const *>(address + offset);
                all_set = all_set && (value == VK_TRUE);
            }
            if (all_set)
            {
                implicit_flags = static_cast<daxa_ImplicitFeatureFlags>(implicit_flags | feature.flag);
            }
        }
        for (auto const & feature : EXPLICIT_FEATURES)
        {
            bool all_set = true;
            for (auto const & offset : feature.vk_feature_offsets)
            {
                VkBool32 value = *reinterpret_cast<VkBool32 const *>(address + offset);
                all_set = all_set && (value == VK_TRUE);
            }
            if (all_set)
            {
                explicit_flags = static_cast<daxa_DeviceExplicitFeatureFlagBits>(explicit_flags | feature.flag);
            }
        }
        return {implicit_flags, explicit_flags};
    }

    auto create_problem_flags(PhysicalDeviceFeaturesStruct const & physical_device_features) -> daxa_MissingRequiredVkFeature
    {
        daxa_MissingRequiredVkFeature problems = {};
        std::byte const * address = reinterpret_cast<std::byte const *>(&physical_device_features);
        for (auto const & feature : REQUIRED_FEATURES)
        {
            VkBool32 value = *reinterpret_cast<VkBool32 const *>(address + feature.offset);
            if (!value)
            {
                problems = feature.problem;
                break;
            }
        }
        return problems;
    }

    void fill_create_features(
        PhysicalDeviceFeaturesStruct & device_create_features,
        daxa_ImplicitFeatureFlags implicit_feature_flags,
        daxa_ExplicitFeatureFlags explicit_feature_flags)
    {
        std::byte * const address = reinterpret_cast<std::byte *>(&device_create_features);
        for (auto const & feature : REQUIRED_FEATURES)
        {
            VkBool32 & value = *reinterpret_cast<VkBool32 *>(address + feature.offset);
            value = VK_TRUE;
        }
        for (auto const & feature : IMPLICIT_FEATURES)
        {
            if (feature.flag & implicit_feature_flags)
            {
                for (auto const & offset : feature.vk_feature_offsets)
                {
                    VkBool32 & value = *reinterpret_cast<VkBool32 *>(address + offset);
                    value = VK_TRUE;
                }
            }
        }
        for (auto const & feature : EXPLICIT_FEATURES)
        {
            if (feature.flag & explicit_feature_flags)
            {
                for (auto const & offset : feature.vk_feature_offsets)
                {
                    VkBool32 & value = *reinterpret_cast<VkBool32 *>(address + offset);
                    value = VK_TRUE;
                }
            }
        }
    }

    void DevicePropertiesStruct::initialize(daxa_DeviceImplicitFeatureFlagBits implicit_features)
    {
        void * chain = {};

        if (implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_PIPELINE)
        {
            physical_device_ray_tracing_pipeline_properties_khr.pNext = chain;
            physical_device_ray_tracing_pipeline_properties_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
            chain = static_cast<void *>(&physical_device_ray_tracing_pipeline_properties_khr);
        }

        if (implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_BASIC_RAY_TRACING)
        {
            physical_device_acceleration_structure_properties_khr.pNext = chain;
            physical_device_acceleration_structure_properties_khr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
            chain = static_cast<void *>(&physical_device_acceleration_structure_properties_khr);
        }

        if (implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_INVOCATION_REORDER)
        {
            physical_device_ray_tracing_invocation_reorder_properties_nv.pNext = chain;
            physical_device_ray_tracing_invocation_reorder_properties_nv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV;
            chain = static_cast<void *>(&physical_device_ray_tracing_invocation_reorder_properties_nv);
        }

        if (implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_MESH_SHADER)
        {
            physical_device_mesh_shader_properties_ext.pNext = chain;
            physical_device_mesh_shader_properties_ext.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT;
            chain = static_cast<void *>(&physical_device_mesh_shader_properties_ext);
        }

        physical_device_properties_2.pNext = chain;
        physical_device_properties_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    }

    void fill_daxa_device_properties(PhysicalDeviceExtensionsStruct const & extensions, PhysicalDeviceFeaturesStruct const & features, VkPhysicalDevice physical_device, daxa_DeviceProperties * out)
    {
        auto flags = create_feature_flags(features);
        out->implicit_features = flags.first;
        out->explicit_features = flags.second;

        out->missing_required_feature = create_problem_flags(features);

        DevicePropertiesStruct properties_struct = {};
        properties_struct.initialize(out->implicit_features);

        vkGetPhysicalDeviceProperties2(physical_device, &properties_struct.physical_device_properties_2);

        // Copy VkPhysicalDeviceProperties to daxa_DeviceProperties beginning.
        // Copy all fields uo to sparseProperties.
        std::memcpy(
            out,
            r_cast<std::byte const *>(&properties_struct.physical_device_properties_2) + sizeof(void *) * 2 /* skip sType and pNext */,
            offsetof(daxa_DeviceProperties, mesh_shader_properties));

        if (out->implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_PIPELINE)
        {
            out->ray_tracing_pipeline_properties.has_value = 1;
            std::memcpy(
                &out->ray_tracing_pipeline_properties.value,
                r_cast<std::byte const *>(&properties_struct.physical_device_ray_tracing_pipeline_properties_khr) + sizeof(void *) * 2, // skip sType and pNext
                sizeof(daxa_RayTracingPipelineProperties));
        }
        if (out->implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_BASIC_RAY_TRACING)
        {
            out->acceleration_structure_properties.has_value = 1;
            std::memcpy(
                &out->acceleration_structure_properties.value,
                r_cast<std::byte const *>(&properties_struct.physical_device_acceleration_structure_properties_khr) + sizeof(void *) * 2, // skip sType and pNext
                sizeof(daxa_AccelerationStructureProperties));
        }
        if (out->implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_RAY_TRACING_INVOCATION_REORDER)
        {
            out->ray_tracing_invocation_reorder_properties.has_value = 1;
            std::memcpy(
                &out->ray_tracing_invocation_reorder_properties.value,
                r_cast<std::byte const *>(&properties_struct.physical_device_ray_tracing_invocation_reorder_properties_nv) + sizeof(void *) * 2, // skip sType and pNext
                sizeof(daxa_RayTracingInvocationReorderProperties));
        }
        if (out->implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_MESH_SHADER)
        {
            out->mesh_shader_properties.has_value = 1;
            std::memcpy(
                &out->mesh_shader_properties.value,
                r_cast<std::byte const *>(&properties_struct.physical_device_mesh_shader_properties_ext) + sizeof(void *) * 2, // skip sType and pNext
                sizeof(daxa_MeshShaderProperties));
            out->mesh_shader_properties.value.prefers_local_invocation_vertex_output = static_cast<daxa_Bool8>(properties_struct.physical_device_mesh_shader_properties_ext.prefersLocalInvocationVertexOutput);
            out->mesh_shader_properties.value.prefers_local_invocation_primitive_output = static_cast<daxa_Bool8>(properties_struct.physical_device_mesh_shader_properties_ext.prefersLocalInvocationPrimitiveOutput);
            out->mesh_shader_properties.value.prefers_compact_vertex_output = static_cast<daxa_Bool8>(properties_struct.physical_device_mesh_shader_properties_ext.prefersCompactVertexOutput);
            out->mesh_shader_properties.value.prefers_compact_primitive_output = static_cast<daxa_Bool8>(properties_struct.physical_device_mesh_shader_properties_ext.prefersCompactPrimitiveOutput);
        }

        u32 queue_family_props_count = 0;
        std::vector<VkQueueFamilyProperties> queue_props;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_props_count, nullptr);
        queue_props.resize(queue_family_props_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_props_count, queue_props.data());
        std::vector<VkBool32> supports_present;
        supports_present.resize(queue_family_props_count);

        out->compute_queue_count = ~0u;
        out->transfer_queue_count = ~0u;
        for (u32 i = 0; i < queue_family_props_count; i++)
        {
            bool const supports_graphics = queue_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;
            bool const supports_compute = queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT;
            bool const supports_transfer = queue_props[i].queueFlags & VK_QUEUE_TRANSFER_BIT;
            if (out->compute_queue_count == ~0u && !supports_graphics && supports_compute && supports_transfer)
            {
                out->compute_queue_count = std::min(queue_props[i].queueCount, DAXA_MAX_COMPUTE_QUEUE_COUNT);
            }
            if (out->transfer_queue_count == ~0u && !supports_graphics && !supports_compute && supports_transfer)
            {
                out->transfer_queue_count = std::min(queue_props[i].queueCount, DAXA_MAX_TRANSFER_QUEUE_COUNT);
            }
        }
        if (out->compute_queue_count == ~0u)
        {
            out->compute_queue_count = 0u;
        }
        if (out->transfer_queue_count == ~0u)
        {
            out->transfer_queue_count = 0u;
        }
    }
} // namespace daxa