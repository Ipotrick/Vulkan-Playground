#pragma once

#include <daxa/daxa.hpp>
#include <daxa/device.hpp>

#include "impl_core.hpp"

#include "cpp_wrapper.hpp"
#include <daxa/c/device.h>

// Adding a new vulkan feature bool to daxa:
//
// 1. first classify the feature
//   - a feature is either required, implicit or explicit
//   - required features are features that will always be enabled, daxa needs them to work
//   - implicit are optional features with no perf overhead, they get enabled when supported
//   - explicit are optional features with perf impact, they are explicitly enabled in device creation
//
// 2. Check if the feature needs new extension
//   - if so, add it to PhysicalDeviceExtensionsStruct
//
// 3. Check if feature needs new physical device feature
//   - DISCLAIMER: some features dont have feature structs, like VK_EXT_conservative_rasterization.
//                 in such cases, add your own feature boolean to PhysicalDeviceFeaturesStruct!
//   - if so, add the struct to PhysicalDeviceFeaturesStruct
//   - also make sure to add initialization code for it in PhysicalDeviceFeaturesStruct::initialize
//   - PLEASE make sure to guard the init behind the extension beeing present in initialize
//
// 4. Now in impl_features.cpp, add the feature to either REQUIRED_FEATURES, IMPLICIT_FEATURES or EXPLICIT_FEATURES
//   - daxa uses these arrays to process features in create_feature_flags, create_problem_flags and fill_create_features
//   - orient yourself on other features.
//   - please encompass as many feature booleans into one daxa feature!
//   - now you will also have to to daxa_MissingRequiredVkFeature, daxa_DeviceImplicitFeatureFlagBits or daxa_DeviceExplicitFeatureFlagBits.
//   - please add it to the c and c++ code.
//
// 5. When adding new functions using a new daxa feature make sure to check if the feature flag is set.
//   - also consider adding a new daxa_Result code
//   - if you add a new daxa_Result code, update the print function in cpp_wrapper

namespace daxa
{
    struct PhysicalDeviceExtensionsStruct
    {
        enum Extension
        {
            physical_device_swapchain_khr,
            physical_device_conservative_rasterization_ext,
            physical_device_pipeline_library_khr,
            physical_device_ray_tracing_maintenance_1_khr,
            physical_device_deferred_host_operation_khr,
            physical_device_acceleration_structure_khr,
            physical_device_ray_tracing_pipeline_khr,
            physical_device_ray_query_khr,
            physical_device_ray_tracing_position_fetch_khr,
            physical_device_extended_dynamic_state_3_ext,
            physical_device_robustness2_ext,
            physical_device_shader_image_atomic_int64_ext,
            physical_device_mesh_shader_ext,
            physical_device_ray_tracing_invocation_reorder_nv,
            physical_device_shader_atomic_float_ext,
            COUNT
        };
        constexpr static std::array<char const *, COUNT> extension_names = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME,
            VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_RAY_QUERY_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME,
            VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
            VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
            VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME,
            VK_EXT_MESH_SHADER_EXTENSION_NAME,
            VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME,
            VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        };
        char const * extension_name_list[COUNT] = {};
        u32 extension_name_list_size = {};
        std::array<bool, COUNT> extensions_present = {};

        auto initialize(VkPhysicalDevice physical_device) -> daxa_Result;
    };

    /// WARNING: DO NOT MOVE THIS STRUCT IN MEMORY, MUST NOT CHANGE ADDRESS!
    struct PhysicalDeviceFeaturesStruct
    {
        VkPhysicalDeviceBufferDeviceAddressFeatures physical_device_buffer_device_address_features = {};
        VkPhysicalDeviceDescriptorIndexingFeatures physical_device_descriptor_indexing_features = {};
        VkPhysicalDeviceHostQueryResetFeatures physical_device_host_query_reset_features = {};
        VkPhysicalDeviceDynamicRenderingFeatures physical_device_dynamic_rendering_features = {};
        VkPhysicalDeviceSynchronization2Features physical_device_synchronization2_features = {};
        VkPhysicalDeviceTimelineSemaphoreFeatures physical_device_timeline_semaphore_features = {};
        VkPhysicalDeviceSubgroupSizeControlFeatures physical_device_subgroup_size_control_features = {};
        VkPhysicalDeviceScalarBlockLayoutFeatures physical_device_scalar_block_layout_features = {};
        VkPhysicalDeviceVariablePointerFeatures physical_device_variable_pointer_features = {};
        VkPhysicalDeviceExtendedDynamicState3FeaturesEXT physical_device_extended_dynamic_state3_features_ext = {};
        VkPhysicalDeviceShaderFloat16Int8Features physical_device_shader_float16_int8_features = {};
        VkPhysicalDeviceRobustness2FeaturesEXT physical_device_robustness2_features_ext = {};
        VkPhysicalDeviceVulkanMemoryModelFeatures physical_device_vulkan_memory_model_features = {};
        VkPhysicalDeviceShaderAtomicInt64Features physical_device_shader_atomic_int64_features = {};
        VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT physical_device_shader_image_atomic_int64_features_ext = {};
        VkPhysicalDeviceMeshShaderFeaturesEXT physical_device_mesh_shader_features_ext = {};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR physical_device_acceleration_structure_features_khr = {};
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR physical_device_ray_tracing_pipeline_features_khr = {};
        VkPhysicalDeviceRayQueryFeaturesKHR physical_device_ray_query_features_khr = {};
        VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR physical_device_ray_tracing_position_fetch_features_khr = {};
        VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV physical_device_ray_tracing_invocation_reorder_features_nv = {};
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT physical_device_shader_atomic_float_features_ext = {};
        VkPhysicalDeviceFeatures2 physical_device_features_2 = {};
        bool conservative_rasterization = {};
        bool swapchain = {};

        void initialize(PhysicalDeviceExtensionsStruct const & extensions);
    };

    auto create_feature_flags(PhysicalDeviceFeaturesStruct const & physical_device_features) -> std::pair<daxa_ImplicitFeatureFlags, daxa_ExplicitFeatureFlags>;

    auto create_problem_flags(PhysicalDeviceFeaturesStruct const & physical_device_features) -> daxa_MissingRequiredVkFeature;

    void fill_create_features(PhysicalDeviceFeaturesStruct & device_create_features, daxa_ImplicitFeatureFlags feature, daxa_ExplicitFeatureFlags explicit_features);

    struct DevicePropertiesStruct
    {
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR physical_device_ray_tracing_pipeline_properties_khr = {};
        VkPhysicalDeviceAccelerationStructurePropertiesKHR physical_device_acceleration_structure_properties_khr = {};
        VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV physical_device_ray_tracing_invocation_reorder_properties_nv = {};
        VkPhysicalDeviceMeshShaderPropertiesEXT physical_device_mesh_shader_properties_ext = {};
        VkPhysicalDeviceProperties2 physical_device_properties_2 = {};

        void initialize(daxa_DeviceImplicitFeatureFlagBits implicit_features);
    };
    
    void fill_daxa_device_properties(PhysicalDeviceExtensionsStruct const & extensions, PhysicalDeviceFeaturesStruct const & features, VkPhysicalDevice physical_device, daxa_DeviceProperties * out);
} // namespace daxa
