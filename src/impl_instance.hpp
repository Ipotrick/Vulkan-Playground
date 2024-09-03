#pragma once

#include "impl_core.hpp"
#include "impl_features.hpp"

#include <daxa/c/instance.h>

struct ImplPhysicalDevice
{
    PhysicalDeviceExtensionsStruct extensions = {};
    PhysicalDeviceFeaturesStruct features = {};
    VkPhysicalDevice vk_handle = {};
};

struct daxa_ImplInstance final : ImplHandle
{
    InstanceInfo info = {};
    std::string engine_name = {};
    std::string app_name = {};
    VkInstance vk_instance = {};

    std::vector<ImplPhysicalDevice> device_internals = {};
    std::vector<daxa_DeviceProperties> device_properties = {};

    static void zero_ref_callback(ImplHandle const * handle);

    auto initialize_physical_devices() -> daxa_Result;
};