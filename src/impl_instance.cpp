#include "impl_core.hpp"
#include "impl_instance.hpp"
#include "impl_device.hpp"
#include "impl_features.hpp"

#include <algorithm>
#include <vector>

// --- Begin API Functions ---

auto daxa_create_instance(daxa_InstanceInfo const * info, daxa_Instance * out_instance) -> daxa_Result
{
    auto ret = daxa_ImplInstance{};
    ret.info = *reinterpret_cast<InstanceInfo const *>(info);
    ret.engine_name = {ret.info.engine_name.data(), ret.info.engine_name.size()};
    ret.info.engine_name = ret.engine_name;
    ret.app_name = {ret.info.app_name.data(), ret.info.app_name.size()};
    ret.info.app_name = ret.app_name;
    std::vector<char const *> required_extensions{};
    required_extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    if ((ret.info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE)
    {
        required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

#if defined(WIN32)
    required_extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(__linux__)
    required_extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#else
// no surface extension
#endif
    // Check existence of extensions:
    std::vector<VkExtensionProperties> instance_extensions = {};
    uint32_t instance_extension_count = {};
    daxa_Result result = static_cast<daxa_Result>(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr));
    _DAXA_RETURN_IF_ERROR(result, result);

    instance_extensions.resize(instance_extension_count);
    result = static_cast<daxa_Result>(vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, instance_extensions.data()));
    _DAXA_RETURN_IF_ERROR(result, result);

    for (auto const * req_ext : required_extensions)
    {
        bool found = false;
        for (auto & instance_extension : instance_extensions)
        {
            if (std::strcmp(req_ext, instance_extension.extensionName) == 0)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            return DAXA_RESULT_ERROR_EXTENSION_NOT_PRESENT;
        }
    }
    VkApplicationInfo const app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = ret.app_name.c_str(),
        .applicationVersion = 0,
        .pEngineName = ret.engine_name.c_str(),
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo const instance_ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = {},
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0u,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(required_extensions.size()),
        .ppEnabledExtensionNames = required_extensions.data(),
    };
    result = static_cast<daxa_Result>(vkCreateInstance(&instance_ci, nullptr, &ret.vk_instance));
    _DAXA_RETURN_IF_ERROR(result, result);

    result = ret.initialize_physical_devices();
    _DAXA_RETURN_IF_ERROR(result, result);

    ret.strong_count = 1;
    *out_instance = new daxa_ImplInstance{};
    **out_instance = std::move(ret);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_ImplInstance::initialize_physical_devices() -> daxa_Result
{
    u32 device_count = {};
    daxa_Result result = static_cast<daxa_Result>(vkEnumeratePhysicalDevices(this->vk_instance, &device_count, nullptr));
    _DAXA_RETURN_IF_ERROR(result, result);

    std::vector<VkPhysicalDevice> vk_physical_devices = {};
    this->device_internals.resize(device_count);
    this->device_properties.resize(device_count);
    vk_physical_devices.resize(device_count);
    result = static_cast<daxa_Result>(vkEnumeratePhysicalDevices(this->vk_instance, &device_count, vk_physical_devices.data()));
    _DAXA_RETURN_IF_ERROR(result, result);

    for (u32 i = 0; i < device_count; ++i)
    {
        auto & internals = this->device_internals[i];
        auto & properties = this->device_properties[i];
        internals.vk_handle = vk_physical_devices[i];

        // Init extensions:
        result = internals.extensions.initialize(internals.vk_handle);
        _DAXA_RETURN_IF_ERROR(result, result);

        // Init features:
        internals.features.initialize(internals.extensions);
        vkGetPhysicalDeviceFeatures2(internals.vk_handle, &internals.features.physical_device_features_2);

        // Init properties:
        fill_daxa_device_properties(internals.extensions, internals.features, internals.vk_handle, &properties);
    }

    return DAXA_RESULT_SUCCESS;
}

auto daxa_instance_create_device(daxa_Instance self, daxa_DeviceInfo const * legacy_info, daxa_Device * out_device) -> daxa_Result
{
    daxa_DeviceInfo2 info = {};
    if (legacy_info->flags & DAXA_DEVICE_FLAG_BUFFER_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT)
    {
        info.explicit_features = DAXA_EXPLICIT_FEATURE_FLAG_ACCELERATION_STRUCTURE_CAPTURE_REPLAY;
    }
    if (legacy_info->flags & DAXA_DEVICE_FLAG_VK_MEMORY_MODEL)
    {
        info.explicit_features = DAXA_EXPLICIT_FEATURE_FLAG_VK_MEMORY_MODEL;
    }
    if (legacy_info->flags & DAXA_DEVICE_FLAG_ROBUST_BUFFER_ACCESS)
    {
        info.explicit_features = DAXA_EXPLICIT_FEATURE_FLAG_ROBUSTNESS_2;
    }
    info.max_allowed_buffers = legacy_info->max_allowed_buffers;
    info.max_allowed_images = legacy_info->max_allowed_images;
    info.max_allowed_samplers = legacy_info->max_allowed_samplers;
    info.max_allowed_acceleration_structures = legacy_info->max_allowed_acceleration_structures;
    info.name = legacy_info->name;

    info.physical_device_index = ~0u;
    struct RatedDevice
    {
        u32 physical_device_index = {};
        i32 rating = {};
    };
    std::vector<RatedDevice> rated_devices = {};
    for (u32 i = 0; i < self->device_properties.size(); ++i)
    {
        auto & props = self->device_properties[i];

        bool no_support_problems = props.missing_required_feature == DAXA_MISSING_REQUIRED_VK_FEATURE_NONE;

        bool matches_info =
            info.max_allowed_buffers <= props.limits.max_descriptor_set_storage_buffers &&
            info.max_allowed_images <= props.limits.max_descriptor_set_storage_images &&
            info.max_allowed_images <= props.limits.max_descriptor_set_sampled_images &&
            info.max_allowed_images <= props.limits.max_descriptor_set_storage_images;
        if (props.acceleration_structure_properties.has_value)
        {
            matches_info = matches_info &&
                           info.max_allowed_acceleration_structures <=
                               props.acceleration_structure_properties.value.max_descriptor_set_update_after_bind_acceleration_structures;
        }

        bool matches_explicit_features = (info.explicit_features & props.explicit_features) == info.explicit_features;

        if (!(no_support_problems && matches_info && matches_explicit_features))
        {
            continue;
        }

        auto rating = legacy_info->selector(&props);

        if (rating <= 0)
        {
            continue;
        }

        rated_devices.push_back({
            .physical_device_index = i,
            .rating = rating,
        });
    }
    
    if (rated_devices.empty())
    {
        _DAXA_RETURN_IF_ERROR(DAXA_RESULT_NO_SUITABLE_DEVICE_FOUND, DAXA_RESULT_NO_SUITABLE_DEVICE_FOUND)
    }

    std::sort(rated_devices.begin(), rated_devices.end(), [](auto a, auto b){ return a.rating > b.rating; });

    u32 chosen_device = rated_devices[0].physical_device_index;
    info.physical_device_index = chosen_device;

    return daxa_instance_create_device_2(self, &info, out_device);
}

void daxa_instance_list_devices_properties(daxa_Instance self, daxa_DeviceProperties const ** properties, daxa_u32 * property_count)
{
    *properties = self->device_properties.data();
    *property_count = static_cast<u32>(self->device_properties.size());
}

auto daxa_instance_choose_device(daxa_Instance self, daxa_ImplicitFeatureFlags desired_implicit_features, daxa_DeviceInfo2 * info) -> daxa_Result
{
    info->physical_device_index = ~0u;
    for (u32 i = 0; i < self->device_properties.size(); ++i)
    {
        auto & props = self->device_properties[i];

        bool const no_support_problems = props.missing_required_feature == DAXA_MISSING_REQUIRED_VK_FEATURE_NONE;

        bool matches_info =
            info->max_allowed_buffers <= props.limits.max_descriptor_set_storage_buffers &&
            info->max_allowed_images <= props.limits.max_descriptor_set_storage_images &&
            info->max_allowed_images <= props.limits.max_descriptor_set_sampled_images &&
            info->max_allowed_images <= props.limits.max_descriptor_set_storage_images;
        if (props.acceleration_structure_properties.has_value)
        {
            matches_info = matches_info &&
                           info->max_allowed_acceleration_structures <=
                               props.acceleration_structure_properties.value.max_descriptor_set_update_after_bind_acceleration_structures;
        }

        bool const matches_explicit_features = (info->explicit_features & props.explicit_features) == info->explicit_features;
        bool const matches_implicit_features = (desired_implicit_features & props.implicit_features) == desired_implicit_features;

        if (no_support_problems && matches_info && matches_explicit_features && matches_implicit_features)
        {
            info->physical_device_index = i;
            return DAXA_RESULT_SUCCESS;
        }
    }
    return DAXA_RESULT_ERROR_NO_SUITABLE_DEVICE_FOUND;
}

auto daxa_instance_create_device_2(daxa_Instance self, daxa_DeviceInfo2 const * info, daxa_Device * out_device) -> daxa_Result
{
    daxa_Result result = {};
    *out_device = new daxa_ImplDevice{};
    defer
    {
        if (result != DAXA_RESULT_SUCCESS)
        {
            delete *out_device;
        }
    };

    auto device_i = info->physical_device_index;

    if (device_i >= self->device_properties.size())
    {
        result = DAXA_RESULT_ERROR_INVALID_DEVICE_INDEX;
    }
    _DAXA_RETURN_IF_ERROR(result, result);

    if (self->device_properties[device_i].missing_required_feature != DAXA_MISSING_REQUIRED_VK_FEATURE_NONE)
    {
        result = DAXA_RESULT_ERROR_DEVICE_NOT_SUPPORTED;
    }
    _DAXA_RETURN_IF_ERROR(result, result);

    if ((info->explicit_features & self->device_properties[device_i].explicit_features) != info->explicit_features)
    {
        result = DAXA_RESULT_ERROR_FEATURE_NOT_PRESENT;
    }
    _DAXA_RETURN_IF_ERROR(result, result);

    result = daxa_ImplDevice::create_2(
        self,
        *info,
        self->device_internals[device_i],
        self->device_properties[device_i],
        *out_device);
    _DAXA_RETURN_IF_ERROR(result, result);

    self->inc_weak_refcnt();
    return result;
}

auto daxa_instance_inc_refcnt(daxa_Instance self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_instance_dec_refcnt(daxa_Instance self) -> u64
{
    return self->dec_refcnt(
        &daxa_ImplInstance::zero_ref_callback,
        self);
}

auto daxa_instance_info(daxa_Instance self) -> daxa_InstanceInfo const *
{
    return reinterpret_cast<daxa_InstanceInfo const *>(&self->info);
}

auto daxa_instance_get_vk_instance(daxa_Instance self) -> VkInstance
{
    return self->vk_instance;
}

// --- End API Functions ---

// --- Begin Internals ---

void daxa_ImplInstance::zero_ref_callback(ImplHandle const * handle)
{
    _DAXA_TEST_PRINT("daxa_ImplInstance::zero_ref_callback\n");
    daxa_Instance self = rc_cast<daxa_Instance>(handle);
    vkDestroyInstance(self->vk_instance, nullptr);
    delete self;
}

// --- End Internals ---