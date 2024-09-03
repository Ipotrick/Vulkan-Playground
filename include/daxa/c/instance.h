#ifndef __DAXA_INSTANCE_H__
#define __DAXA_INSTANCE_H__

#include <daxa/c/device.h>

typedef daxa_Flags daxa_InstanceFlags;
static const daxa_InstanceFlags DAXA_INSTANCE_FLAG_DEBUG_UTIL = 0x1;
static const daxa_InstanceFlags DAXA_INSTANCE_FLAG_PARENT_MUST_OUTLIVE_CHILD = 0x2;

typedef struct
{
    daxa_InstanceFlags flags;
    daxa_SmallString engine_name;
    daxa_SmallString app_name;
} daxa_InstanceInfo;

static const daxa_InstanceInfo DAXA_DEFAULT_INSTANCE_INFO = {
    .flags = DAXA_INSTANCE_FLAG_DEBUG_UTIL,
    .engine_name = {.data = "daxa", .size = 4},
    .app_name = {.data = "daxa app", .size = 8},
};

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_create_instance(daxa_InstanceInfo const * info, daxa_Instance * out_instance);

/// WARNING: DEPRECATED, use daxa_instance_create_device_2 instead!
DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_instance_create_device(daxa_Instance instance, daxa_DeviceInfo const * info, daxa_Device * out_device);

DAXA_EXPORT DAXA_NO_DISCARD daxa_Result
daxa_instance_create_device_2(daxa_Instance instance, daxa_DeviceInfo2 const * info, daxa_Device * out_device);

// Can be used to autofill the physical_device_index in a partially filled daxa_DeviceInfo2.
DAXA_EXPORT daxa_Result
daxa_instance_choose_device(daxa_Instance instance, daxa_ImplicitFeatureFlags desired_implicit_features, daxa_DeviceInfo2 * info);

// Returns previous ref count.
DAXA_EXPORT uint64_t
daxa_instance_inc_refcnt(daxa_Instance instance);
// Returns previous ref count.
DAXA_EXPORT uint64_t
daxa_instance_dec_refcnt(daxa_Instance instance);

DAXA_EXPORT daxa_InstanceInfo const *
daxa_instance_info(daxa_Instance instance);

DAXA_EXPORT VkInstance
daxa_instance_get_vk_instance(daxa_Instance instance);

DAXA_EXPORT void
daxa_instance_list_devices_properties(daxa_Instance instance, daxa_DeviceProperties const** properties, daxa_u32 * property_count);

#endif // #ifndef __DAXA_INSTANCE_H__
