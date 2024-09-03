#include "impl_core.hpp"

#include <daxa/c/daxa.h>
#include <daxa/daxa.hpp>

#include <chrono>
#include <iostream>
#include <utility>
#include <fmt/format.h>
#include <bit>

#include "impl_instance.hpp"
#include "impl_device.hpp"

static_assert(sizeof(daxa::Queue) == sizeof(daxa_Queue));
static_assert(alignof(daxa::Queue) == alignof(daxa_Queue));

// --- Begin Helpers ---

auto daxa_result_to_string(daxa_Result result) -> std::string_view
{
    switch (result)
    {
    case daxa_Result::DAXA_RESULT_SUCCESS: return "DAXA_RESULT_SUCCESS";
    case daxa_Result::DAXA_RESULT_NOT_READY: return "DAXA_RESULT_NOT_READY";
    case daxa_Result::DAXA_RESULT_TIMEOUT: return "DAXA_RESULT_TIMEOUT";
    case daxa_Result::DAXA_RESULT_EVENT_SET: return "DAXA_RESULT_EVENT_SET";
    case daxa_Result::DAXA_RESULT_EVENT_RESET: return "DAXA_RESULT_EVENT_RESET";
    case daxa_Result::DAXA_RESULT_INCOMPLETE: return "DAXA_RESULT_INCOMPLETE";
    case daxa_Result::DAXA_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "DAXA_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case daxa_Result::DAXA_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "DAXA_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case daxa_Result::DAXA_RESULT_ERROR_INITIALIZATION_FAILED: return "DAXA_RESULT_ERROR_INITIALIZATION_FAILED";
    case daxa_Result::DAXA_RESULT_ERROR_DEVICE_LOST: return "DAXA_RESULT_ERROR_DEVICE_LOST";
    case daxa_Result::DAXA_RESULT_ERROR_MEMORY_MAP_FAILED: return "DAXA_RESULT_ERROR_MEMORY_MAP_FAILED";
    case daxa_Result::DAXA_RESULT_ERROR_LAYER_NOT_PRESENT: return "DAXA_RESULT_ERROR_LAYER_NOT_PRESENT";
    case daxa_Result::DAXA_RESULT_ERROR_EXTENSION_NOT_PRESENT: return "DAXA_RESULT_ERROR_EXTENSION_NOT_PRESENT";
    case daxa_Result::DAXA_RESULT_ERROR_FEATURE_NOT_PRESENT: return "DAXA_RESULT_ERROR_FEATURE_NOT_PRESENT";
    case daxa_Result::DAXA_RESULT_ERROR_INCOMPATIBLE_DRIVER: return "DAXA_RESULT_ERROR_INCOMPATIBLE_DRIVER";
    case daxa_Result::DAXA_RESULT_ERROR_TOO_MANY_OBJECTS: return "DAXA_RESULT_ERROR_TOO_MANY_OBJECTS";
    case daxa_Result::DAXA_RESULT_ERROR_FORMAT_NOT_SUPPORTED: return "DAXA_RESULT_ERROR_FORMAT_NOT_SUPPORTED";
    case daxa_Result::DAXA_RESULT_ERROR_FRAGMENTED_POOL: return "DAXA_RESULT_ERROR_FRAGMENTED_POOL";
    case daxa_Result::DAXA_RESULT_ERROR_UNKNOWN: return "DAXA_RESULT_ERROR_UNKNOWN";
    case daxa_Result::DAXA_RESULT_ERROR_OUT_OF_POOL_MEMORY: return "DAXA_RESULT_ERROR_OUT_OF_POOL_MEMORY";
    case daxa_Result::DAXA_RESULT_ERROR_INVALID_EXTERNAL_HANDLE: return "DAXA_RESULT_ERROR_INVALID_EXTERNAL_HANDLE";
    case daxa_Result::DAXA_RESULT_ERROR_FRAGMENTATION: return "DAXA_RESULT_ERROR_FRAGMENTATION";
    case daxa_Result::DAXA_RESULT_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "DAXA_RESULT_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
    case daxa_Result::DAXA_RESULT_PIPELINE_COMPILE_REQUIRED: return "DAXA_RESULT_PIPELINE_COMPILE_REQUIRED";
    case daxa_Result::DAXA_RESULT_ERROR_SURFACE_LOST_KHR: return "DAXA_RESULT_ERROR_SURFACE_LOST_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "DAXA_RESULT_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case daxa_Result::DAXA_RESULT_SUBOPTIMAL_KHR: return "DAXA_RESULT_SUBOPTIMAL_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_OUT_OF_DATE_KHR: return "DAXA_RESULT_ERROR_OUT_OF_DATE_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "DAXA_RESULT_ERROR_INCOMPATIBLE_DISPLAY_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_VALIDATION_FAILED_EXT: return "DAXA_RESULT_ERROR_VALIDATION_FAILED_EXT";
    case daxa_Result::DAXA_RESULT_ERROR_INVALID_SHADER_NV: return "DAXA_RESULT_ERROR_INVALID_SHADER_NV";
    case daxa_Result::DAXA_RESULT_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR: return "DAXA_RESULT_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR: return "DAXA_RESULT_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR: return "DAXA_RESULT_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR: return "DAXA_RESULT_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR: return "DAXA_RESULT_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR: return "DAXA_RESULT_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: return "DAXA_RESULT_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
    case daxa_Result::DAXA_RESULT_ERROR_NOT_PERMITTED_KHR: return "DAXA_RESULT_ERROR_NOT_PERMITTED_KHR";
    case daxa_Result::DAXA_RESULT_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: return "DAXA_RESULT_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
    case daxa_Result::DAXA_RESULT_THREAD_IDLE_KHR: return "DAXA_RESULT_THREAD_IDLE_KHR";
    case daxa_Result::DAXA_RESULT_THREAD_DONE_KHR: return "DAXA_RESULT_THREAD_DONE_KHR";
    case daxa_Result::DAXA_RESULT_OPERATION_DEFERRED_KHR: return "DAXA_RESULT_OPERATION_DEFERRED_KHR";
    case daxa_Result::DAXA_RESULT_OPERATION_NOT_DEFERRED_KHR: return "DAXA_RESULT_OPERATION_NOT_DEFERRED_KHR";
    case daxa_Result::DAXA_RESULT_MISSING_EXTENSION: return "DAXA_RESULT_MISSING_EXTENSION";
    case daxa_Result::DAXA_RESULT_INVALID_BUFFER_ID: return "DAXA_RESULT_INVALID_BUFFER_ID";
    case daxa_Result::DAXA_RESULT_INVALID_IMAGE_ID: return "DAXA_RESULT_INVALID_IMAGE_ID";
    case daxa_Result::DAXA_RESULT_INVALID_IMAGE_VIEW_ID: return "DAXA_RESULT_INVALID_IMAGE_VIEW_ID";
    case daxa_Result::DAXA_RESULT_INVALID_SAMPLER_ID: return "DAXA_RESULT_INVALID_SAMPLER_ID";
    case daxa_Result::DAXA_RESULT_BUFFER_DOUBLE_FREE: return "DAXA_RESULT_BUFFER_DOUBLE_FREE";
    case daxa_Result::DAXA_RESULT_IMAGE_DOUBLE_FREE: return "DAXA_RESULT_IMAGE_DOUBLE_FREE";
    case daxa_Result::DAXA_RESULT_IMAGE_VIEW_DOUBLE_FREE: return "DAXA_RESULT_IMAGE_VIEW_DOUBLE_FREE";
    case daxa_Result::DAXA_RESULT_SAMPLER_DOUBLE_FREE: return "DAXA_RESULT_SAMPLER_DOUBLE_FREE";
    case daxa_Result::DAXA_RESULT_INVALID_BUFFER_INFO: return "DAXA_RESULT_INVALID_BUFFER_INFO";
    case daxa_Result::DAXA_RESULT_INVALID_IMAGE_INFO: return "DAXA_RESULT_INVALID_IMAGE_INFO";
    case daxa_Result::DAXA_RESULT_INVALID_IMAGE_VIEW_INFO: return "DAXA_RESULT_INVALID_IMAGE_VIEW_INFO";
    case daxa_Result::DAXA_RESULT_INVALID_SAMPLER_INFO: return "DAXA_RESULT_INVALID_SAMPLER_INFO";
    case daxa_Result::DAXA_RESULT_COMMAND_LIST_COMPLETED: return "DAXA_RESULT_COMMAND_LIST_COMPLETED";
    case daxa_Result::DAXA_RESULT_COMMAND_LIST_NOT_COMPLETED: return "DAXA_RESULT_COMMAND_LIST_NOT_COMPLETED";
    case daxa_Result::DAXA_RESULT_INVALID_CLEAR_VALUE: return "DAXA_RESULT_INVALID_CLEAR_VALUE";
    case daxa_Result::DAXA_RESULT_BUFFER_NOT_HOST_VISIBLE: return "DAXA_RESULT_BUFFER_NOT_HOST_VISIBLE";
    case daxa_Result::DAXA_RESULT_BUFFER_NOT_DEVICE_VISIBLE: return "DAXA_RESULT_BUFFER_NOT_DEVICE_VISIBLE";
    case daxa_Result::DAXA_RESULT_INCOMPLETE_COMMAND_LIST: return "DAXA_RESULT_INCOMPLETE_COMMAND_LIST";
    case daxa_Result::DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_BUFFER_COUNT: return "DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_BUFFER_COUNT";
    case daxa_Result::DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_IMAGE_COUNT: return "DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_IMAGE_COUNT";
    case daxa_Result::DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_SAMPLER_COUNT: return "DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_SAMPLER_COUNT";
    case daxa_Result::DAXA_RESULT_FAILED_TO_CREATE_NULL_BUFFER: return "DAXA_RESULT_FAILED_TO_CREATE_NULL_BUFFER";
    case daxa_Result::DAXA_RESULT_FAILED_TO_CREATE_NULL_IMAGE: return "DAXA_RESULT_FAILED_TO_CREATE_NULL_IMAGE";
    case daxa_Result::DAXA_RESULT_FAILED_TO_CREATE_NULL_IMAGE_VIEW: return "DAXA_RESULT_FAILED_TO_CREATE_NULL_IMAGE_VIEW";
    case daxa_Result::DAXA_RESULT_FAILED_TO_CREATE_NULL_SAMPLER: return "DAXA_RESULT_FAILED_TO_CREATE_NULL_SAMPLER";
    case daxa_Result::DAXA_RESULT_FAILED_TO_CREATE_BUFFER: return "DAXA_RESULT_FAILED_TO_CREATE_BUFFER";
    case daxa_Result::DAXA_RESULT_FAILED_TO_CREATE_IMAGE: return "DAXA_RESULT_FAILED_TO_CREATE_IMAGE";
    case daxa_Result::DAXA_RESULT_FAILED_TO_CREATE_IMAGE_VIEW: return "DAXA_RESULT_FAILED_TO_CREATE_IMAGE_VIEW";
    case daxa_Result::DAXA_RESULT_FAILED_TO_CREATE_DEFAULT_IMAGE_VIEW: return "DAXA_RESULT_FAILED_TO_CREATE_DEFAULT_IMAGE_VIEW";
    case daxa_Result::DAXA_RESULT_FAILED_TO_CREATE_SAMPLER: return "DAXA_RESULT_FAILED_TO_CREATE_SAMPLER";
    case daxa_Result::DAXA_RESULT_FAILED_TO_CREATE_BDA_BUFFER: return "DAXA_RESULT_FAILED_TO_CREATE_BDA_BUFFER";
    case daxa_Result::DAXA_RESULT_FAILED_TO_SUBMIT_DEVICE_INIT_COMMANDS: return "DAXA_RESULT_FAILED_TO_SUBMIT_DEVICE_INIT_COMMANDS";
    case daxa_Result::DAXA_RESULT_INVALID_BUFFER_RANGE: return "DAXA_RESULT_INVALID_BUFFER_RANGE";
    case daxa_Result::DAXA_RESULT_INVALID_BUFFER_OFFSET: return "DAXA_RESULT_INVALID_BUFFER_OFFSET";
    case daxa_Result::DAXA_RESULT_NO_SUITABLE_FORMAT_FOUND: return "DAXA_RESULT_NO_SUITABLE_FORMAT_FOUND";
    case daxa_Result::DAXA_RESULT_RANGE_OUT_OF_BOUNDS: return "DAXA_RESULT_RANGE_OUT_OF_BOUNDS";
    case daxa_Result::DAXA_RESULT_NO_SUITABLE_DEVICE_FOUND: return "DAXA_RESULT_NO_SUITABLE_DEVICE_FOUND";
    case daxa_Result::DAXA_RESULT_EXCEEDED_MAX_BUFFERS: return "DAXA_RESULT_EXCEEDED_MAX_BUFFERS";
    case daxa_Result::DAXA_RESULT_EXCEEDED_MAX_IMAGES: return "DAXA_RESULT_EXCEEDED_MAX_IMAGES";
    case daxa_Result::DAXA_RESULT_EXCEEDED_MAX_IMAGE_VIEWS: return "DAXA_RESULT_EXCEEDED_MAX_IMAGE_VIEWS";
    case daxa_Result::DAXA_RESULT_EXCEEDED_MAX_SAMPLERS: return "DAXA_RESULT_EXCEEDED_MAX_SAMPLERS";
    case daxa_Result::DAXA_RESULT_DEVICE_SURFACE_UNSUPPORTED_PRESENT_MODE: return "DAXA_RESULT_DEVICE_SURFACE_UNSUPPORTED_PRESENT_MODE";
    case daxa_Result::DAXA_RESULT_COMMAND_REFERENCES_INVALID_BUFFER_ID: return "DAXA_RESULT_COMMAND_REFERENCES_INVALID_BUFFER_ID";
    case daxa_Result::DAXA_RESULT_COMMAND_REFERENCES_INVALID_IMAGE_ID: return "DAXA_RESULT_COMMAND_REFERENCES_INVALID_IMAGE_ID";
    case daxa_Result::DAXA_RESULT_COMMAND_REFERENCES_INVALID_IMAGE_VIEW_ID: return "DAXA_RESULT_COMMAND_REFERENCES_INVALID_IMAGE_VIEW_ID";
    case daxa_Result::DAXA_RESULT_COMMAND_REFERENCES_INVALID_SAMPLER_ID: return "DAXA_RESULT_COMMAND_REFERENCES_INVALID_SAMPLER_ID";
    case daxa_Result::DAXA_RESULT_INVALID_ACCELERATION_STRUCTURE_ID: return "DAXA_RESULT_INVALID_ACCELERATION_STRUCTURE_ID";
    case daxa_Result::DAXA_RESULT_EXCEEDED_MAX_ACCELERATION_STRUCTURES: return "DAXA_RESULT_EXCEEDED_MAX_ACCELERATION_STRUCTURES";
    case daxa_Result::DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_RAYTRACING: return "DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_RAYTRACING";
    case daxa_Result::DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_MESH_SHADER: return "DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_MESH_SHADER";
    case daxa_Result::DAXA_RESULT_INVALID_TLAS_ID: return "DAXA_RESULT_INVALID_TLAS_ID";
    case daxa_Result::DAXA_RESULT_INVALID_BLAS_ID: return "DAXA_RESULT_INVALID_BLAS_ID";
    case daxa_Result::DAXA_RESULT_INVALID_WITHOUT_ENABLING_RAY_TRACING: return "DAXA_RESULT_INVALID_WITHOUT_ENABLING_RAY_TRACING";
    case daxa_Result::DAXA_RESULT_NO_COMPUTE_PIPELINE_BOUND: return "DAXA_RESULT_NO_COMPUTE_PIPELINE_BOUND";
    case daxa_Result::DAXA_RESULT_NO_RASTER_PIPELINE_BOUND: return "DAXA_RESULT_NO_RASTER_PIPELINE_BOUND";
    case daxa_Result::DAXA_RESULT_NO_RAYTRACING_PIPELINE_BOUND: return "DAXA_RESULT_NO_RAYTRACING_PIPELINE_BOUND";
    case daxa_Result::DAXA_RESULT_NO_PIPELINE_BOUND: return "DAXA_RESULT_NO_PIPELINE_BOUND";
    case daxa_Result::DAXA_RESULT_PUSHCONSTANT_RANGE_EXCEEDED: return "DAXA_RESULT_PUSHCONSTANT_RANGE_EXCEEDED";
    case daxa_Result::DAXA_RESULT_MESH_SHADER_NOT_DEVICE_ENABLED: return "DAXA_RESULT_MESH_SHADER_NOT_DEVICE_ENABLED";
    case daxa_Result::DAXA_RESULT_ERROR_COPY_OUT_OF_BOUNDS: return "DAXA_RESULT_ERROR_COPY_OUT_OF_BOUNDS";
    case daxa_Result::DAXA_RESULT_ERROR_NO_GRAPHICS_QUEUE_FOUND: return "DAXA_RESULT_ERROR_NO_GRAPHICS_QUEUE_FOUND";
    case daxa_Result::DAXA_RESULT_ERROR_COULD_NOT_QUERY_QUEUE: return "DAXA_RESULT_ERROR_COULD_NOT_QUERY_QUEUE";
    case daxa_Result::DAXA_RESULT_ERROR_INVALID_QUEUE: return "DAXA_RESULT_ERROR_INVALID_QUEUE";
    case daxa_Result::DAXA_RESULT_ERROR_CMD_LIST_SUBMIT_QUEUE_FAMILY_MISMATCH: return "DAXA_RESULT_ERROR_CMD_LIST_SUBMIT_QUEUE_FAMILY_MISMATCH";
    case daxa_Result::DAXA_RESULT_ERROR_PRESENT_QUEUE_FAMILY_MISMATCH: return "DAXA_RESULT_ERROR_PRESENT_QUEUE_FAMILY_MISMATCH";
    case daxa_Result::DAXA_RESULT_ERROR_INVALID_QUEUE_FAMILY: return "DAXA_RESULT_ERROR_INVALID_QUEUE_FAMILY";
    case daxa_Result::DAXA_RESULT_ERROR_INVALID_DEVICE_INDEX: return "DAXA_RESULT_ERROR_INVALID_DEVICE_INDEX";
    case daxa_Result::DAXA_RESULT_ERROR_DEVICE_NOT_SUPPORTED: return "DAXA_RESULT_ERROR_DEVICE_NOT_SUPPORTED";
    case daxa_Result::DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_ACCELERATION_STRUCTURE_COUNT: return "DAXA_RESULT_DEVICE_DOES_NOT_SUPPORT_ACCELERATION_STRUCTURE_COUNT";
    case daxa_Result::DAXA_RESULT_ERROR_NO_SUITABLE_DEVICE_FOUND: return "DAXA_RESULT_ERROR_NO_SUITABLE_DEVICE_FOUND";
    case daxa_Result::DAXA_RESULT_MAX_ENUM: return "DAXA_RESULT_MAX_ENUM";
    default: return "UNIMPLEMENTED";
    }
};

template <usize N = 1>
void check_result(daxa_Result result, char const * message, std::array<daxa_Result, N> allowed_codes = {DAXA_RESULT_SUCCESS})
{
    bool result_allowed = false;
    for (auto allowed_code : allowed_codes)
    {
        result_allowed = (result_allowed || allowed_code == result);
    }
    if (!result_allowed)
    {
        std::cout << fmt::format(
                         "[[DAXA ASSERT FAILURE]]: error code: {}({}), {}.\n\n",
                         daxa_result_to_string(result),
                         std::bit_cast<i32>(result),
                         message)
                  << std::flush;
        throw std::runtime_error({});
    }
}

// --- End Helpers ---

namespace daxa
{
    /// --- Begin Instance ---

    auto create_instance(InstanceInfo const & info) -> Instance
    {
        Instance ret = {};
        check_result(daxa_create_instance(
                         r_cast<daxa_InstanceInfo const *>(&info),
                         r_cast<daxa_Instance *>(&ret)),
                     "failed to create instance");
        return ret;
    }

    auto Instance::create_device(DeviceInfo const & info) -> Device
    {
        Device ret = {};
        check_result(daxa_instance_create_device(
                         r_cast<daxa_Instance>(this->object),
                         r_cast<daxa_DeviceInfo const *>(&info),
                         r_cast<daxa_Device *>(&ret)),
                     "failed to create device");
        return ret;
    }

    auto Instance::create_device_2(DeviceInfo2 const & info) -> Device
    {
        Device ret = {};
        check_result(daxa_instance_create_device_2(
                         r_cast<daxa_Instance>(this->object),
                         r_cast<daxa_DeviceInfo2 const *>(&info),
                         r_cast<daxa_Device *>(&ret)),
                     "failed to create device");
        return ret;
    }

    auto Instance::choose_device(ImplicitFeatureFlags desired_features, DeviceInfo2 const& p_info) -> DeviceInfo2
    {
        auto info = p_info;
        check_result(daxa_instance_choose_device(
                         r_cast<daxa_Instance>(this->object),
                         static_cast<daxa_ImplicitFeatureFlags>(desired_features.data),
                         r_cast<daxa_DeviceInfo2 *>(&info)),
                     "failed to find fitting device");
        return info;
    }
    
    auto Instance::list_devices_properties() -> std::span<DeviceProperties const>
    {
        DeviceProperties const * data = {};
        u32 size = {};
        daxa_instance_list_devices_properties(r_cast<daxa_Instance>(this->object), r_cast<daxa_DeviceProperties const**>(&data), &size);
        return {data, size};
    }

    auto Instance::info() const -> InstanceInfo const &
    {
        return *r_cast<InstanceInfo const *>(daxa_instance_info(rc_cast<daxa_Instance>(this->object)));
    }

    auto Instance::inc_refcnt(ImplHandle const * object) -> u64
    {
        _DAXA_TEST_PRINT("instance inc refcnt\n");
        return daxa_instance_inc_refcnt(rc_cast<daxa_Instance>(object));
    }

    auto Instance::dec_refcnt(ImplHandle const * object) -> u64
    {
        _DAXA_TEST_PRINT("instance dec refcnt\n");
        return daxa_instance_dec_refcnt(rc_cast<daxa_Instance>(object));
    }

    /// --- End Instance ---

    /// --- Begin Device ---

    auto default_device_score(DeviceProperties const & device_props) -> i32
    {
        return daxa_default_device_score(r_cast<daxa_DeviceProperties const *>(&device_props));
    }

    auto Device::create_memory(MemoryBlockInfo const & info) -> MemoryBlock
    {
        MemoryBlock ret = {};
        check_result(daxa_dvc_create_memory(
                         r_cast<daxa_Device>(this->object),
                         r_cast<daxa_MemoryBlockInfo const *>(&info),
                         r_cast<daxa_MemoryBlock *>(&ret)),
                     "failed to create memory block");
        return ret;
    }

    auto Device::buffer_memory_requirements(BufferInfo const & info) const -> MemoryRequirements
    {
        return std::bit_cast<MemoryRequirements>(
            daxa_dvc_buffer_memory_requirements(
                rc_cast<daxa_Device>(this->object),
                r_cast<daxa_BufferInfo const *>(&info)));
    }

    auto Device::image_memory_requirements(ImageInfo const & info) const -> MemoryRequirements
    {
        return std::bit_cast<MemoryRequirements>(
            daxa_dvc_image_memory_requirements(
                rc_cast<daxa_Device>(this->object),
                r_cast<daxa_ImageInfo const *>(&info)));
    }

    auto Device::tlas_build_sizes(TlasBuildInfo const & info)
        -> AccelerationStructureBuildSizesInfo
    {
        AccelerationStructureBuildSizesInfo ret = {};
        check_result(daxa_dvc_get_tlas_build_sizes(
                         rc_cast<daxa_Device>(this->object),
                         r_cast<daxa_TlasBuildInfo const *>(&info),
                         r_cast<daxa_AccelerationStructureBuildSizesInfo *>(&ret)),
                     "failed to get tlas build sizes");
        return ret;
    }

    auto Device::blas_build_sizes(BlasBuildInfo const & info)
        -> AccelerationStructureBuildSizesInfo
    {
        AccelerationStructureBuildSizesInfo ret = {};
        check_result(daxa_dvc_get_blas_build_sizes(
                         rc_cast<daxa_Device>(this->object),
                         r_cast<daxa_BlasBuildInfo const *>(&info),
                         r_cast<daxa_AccelerationStructureBuildSizesInfo *>(&ret)),
                     "failed to get blas build sizes");
        return ret;
    }

#define DAXA_DECL_GPU_RES_FN(Name, name)                                \
    auto Device::create_##name(Name##Info const & info) -> Name##Id     \
    {                                                                   \
        Name##Id id = {};                                               \
        check_result(                                                   \
            daxa_dvc_create_##name(                                     \
                r_cast<daxa_Device>(this->object),                      \
                r_cast<daxa_##Name##Info const *>(&info),               \
                r_cast<daxa_##Name##Id *>(&id)),                        \
            "failed to create " #name);                                 \
        return id;                                                      \
    }                                                                   \
    void Device::destroy_##name(Name##Id id)                            \
    {                                                                   \
        auto result = daxa_dvc_destroy_##name(                          \
            r_cast<daxa_Device>(this->object),                          \
            static_cast<daxa_##Name##Id>(id));                          \
        check_result(result, "invalid resource id");                    \
    }                                                                   \
    auto Device::is_##name##_id_valid(Name##Id id) const -> bool        \
    {                                                                   \
        return daxa_dvc_is_##name##_valid(                              \
            rc_cast<daxa_Device>(this->object),                         \
            static_cast<daxa_##Name##Id>(id));                          \
    }                                                                   \
    auto Device::name##_info(Name##Id id) const -> Optional<Name##Info> \
    {                                                                   \
        Name##Info info = {};                                           \
        auto result = daxa_dvc_info_##name(                             \
            rc_cast<daxa_Device>(this->object),                         \
            static_cast<daxa_##Name##Id>(id),                           \
            r_cast<daxa_##Name##Info *>(&info));                        \
        if (result == DAXA_RESULT_SUCCESS)                              \
        {                                                               \
            return {info};                                              \
        }                                                               \
        return {};                                                      \
    }

    auto Device::create_buffer_from_memory_block(MemoryBlockBufferInfo const & info) -> BufferId
    {
        BufferId id = {};
        check_result(
            daxa_dvc_create_buffer_from_memory_block(
                r_cast<daxa_Device>(this->object),
                r_cast<daxa_MemoryBlockBufferInfo const *>(&info),
                r_cast<daxa_BufferId *>(&id)),
            "failed to create buffer from memory block");
        return id;
    }
    auto Device::create_image_from_memory_block(MemoryBlockImageInfo const & info) -> ImageId
    {
        ImageId id = {};
        check_result(
            daxa_dvc_create_image_from_block(
                r_cast<daxa_Device>(this->object),
                r_cast<daxa_MemoryBlockImageInfo const *>(&info),
                r_cast<daxa_ImageId *>(&id)),
            "failed to create image from memory block");
        return id;
    }
    auto Device::create_tlas_from_buffer(BufferTlasInfo const & info) -> TlasId
    {
        TlasId id = {};
        check_result(
            daxa_dvc_create_tlas_from_buffer(
                r_cast<daxa_Device>(this->object),
                r_cast<daxa_BufferTlasInfo const *>(&info),
                r_cast<daxa_TlasId *>(&id)),
            "failed to create tlas from buffer");
        return id;
    }
    auto Device::create_blas_from_buffer(BufferBlasInfo const & info) -> BlasId
    {
        BlasId id = {};
        check_result(
            daxa_dvc_create_blas_from_buffer(
                r_cast<daxa_Device>(this->object),
                r_cast<daxa_BufferBlasInfo const *>(&info),
                r_cast<daxa_BlasId *>(&id)),
            "failed to create blas from buffer");
        return id;
    }
    DAXA_DECL_GPU_RES_FN(Buffer, buffer)
    DAXA_DECL_GPU_RES_FN(Image, image)
    DAXA_DECL_GPU_RES_FN(ImageView, image_view)
    DAXA_DECL_GPU_RES_FN(Sampler, sampler)
    DAXA_DECL_GPU_RES_FN(Tlas, tlas)
    DAXA_DECL_GPU_RES_FN(Blas, blas)

    auto Device::buffer_device_address(BufferId id) const -> Optional<DeviceAddress>
    {
        DeviceAddress ret = 0;
        auto result = daxa_dvc_buffer_device_address(
            rc_cast<daxa_Device>(this->object),
            static_cast<daxa_BufferId>(id),
            r_cast<daxa_DeviceAddress *>(&ret));
        if (result == DAXA_RESULT_SUCCESS)
        {
            return {ret};
        }
        check_result(result, "failed to get device address", std::array{DAXA_RESULT_SUCCESS, DAXA_RESULT_INVALID_BUFFER_ID});
        return {};
    }

    auto Device::tlas_device_address(TlasId id) const -> Optional<DeviceAddress>
    {
        DeviceAddress ret = 0;
        auto result = daxa_dvc_tlas_device_address(
            rc_cast<daxa_Device>(this->object),
            static_cast<daxa_TlasId>(id),
            r_cast<daxa_DeviceAddress *>(&ret));
        if (result == DAXA_RESULT_SUCCESS)
        {
            return {ret};
        }
        check_result(result, "failed to get device address", std::array{DAXA_RESULT_SUCCESS, DAXA_RESULT_INVALID_TLAS_ID});
        return {};
    }

    auto Device::blas_device_address(BlasId id) const -> Optional<DeviceAddress>
    {
        DeviceAddress ret = 0;
        auto result = daxa_dvc_blas_device_address(
            rc_cast<daxa_Device>(this->object),
            static_cast<daxa_BlasId>(id),
            r_cast<daxa_DeviceAddress *>(&ret));
        if (result == DAXA_RESULT_SUCCESS)
        {
            return {ret};
        }
        check_result(result, "failed to get device address", std::array{DAXA_RESULT_SUCCESS, DAXA_RESULT_INVALID_BLAS_ID});
        return {};
    }

    auto Device::buffer_host_address(BufferId id) const -> Optional<std::byte *>
    {
        std::byte * ret = nullptr;
        auto result = daxa_dvc_buffer_host_address(
            rc_cast<daxa_Device>(this->object),
            static_cast<daxa_BufferId>(id),
            r_cast<void **>(&ret));
        if (result == DAXA_RESULT_SUCCESS)
        {
            return {ret};
        }
        check_result(result, "failed to get host address", std::array{DAXA_RESULT_SUCCESS, DAXA_RESULT_INVALID_BUFFER_ID});
        return {};
    }

#define DAXA_DECL_DVC_CREATE_FN(Name, name)                        \
    auto Device::create_##name(Name##Info const & info) -> Name    \
    {                                                              \
        Name ret = {};                                             \
        check_result(daxa_dvc_create_##name(                       \
                         r_cast<daxa_Device>(this->object),        \
                         r_cast<daxa_##Name##Info const *>(&info), \
                         r_cast<daxa_##Name *>(&ret)),             \
                     "failed to create " #name);                   \
        return ret;                                                \
    }

    auto Device::create_command_recorder(CommandRecorderInfo const & info) -> CommandRecorder
    {
        if (info.queue_family != daxa::QueueFamily::MAIN)
        {
            std::cout << "[[DAXA ASSERT FAILURE]]: queue family must be main for a generic command recorder.\n\n"
                      << std::flush;
            throw std::runtime_error({});
        }
        CommandRecorder ret = {};
        check_result(daxa_dvc_create_command_recorder(
                         r_cast<daxa_Device>(this->object),
                         r_cast<daxa_CommandRecorderInfo const *>(&info),
                         r_cast<daxa_CommandRecorder *>(&ret)),
                     "failed to create command recorder");
        return ret;
    }

    auto Device::create_compute_command_recorder(CommandRecorderInfo const & info) -> ComputeCommandRecorder
    {
        if (info.queue_family != daxa::QueueFamily::MAIN && info.queue_family != daxa::QueueFamily::COMPUTE)
        {
            std::cout << "[[DAXA ASSERT FAILURE]]: queue family must be either main or compute for a compute command recorder.\n\n"
                      << std::flush;
            throw std::runtime_error({});
        }
        ComputeCommandRecorder ret = {};
        check_result(daxa_dvc_create_command_recorder(
                         r_cast<daxa_Device>(this->object),
                         r_cast<daxa_CommandRecorderInfo const *>(&info),
                         r_cast<daxa_CommandRecorder *>(&ret)),
                     "failed to create command recorder");
        return ret;
    }

    auto Device::create_transfer_command_recorder(CommandRecorderInfo const & info) -> TransferCommandRecorder
    {
        TransferCommandRecorder ret = {};
        check_result(daxa_dvc_create_command_recorder(
                         r_cast<daxa_Device>(this->object),
                         r_cast<daxa_CommandRecorderInfo const *>(&info),
                         r_cast<daxa_CommandRecorder *>(&ret)),
                     "failed to create command recorder");
        return ret;
    }

    DAXA_DECL_DVC_CREATE_FN(RasterPipeline, raster_pipeline)
    DAXA_DECL_DVC_CREATE_FN(ComputePipeline, compute_pipeline)
    DAXA_DECL_DVC_CREATE_FN(RayTracingPipeline, ray_tracing_pipeline)
    DAXA_DECL_DVC_CREATE_FN(Swapchain, swapchain)
    DAXA_DECL_DVC_CREATE_FN(BinarySemaphore, binary_semaphore)
    DAXA_DECL_DVC_CREATE_FN(TimelineSemaphore, timeline_semaphore)
    DAXA_DECL_DVC_CREATE_FN(Event, event)
    DAXA_DECL_DVC_CREATE_FN(TimelineQueryPool, timeline_query_pool)

    auto Device::info() const -> DeviceInfo2 const &
    {
        return *r_cast<DeviceInfo2 const *>(daxa_dvc_info(rc_cast<daxa_Device>(this->object)));
    }

    void Device::wait_idle()
    {
        auto result = daxa_dvc_wait_idle(r_cast<daxa_Device>(this->object));
        check_result(result, "failed to wait idle device");
    }

    void Device::queue_wait_idle(Queue queue)
    {
        auto result = daxa_dvc_queue_wait_idle(r_cast<daxa_Device>(this->object), std::bit_cast<daxa_Queue>(queue));
        check_result(result, "failed to queue wait idle device");
    }

    auto Device::queue_count(QueueFamily queue_family) -> u32
    {
        u32 out_value = {};
        auto result = daxa_dvc_queue_count(r_cast<daxa_Device>(this->object), static_cast<daxa_QueueFamily>(queue_family), &out_value);
        check_result(result, "failed to get queue count");
        return out_value;
    }

    void Device::submit_commands(CommandSubmitInfo const & submit_info)
    {
        daxa_CommandSubmitInfo const c_submit_info = {
            .queue = std::bit_cast<daxa_Queue>(submit_info.queue),
            .wait_stages = static_cast<VkPipelineStageFlags>(submit_info.wait_stages.data),
            .command_lists = reinterpret_cast<daxa_ExecutableCommandList const *>(submit_info.command_lists.data()),
            .command_list_count = submit_info.command_lists.size(),
            .wait_binary_semaphores = reinterpret_cast<daxa_BinarySemaphore const *>(submit_info.wait_binary_semaphores.data()),
            .wait_binary_semaphore_count = submit_info.wait_binary_semaphores.size(),
            .signal_binary_semaphores = reinterpret_cast<daxa_BinarySemaphore const *>(submit_info.signal_binary_semaphores.data()),
            .signal_binary_semaphore_count = submit_info.signal_binary_semaphores.size(),
            .wait_timeline_semaphores = reinterpret_cast<daxa_TimelinePair const *>(submit_info.wait_timeline_semaphores.data()),
            .wait_timeline_semaphore_count = submit_info.wait_timeline_semaphores.size(),
            .signal_timeline_semaphores = reinterpret_cast<daxa_TimelinePair const *>(submit_info.signal_timeline_semaphores.data()),
            .signal_timeline_semaphore_count = submit_info.signal_timeline_semaphores.size(),
        };
        check_result(
            daxa_dvc_submit(r_cast<daxa_Device>(this->object), &c_submit_info),
            "failed to submit commands");
    }

    void Device::present_frame(PresentInfo const & info)
    {
        daxa_PresentInfo const c_present_info = {
            .wait_binary_semaphores = reinterpret_cast<daxa_BinarySemaphore const *>(info.wait_binary_semaphores.data()),
            .wait_binary_semaphore_count = info.wait_binary_semaphores.size(),
            .swapchain = *reinterpret_cast<daxa_Swapchain const *>(&info.swapchain),
        };
        check_result(
            daxa_dvc_present(r_cast<daxa_Device>(this->object), &c_present_info),
            "failed to present frame", std::array{DAXA_RESULT_SUCCESS, DAXA_RESULT_SUBOPTIMAL_KHR, DAXA_RESULT_ERROR_OUT_OF_DATE_KHR});
    }

    void Device::collect_garbage()
    {
        check_result(
            daxa_dvc_collect_garbage(r_cast<daxa_Device>(this->object)),
            "failed to collect garbage");
    }

    auto Device::properties() const -> DeviceProperties const &
    {
        return *r_cast<DeviceProperties const *>(daxa_dvc_properties(rc_cast<daxa_Device>(object)));
    }

    auto Device::get_supported_present_modes(NativeWindowHandle native_handle, NativeWindowPlatform native_platform) const -> std::vector<PresentMode>
    {
        auto * c_device = rc_cast<daxa_Device>(object);
        VkSurfaceKHR surface = {};
        auto result = create_surface(
            c_device->instance,
            std::bit_cast<daxa_NativeWindowHandle>(native_handle),
            std::bit_cast<daxa_NativeWindowPlatform>(native_platform),
            &surface);
        check_result(result, "could not create surface");

        u32 present_mode_count = {};
        auto vk_result = vkGetPhysicalDeviceSurfacePresentModesKHR(
            c_device->vk_physical_device,
            surface,
            &present_mode_count,
            nullptr);
        if (vk_result != VK_SUCCESS)
        {
            vkDestroySurfaceKHR(c_device->instance->vk_instance, surface, nullptr);
            check_result(std::bit_cast<daxa_Result>(vk_result), "failed to query present modes");
        }
        std::vector<PresentMode> ret = {};
        ret.resize(static_cast<usize>(present_mode_count));
        vk_result = vkGetPhysicalDeviceSurfacePresentModesKHR(
            c_device->vk_physical_device,
            surface,
            &present_mode_count,
            r_cast<VkPresentModeKHR *>(ret.data()));
        if (vk_result != VK_SUCCESS)
        {
            vkDestroySurfaceKHR(c_device->instance->vk_instance, surface, nullptr);
            check_result(std::bit_cast<daxa_Result>(vk_result), "failed to query present modes");
        }
        vkDestroySurfaceKHR(c_device->instance->vk_instance, surface, nullptr);
        return ret;
    }

    auto Device::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_dvc_inc_refcnt(rc_cast<daxa_Device>(object));
    }

    auto Device::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_dvc_dec_refcnt(rc_cast<daxa_Device>(object));
    }

    /// --- End Device ---

    /// --- Begin BinarySemaphore ---

    auto BinarySemaphore::info() const -> BinarySemaphoreInfo const &
    {
        return *r_cast<BinarySemaphoreInfo const *>(daxa_binary_semaphore_info(rc_cast<daxa_BinarySemaphore>(this->object)));
    }

    auto BinarySemaphore::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_binary_semaphore_inc_refcnt(rc_cast<daxa_BinarySemaphore>(object));
    }

    auto BinarySemaphore::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_binary_semaphore_dec_refcnt(rc_cast<daxa_BinarySemaphore>(object));
    }

    /// --- End BinarySemaphore ---

    /// --- Begin TimelineSemaphore ---

    auto TimelineSemaphore::info() const -> TimelineSemaphoreInfo const &
    {
        return *r_cast<TimelineSemaphoreInfo const *>(daxa_timeline_semaphore_info(rc_cast<daxa_TimelineSemaphore>(this->object)));
    }

    auto TimelineSemaphore::value() const -> u64
    {
        u64 ret = {};
        check_result(
            daxa_timeline_semaphore_get_value(rc_cast<daxa_TimelineSemaphore>(this->object), &ret),
            "failed to get timeline value");
        return ret;
    }

    void TimelineSemaphore::set_value(u64 value)
    {
        check_result(
            daxa_timeline_semaphore_set_value(r_cast<daxa_TimelineSemaphore>(this->object), value),
            "failed to set timeline value");
    }

    auto TimelineSemaphore::wait_for_value(u64 value, u64 timeout_nanos) -> bool
    {
        auto result = daxa_timeline_semaphore_wait_for_value(r_cast<daxa_TimelineSemaphore>(this->object), value, timeout_nanos);
        DAXA_DBG_ASSERT_TRUE_M(result == DAXA_RESULT_SUCCESS || result == DAXA_RESULT_TIMEOUT, "failed to wait on timeline");
        return result == DAXA_RESULT_SUCCESS;
    }

    auto TimelineSemaphore::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_timeline_semaphore_inc_refcnt(rc_cast<daxa_TimelineSemaphore>(object));
    }

    auto TimelineSemaphore::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_timeline_semaphore_dec_refcnt(rc_cast<daxa_TimelineSemaphore>(object));
    }

    /// --- End TimelineSemaphore ---

    /// --- Begin Event

    auto Event::info() const -> EventInfo const &
    {
        return *reinterpret_cast<EventInfo const *>(daxa_event_info(rc_cast<daxa_Event>(this->object)));
    }

    auto Event::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_event_inc_refcnt(rc_cast<daxa_Event>(object));
    }

    auto Event::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_event_dec_refcnt(rc_cast<daxa_Event>(object));
    }

    /// --- End Event

    /// --- Begin MemoryBlock

    auto MemoryBlock::info() -> MemoryBlockInfo const &
    {
        return *r_cast<MemoryBlockInfo const *>(daxa_memory_block_info(r_cast<daxa_MemoryBlock>(this->object)));
    }

    auto MemoryBlock::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_memory_block_inc_refcnt(rc_cast<daxa_MemoryBlock>(object));
    }

    auto MemoryBlock::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_memory_block_dec_refcnt(rc_cast<daxa_MemoryBlock>(object));
    }

    /// --- End MemoryBlock

    /// --- Begin TimelineQueryPool ---

    auto TimelineQueryPool::info() const -> TimelineQueryPoolInfo const &
    {
        return *r_cast<TimelineQueryPoolInfo const *>(daxa_timeline_query_pool_info(rc_cast<daxa_TimelineQueryPool>(this->object)));
    }

    auto TimelineQueryPool::get_query_results(u32 start_index, u32 count) -> std::vector<u64>
    {
        std::vector<u64> ret = {};
        ret.resize(count * 2);
        check_result(
            daxa_timeline_query_pool_query_results(rc_cast<daxa_TimelineQueryPool>(this->object), start_index, count, ret.data()),
            "failed to query results of timeline query pool", std::array{DAXA_RESULT_SUCCESS, DAXA_RESULT_NOT_READY});
        return ret;
    }

    auto TimelineQueryPool::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_timeline_query_pool_inc_refcnt(rc_cast<daxa_TimelineQueryPool>(object));
    }
    auto TimelineQueryPool::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_timeline_query_pool_dec_refcnt(rc_cast<daxa_TimelineQueryPool>(object));
    }

    /// --- End TimelineQueryPool ---

    /// --- Begin Swapchain ---

    void Swapchain::resize()
    {
        check_result(
            daxa_swp_resize(r_cast<daxa_Swapchain>(this->object)),
            "failed to resize swapchain", std::array{DAXA_RESULT_SUCCESS, DAXA_RESULT_ERROR_OUT_OF_DATE_KHR});
    }

    void Swapchain::set_present_mode(PresentMode present_mode)
    {
        check_result(
            daxa_swp_set_present_mode(r_cast<daxa_Swapchain>(this->object), std::bit_cast<VkPresentModeKHR>(present_mode)),
            "failed to set swapchain present mode");
    }

    auto Swapchain::acquire_next_image() -> ImageId
    {
        ImageId ret = {};
        auto result = daxa_swp_acquire_next_image(r_cast<daxa_Swapchain>(this->object), r_cast<daxa_ImageId *>(&ret));
        if (result == DAXA_RESULT_ERROR_OUT_OF_DATE_KHR ||
            result == DAXA_RESULT_ERROR_SURFACE_LOST_KHR ||
            result == DAXA_RESULT_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT)
        {
            return {};
        }
        else if (result == DAXA_RESULT_SUCCESS)
        {
            return ret;
        }
        else
        {
            check_result(result, "failed to acquire next swapchain image");
        }
        return {};
    }

    auto Swapchain::current_acquire_semaphore() const -> BinarySemaphore const &
    {
        return *rc_cast<BinarySemaphore *>(daxa_swp_current_acquire_semaphore(rc_cast<daxa_Swapchain>(this->object)));
    }

    auto Swapchain::current_present_semaphore() const -> BinarySemaphore const &
    {
        return *rc_cast<BinarySemaphore *>(daxa_swp_current_present_semaphore(rc_cast<daxa_Swapchain>(this->object)));
    }

    auto Swapchain::current_cpu_timeline_value() const -> u64
    {
        return daxa_swp_current_cpu_timeline_value(rc_cast<daxa_Swapchain>(this->object));
    }

    auto Swapchain::gpu_timeline_semaphore() const -> TimelineSemaphore const &
    {
        return *rc_cast<TimelineSemaphore *>(daxa_swp_gpu_timeline_semaphore(rc_cast<daxa_Swapchain>(this->object)));
    }

    auto Swapchain::current_timeline_pair() const -> std::pair<TimelineSemaphore, u64>
    {
        auto gpu_value = *r_cast<TimelineSemaphore const *>(daxa_swp_gpu_timeline_semaphore(rc_cast<daxa_Swapchain>(this->object)));
        auto cpu_value = daxa_swp_current_cpu_timeline_value(rc_cast<daxa_Swapchain>(this->object));
        return std::pair{gpu_value, cpu_value};
    }

    auto Swapchain::info() const -> SwapchainInfo const &
    {
        return *r_cast<SwapchainInfo const *>(daxa_swp_info(rc_cast<daxa_Swapchain>(this->object)));
    }

    auto Swapchain::get_surface_extent() const -> Extent2D
    {
        return std::bit_cast<Extent2D>(daxa_swp_get_surface_extent(rc_cast<daxa_Swapchain>(this->object)));
    }

    auto Swapchain::get_format() const -> Format
    {
        return std::bit_cast<Format>(daxa_swp_get_format(rc_cast<daxa_Swapchain>(this->object)));
    }

    auto Swapchain::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_swp_inc_refcnt(rc_cast<daxa_Swapchain>(object));
    }

    auto Swapchain::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_swp_dec_refcnt(rc_cast<daxa_Swapchain>(object));
    }

    /// --- End Swapchain ---

    /// --- Begin Pipelines

    auto RayTracingPipeline::info() const -> RayTracingPipelineInfo const &
    {
        return *r_cast<RayTracingPipelineInfo const *>(rc_cast<daxa_RayTracingPipeline>(this->object));
    }

    auto RayTracingPipeline::create_default_sbt() const -> SbtPair
    {
        auto result = SbtPair{};
        auto daxa_res = daxa_ray_tracing_pipeline_create_default_sbt(
            rc_cast<daxa_RayTracingPipeline>(this->object),
            r_cast<daxa_RayTracingShaderBindingTable *>(&result.table),
            r_cast<daxa_BufferId *>(&result.buffer));
        check_result(daxa_res, "failed in create_default_sbt");
        return result;
    }

    void RayTracingPipeline::get_shader_group_handles(void * out_blob) const
    {
        auto daxa_res = daxa_ray_tracing_pipeline_get_shader_group_handles(
            rc_cast<daxa_RayTracingPipeline>(this->object), out_blob);
        check_result(daxa_res, "failed in get_shader_group_handles");
    }

    auto RayTracingPipeline::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_ray_tracing_pipeline_inc_refcnt(rc_cast<daxa_RayTracingPipeline>(object));
    }

    auto RayTracingPipeline::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_ray_tracing_pipeline_dec_refcnt(rc_cast<daxa_RayTracingPipeline>(object));
    }

    auto ComputePipeline::info() const -> ComputePipelineInfo const &
    {
        return *r_cast<ComputePipelineInfo const *>(rc_cast<daxa_ComputePipeline>(this->object));
    }

    auto ComputePipeline::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_compute_pipeline_inc_refcnt(rc_cast<daxa_ComputePipeline>(object));
    }

    auto ComputePipeline::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_compute_pipeline_dec_refcnt(rc_cast<daxa_ComputePipeline>(object));
    }

    auto RasterPipeline::info() const -> RasterPipelineInfo const &
    {
        return *r_cast<RasterPipelineInfo const *>(rc_cast<daxa_RasterPipeline>(this->object));
    }

    auto RasterPipeline::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_raster_pipeline_inc_refcnt(rc_cast<daxa_RasterPipeline>(object));
    }

    auto RasterPipeline::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_raster_pipeline_dec_refcnt(rc_cast<daxa_RasterPipeline>(object));
    }

    /// --- End Pipelines

    /// --- Begin ExecutableCommandList

    auto ExecutableCommandList::inc_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_executable_commands_inc_refcnt(rc_cast<daxa_ExecutableCommandList>(object));
    }

    auto ExecutableCommandList::dec_refcnt(ImplHandle const * object) -> u64
    {
        return daxa_executable_commands_dec_refcnt(rc_cast<daxa_ExecutableCommandList>(object));
    }

    /// --- End Executable Commands

    /// --- Begin RenderCommandBuffer

#define DAXA_DECL_RENDER_COMMAND_LIST_WRAPPER(name, Info) \
    void RenderCommandRecorder::name(Info const & info)   \
    {                                                     \
        daxa_cmd_##name(                                  \
            this->internal,                               \
            r_cast<daxa_##Info const *>(&info));          \
    }
#define DAXA_DECL_RENDER_COMMAND_LIST_WRAPPER_CHECK_RESULT(name, Info) \
    void RenderCommandRecorder::name(Info const & info)                \
    {                                                                  \
        auto result = daxa_cmd_##name(                                 \
            this->internal,                                            \
            r_cast<daxa_##Info const *>(&info));                       \
        check_result(result, "failed in " #name);                      \
    }

    RenderCommandRecorder::RenderCommandRecorder(RenderCommandRecorder && other) : internal{}
    {
        std::swap(this->internal, other.internal);
    }

    auto RenderCommandRecorder::operator=(RenderCommandRecorder && other) -> RenderCommandRecorder &
    {
        if (internal != nullptr)
        {
            daxa_destroy_command_recorder(this->internal);
            this->internal = {};
        }
        std::swap(this->internal, other.internal);
        return *this;
    }

    RenderCommandRecorder::~RenderCommandRecorder()
    {
        if (this->internal != nullptr)
        {
            daxa_destroy_command_recorder(this->internal);
            this->internal = {};
        }
    }

    auto RenderCommandRecorder::end_renderpass() && -> CommandRecorder
    {
        daxa_cmd_end_renderpass(this->internal);
        CommandRecorder ret = {};
        ret.internal = this->internal;
        this->internal = {};
        return ret;
    }

    void RenderCommandRecorder::set_viewport(ViewportInfo const & info)
    {
        daxa_cmd_set_viewport(
            this->internal,
            r_cast<VkViewport const *>(&info));
    }

    void RenderCommandRecorder::set_scissor(Rect2D const & info)
    {
        daxa_cmd_set_scissor(
            this->internal,
            r_cast<VkRect2D const *>(&info));
    }

    void RenderCommandRecorder::set_rasterization_samples(RasterizationSamples info)
    {
        auto result = daxa_cmd_set_rasterization_samples(this->internal, static_cast<VkSampleCountFlagBits>(info));
        check_result(result, "failed in set_rasterization_samples");
    }

    DAXA_DECL_RENDER_COMMAND_LIST_WRAPPER(set_depth_bias, DepthBiasInfo)
    DAXA_DECL_RENDER_COMMAND_LIST_WRAPPER_CHECK_RESULT(set_index_buffer, SetIndexBufferInfo)
    DAXA_DECL_RENDER_COMMAND_LIST_WRAPPER(draw, DrawInfo)
    DAXA_DECL_RENDER_COMMAND_LIST_WRAPPER(draw_indexed, DrawIndexedInfo)
    DAXA_DECL_RENDER_COMMAND_LIST_WRAPPER_CHECK_RESULT(draw_indirect, DrawIndirectInfo)
    DAXA_DECL_RENDER_COMMAND_LIST_WRAPPER_CHECK_RESULT(draw_indirect_count, DrawIndirectCountInfo)

    void RenderCommandRecorder::draw_mesh_tasks(u32 x, u32 y, u32 z)
    {
        daxa_cmd_draw_mesh_tasks(
            this->internal,
            x, y, z);
    }
    DAXA_DECL_RENDER_COMMAND_LIST_WRAPPER_CHECK_RESULT(draw_mesh_tasks_indirect, DrawMeshTasksIndirectInfo)
    DAXA_DECL_RENDER_COMMAND_LIST_WRAPPER_CHECK_RESULT(draw_mesh_tasks_indirect_count, DrawMeshTasksIndirectCountInfo)

    void RenderCommandRecorder::set_pipeline(RasterPipeline const & pipeline)
    {
        daxa_cmd_set_raster_pipeline(
            this->internal,
            *r_cast<daxa_RasterPipeline const *>(&pipeline));
    }

    void RenderCommandRecorder::push_constant_vptr(PushConstantInfo const & info)
    {
        auto c_info = std::bit_cast<daxa_PushConstantInfo>(info);
        auto result = daxa_cmd_push_constant(
            this->internal, &c_info);
        check_result(result, "failed in push_constant_vptr");
    }

    /// --- End RenderCommandBuffer

    /// --- Begin CommandRecorder ---

#define DAXA_DECL_COMMAND_LIST_WRAPPER(recorder_type, name, Info) \
    void recorder_type::name(Info const & info)                   \
    {                                                             \
        daxa_cmd_##name(                                          \
            this->internal,                                       \
            r_cast<daxa_##Info const *>(&info));                  \
    }
#define DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(recorder_type, name, Info) \
    void recorder_type::name(Info const & info)                                \
    {                                                                          \
        auto result = daxa_cmd_##name(                                         \
            this->internal,                                                    \
            r_cast<daxa_##Info const *>(&info));                               \
        check_result(result, "failed in " #name);                              \
    }

    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(TransferCommandRecorder, copy_buffer_to_buffer, BufferCopyInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(TransferCommandRecorder, copy_buffer_to_image, BufferImageCopyInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(TransferCommandRecorder, copy_image_to_buffer, ImageBufferCopyInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(TransferCommandRecorder, copy_image_to_image, ImageCopyInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(TransferCommandRecorder, blit_image_to_image, ImageBlitInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(TransferCommandRecorder, clear_buffer, BufferClearInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(TransferCommandRecorder, clear_image, ImageClearInfo)
    void ComputeCommandRecorder::build_acceleration_structures(BuildAccelerationStructuresInfo const & info)
    {
        auto result = daxa_cmd_build_acceleration_structures(
            this->internal,
            r_cast<daxa_BuildAccelerationStucturesInfo const *>(&info));
        check_result(result, "failed to build acceleration structures");
    }
    DAXA_DECL_COMMAND_LIST_WRAPPER(TransferCommandRecorder, pipeline_barrier, MemoryBarrierInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(TransferCommandRecorder, pipeline_barrier_image_transition, ImageMemoryBarrierInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER(TransferCommandRecorder, signal_event, EventSignalInfo)

    void TransferCommandRecorder::wait_events(std::span<EventWaitInfo const> const & infos)
    {
        daxa_cmd_wait_events(
            this->internal, r_cast<daxa_EventSignalInfo const *>(infos.data()), infos.size());
    }

    DAXA_DECL_COMMAND_LIST_WRAPPER(TransferCommandRecorder, wait_event, EventWaitInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER(TransferCommandRecorder, reset_event, ResetEventInfo)

#define DAXA_DECL_COMMAND_LIST_DESTROY_DEFERRED_FN(name, Name)           \
    void TransferCommandRecorder::destroy_##name##_deferred(Name##Id id) \
    {                                                                    \
        auto result = daxa_cmd_destroy_##name##_deferred(                \
            this->internal,                                              \
            static_cast<daxa_##Name##Id>(id));                           \
        check_result(result, "failed to destroy " #name);                \
    }
    DAXA_DECL_COMMAND_LIST_DESTROY_DEFERRED_FN(buffer, Buffer)
    DAXA_DECL_COMMAND_LIST_DESTROY_DEFERRED_FN(image, Image)
    DAXA_DECL_COMMAND_LIST_DESTROY_DEFERRED_FN(image_view, ImageView)
    DAXA_DECL_COMMAND_LIST_DESTROY_DEFERRED_FN(sampler, Sampler)

    void ComputeCommandRecorder::push_constant_vptr(PushConstantInfo const & info)
    {
        auto const c_info = std::bit_cast<daxa_PushConstantInfo>(info);
        auto result = daxa_cmd_push_constant(
            this->internal, &c_info);
        check_result(result, "failed in push_constant_vptr");
    }

    void ComputeCommandRecorder::set_pipeline(ComputePipeline const & pipeline)
    {
        daxa_cmd_set_compute_pipeline(
            this->internal,
            *r_cast<daxa_ComputePipeline const *>(&pipeline));
    }

    void ComputeCommandRecorder::dispatch(DispatchInfo const & info)
    {
        auto result = daxa_cmd_dispatch(
            this->internal,
            r_cast<daxa_DispatchInfo const *>(&info));
        check_result(result, "failed in dispatch");
    }

    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(ComputeCommandRecorder, dispatch_indirect, DispatchIndirectInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(ComputeCommandRecorder, trace_rays, TraceRaysInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER_CHECK_RESULT(ComputeCommandRecorder, trace_rays_indirect, TraceRaysIndirectInfo)

    void ComputeCommandRecorder::set_pipeline(RayTracingPipeline const & pipeline)
    {
        daxa_cmd_set_ray_tracing_pipeline(
            this->internal,
            *r_cast<daxa_RayTracingPipeline const *>(&pipeline));
    }

    auto CommandRecorder::begin_renderpass(RenderPassBeginInfo const & info) && -> RenderCommandRecorder
    {
        auto result = daxa_cmd_begin_renderpass(
            this->internal,
            r_cast<daxa_RenderPassBeginInfo const *>(&info));
        check_result(result, "failed to begin renderpass");
        RenderCommandRecorder ret = {};
        ret.internal = this->internal;
        this->internal = {};
        return ret;
    }
    DAXA_DECL_COMMAND_LIST_WRAPPER(TransferCommandRecorder, write_timestamp, WriteTimestampInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER(TransferCommandRecorder, reset_timestamps, ResetTimestampsInfo)
    DAXA_DECL_COMMAND_LIST_WRAPPER(TransferCommandRecorder, begin_label, CommandLabelInfo)

    void TransferCommandRecorder::end_label()
    {
        daxa_cmd_end_label(this->internal);
    }

    auto TransferCommandRecorder::complete_current_commands() -> ExecutableCommandList
    {
        ExecutableCommandList ret = {};
        auto result = daxa_cmd_complete_current_commands(this->internal, r_cast<daxa_ExecutableCommandList *>(&ret));
        check_result(result, "failed to complete current commands");
        return ret;
    }

    auto TransferCommandRecorder::info() const -> CommandRecorderInfo const &
    {
        return *r_cast<CommandRecorderInfo const *>(daxa_cmd_info(*rc_cast<daxa_CommandRecorder *>(this)));
    }

    TransferCommandRecorder::~TransferCommandRecorder()
    {
        if (this->internal != nullptr)
        {
            daxa_destroy_command_recorder(this->internal);
            this->internal = {};
        }
    }

    TransferCommandRecorder::TransferCommandRecorder(TransferCommandRecorder && other) : internal{}
    {
        std::swap(this->internal, other.internal);
    }

    auto TransferCommandRecorder::operator=(TransferCommandRecorder && other) -> TransferCommandRecorder &
    {
        if (internal != nullptr)
        {
            daxa_destroy_command_recorder(this->internal);
            this->internal = {};
        }
        std::swap(this->internal, other.internal);
        return *this;
    }

    /// --- End CommandRecorder ---

    /// --- Begin to_string ---

    auto to_string(MemoryBarrierInfo const & info) -> std::string
    {
        return fmt::format("access: ({}) -> ({})", to_string(info.src_access), to_string(info.dst_access));
    }

    auto to_string(ImageMemoryBarrierInfo const & info) -> std::string
    {
        return fmt::format("access: ({}) -> ({}), layout: ({}) -> ({}), slice: {}, id: {}",
                           to_string(info.src_access),
                           to_string(info.dst_access),
                           to_string(info.src_layout),
                           to_string(info.dst_layout),
                           to_string(info.image_slice),
                           to_string(info.image_id));
    }

    auto to_string(AccessTypeFlags flags) -> std::string
    {
        if (flags == AccessTypeFlagBits::NONE)
        {
            return "NONE";
        }

        std::string ret = {};
        if ((flags & AccessTypeFlagBits::READ) != AccessTypeFlagBits::NONE)
        {
            // if (ret.size() != 0)
            // {
            //     ret += " | ";
            // }
            ret += "READ";
        }
        if ((flags & AccessTypeFlagBits::WRITE) != AccessTypeFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "WRITE";
        }
        return ret;
    }

    auto to_string(ImageLayout layout) -> std::string_view
    {
        switch (layout)
        {
        case ImageLayout::UNDEFINED: return "UNDEFINED";
        case ImageLayout::GENERAL: return "GENERAL";
        case ImageLayout::TRANSFER_SRC_OPTIMAL: return "TRANSFER_SRC_OPTIMAL";
        case ImageLayout::TRANSFER_DST_OPTIMAL: return "TRANSFER_DST_OPTIMAL";
        case ImageLayout::READ_ONLY_OPTIMAL: return "READ_ONLY_OPTIMAL";
        case ImageLayout::ATTACHMENT_OPTIMAL: return "ATTACHMENT_OPTIMAL";
        case ImageLayout::PRESENT_SRC: return "PRESENT_SRC";
        default: DAXA_DBG_ASSERT_TRUE_M(false, "invalid ImageLayout");
        }
        return "invalid ImageLayout";
    }

    auto to_string(ImageUsageFlags const & flags) -> std::string
    {
        if (flags == ImageUsageFlagBits::NONE)
        {
            return "NONE";
        }

        std::string ret = {};

        if ((flags & ImageUsageFlagBits::NONE) != ImageUsageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "NONE";
        }
        if ((flags & ImageUsageFlagBits::TRANSFER_SRC) != ImageUsageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "TRANSFER_SRC";
        }
        if ((flags & ImageUsageFlagBits::TRANSFER_DST) != ImageUsageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "TRANSFER_DST";
        }
        if ((flags & ImageUsageFlagBits::SHADER_SAMPLED) != ImageUsageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "SHADER_SAMPLED";
        }
        if ((flags & ImageUsageFlagBits::SHADER_STORAGE) != ImageUsageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "SHADER_STORAGE";
        }
        if ((flags & ImageUsageFlagBits::COLOR_ATTACHMENT) != ImageUsageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "COLOR_ATTACHMENT";
        }
        if ((flags & ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT) != ImageUsageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "DEPTH_STENCIL_ATTACHMENT";
        }
        if ((flags & ImageUsageFlagBits::TRANSIENT_ATTACHMENT) != ImageUsageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "TRANSIENT_ATTACHMENT";
        }
        if ((flags & ImageUsageFlagBits::FRAGMENT_DENSITY_MAP) != ImageUsageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "FRAGMENT_DENSITY_MAP";
        }
        if ((flags & ImageUsageFlagBits::FRAGMENT_SHADING_RATE_ATTACHMENT) != ImageUsageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "FRAGMENT_SHADING_RATE_ATTACHMENT";
        }
        return ret;
    }

    auto to_string(Format format) -> std::string_view
    {
        switch (format)
        {
        case Format::UNDEFINED: return "UNDEFINED";
        case Format::R4G4_UNORM_PACK8: return "R4G4_UNORM_PACK8";
        case Format::R4G4B4A4_UNORM_PACK16: return "R4G4B4A4_UNORM_PACK16";
        case Format::B4G4R4A4_UNORM_PACK16: return "B4G4R4A4_UNORM_PACK16";
        case Format::R5G6B5_UNORM_PACK16: return "R5G6B5_UNORM_PACK16";
        case Format::B5G6R5_UNORM_PACK16: return "B5G6R5_UNORM_PACK16";
        case Format::R5G5B5A1_UNORM_PACK16: return "R5G5B5A1_UNORM_PACK16";
        case Format::B5G5R5A1_UNORM_PACK16: return "B5G5R5A1_UNORM_PACK16";
        case Format::A1R5G5B5_UNORM_PACK16: return "A1R5G5B5_UNORM_PACK16";
        case Format::R8_UNORM: return "R8_UNORM";
        case Format::R8_SNORM: return "R8_SNORM";
        case Format::R8_USCALED: return "R8_USCALED";
        case Format::R8_SSCALED: return "R8_SSCALED";
        case Format::R8_UINT: return "R8_UINT";
        case Format::R8_SINT: return "R8_SINT";
        case Format::R8_SRGB: return "R8_SRGB";
        case Format::R8G8_UNORM: return "R8G8_UNORM";
        case Format::R8G8_SNORM: return "R8G8_SNORM";
        case Format::R8G8_USCALED: return "R8G8_USCALED";
        case Format::R8G8_SSCALED: return "R8G8_SSCALED";
        case Format::R8G8_UINT: return "R8G8_UINT";
        case Format::R8G8_SINT: return "R8G8_SINT";
        case Format::R8G8_SRGB: return "R8G8_SRGB";
        case Format::R8G8B8_UNORM: return "R8G8B8_UNORM";
        case Format::R8G8B8_SNORM: return "R8G8B8_SNORM";
        case Format::R8G8B8_USCALED: return "R8G8B8_USCALED";
        case Format::R8G8B8_SSCALED: return "R8G8B8_SSCALED";
        case Format::R8G8B8_UINT: return "R8G8B8_UINT";
        case Format::R8G8B8_SINT: return "R8G8B8_SINT";
        case Format::R8G8B8_SRGB: return "R8G8B8_SRGB";
        case Format::B8G8R8_UNORM: return "B8G8R8_UNORM";
        case Format::B8G8R8_SNORM: return "B8G8R8_SNORM";
        case Format::B8G8R8_USCALED: return "B8G8R8_USCALED";
        case Format::B8G8R8_SSCALED: return "B8G8R8_SSCALED";
        case Format::B8G8R8_UINT: return "B8G8R8_UINT";
        case Format::B8G8R8_SINT: return "B8G8R8_SINT";
        case Format::B8G8R8_SRGB: return "B8G8R8_SRGB";
        case Format::R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case Format::R8G8B8A8_SNORM: return "R8G8B8A8_SNORM";
        case Format::R8G8B8A8_USCALED: return "R8G8B8A8_USCALED";
        case Format::R8G8B8A8_SSCALED: return "R8G8B8A8_SSCALED";
        case Format::R8G8B8A8_UINT: return "R8G8B8A8_UINT";
        case Format::R8G8B8A8_SINT: return "R8G8B8A8_SINT";
        case Format::R8G8B8A8_SRGB: return "R8G8B8A8_SRGB";
        case Format::B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
        case Format::B8G8R8A8_SNORM: return "B8G8R8A8_SNORM";
        case Format::B8G8R8A8_USCALED: return "B8G8R8A8_USCALED";
        case Format::B8G8R8A8_SSCALED: return "B8G8R8A8_SSCALED";
        case Format::B8G8R8A8_UINT: return "B8G8R8A8_UINT";
        case Format::B8G8R8A8_SINT: return "B8G8R8A8_SINT";
        case Format::B8G8R8A8_SRGB: return "B8G8R8A8_SRGB";
        case Format::A8B8G8R8_UNORM_PACK32: return "A8B8G8R8_UNORM_PACK32";
        case Format::A8B8G8R8_SNORM_PACK32: return "A8B8G8R8_SNORM_PACK32";
        case Format::A8B8G8R8_USCALED_PACK32: return "A8B8G8R8_USCALED_PACK32";
        case Format::A8B8G8R8_SSCALED_PACK32: return "A8B8G8R8_SSCALED_PACK32";
        case Format::A8B8G8R8_UINT_PACK32: return "A8B8G8R8_UINT_PACK32";
        case Format::A8B8G8R8_SINT_PACK32: return "A8B8G8R8_SINT_PACK32";
        case Format::A8B8G8R8_SRGB_PACK32: return "A8B8G8R8_SRGB_PACK32";
        case Format::A2R10G10B10_UNORM_PACK32: return "A2R10G10B10_UNORM_PACK32";
        case Format::A2R10G10B10_SNORM_PACK32: return "A2R10G10B10_SNORM_PACK32";
        case Format::A2R10G10B10_USCALED_PACK32: return "A2R10G10B10_USCALED_PACK32";
        case Format::A2R10G10B10_SSCALED_PACK32: return "A2R10G10B10_SSCALED_PACK32";
        case Format::A2R10G10B10_UINT_PACK32: return "A2R10G10B10_UINT_PACK32";
        case Format::A2R10G10B10_SINT_PACK32: return "A2R10G10B10_SINT_PACK32";
        case Format::A2B10G10R10_UNORM_PACK32: return "A2B10G10R10_UNORM_PACK32";
        case Format::A2B10G10R10_SNORM_PACK32: return "A2B10G10R10_SNORM_PACK32";
        case Format::A2B10G10R10_USCALED_PACK32: return "A2B10G10R10_USCALED_PACK32";
        case Format::A2B10G10R10_SSCALED_PACK32: return "A2B10G10R10_SSCALED_PACK32";
        case Format::A2B10G10R10_UINT_PACK32: return "A2B10G10R10_UINT_PACK32";
        case Format::A2B10G10R10_SINT_PACK32: return "A2B10G10R10_SINT_PACK32";
        case Format::R16_UNORM: return "R16_UNORM";
        case Format::R16_SNORM: return "R16_SNORM";
        case Format::R16_USCALED: return "R16_USCALED";
        case Format::R16_SSCALED: return "R16_SSCALED";
        case Format::R16_UINT: return "R16_UINT";
        case Format::R16_SINT: return "R16_SINT";
        case Format::R16_SFLOAT: return "R16_SFLOAT";
        case Format::R16G16_UNORM: return "R16G16_UNORM";
        case Format::R16G16_SNORM: return "R16G16_SNORM";
        case Format::R16G16_USCALED: return "R16G16_USCALED";
        case Format::R16G16_SSCALED: return "R16G16_SSCALED";
        case Format::R16G16_UINT: return "R16G16_UINT";
        case Format::R16G16_SINT: return "R16G16_SINT";
        case Format::R16G16_SFLOAT: return "R16G16_SFLOAT";
        case Format::R16G16B16_UNORM: return "R16G16B16_UNORM";
        case Format::R16G16B16_SNORM: return "R16G16B16_SNORM";
        case Format::R16G16B16_USCALED: return "R16G16B16_USCALED";
        case Format::R16G16B16_SSCALED: return "R16G16B16_SSCALED";
        case Format::R16G16B16_UINT: return "R16G16B16_UINT";
        case Format::R16G16B16_SINT: return "R16G16B16_SINT";
        case Format::R16G16B16_SFLOAT: return "R16G16B16_SFLOAT";
        case Format::R16G16B16A16_UNORM: return "R16G16B16A16_UNORM";
        case Format::R16G16B16A16_SNORM: return "R16G16B16A16_SNORM";
        case Format::R16G16B16A16_USCALED: return "R16G16B16A16_USCALED";
        case Format::R16G16B16A16_SSCALED: return "R16G16B16A16_SSCALED";
        case Format::R16G16B16A16_UINT: return "R16G16B16A16_UINT";
        case Format::R16G16B16A16_SINT: return "R16G16B16A16_SINT";
        case Format::R16G16B16A16_SFLOAT: return "R16G16B16A16_SFLOAT";
        case Format::R32_UINT: return "R32_UINT";
        case Format::R32_SINT: return "R32_SINT";
        case Format::R32_SFLOAT: return "R32_SFLOAT";
        case Format::R32G32_UINT: return "R32G32_UINT";
        case Format::R32G32_SINT: return "R32G32_SINT";
        case Format::R32G32_SFLOAT: return "R32G32_SFLOAT";
        case Format::R32G32B32_UINT: return "R32G32B32_UINT";
        case Format::R32G32B32_SINT: return "R32G32B32_SINT";
        case Format::R32G32B32_SFLOAT: return "R32G32B32_SFLOAT";
        case Format::R32G32B32A32_UINT: return "R32G32B32A32_UINT";
        case Format::R32G32B32A32_SINT: return "R32G32B32A32_SINT";
        case Format::R32G32B32A32_SFLOAT: return "R32G32B32A32_SFLOAT";
        case Format::R64_UINT: return "R64_UINT";
        case Format::R64_SINT: return "R64_SINT";
        case Format::R64_SFLOAT: return "R64_SFLOAT";
        case Format::R64G64_UINT: return "R64G64_UINT";
        case Format::R64G64_SINT: return "R64G64_SINT";
        case Format::R64G64_SFLOAT: return "R64G64_SFLOAT";
        case Format::R64G64B64_UINT: return "R64G64B64_UINT";
        case Format::R64G64B64_SINT: return "R64G64B64_SINT";
        case Format::R64G64B64_SFLOAT: return "R64G64B64_SFLOAT";
        case Format::R64G64B64A64_UINT: return "R64G64B64A64_UINT";
        case Format::R64G64B64A64_SINT: return "R64G64B64A64_SINT";
        case Format::R64G64B64A64_SFLOAT: return "R64G64B64A64_SFLOAT";
        case Format::B10G11R11_UFLOAT_PACK32: return "B10G11R11_UFLOAT_PACK32";
        case Format::E5B9G9R9_UFLOAT_PACK32: return "E5B9G9R9_UFLOAT_PACK32";
        case Format::D16_UNORM: return "D16_UNORM";
        case Format::X8_D24_UNORM_PACK32: return "X8_D24_UNORM_PACK32";
        case Format::D32_SFLOAT: return "D32_SFLOAT";
        case Format::S8_UINT: return "S8_UINT";
        case Format::D16_UNORM_S8_UINT: return "D16_UNORM_S8_UINT";
        case Format::D24_UNORM_S8_UINT: return "D24_UNORM_S8_UINT";
        case Format::D32_SFLOAT_S8_UINT: return "D32_SFLOAT_S8_UINT";
        case Format::BC1_RGB_UNORM_BLOCK: return "BC1_RGB_UNORM_BLOCK";
        case Format::BC1_RGB_SRGB_BLOCK: return "BC1_RGB_SRGB_BLOCK";
        case Format::BC1_RGBA_UNORM_BLOCK: return "BC1_RGBA_UNORM_BLOCK";
        case Format::BC1_RGBA_SRGB_BLOCK: return "BC1_RGBA_SRGB_BLOCK";
        case Format::BC2_UNORM_BLOCK: return "BC2_UNORM_BLOCK";
        case Format::BC2_SRGB_BLOCK: return "BC2_SRGB_BLOCK";
        case Format::BC3_UNORM_BLOCK: return "BC3_UNORM_BLOCK";
        case Format::BC3_SRGB_BLOCK: return "BC3_SRGB_BLOCK";
        case Format::BC4_UNORM_BLOCK: return "BC4_UNORM_BLOCK";
        case Format::BC4_SNORM_BLOCK: return "BC4_SNORM_BLOCK";
        case Format::BC5_UNORM_BLOCK: return "BC5_UNORM_BLOCK";
        case Format::BC5_SNORM_BLOCK: return "BC5_SNORM_BLOCK";
        case Format::BC6H_UFLOAT_BLOCK: return "BC6H_UFLOAT_BLOCK";
        case Format::BC6H_SFLOAT_BLOCK: return "BC6H_SFLOAT_BLOCK";
        case Format::BC7_UNORM_BLOCK: return "BC7_UNORM_BLOCK";
        case Format::BC7_SRGB_BLOCK: return "BC7_SRGB_BLOCK";
        case Format::ETC2_R8G8B8_UNORM_BLOCK: return "ETC2_R8G8B8_UNORM_BLOCK";
        case Format::ETC2_R8G8B8_SRGB_BLOCK: return "ETC2_R8G8B8_SRGB_BLOCK";
        case Format::ETC2_R8G8B8A1_UNORM_BLOCK: return "ETC2_R8G8B8A1_UNORM_BLOCK";
        case Format::ETC2_R8G8B8A1_SRGB_BLOCK: return "ETC2_R8G8B8A1_SRGB_BLOCK";
        case Format::ETC2_R8G8B8A8_UNORM_BLOCK: return "ETC2_R8G8B8A8_UNORM_BLOCK";
        case Format::ETC2_R8G8B8A8_SRGB_BLOCK: return "ETC2_R8G8B8A8_SRGB_BLOCK";
        case Format::EAC_R11_UNORM_BLOCK: return "EAC_R11_UNORM_BLOCK";
        case Format::EAC_R11_SNORM_BLOCK: return "EAC_R11_SNORM_BLOCK";
        case Format::EAC_R11G11_UNORM_BLOCK: return "EAC_R11G11_UNORM_BLOCK";
        case Format::EAC_R11G11_SNORM_BLOCK: return "EAC_R11G11_SNORM_BLOCK";
        case Format::ASTC_4x4_UNORM_BLOCK: return "ASTC_4x4_UNORM_BLOCK";
        case Format::ASTC_4x4_SRGB_BLOCK: return "ASTC_4x4_SRGB_BLOCK";
        case Format::ASTC_5x4_UNORM_BLOCK: return "ASTC_5x4_UNORM_BLOCK";
        case Format::ASTC_5x4_SRGB_BLOCK: return "ASTC_5x4_SRGB_BLOCK";
        case Format::ASTC_5x5_UNORM_BLOCK: return "ASTC_5x5_UNORM_BLOCK";
        case Format::ASTC_5x5_SRGB_BLOCK: return "ASTC_5x5_SRGB_BLOCK";
        case Format::ASTC_6x5_UNORM_BLOCK: return "ASTC_6x5_UNORM_BLOCK";
        case Format::ASTC_6x5_SRGB_BLOCK: return "ASTC_6x5_SRGB_BLOCK";
        case Format::ASTC_6x6_UNORM_BLOCK: return "ASTC_6x6_UNORM_BLOCK";
        case Format::ASTC_6x6_SRGB_BLOCK: return "ASTC_6x6_SRGB_BLOCK";
        case Format::ASTC_8x5_UNORM_BLOCK: return "ASTC_8x5_UNORM_BLOCK";
        case Format::ASTC_8x5_SRGB_BLOCK: return "ASTC_8x5_SRGB_BLOCK";
        case Format::ASTC_8x6_UNORM_BLOCK: return "ASTC_8x6_UNORM_BLOCK";
        case Format::ASTC_8x6_SRGB_BLOCK: return "ASTC_8x6_SRGB_BLOCK";
        case Format::ASTC_8x8_UNORM_BLOCK: return "ASTC_8x8_UNORM_BLOCK";
        case Format::ASTC_8x8_SRGB_BLOCK: return "ASTC_8x8_SRGB_BLOCK";
        case Format::ASTC_10x5_UNORM_BLOCK: return "ASTC_10x5_UNORM_BLOCK";
        case Format::ASTC_10x5_SRGB_BLOCK: return "ASTC_10x5_SRGB_BLOCK";
        case Format::ASTC_10x6_UNORM_BLOCK: return "ASTC_10x6_UNORM_BLOCK";
        case Format::ASTC_10x6_SRGB_BLOCK: return "ASTC_10x6_SRGB_BLOCK";
        case Format::ASTC_10x8_UNORM_BLOCK: return "ASTC_10x8_UNORM_BLOCK";
        case Format::ASTC_10x8_SRGB_BLOCK: return "ASTC_10x8_SRGB_BLOCK";
        case Format::ASTC_10x10_UNORM_BLOCK: return "ASTC_10x10_UNORM_BLOCK";
        case Format::ASTC_10x10_SRGB_BLOCK: return "ASTC_10x10_SRGB_BLOCK";
        case Format::ASTC_12x10_UNORM_BLOCK: return "ASTC_12x10_UNORM_BLOCK";
        case Format::ASTC_12x10_SRGB_BLOCK: return "ASTC_12x10_SRGB_BLOCK";
        case Format::ASTC_12x12_UNORM_BLOCK: return "ASTC_12x12_UNORM_BLOCK";
        case Format::ASTC_12x12_SRGB_BLOCK: return "ASTC_12x12_SRGB_BLOCK";
        case Format::G8B8G8R8_422_UNORM: return "G8B8G8R8_422_UNORM";
        case Format::B8G8R8G8_422_UNORM: return "B8G8R8G8_422_UNORM";
        case Format::G8_B8_R8_3PLANE_420_UNORM: return "G8_B8_R8_3PLANE_420_UNORM";
        case Format::G8_B8R8_2PLANE_420_UNORM: return "G8_B8R8_2PLANE_420_UNORM";
        case Format::G8_B8_R8_3PLANE_422_UNORM: return "G8_B8_R8_3PLANE_422_UNORM";
        case Format::G8_B8R8_2PLANE_422_UNORM: return "G8_B8R8_2PLANE_422_UNORM";
        case Format::G8_B8_R8_3PLANE_444_UNORM: return "G8_B8_R8_3PLANE_444_UNORM";
        case Format::R10X6_UNORM_PACK16: return "R10X6_UNORM_PACK16";
        case Format::R10X6G10X6_UNORM_2PACK16: return "R10X6G10X6_UNORM_2PACK16";
        case Format::R10X6G10X6B10X6A10X6_UNORM_4PACK16: return "R10X6G10X6B10X6A10X6_UNORM_4PACK16";
        case Format::G10X6B10X6G10X6R10X6_422_UNORM_4PACK16: return "G10X6B10X6G10X6R10X6_422_UNORM_4PACK16";
        case Format::B10X6G10X6R10X6G10X6_422_UNORM_4PACK16: return "B10X6G10X6R10X6G10X6_422_UNORM_4PACK16";
        case Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16: return "G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16";
        case Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16: return "G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16";
        case Format::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16: return "G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16";
        case Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16: return "G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16";
        case Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16: return "G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16";
        case Format::R12X4_UNORM_PACK16: return "R12X4_UNORM_PACK16";
        case Format::R12X4G12X4_UNORM_2PACK16: return "R12X4G12X4_UNORM_2PACK16";
        case Format::R12X4G12X4B12X4A12X4_UNORM_4PACK16: return "R12X4G12X4B12X4A12X4_UNORM_4PACK16";
        case Format::G12X4B12X4G12X4R12X4_422_UNORM_4PACK16: return "G12X4B12X4G12X4R12X4_422_UNORM_4PACK16";
        case Format::B12X4G12X4R12X4G12X4_422_UNORM_4PACK16: return "B12X4G12X4R12X4G12X4_422_UNORM_4PACK16";
        case Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16: return "G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16";
        case Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16: return "G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16";
        case Format::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16: return "G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16";
        case Format::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16: return "G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16";
        case Format::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16: return "G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16";
        case Format::G16B16G16R16_422_UNORM: return "G16B16G16R16_422_UNORM";
        case Format::B16G16R16G16_422_UNORM: return "B16G16R16G16_422_UNORM";
        case Format::G16_B16_R16_3PLANE_420_UNORM: return "G16_B16_R16_3PLANE_420_UNORM";
        case Format::G16_B16R16_2PLANE_420_UNORM: return "G16_B16R16_2PLANE_420_UNORM";
        case Format::G16_B16_R16_3PLANE_422_UNORM: return "G16_B16_R16_3PLANE_422_UNORM";
        case Format::G16_B16R16_2PLANE_422_UNORM: return "G16_B16R16_2PLANE_422_UNORM";
        case Format::G16_B16_R16_3PLANE_444_UNORM: return "G16_B16_R16_3PLANE_444_UNORM";
        case Format::G8_B8R8_2PLANE_444_UNORM: return "G8_B8R8_2PLANE_444_UNORM";
        case Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16: return "G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16";
        case Format::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16: return "G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16";
        case Format::G16_B16R16_2PLANE_444_UNORM: return "G16_B16R16_2PLANE_444_UNORM";
        case Format::A4R4G4B4_UNORM_PACK16: return "A4R4G4B4_UNORM_PACK16";
        case Format::A4B4G4R4_UNORM_PACK16: return "A4B4G4R4_UNORM_PACK16";
        case Format::ASTC_4x4_SFLOAT_BLOCK: return "ASTC_4x4_SFLOAT_BLOCK";
        case Format::ASTC_5x4_SFLOAT_BLOCK: return "ASTC_5x4_SFLOAT_BLOCK";
        case Format::ASTC_5x5_SFLOAT_BLOCK: return "ASTC_5x5_SFLOAT_BLOCK";
        case Format::ASTC_6x5_SFLOAT_BLOCK: return "ASTC_6x5_SFLOAT_BLOCK";
        case Format::ASTC_6x6_SFLOAT_BLOCK: return "ASTC_6x6_SFLOAT_BLOCK";
        case Format::ASTC_8x5_SFLOAT_BLOCK: return "ASTC_8x5_SFLOAT_BLOCK";
        case Format::ASTC_8x6_SFLOAT_BLOCK: return "ASTC_8x6_SFLOAT_BLOCK";
        case Format::ASTC_8x8_SFLOAT_BLOCK: return "ASTC_8x8_SFLOAT_BLOCK";
        case Format::ASTC_10x5_SFLOAT_BLOCK: return "ASTC_10x5_SFLOAT_BLOCK";
        case Format::ASTC_10x6_SFLOAT_BLOCK: return "ASTC_10x6_SFLOAT_BLOCK";
        case Format::ASTC_10x8_SFLOAT_BLOCK: return "ASTC_10x8_SFLOAT_BLOCK";
        case Format::ASTC_10x10_SFLOAT_BLOCK: return "ASTC_10x10_SFLOAT_BLOCK";
        case Format::ASTC_12x10_SFLOAT_BLOCK: return "ASTC_12x10_SFLOAT_BLOCK";
        case Format::ASTC_12x12_SFLOAT_BLOCK: return "ASTC_12x12_SFLOAT_BLOCK";
        case Format::PVRTC1_2BPP_UNORM_BLOCK_IMG: return "PVRTC1_2BPP_UNORM_BLOCK_IMG";
        case Format::PVRTC1_4BPP_UNORM_BLOCK_IMG: return "PVRTC1_4BPP_UNORM_BLOCK_IMG";
        case Format::PVRTC2_2BPP_UNORM_BLOCK_IMG: return "PVRTC2_2BPP_UNORM_BLOCK_IMG";
        case Format::PVRTC2_4BPP_UNORM_BLOCK_IMG: return "PVRTC2_4BPP_UNORM_BLOCK_IMG";
        case Format::PVRTC1_2BPP_SRGB_BLOCK_IMG: return "PVRTC1_2BPP_SRGB_BLOCK_IMG";
        case Format::PVRTC1_4BPP_SRGB_BLOCK_IMG: return "PVRTC1_4BPP_SRGB_BLOCK_IMG";
        case Format::PVRTC2_2BPP_SRGB_BLOCK_IMG: return "PVRTC2_2BPP_SRGB_BLOCK_IMG";
        case Format::PVRTC2_4BPP_SRGB_BLOCK_IMG: return "PVRTC2_4BPP_SRGB_BLOCK_IMG";
        default: return "unknown";
        }
    }

    auto to_string(ImageMipArraySlice slice) -> std::string
    {
        return fmt::format("mips: {}-{}, layers: {}-{}",
                           slice.base_mip_level,
                           slice.base_mip_level + slice.level_count - 1,
                           slice.base_array_layer,
                           slice.base_array_layer + slice.layer_count - 1);
    }

    auto to_string(ImageArraySlice slice) -> std::string
    {
        return fmt::format("mip: {}, layers: {}-{}",
                           slice.mip_level,
                           slice.base_array_layer,
                           slice.base_array_layer + slice.layer_count - 1);
    }

    auto to_string(ImageSlice slice) -> std::string
    {
        return fmt::format(" mip: {}, layer: {}",
                           slice.mip_level,
                           slice.array_layer);
    }

    auto to_string(PipelineStageFlags flags) -> std::string
    {
        if (flags == PipelineStageFlagBits::NONE)
        {
            return "NONE";
        }

        std::string ret = {};

        if ((flags & PipelineStageFlagBits::TOP_OF_PIPE) != PipelineStageFlagBits::NONE)
        {
            ret += "TOP_OF_PIPE";
        }
        if ((flags & PipelineStageFlagBits::DRAW_INDIRECT) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "DRAW_INDIRECT";
        }
        if ((flags & PipelineStageFlagBits::VERTEX_SHADER) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "VERTEX_SHADER";
        }
        if ((flags & PipelineStageFlagBits::TASK_SHADER) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "TASK_SHADER";
        }
        if ((flags & PipelineStageFlagBits::MESH_SHADER) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "MESH_SHADER";
        }
        if ((flags & PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "TESSELLATION_CONTROL_SHADER";
        }
        if ((flags & PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "TESSELLATION_EVALUATION_SHADER";
        }
        if ((flags & PipelineStageFlagBits::GEOMETRY_SHADER) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "GEOMETRY_SHADER";
        }
        if ((flags & PipelineStageFlagBits::FRAGMENT_SHADER) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "FRAGMENT_SHADER";
        }
        if ((flags & PipelineStageFlagBits::EARLY_FRAGMENT_TESTS) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "EARLY_FRAGMENT_TESTS";
        }
        if ((flags & PipelineStageFlagBits::LATE_FRAGMENT_TESTS) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "LATE_FRAGMENT_TESTS";
        }
        if ((flags & PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "COLOR_ATTACHMENT_OUTPUT";
        }
        if ((flags & PipelineStageFlagBits::COMPUTE_SHADER) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "COMPUTE";
        }
        if ((flags & PipelineStageFlagBits::RAY_TRACING_SHADER) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "RAY_TRACING_SHADER";
        }
        if ((flags & PipelineStageFlagBits::TRANSFER) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "TRANSFER";
        }
        if ((flags & PipelineStageFlagBits::BOTTOM_OF_PIPE) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "BOTTOM_OF_PIPE";
        }
        if ((flags & PipelineStageFlagBits::HOST) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "HOST";
        }
        if ((flags & PipelineStageFlagBits::ALL_GRAPHICS) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "ALL_GRAPHICS";
        }
        if ((flags & PipelineStageFlagBits::ALL_COMMANDS) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "ALL_COMMANDS";
        }
        if ((flags & PipelineStageFlagBits::COPY) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "COPY";
        }
        if ((flags & PipelineStageFlagBits::RESOLVE) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "RESOLVE";
        }
        if ((flags & PipelineStageFlagBits::BLIT) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "BLIT";
        }
        if ((flags & PipelineStageFlagBits::CLEAR) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "CLEAR";
        }
        if ((flags & PipelineStageFlagBits::INDEX_INPUT) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "INDEX_INPUT";
        }
        if ((flags & PipelineStageFlagBits::PRE_RASTERIZATION_SHADERS) != PipelineStageFlagBits::NONE)
        {
            if (!ret.empty())
            {
                ret += " | ";
            }
            ret += "PRE_RASTERIZATION_SHADERS";
        }
        return ret;
    }

    auto to_string(Access access) -> std::string
    {
        return fmt::format("stages: {}, type: {}", to_string(access.stages), to_string(access.type));
    }

    auto to_string(QueueFamily family) -> std::string_view
    {
        switch (family)
        {
        case QueueFamily::MAIN: return "MAIN";
        case QueueFamily::COMPUTE: return "COMPUTE";
        case QueueFamily::TRANSFER: return "TRANSFER";
        };
    }

    /// --- End to_string ---

    // --- Begin Misc ---

    auto ImageMipArraySlice::contains(ImageMipArraySlice const & slice) const -> bool
    {
        u32 const a_mip_p0 = this->base_mip_level;
        u32 const a_mip_p1 = this->base_mip_level + this->level_count - 1;
        u32 const b_mip_p0 = slice.base_mip_level;
        u32 const b_mip_p1 = slice.base_mip_level + slice.level_count - 1;

        u32 const a_arr_p0 = this->base_array_layer;
        u32 const a_arr_p1 = this->base_array_layer + this->layer_count - 1;
        u32 const b_arr_p0 = slice.base_array_layer;
        u32 const b_arr_p1 = slice.base_array_layer + slice.layer_count - 1;

        return b_mip_p0 >= a_mip_p0 &&
               b_mip_p1 <= a_mip_p1 &&
               b_arr_p0 >= a_arr_p0 &&
               b_arr_p1 <= a_arr_p1;
    }
    auto ImageMipArraySlice::intersects(ImageMipArraySlice const & slice) const -> bool
    {
        u32 const a_mip_p0 = this->base_mip_level;
        u32 const a_mip_p1 = this->base_mip_level + this->level_count - 1;
        u32 const b_mip_p0 = slice.base_mip_level;
        u32 const b_mip_p1 = slice.base_mip_level + slice.level_count - 1;

        u32 const a_arr_p0 = this->base_array_layer;
        u32 const a_arr_p1 = this->base_array_layer + this->layer_count - 1;
        u32 const b_arr_p0 = slice.base_array_layer;
        u32 const b_arr_p1 = slice.base_array_layer + slice.layer_count - 1;

        bool const mip_disjoint = (a_mip_p1 < b_mip_p0) || (b_mip_p1 < a_mip_p0);
        bool const arr_disjoint = (a_arr_p1 < b_arr_p0) || (b_arr_p1 < a_arr_p0);

        return !mip_disjoint && !arr_disjoint;
    }
    auto ImageMipArraySlice::intersect(ImageMipArraySlice const & slice) const -> ImageMipArraySlice
    {
        u32 const a_mip_p0 = this->base_mip_level;
        u32 const a_mip_p1 = this->base_mip_level + this->level_count - 1;
        u32 const b_mip_p0 = slice.base_mip_level;
        u32 const b_mip_p1 = slice.base_mip_level + slice.level_count - 1;
        u32 const max_mip_p0 = std::max(a_mip_p0, b_mip_p0);
        u32 const min_mip_p1 = std::min(a_mip_p1, b_mip_p1);

        u32 const a_arr_p0 = this->base_array_layer;
        u32 const a_arr_p1 = this->base_array_layer + this->layer_count - 1;
        u32 const b_arr_p0 = slice.base_array_layer;
        u32 const b_arr_p1 = slice.base_array_layer + slice.layer_count - 1;
        u32 const max_arr_p0 = std::max(a_arr_p0, b_arr_p0);
        u32 const min_arr_p1 = std::min(a_arr_p1, b_arr_p1);

        // NOTE(grundlett): This multiplication at the end is to cancel out
        // the potential underflow of unsigned integers. Since the p1 could
        // could technically be less than the p0, this means that after doing
        // p1 + 1 - p0, you should get a "negative" number.
        u32 const mip_n = (min_mip_p1 + 1 - max_mip_p0) * static_cast<u32>(max_mip_p0 <= min_mip_p1);
        u32 const arr_n = (min_arr_p1 + 1 - max_arr_p0) * static_cast<u32>(max_arr_p0 <= min_arr_p1);

        return ImageMipArraySlice{
            .base_mip_level = max_mip_p0,
            .level_count = mip_n,
            .base_array_layer = max_arr_p0,
            .layer_count = arr_n,
        };
    }
    auto ImageMipArraySlice::subtract(ImageMipArraySlice const & slice) const -> std::tuple<std::array<ImageMipArraySlice, 4>, usize>
    {
        u32 const a_mip_p0 = this->base_mip_level;
        u32 const a_mip_p1 = this->base_mip_level + this->level_count - 1;
        u32 const b_mip_p0 = slice.base_mip_level;
        u32 const b_mip_p1 = slice.base_mip_level + slice.level_count - 1;

        u32 const a_arr_p0 = this->base_array_layer;
        u32 const a_arr_p1 = this->base_array_layer + this->layer_count - 1;
        u32 const b_arr_p0 = slice.base_array_layer;
        u32 const b_arr_p1 = slice.base_array_layer + slice.layer_count - 1;

        u32 const mip_case = static_cast<u32>(b_mip_p1 < a_mip_p1) + static_cast<u32>(b_mip_p0 > a_mip_p0) * 2;
        u32 const arr_case = static_cast<u32>(b_arr_p1 < a_arr_p1) + static_cast<u32>(b_arr_p0 > a_arr_p0) * 2;

        std::tuple<std::array<ImageMipArraySlice, 4>, usize> result = {};
        if (!this->intersects(slice))
        {
            auto & [result_rects, result_n] = result;
            result_n = 1;
            result_rects[0] = *this;
            return result;
        }

        // clang-format off
        //
        //     mips ➡️
        // arrays       0              1          2            3
        //  ⬇️
        //
        //           ▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓             ▓▓▓▓       ▓▓  
        //  0      A ▓▓██████▓▓   B ▓▓██░░░░   C ░░░░██▓▓   D ░░██░░
        //           ▓▓██████▓▓     ▓▓██░░░░     ░░░░██▓▓     ░░██░░
        //           ▓▓██████▓▓     ▓▓██░░░░     ░░░░██▓▓     ░░██░░
        //           ▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓             ▓▓▓▓       ▓▓  
        //
        //           ▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓             ▓▓▓▓       ▓▓  
        //  1      E ▓▓██████▓▓   F ▓▓██░░░░   G ░░░░██▓▓   H ░░██░░
        //             ░░░░░░         ░░░░░░     ░░░░░░       ░░░░░░
        //             ░░░░░░         ░░░░░░     ░░░░░░       ░░░░░░
        //
        //  3      I   ░░░░░░     J   ░░░░░░   K ░░░░░░     L ░░░░░░
        //             ░░░░░░         ░░░░░░     ░░░░░░       ░░░░░░
        //           ▓▓██████▓▓     ▓▓██░░░░     ░░░░██▓▓     ░░██░░
        //           ▓▓▓▓▓▓▓▓▓▓     ▓▓▓▓             ▓▓▓▓       ▓▓  
        //
        //  2      M   ░░░░░░     N   ░░░░░░   O ░░░░░░     P ░░░░░░
        //           ▓▓██████▓▓     ▓▓██░░░░     ░░░░██▓▓     ░░██░░
        //             ░░░░░░         ░░░░░░     ░░░░░░       ░░░░░░
        //
        // clang-format on

        // clang-format off
        static constexpr std::array<usize, 16> rect_n {
            0, 1, 1, 2,
            1, 2, 2, 3,
            1, 2, 2, 3,
            2, 3, 3, 4,
        };

        #define NO_RBC {0, 0}
        struct RectBCIndices {
            usize mip_i;
            usize arr_i;
        };
        //   0      1      2      3      4      5
        // b1>a1  a0>b0  a0>a1  a0>b1  b0>b1  b0>a1
        static constexpr std::array<std::array<RectBCIndices, 4>, 16> bc_indices = {{
            {{NO_RBC, NO_RBC, NO_RBC, NO_RBC}},   {{{0, 2}, NO_RBC, NO_RBC, NO_RBC}},   {{{1, 2}, NO_RBC, NO_RBC, NO_RBC}},   {{{1, 2}, {0, 2}, NO_RBC, NO_RBC}},
            {{{2, 0}, NO_RBC, NO_RBC, NO_RBC}},   {{{0, 3}, {2, 0}, NO_RBC, NO_RBC}},   {{{1, 3}, {2, 0}, NO_RBC, NO_RBC}},   {{{1, 3}, {0, 3}, {2, 0}, NO_RBC}},
            {{{2, 1}, NO_RBC, NO_RBC, NO_RBC}},   {{{2, 1}, {0, 5}, NO_RBC, NO_RBC}},   {{{2, 1}, {1, 5}, NO_RBC, NO_RBC}},   {{{2, 1}, {1, 5}, {0, 5}, NO_RBC}},
            {{{2, 1}, {2, 0}, NO_RBC, NO_RBC}},   {{{2, 1}, {0, 4}, {2, 0}, NO_RBC}},   {{{2, 1}, {1, 4}, {2, 0}, NO_RBC}},   {{{2, 1}, {1, 4}, {0, 4}, {2, 0}}},
        }};
        // clang-format on

        struct BaseAndCount
        {
            u32 base;
            u32 count;
        };
        std::array<BaseAndCount, 3> const mip_bc{
            BaseAndCount{.base = b_mip_p1 + 1, .count = (a_mip_p1 + 1) - (b_mip_p1 + 1)}, // b1 -> a1
            BaseAndCount{.base = a_mip_p0, .count = b_mip_p0 - a_mip_p0},                 // a0 -> b0
            BaseAndCount{.base = a_mip_p0, .count = (a_mip_p1 + 1) - a_mip_p0},           // a0 -> a1
        };
        std::array<BaseAndCount, 6> const arr_bc{
            BaseAndCount{.base = b_arr_p1 + 1, .count = (a_arr_p1 + 1) - (b_arr_p1 + 1)}, // b1 -> a1
            BaseAndCount{.base = a_arr_p0, .count = b_arr_p0 - a_arr_p0},                 // a0 -> b0
            BaseAndCount{.base = a_arr_p0, .count = (a_arr_p1 + 1) - a_arr_p0},           // a0 -> a1
            BaseAndCount{.base = a_arr_p0, .count = (b_arr_p1 + 1) - a_arr_p0},           // a0 -> b1
            BaseAndCount{.base = b_arr_p0, .count = (b_arr_p1 + 1) - b_arr_p0},           // b0 -> b1
            BaseAndCount{.base = b_arr_p0, .count = (a_arr_p1 + 1) - b_arr_p0},           // b0 -> a1
        };

        usize const result_index = mip_case + arr_case * 4;
        usize const result_rect_n = rect_n.at(result_index);
        auto const & bc = bc_indices.at(result_index);
        std::get<1>(result) = result_rect_n;

        for (usize i = 0; i < result_rect_n; ++i)
        {
            auto & rect_i = std::get<0>(result)[i];
            auto const & bc_i = bc.at(i);
            rect_i = *this;
            rect_i.base_mip_level = mip_bc.at(bc_i.mip_i).base;
            rect_i.level_count = mip_bc.at(bc_i.mip_i).count;
            rect_i.base_array_layer = arr_bc.at(bc_i.arr_i).base;
            rect_i.layer_count = arr_bc.at(bc_i.arr_i).count;
        }

        return result;
    }

    auto ImageArraySlice::slice(ImageMipArraySlice const & mip_array_slice, u32 mip_level) -> ImageArraySlice
    {
        DAXA_DBG_ASSERT_TRUE_M(mip_level >= mip_array_slice.base_mip_level && mip_level < (mip_array_slice.base_mip_level + mip_array_slice.level_count), "slices mip level must be contained in initial slice");

        return ImageArraySlice{
            .mip_level = mip_level,
            .base_array_layer = mip_array_slice.base_array_layer,
            .layer_count = mip_array_slice.layer_count,
        };
    }

    auto ImageArraySlice::contained_in(ImageMipArraySlice const & slice) const -> bool
    {
        return this->mip_level >= slice.base_mip_level &&
               this->mip_level < (slice.base_mip_level + slice.level_count) &&
               this->base_array_layer >= slice.base_array_layer &&
               (this->base_array_layer + this->layer_count) <= (slice.base_array_layer + slice.layer_count);
    }

    auto slice(ImageArraySlice const & mip_array_slice, u32 array_layer) -> ImageSlice
    {
        DAXA_DBG_ASSERT_TRUE_M(array_layer >= mip_array_slice.base_array_layer && array_layer < (mip_array_slice.base_array_layer + mip_array_slice.layer_count), "slices array layer must be contained in initial slice");

        return ImageSlice{
            .mip_level = mip_array_slice.mip_level,
            .array_layer = array_layer,
        };
    }

    auto ImageSlice::contained_in(ImageMipArraySlice const & slice) const -> bool
    {
        return this->mip_level >= slice.base_mip_level &&
               this->mip_level < (slice.base_mip_level + slice.level_count) &&
               this->array_layer >= slice.base_array_layer &&
               array_layer < (slice.base_array_layer + slice.layer_count);
    }

    auto ImageSlice::contained_in(ImageArraySlice const & slice) const -> bool
    {
        return this->mip_level == slice.mip_level &&
               this->array_layer >= slice.base_array_layer &&
               array_layer < (slice.base_array_layer + slice.layer_count);
    }

    auto operator|(Access const & a, Access const & b) -> Access
    {
        return Access{.stages = a.stages | b.stages, .type = a.type | b.type};
    }

    auto operator&(Access const & a, Access const & b) -> Access
    {
        return Access{.stages = a.stages & b.stages, .type = a.type & b.type};
    }

    // -- End Misc ---
} // namespace daxa
