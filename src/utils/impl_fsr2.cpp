#if DAXA_BUILT_WITH_UTILS_FSR2

#include "impl_fsr2.hpp"

#include <cstring>
#include <utility>

namespace daxa
{
    Fsr2Context::Fsr2Context(UpscaleInstanceInfo const & info)
    {
        this->object = new ImplFsr2Context(info);
    }

    void Fsr2Context::resize(UpscaleSizeInfo const & info)
    {
        auto & impl = *r_cast<ImplFsr2Context *>(this->object);
        impl.resize(info);
    }

    void Fsr2Context::upscale(CommandRecorder & command_list, UpscaleInfo const & info)
    {
        auto & impl = *r_cast<ImplFsr2Context *>(this->object);
        impl.upscale(command_list, info);
    }

    auto Fsr2Context::get_jitter(u64 index) const -> daxa_f32vec2
    {
        auto const & impl = *r_cast<ImplFsr2Context *>(this->object);
        daxa_f32vec2 result{};
        i32 const jitter_phase_count = ffxFsr2GetJitterPhaseCount(static_cast<i32>(impl.info.size_info.render_size_x), static_cast<i32>(impl.info.size_info.display_size_x));
        ffxFsr2GetJitterOffset(&result.x, &result.y, static_cast<i32>(index), jitter_phase_count);
        return result;
    }

    ImplFsr2Context::ImplFsr2Context(UpscaleInstanceInfo a_info)
        : info{std::move(a_info)}
    {
    }

    ImplFsr2Context::~ImplFsr2Context() // NOLINT(bugprone-exception-escape)
    {
        destroy_resizable_resources();
    }

    void ImplFsr2Context::create_resizable_resources()
    {
        destroy_resizable_resources();
        fsr2_context_description.maxRenderSize.width = this->info.size_info.render_size_x;
        fsr2_context_description.maxRenderSize.height = this->info.size_info.render_size_y;
        fsr2_context_description.displaySize.width = this->info.size_info.display_size_x;
        fsr2_context_description.displaySize.height = this->info.size_info.display_size_y;

        // Setup VK interface.
        auto logical_device = daxa_dvc_get_vk_device(*reinterpret_cast<daxa_Device *>(&this->info.device));
        auto physical_device = daxa_dvc_get_vk_physical_device(*reinterpret_cast<daxa_Device *>(&this->info.device));

        usize const scratch_buffer_size = ffxFsr2GetScratchMemorySizeVK(physical_device);
        scratch_buffer = malloc(scratch_buffer_size);
        {
            [[maybe_unused]] FfxErrorCode const err = ffxFsr2GetInterfaceVK(&fsr2_context_description.callbacks, scratch_buffer, scratch_buffer_size, physical_device, vkGetDeviceProcAddr);
            DAXA_DBG_ASSERT_TRUE_M(err == FFX_OK, "FSR2 Failed to create the Vulkan interface");
        }

        // set up for later, when we resize
        fsr2_context_description.device = ffxGetDeviceVK(logical_device);
        fsr2_context_description.flags = {};
        fsr2_context_description.flags |= FFX_FSR2_ENABLE_AUTO_EXPOSURE;
        if (this->info.color_hdr) {
            fsr2_context_description.flags |= FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE;
        }
        if (this->info.depth_inv) {
            fsr2_context_description.flags |= FFX_FSR2_ENABLE_DEPTH_INVERTED;
        }
        if (this->info.depth_inf) {
            fsr2_context_description.flags |= FFX_FSR2_ENABLE_DEPTH_INFINITE;
        }

        {
            [[maybe_unused]] FfxErrorCode const err = ffxFsr2ContextCreate(&fsr2_context, &fsr2_context_description);
            DAXA_DBG_ASSERT_TRUE_M(err == FFX_OK, "FSR2 Failed to create the FSR context");
        }
        initialized = true;
    }

    void ImplFsr2Context::destroy_resizable_resources()
    {
        if (initialized)
        {
            [[maybe_unused]] FfxErrorCode const err = ffxFsr2ContextDestroy(&fsr2_context);
            DAXA_DBG_ASSERT_TRUE_M(err == FFX_OK, "FSR2 Failed to destroy the FSR context");
            free(scratch_buffer);
        }
    }

    void ImplFsr2Context::resize(UpscaleSizeInfo const & resize_info)
    {
        this->info.size_info = resize_info;
        create_resizable_resources();
    }

    void ImplFsr2Context::upscale(CommandRecorder & recorder, UpscaleInfo const & upscale_info)
    {
#define HANDLE_RES(expression)          \
    {                                   \
        auto res = expression;          \
        if (res != DAXA_RESULT_SUCCESS) \
        {                               \
            return;                     \
        }                               \
    }

        VkImage color_vk_image = {}, depth_vk_image = {}, velocity_vk_image = {}, output_vk_image = {};
        HANDLE_RES(daxa_dvc_get_vk_image(*reinterpret_cast<daxa_Device *>(&this->info.device), std::bit_cast<daxa_ImageId>(upscale_info.color), &color_vk_image));
        HANDLE_RES(daxa_dvc_get_vk_image(*reinterpret_cast<daxa_Device *>(&this->info.device), std::bit_cast<daxa_ImageId>(upscale_info.depth), &depth_vk_image));
        HANDLE_RES(daxa_dvc_get_vk_image(*reinterpret_cast<daxa_Device *>(&this->info.device), std::bit_cast<daxa_ImageId>(upscale_info.motion_vectors), &velocity_vk_image));
        HANDLE_RES(daxa_dvc_get_vk_image(*reinterpret_cast<daxa_Device *>(&this->info.device), std::bit_cast<daxa_ImageId>(upscale_info.output), &output_vk_image));

        VkImageView color_vk_image_view = {}, depth_vk_image_view = {}, velocity_vk_image_view = {}, output_vk_image_view = {};
        HANDLE_RES(daxa_dvc_get_vk_image_view(*reinterpret_cast<daxa_Device *>(&this->info.device), std::bit_cast<daxa_ImageViewId>(upscale_info.color.default_view()), &color_vk_image_view));
        HANDLE_RES(daxa_dvc_get_vk_image_view(*reinterpret_cast<daxa_Device *>(&this->info.device), std::bit_cast<daxa_ImageViewId>(upscale_info.depth.default_view()), &depth_vk_image_view));
        HANDLE_RES(daxa_dvc_get_vk_image_view(*reinterpret_cast<daxa_Device *>(&this->info.device), std::bit_cast<daxa_ImageViewId>(upscale_info.motion_vectors.default_view()), &velocity_vk_image_view));
        HANDLE_RES(daxa_dvc_get_vk_image_view(*reinterpret_cast<daxa_Device *>(&this->info.device), std::bit_cast<daxa_ImageViewId>(upscale_info.output.default_view()), &output_vk_image_view));

        auto const & color_extent = this->info.device.image_info(upscale_info.color).value().size;
        auto const & depth_extent = this->info.device.image_info(upscale_info.depth).value().size;
        auto const & velocity_extent = this->info.device.image_info(upscale_info.motion_vectors).value().size;
        auto const & output_extent = this->info.device.image_info(upscale_info.output).value().size;

        auto const color_format = static_cast<VkFormat>(this->info.device.image_info(upscale_info.color).value().format);
        auto const depth_format = static_cast<VkFormat>(this->info.device.image_info(upscale_info.depth).value().format);
        auto const velocity_format = static_cast<VkFormat>(this->info.device.image_info(upscale_info.motion_vectors).value().format);
        auto const output_format = static_cast<VkFormat>(this->info.device.image_info(upscale_info.output).value().format);

        wchar_t fsr_input_color[] = L"FSR2_InputColor";
        wchar_t fsr_input_depth[] = L"FSR2_InputDepth";
        wchar_t fsr_input_velocity[] = L"FSR2_InputMotionVectors";
        wchar_t fsr_input_exposure[] = L"FSR2_InputExposure";
        wchar_t fsr_output_upscaled_color[] = L"FSR2_OutputUpscaledColor";

        FfxFsr2DispatchDescription dispatch_description = {};

        auto cmd_buffer = daxa_cmd_get_vk_command_buffer(*reinterpret_cast<daxa_CommandRecorder *>(&recorder));
        dispatch_description.commandList = ffxGetCommandListVK(cmd_buffer);

        dispatch_description.color = ffxGetTextureResourceVK(
            &fsr2_context, color_vk_image, color_vk_image_view,
            color_extent.x, color_extent.y,
            color_format, fsr_input_color);

        dispatch_description.depth = ffxGetTextureResourceVK(
            &fsr2_context, depth_vk_image, depth_vk_image_view,
            depth_extent.x, depth_extent.y,
            depth_format, fsr_input_depth);

        dispatch_description.motionVectors = ffxGetTextureResourceVK(
            &fsr2_context, velocity_vk_image, velocity_vk_image_view,
            velocity_extent.x, velocity_extent.y,
            velocity_format, fsr_input_velocity);

        dispatch_description.exposure = ffxGetTextureResourceVK(&fsr2_context, nullptr, nullptr, 1, 1, VK_FORMAT_UNDEFINED, fsr_input_exposure);

        dispatch_description.output = ffxGetTextureResourceVK(
            &fsr2_context, output_vk_image, output_vk_image_view,
            output_extent.x, output_extent.y,
            output_format, fsr_output_upscaled_color, FFX_RESOURCE_STATE_UNORDERED_ACCESS);

        dispatch_description.jitterOffset.x = upscale_info.jitter.x;
        dispatch_description.jitterOffset.y = upscale_info.jitter.y;
        dispatch_description.motionVectorScale.x = static_cast<daxa_f32>(velocity_extent.x);
        dispatch_description.motionVectorScale.y = static_cast<daxa_f32>(velocity_extent.y);
        dispatch_description.reset = upscale_info.should_reset;
        dispatch_description.enableSharpening = upscale_info.should_sharpen;
        dispatch_description.sharpness = upscale_info.sharpening;
        dispatch_description.frameTimeDelta = upscale_info.delta_time * 1000.0f;
        dispatch_description.preExposure = 1.0f;
        dispatch_description.renderSize.width = color_extent.x;
        dispatch_description.renderSize.height = color_extent.y;
        dispatch_description.cameraFar = upscale_info.camera_info.far_plane;
        dispatch_description.cameraNear = upscale_info.camera_info.near_plane;
        dispatch_description.cameraFovAngleVertical = upscale_info.camera_info.vertical_fov;

        [[maybe_unused]] FfxErrorCode const err = ffxFsr2ContextDispatch(&fsr2_context, &dispatch_description);
        DAXA_DBG_ASSERT_TRUE_M(err == FFX_OK, "FSR2 Failed dispatch");
    }

    void ImplFsr2Context::zero_ref_callback(ImplHandle const * handle)
    {
        auto self = r_cast<ImplFsr2Context const *>(handle);
        delete self;
    }

    auto Fsr2Context::inc_refcnt(ImplHandle const * object) -> u64
    {
        return object->inc_refcnt();
    }

    auto Fsr2Context::dec_refcnt(ImplHandle const * object) -> u64
    {
        return object->dec_refcnt(
            ImplFsr2Context::zero_ref_callback,
            nullptr);
    }
} // namespace daxa

#endif
