#if DAXA_BUILT_WITH_UTILS_TASK_GRAPH

#include <algorithm>
#include <iostream>
#include <set>

#include <utility>

#include "impl_task_graph.hpp"
#include "impl_task_graph_debug.hpp"

namespace daxa
{
    auto TaskInterface::get(TaskBufferAttachmentIndex index) const -> TaskBufferAttachmentInfo const &
    {
        return attachment_infos[index.value].value.buffer;
    }

    auto TaskInterface::get(TaskBufferView view) const -> TaskBufferAttachmentInfo const &
    {
        auto iter = std::find_if(attachment_infos.begin(), attachment_infos.end(), [&](auto const & other)
                                 { 
            if (other.type == TaskAttachmentType::BUFFER)
            {
                return other.value.buffer.view == view || other.value.buffer.translated_view == view;
            }
            return false; });
        DAXA_DBG_ASSERT_TRUE_M(iter != attachment_infos.end(), "Detected invalid task buffer view as index for attachment!");

        return iter->value.buffer;
    }

    auto TaskInterface::get(TaskBlasAttachmentIndex index) const -> TaskBlasAttachmentInfo const &
    {
        return attachment_infos[index.value].value.blas;
    }

    auto TaskInterface::get(TaskBlasView view) const -> TaskBlasAttachmentInfo const &
    {
        auto iter = std::find_if(attachment_infos.begin(), attachment_infos.end(), [&](auto const & other)
                                 { 
            if (other.type == TaskAttachmentType::BLAS)
            {
                return other.value.blas.view == view || other.value.blas.translated_view == view;
            }
            return false; });
        DAXA_DBG_ASSERT_TRUE_M(iter != attachment_infos.end(), "Detected invalid task blas view as index for attachment!");

        return iter->value.blas;
    }

    auto TaskInterface::get(TaskTlasAttachmentIndex index) const -> TaskTlasAttachmentInfo const &
    {
        return attachment_infos[index.value].value.tlas;
    }

    auto TaskInterface::get(TaskTlasView view) const -> TaskTlasAttachmentInfo const &
    {
        auto iter = std::find_if(attachment_infos.begin(), attachment_infos.end(), [&](auto const & other)
                                 { 
            if (other.type == TaskAttachmentType::TLAS)
            {
                return other.value.tlas.view == view || other.value.tlas.translated_view == view;
            }
            return false; });
        DAXA_DBG_ASSERT_TRUE_M(iter != attachment_infos.end(), "Detected invalid task tlas view as index for attachment!");

        return iter->value.tlas;
    }

    auto TaskInterface::get(TaskImageAttachmentIndex index) const -> TaskImageAttachmentInfo const &
    {
        return attachment_infos[index.value].value.image;
    }

    auto TaskInterface::get(TaskImageView view) const -> TaskImageAttachmentInfo const &
    {
        auto iter = std::find_if(attachment_infos.begin(), attachment_infos.end(), [&](auto const & other)
                                 { 
            if (other.type == TaskAttachmentType::IMAGE)
            {
                return other.value.image.view == view || other.value.image.translated_view == view;
            }
            return false; });
        DAXA_DBG_ASSERT_TRUE_M(iter != attachment_infos.end(), "Detected invalid task buffer view as index for attachment!");
        return iter->value.image;
    }

    auto TaskInterface::get(usize index) const -> TaskAttachmentInfo const &
    {
        return attachment_infos[index];
    }

    auto to_string(TaskGPUResourceView const & id) -> std::string
    {
        return fmt::format("tg idx: {}, index: {}", id.task_graph_index, id.index);
    }

    auto static constexpr access_to_usage(TaskImageAccess const & access) -> ImageUsageFlags
    {
        switch (access)
        {
        case TaskImageAccess::GRAPHICS_SHADER_SAMPLED: [[fallthrough]];
        case TaskImageAccess::VERTEX_SHADER_SAMPLED: [[fallthrough]];
        case TaskImageAccess::TESSELLATION_CONTROL_SHADER_SAMPLED: [[fallthrough]];
        case TaskImageAccess::TESSELLATION_EVALUATION_SHADER_SAMPLED: [[fallthrough]];
        case TaskImageAccess::GEOMETRY_SHADER_SAMPLED: [[fallthrough]];
        case TaskImageAccess::FRAGMENT_SHADER_SAMPLED: [[fallthrough]];
        case TaskImageAccess::TASK_SHADER_SAMPLED: [[fallthrough]];
        case TaskImageAccess::MESH_SHADER_SAMPLED: [[fallthrough]];
        case TaskImageAccess::COMPUTE_SHADER_SAMPLED:
        case TaskImageAccess::RAY_TRACING_SHADER_SAMPLED:
            return ImageUsageFlagBits::SHADER_SAMPLED;
        case TaskImageAccess::GRAPHICS_SHADER_STORAGE_WRITE_ONLY: [[fallthrough]];
        case TaskImageAccess::GRAPHICS_SHADER_STORAGE_READ_ONLY: [[fallthrough]];
        case TaskImageAccess::GRAPHICS_SHADER_STORAGE_READ_WRITE: [[fallthrough]];
        case TaskImageAccess::GRAPHICS_SHADER_STORAGE_READ_WRITE_CONCURRENT: [[fallthrough]];
        case TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY: [[fallthrough]];
        case TaskImageAccess::COMPUTE_SHADER_STORAGE_READ_ONLY: [[fallthrough]];
        case TaskImageAccess::COMPUTE_SHADER_STORAGE_READ_WRITE: [[fallthrough]];
        case TaskImageAccess::COMPUTE_SHADER_STORAGE_READ_WRITE_CONCURRENT: [[fallthrough]];
        case TaskImageAccess::RAY_TRACING_SHADER_STORAGE_WRITE_ONLY: [[fallthrough]];
        case TaskImageAccess::RAY_TRACING_SHADER_STORAGE_READ_ONLY: [[fallthrough]];
        case TaskImageAccess::RAY_TRACING_SHADER_STORAGE_READ_WRITE: [[fallthrough]];
        case TaskImageAccess::RAY_TRACING_SHADER_STORAGE_READ_WRITE_CONCURRENT: [[fallthrough]];
        case TaskImageAccess::VERTEX_SHADER_STORAGE_WRITE_ONLY: [[fallthrough]];
        case TaskImageAccess::VERTEX_SHADER_STORAGE_READ_ONLY: [[fallthrough]];
        case TaskImageAccess::VERTEX_SHADER_STORAGE_READ_WRITE: [[fallthrough]];
        case TaskImageAccess::VERTEX_SHADER_STORAGE_READ_WRITE_CONCURRENT: [[fallthrough]];
        case TaskImageAccess::TASK_SHADER_STORAGE_WRITE_ONLY: [[fallthrough]];
        case TaskImageAccess::TASK_SHADER_STORAGE_READ_ONLY: [[fallthrough]];
        case TaskImageAccess::TASK_SHADER_STORAGE_READ_WRITE: [[fallthrough]];
        case TaskImageAccess::TASK_SHADER_STORAGE_READ_WRITE_CONCURRENT: [[fallthrough]];
        case TaskImageAccess::MESH_SHADER_STORAGE_WRITE_ONLY: [[fallthrough]];
        case TaskImageAccess::MESH_SHADER_STORAGE_READ_ONLY: [[fallthrough]];
        case TaskImageAccess::MESH_SHADER_STORAGE_READ_WRITE: [[fallthrough]];
        case TaskImageAccess::MESH_SHADER_STORAGE_READ_WRITE_CONCURRENT: [[fallthrough]];
        case TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_WRITE_ONLY: [[fallthrough]];
        case TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_READ_ONLY: [[fallthrough]];
        case TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_READ_WRITE: [[fallthrough]];
        case TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_READ_WRITE_CONCURRENT: [[fallthrough]];
        case TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_WRITE_ONLY: [[fallthrough]];
        case TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_READ_ONLY: [[fallthrough]];
        case TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_READ_WRITE: [[fallthrough]];
        case TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_READ_WRITE_CONCURRENT: [[fallthrough]];
        case TaskImageAccess::GEOMETRY_SHADER_STORAGE_WRITE_ONLY: [[fallthrough]];
        case TaskImageAccess::GEOMETRY_SHADER_STORAGE_READ_ONLY: [[fallthrough]];
        case TaskImageAccess::GEOMETRY_SHADER_STORAGE_READ_WRITE: [[fallthrough]];
        case TaskImageAccess::GEOMETRY_SHADER_STORAGE_READ_WRITE_CONCURRENT: [[fallthrough]];
        case TaskImageAccess::FRAGMENT_SHADER_STORAGE_WRITE_ONLY: [[fallthrough]];
        case TaskImageAccess::FRAGMENT_SHADER_STORAGE_READ_ONLY: [[fallthrough]];
        case TaskImageAccess::FRAGMENT_SHADER_STORAGE_READ_WRITE: [[fallthrough]];
        case TaskImageAccess::FRAGMENT_SHADER_STORAGE_READ_WRITE_CONCURRENT:
            return ImageUsageFlagBits::SHADER_STORAGE;
        case TaskImageAccess::TRANSFER_READ:
            return ImageUsageFlagBits::TRANSFER_SRC;
        case TaskImageAccess::TRANSFER_WRITE:
            return ImageUsageFlagBits::TRANSFER_DST;
        // NOTE(msakmary) - not fully sure about the resolve being color attachment usage
        // this is the best I could deduce from vulkan docs
        case TaskImageAccess::RESOLVE_WRITE: [[fallthrough]];
        case TaskImageAccess::COLOR_ATTACHMENT:
            return ImageUsageFlagBits::COLOR_ATTACHMENT;
        case TaskImageAccess::DEPTH_ATTACHMENT: [[fallthrough]];
        case TaskImageAccess::STENCIL_ATTACHMENT: [[fallthrough]];
        case TaskImageAccess::DEPTH_STENCIL_ATTACHMENT:
            return ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT;
        case TaskImageAccess::DEPTH_ATTACHMENT_READ: [[fallthrough]];
        case TaskImageAccess::STENCIL_ATTACHMENT_READ: [[fallthrough]];
        case TaskImageAccess::DEPTH_STENCIL_ATTACHMENT_READ:
            return ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT;
        case TaskImageAccess::PRESENT: [[fallthrough]];
        case TaskImageAccess::NONE: [[fallthrough]];
        default:
            return ImageUsageFlagBits::NONE;
        }
    }

    auto static constexpr view_type_to_create_flags(ImageViewType const & view_type) -> ImageCreateFlags
    {
        // NOTE: Do we need to handle other flags, such as compatible array?
        if (view_type == ImageViewType::CUBE)
        {
            return ImageCreateFlagBits::COMPATIBLE_CUBE;
        }
        else
        {
            return ImageCreateFlagBits::NONE;
        }
    }

    auto task_image_access_to_layout_access(TaskImageAccess const & access) -> std::tuple<ImageLayout, Access, TaskAccessConcurrency>
    {
        switch (access)
        {
        case TaskImageAccess::NONE: return {ImageLayout::UNDEFINED, {PipelineStageFlagBits::NONE, AccessTypeFlagBits::NONE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::GRAPHICS_SHADER_SAMPLED: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::ALL_GRAPHICS, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::GRAPHICS_SHADER_STORAGE_WRITE_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::PipelineStageFlagBits::ALL_GRAPHICS, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::GRAPHICS_SHADER_STORAGE_READ_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::PipelineStageFlagBits::ALL_GRAPHICS, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::GRAPHICS_SHADER_STORAGE_READ_WRITE: return {ImageLayout::GENERAL, {PipelineStageFlagBits::PipelineStageFlagBits::ALL_GRAPHICS, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::GRAPHICS_SHADER_STORAGE_READ_WRITE_CONCURRENT: return {ImageLayout::GENERAL, {PipelineStageFlagBits::PipelineStageFlagBits::ALL_GRAPHICS, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::COMPUTE_SHADER_SAMPLED: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::COMPUTE_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::COMPUTE_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::COMPUTE_SHADER_STORAGE_READ_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::COMPUTE_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::COMPUTE_SHADER_STORAGE_READ_WRITE: return {ImageLayout::GENERAL, {PipelineStageFlagBits::COMPUTE_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::COMPUTE_SHADER_STORAGE_READ_WRITE_CONCURRENT: return {ImageLayout::GENERAL, {PipelineStageFlagBits::COMPUTE_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::RAY_TRACING_SHADER_SAMPLED: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::RAY_TRACING_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::RAY_TRACING_SHADER_STORAGE_WRITE_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::RAY_TRACING_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::RAY_TRACING_SHADER_STORAGE_READ_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::RAY_TRACING_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::RAY_TRACING_SHADER_STORAGE_READ_WRITE: return {ImageLayout::GENERAL, {PipelineStageFlagBits::RAY_TRACING_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::RAY_TRACING_SHADER_STORAGE_READ_WRITE_CONCURRENT: return {ImageLayout::GENERAL, {PipelineStageFlagBits::RAY_TRACING_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::VERTEX_SHADER_SAMPLED: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::VERTEX_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::VERTEX_SHADER_STORAGE_WRITE_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::VERTEX_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::VERTEX_SHADER_STORAGE_READ_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::VERTEX_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::VERTEX_SHADER_STORAGE_READ_WRITE: return {ImageLayout::GENERAL, {PipelineStageFlagBits::VERTEX_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::VERTEX_SHADER_STORAGE_READ_WRITE_CONCURRENT: return {ImageLayout::GENERAL, {PipelineStageFlagBits::VERTEX_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::TASK_SHADER_SAMPLED: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::TASK_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::TASK_SHADER_STORAGE_WRITE_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TASK_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::TASK_SHADER_STORAGE_READ_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TASK_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::TASK_SHADER_STORAGE_READ_WRITE: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TASK_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::TASK_SHADER_STORAGE_READ_WRITE_CONCURRENT: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TASK_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::MESH_SHADER_SAMPLED: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::MESH_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::MESH_SHADER_STORAGE_WRITE_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::MESH_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::MESH_SHADER_STORAGE_READ_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::MESH_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::MESH_SHADER_STORAGE_READ_WRITE: return {ImageLayout::GENERAL, {PipelineStageFlagBits::MESH_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::MESH_SHADER_STORAGE_READ_WRITE_CONCURRENT: return {ImageLayout::GENERAL, {PipelineStageFlagBits::MESH_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::TESSELLATION_CONTROL_SHADER_SAMPLED: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_WRITE_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_READ_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_READ_WRITE: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_READ_WRITE_CONCURRENT: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::TESSELLATION_EVALUATION_SHADER_SAMPLED: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_WRITE_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_READ_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_READ_WRITE: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_READ_WRITE_CONCURRENT: return {ImageLayout::GENERAL, {PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::GEOMETRY_SHADER_SAMPLED: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::GEOMETRY_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::GEOMETRY_SHADER_STORAGE_WRITE_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::GEOMETRY_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::GEOMETRY_SHADER_STORAGE_READ_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::GEOMETRY_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::GEOMETRY_SHADER_STORAGE_READ_WRITE: return {ImageLayout::GENERAL, {PipelineStageFlagBits::GEOMETRY_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::GEOMETRY_SHADER_STORAGE_READ_WRITE_CONCURRENT: return {ImageLayout::GENERAL, {PipelineStageFlagBits::GEOMETRY_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::FRAGMENT_SHADER_SAMPLED: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::FRAGMENT_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::FRAGMENT_SHADER_STORAGE_WRITE_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::FRAGMENT_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::FRAGMENT_SHADER_STORAGE_READ_ONLY: return {ImageLayout::GENERAL, {PipelineStageFlagBits::FRAGMENT_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::FRAGMENT_SHADER_STORAGE_READ_WRITE: return {ImageLayout::GENERAL, {PipelineStageFlagBits::FRAGMENT_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::FRAGMENT_SHADER_STORAGE_READ_WRITE_CONCURRENT: return {ImageLayout::GENERAL, {PipelineStageFlagBits::FRAGMENT_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::TRANSFER_READ: return {ImageLayout::TRANSFER_SRC_OPTIMAL, {PipelineStageFlagBits::TRANSFER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::TRANSFER_WRITE: return {ImageLayout::TRANSFER_DST_OPTIMAL, {PipelineStageFlagBits::TRANSFER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::COLOR_ATTACHMENT: return {ImageLayout::ATTACHMENT_OPTIMAL, {PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::DEPTH_ATTACHMENT:
            [[fallthrough]];
        case TaskImageAccess::STENCIL_ATTACHMENT:
            [[fallthrough]];
        case TaskImageAccess::DEPTH_STENCIL_ATTACHMENT: return {ImageLayout::ATTACHMENT_OPTIMAL, {PipelineStageFlagBits::EARLY_FRAGMENT_TESTS | PipelineStageFlagBits::LATE_FRAGMENT_TESTS, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::DEPTH_ATTACHMENT_READ:
            [[fallthrough]];
        case TaskImageAccess::STENCIL_ATTACHMENT_READ:
            [[fallthrough]];
        case TaskImageAccess::DEPTH_STENCIL_ATTACHMENT_READ: return {ImageLayout::READ_ONLY_OPTIMAL, {PipelineStageFlagBits::EARLY_FRAGMENT_TESTS | PipelineStageFlagBits::LATE_FRAGMENT_TESTS, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskImageAccess::RESOLVE_WRITE: return {ImageLayout::ATTACHMENT_OPTIMAL, {PipelineStageFlagBits::RESOLVE, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskImageAccess::PRESENT: return {ImageLayout::PRESENT_SRC, {PipelineStageFlagBits::ALL_COMMANDS, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        default: DAXA_DBG_ASSERT_TRUE_M(false, "unreachable");
        }
        return {};
    }

    auto task_buffer_access_to_access(TaskBufferAccess const & access) -> std::pair<Access, TaskAccessConcurrency>
    {
        switch (access)
        {
        case TaskBufferAccess::NONE: return {{PipelineStageFlagBits::NONE, AccessTypeFlagBits::NONE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::READ: return {{PipelineStageFlagBits::ALL_COMMANDS, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::WRITE: return {{PipelineStageFlagBits::ALL_COMMANDS, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::READ_WRITE: return {{PipelineStageFlagBits::ALL_COMMANDS, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::GRAPHICS_SHADER_READ: return {{PipelineStageFlagBits::ALL_GRAPHICS, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::GRAPHICS_SHADER_WRITE: return {{PipelineStageFlagBits::PipelineStageFlagBits::ALL_GRAPHICS, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::GRAPHICS_SHADER_READ_WRITE: return {{PipelineStageFlagBits::PipelineStageFlagBits::ALL_GRAPHICS, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::GRAPHICS_SHADER_READ_WRITE_CONCURRENT: return {{PipelineStageFlagBits::PipelineStageFlagBits::ALL_GRAPHICS, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::COMPUTE_SHADER_READ: return {{PipelineStageFlagBits::COMPUTE_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::COMPUTE_SHADER_WRITE: return {{PipelineStageFlagBits::COMPUTE_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::COMPUTE_SHADER_READ_WRITE: return {{PipelineStageFlagBits::COMPUTE_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::COMPUTE_SHADER_READ_WRITE_CONCURRENT: return {{PipelineStageFlagBits::COMPUTE_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::RAY_TRACING_SHADER_READ: return {{PipelineStageFlagBits::RAY_TRACING_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::RAY_TRACING_SHADER_WRITE: return {{PipelineStageFlagBits::RAY_TRACING_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::RAY_TRACING_SHADER_READ_WRITE: return {{PipelineStageFlagBits::RAY_TRACING_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::RAY_TRACING_SHADER_READ_WRITE_CONCURRENT: return {{PipelineStageFlagBits::RAY_TRACING_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::VERTEX_SHADER_READ: return {{PipelineStageFlagBits::VERTEX_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::VERTEX_SHADER_WRITE: return {{PipelineStageFlagBits::VERTEX_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::VERTEX_SHADER_READ_WRITE: return {{PipelineStageFlagBits::VERTEX_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::VERTEX_SHADER_READ_WRITE_CONCURRENT: return {{PipelineStageFlagBits::VERTEX_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::TASK_SHADER_READ: return {{PipelineStageFlagBits::TASK_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::TASK_SHADER_WRITE: return {{PipelineStageFlagBits::TASK_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::TASK_SHADER_READ_WRITE: return {{PipelineStageFlagBits::TASK_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::TASK_SHADER_READ_WRITE_CONCURRENT: return {{PipelineStageFlagBits::TASK_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::MESH_SHADER_READ: return {{PipelineStageFlagBits::MESH_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::MESH_SHADER_WRITE: return {{PipelineStageFlagBits::MESH_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::MESH_SHADER_READ_WRITE: return {{PipelineStageFlagBits::MESH_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::MESH_SHADER_READ_WRITE_CONCURRENT: return {{PipelineStageFlagBits::MESH_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::TESSELLATION_CONTROL_SHADER_READ: return {{PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::TESSELLATION_EVALUATION_SHADER_READ: return {{PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::GEOMETRY_SHADER_READ: return {{PipelineStageFlagBits::GEOMETRY_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::FRAGMENT_SHADER_READ: return {{PipelineStageFlagBits::FRAGMENT_SHADER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::TESSELLATION_CONTROL_SHADER_WRITE: return {{PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::TESSELLATION_EVALUATION_SHADER_WRITE: return {{PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::GEOMETRY_SHADER_WRITE: return {{PipelineStageFlagBits::GEOMETRY_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::FRAGMENT_SHADER_WRITE: return {{PipelineStageFlagBits::FRAGMENT_SHADER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::TESSELLATION_CONTROL_SHADER_READ_WRITE: return {{PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::TESSELLATION_CONTROL_SHADER_READ_WRITE_CONCURRENT: return {{PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::TESSELLATION_EVALUATION_SHADER_READ_WRITE: return {{PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::TESSELLATION_EVALUATION_SHADER_READ_WRITE_CONCURRENT: return {{PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::GEOMETRY_SHADER_READ_WRITE: return {{PipelineStageFlagBits::GEOMETRY_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::GEOMETRY_SHADER_READ_WRITE_CONCURRENT: return {{PipelineStageFlagBits::GEOMETRY_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::FRAGMENT_SHADER_READ_WRITE: return {{PipelineStageFlagBits::FRAGMENT_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::FRAGMENT_SHADER_READ_WRITE_CONCURRENT: return {{PipelineStageFlagBits::FRAGMENT_SHADER, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::TRANSFER_READ: return {{PipelineStageFlagBits::TRANSFER, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::TRANSFER_WRITE: return {{PipelineStageFlagBits::TRANSFER, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::HOST_TRANSFER_READ: return {{PipelineStageFlagBits::HOST, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::HOST_TRANSFER_WRITE: return {{PipelineStageFlagBits::HOST, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::INDEX_READ: return {{PipelineStageFlagBits::INDEX_INPUT, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::DRAW_INDIRECT_INFO_READ: return {{PipelineStageFlagBits::DRAW_INDIRECT, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::ACCELERATION_STRUCTURE_BUILD_READ: return {{PipelineStageFlagBits::ACCELERATION_STRUCTURE_BUILD, AccessTypeFlagBits::READ}, TaskAccessConcurrency::CONCURRENT};
        case TaskBufferAccess::ACCELERATION_STRUCTURE_BUILD_WRITE: return {{PipelineStageFlagBits::ACCELERATION_STRUCTURE_BUILD, AccessTypeFlagBits::WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        case TaskBufferAccess::ACCELERATION_STRUCTURE_BUILD_READ_WRITE: return {{PipelineStageFlagBits::ACCELERATION_STRUCTURE_BUILD, AccessTypeFlagBits::READ_WRITE}, TaskAccessConcurrency::EXCLUSIVE};
        default: DAXA_DBG_ASSERT_TRUE_M(false, "unreachable");
        }
        return {};
    }

    auto TaskGPUResourceView::is_empty() const -> bool
    {
        return index == 0 && task_graph_index == 0;
    }

    auto TaskGPUResourceView::is_persistent() const -> bool
    {
        return task_graph_index == std::numeric_limits<u32>::max() && !is_null();
    }

    auto TaskGPUResourceView::is_null() const -> bool
    {
        return
            task_graph_index == std::numeric_limits<u32>::max() &&
            index == std::numeric_limits<u32>::max();
    }

    auto to_string(TaskBufferAccess const & usage) -> std::string_view
    {
        switch (usage)
        {
        case daxa::TaskBufferAccess::NONE: return std::string_view{"NONE"};
        case daxa::TaskBufferAccess::GRAPHICS_SHADER_READ: return std::string_view{"GRAPHICS_SHADER_READ"};
        case daxa::TaskBufferAccess::GRAPHICS_SHADER_WRITE: return std::string_view{"GRAPHICS_SHADER_WRITE"};
        case daxa::TaskBufferAccess::GRAPHICS_SHADER_READ_WRITE: return std::string_view{"GRAPHICS_SHADER_READ_WRITE"};
        case daxa::TaskBufferAccess::COMPUTE_SHADER_READ: return std::string_view{"COMPUTE_SHADER_READ"};
        case daxa::TaskBufferAccess::COMPUTE_SHADER_WRITE: return std::string_view{"COMPUTE_SHADER_WRITE"};
        case daxa::TaskBufferAccess::COMPUTE_SHADER_READ_WRITE: return std::string_view{"COMPUTE_SHADER_READ_WRITE"};
        case daxa::TaskBufferAccess::RAY_TRACING_SHADER_READ: return std::string_view{"RAY_TRACING_SHADER_READ"};
        case daxa::TaskBufferAccess::RAY_TRACING_SHADER_WRITE: return std::string_view{"RAY_TRACING_SHADER_WRITE"};
        case daxa::TaskBufferAccess::RAY_TRACING_SHADER_READ_WRITE: return std::string_view{"RAY_TRACING_SHADER_READ_WRITE"};
        case daxa::TaskBufferAccess::TASK_SHADER_READ: return std::string_view{"TASK_SHADER_READ"};
        case daxa::TaskBufferAccess::TASK_SHADER_WRITE: return std::string_view{"TASK_SHADER_WRITE"};
        case daxa::TaskBufferAccess::TASK_SHADER_READ_WRITE: return std::string_view{"TASK_SHADER_READ_WRITE"};
        case daxa::TaskBufferAccess::MESH_SHADER_READ: return std::string_view{"MESH_SHADER_READ"};
        case daxa::TaskBufferAccess::MESH_SHADER_WRITE: return std::string_view{"MESH_SHADER_WRITE"};
        case daxa::TaskBufferAccess::MESH_SHADER_READ_WRITE: return std::string_view{"MESH_SHADER_READ_WRITE"};
        case daxa::TaskBufferAccess::VERTEX_SHADER_READ: return std::string_view{"VERTEX_SHADER_READ"};
        case daxa::TaskBufferAccess::VERTEX_SHADER_WRITE: return std::string_view{"VERTEX_SHADER_WRITE"};
        case daxa::TaskBufferAccess::VERTEX_SHADER_READ_WRITE: return std::string_view{"VERTEX_SHADER_READ_WRITE"};
        case daxa::TaskBufferAccess::TESSELLATION_CONTROL_SHADER_READ: return std::string_view{"TESSELLATION_CONTROL_SHADER_READ"};
        case daxa::TaskBufferAccess::TESSELLATION_CONTROL_SHADER_WRITE: return std::string_view{"TESSELLATION_CONTROL_SHADER_WRITE"};
        case daxa::TaskBufferAccess::TESSELLATION_CONTROL_SHADER_READ_WRITE: return std::string_view{"TESSELLATION_CONTROL_SHADER_READ_WRITE"};
        case daxa::TaskBufferAccess::TESSELLATION_EVALUATION_SHADER_READ: return std::string_view{"TESSELLATION_EVALUATION_SHADER_READ"};
        case daxa::TaskBufferAccess::TESSELLATION_EVALUATION_SHADER_WRITE: return std::string_view{"TESSELLATION_EVALUATION_SHADER_WRITE"};
        case daxa::TaskBufferAccess::TESSELLATION_EVALUATION_SHADER_READ_WRITE: return std::string_view{"TESSELLATION_EVALUATION_SHADER_READ_WRITE"};
        case daxa::TaskBufferAccess::GEOMETRY_SHADER_READ: return std::string_view{"GEOMETRY_SHADER_READ"};
        case daxa::TaskBufferAccess::GEOMETRY_SHADER_WRITE: return std::string_view{"GEOMETRY_SHADER_WRITE"};
        case daxa::TaskBufferAccess::GEOMETRY_SHADER_READ_WRITE: return std::string_view{"GEOMETRY_SHADER_READ_WRITE"};
        case daxa::TaskBufferAccess::FRAGMENT_SHADER_READ: return std::string_view{"FRAGMENT_SHADER_READ"};
        case daxa::TaskBufferAccess::FRAGMENT_SHADER_WRITE: return std::string_view{"FRAGMENT_SHADER_WRITE"};
        case daxa::TaskBufferAccess::FRAGMENT_SHADER_READ_WRITE: return std::string_view{"FRAGMENT_SHADER_READ_WRITE"};
        case daxa::TaskBufferAccess::INDEX_READ: return std::string_view{"INDEX_READ"};
        case daxa::TaskBufferAccess::DRAW_INDIRECT_INFO_READ: return std::string_view{"DRAW_INDIRECT_INFO_READ"};
        case daxa::TaskBufferAccess::TRANSFER_READ: return std::string_view{"TRANSFER_READ"};
        case daxa::TaskBufferAccess::TRANSFER_WRITE: return std::string_view{"TRANSFER_WRITE"};
        case daxa::TaskBufferAccess::HOST_TRANSFER_READ: return std::string_view{"HOST_TRANSFER_READ"};
        case daxa::TaskBufferAccess::HOST_TRANSFER_WRITE: return std::string_view{"HOST_TRANSFER_WRITE"};
        case daxa::TaskBufferAccess::ACCELERATION_STRUCTURE_BUILD_READ: return std::string_view{"HOST_TACCELERATION_STRUCTURE_BUILD_READRANSFER_WRITE"};
        case daxa::TaskBufferAccess::ACCELERATION_STRUCTURE_BUILD_WRITE: return std::string_view{"ACCELERATION_STRUCTURE_BUILD_WRITE"};
        case daxa::TaskBufferAccess::ACCELERATION_STRUCTURE_BUILD_READ_WRITE: return std::string_view{"ACCELERATION_STRUCTURE_BUILD_READ_WRITE"};
        case daxa::TaskBufferAccess::MAX_ENUM: return std::string_view{"MAX_ENUM"};
        default: DAXA_DBG_ASSERT_TRUE_M(false, "unreachable");
        }
        return "invalid";
    }

    auto to_string(TaskBlasAccess const & usage) -> std::string_view
    {
        return to_string(static_cast<TaskBufferAccess>(usage));
    }

    auto to_string(TaskTlasAccess const & usage) -> std::string_view
    {
        return to_string(static_cast<TaskBufferAccess>(usage));
    }

    auto to_string(TaskImageAccess const & usage) -> std::string_view
    {
        switch (usage)
        {
        case daxa::TaskImageAccess::NONE: return std::string_view{"NONE"};
        case daxa::TaskImageAccess::GRAPHICS_SHADER_SAMPLED: return std::string_view{"GRAPHICS_SHADER_SAMPLED"};
        case daxa::TaskImageAccess::GRAPHICS_SHADER_STORAGE_READ_WRITE: return std::string_view{"GRAPHICS_SHADER_STORAGE_READ_WRITE"};
        case daxa::TaskImageAccess::GRAPHICS_SHADER_STORAGE_WRITE_ONLY: return std::string_view{"GRAPHICS_SHADER_STORAGE_WRITE_ONLY"};
        case daxa::TaskImageAccess::GRAPHICS_SHADER_STORAGE_READ_ONLY: return std::string_view{"GRAPHICS_SHADER_STORAGE_READ_ONLY"};
        case daxa::TaskImageAccess::COMPUTE_SHADER_SAMPLED: return std::string_view{"COMPUTE_SHADER_SAMPLED"};
        case daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_WRITE_ONLY: return std::string_view{"COMPUTE_SHADER_STORAGE_WRITE_ONLY"};
        case daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_READ_ONLY: return std::string_view{"COMPUTE_SHADER_STORAGE_READ_ONLY"};
        case daxa::TaskImageAccess::COMPUTE_SHADER_STORAGE_READ_WRITE: return std::string_view{"COMPUTE_SHADER_STORAGE_READ_WRITE"};
        case daxa::TaskImageAccess::RAY_TRACING_SHADER_SAMPLED: return std::string_view{"RAY_TRACING_SHADER_SAMPLED"};
        case daxa::TaskImageAccess::RAY_TRACING_SHADER_STORAGE_WRITE_ONLY: return std::string_view{"RAY_TRACING_SHADER_STORAGE_WRITE_ONLY"};
        case daxa::TaskImageAccess::RAY_TRACING_SHADER_STORAGE_READ_ONLY: return std::string_view{"RAY_TRACING_SHADER_STORAGE_READ_ONLY"};
        case daxa::TaskImageAccess::RAY_TRACING_SHADER_STORAGE_READ_WRITE: return std::string_view{"RAY_TRACING_SHADER_STORAGE_READ_WRITE"};
        case daxa::TaskImageAccess::TASK_SHADER_SAMPLED: return std::string_view{"TASK_SHADER_SAMPLED"};
        case daxa::TaskImageAccess::TASK_SHADER_STORAGE_WRITE_ONLY: return std::string_view{"TASK_SHADER_STORAGE_WRITE_ONLY"};
        case daxa::TaskImageAccess::TASK_SHADER_STORAGE_READ_ONLY: return std::string_view{"TASK_SHADER_STORAGE_READ_ONLY"};
        case daxa::TaskImageAccess::TASK_SHADER_STORAGE_READ_WRITE: return std::string_view{"TASK_SHADER_STORAGE_READ_WRITE"};
        case daxa::TaskImageAccess::MESH_SHADER_SAMPLED: return std::string_view{"MESH_SHADER_SAMPLED"};
        case daxa::TaskImageAccess::MESH_SHADER_STORAGE_WRITE_ONLY: return std::string_view{"MESH_SHADER_STORAGE_WRITE_ONLY"};
        case daxa::TaskImageAccess::MESH_SHADER_STORAGE_READ_ONLY: return std::string_view{"MESH_SHADER_STORAGE_READ_ONLY"};
        case daxa::TaskImageAccess::MESH_SHADER_STORAGE_READ_WRITE: return std::string_view{"MESH_SHADER_STORAGE_READ_WRITE"};
        case daxa::TaskImageAccess::VERTEX_SHADER_SAMPLED: return std::string_view{"VERTEX_SHADER_SAMPLED"};
        case daxa::TaskImageAccess::VERTEX_SHADER_STORAGE_WRITE_ONLY: return std::string_view{"VERTEX_SHADER_STORAGE_WRITE_ONLY"};
        case daxa::TaskImageAccess::VERTEX_SHADER_STORAGE_READ_ONLY: return std::string_view{"VERTEX_SHADER_STORAGE_READ_ONLY"};
        case daxa::TaskImageAccess::VERTEX_SHADER_STORAGE_READ_WRITE: return std::string_view{"VERTEX_SHADER_STORAGE_READ_WRITE"};
        case daxa::TaskImageAccess::TESSELLATION_CONTROL_SHADER_SAMPLED: return std::string_view{"TESSELLATION_CONTROL_SHADER_SAMPLED"};
        case daxa::TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_WRITE_ONLY: return std::string_view{"TESSELLATION_CONTROL_SHADER_STORAGE_WRITE_ONLY"};
        case daxa::TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_READ_ONLY: return std::string_view{"TESSELLATION_CONTROL_SHADER_STORAGE_READ_ONLY"};
        case daxa::TaskImageAccess::TESSELLATION_CONTROL_SHADER_STORAGE_READ_WRITE: return std::string_view{"TESSELLATION_CONTROL_SHADER_STORAGE_READ_WRITE"};
        case daxa::TaskImageAccess::TESSELLATION_EVALUATION_SHADER_SAMPLED: return std::string_view{"TESSELLATION_EVALUATION_SHADER_SAMPLED"};
        case daxa::TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_WRITE_ONLY: return std::string_view{"TESSELLATION_EVALUATION_SHADER_STORAGE_WRITE_ONLY"};
        case daxa::TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_READ_ONLY: return std::string_view{"TESSELLATION_EVALUATION_SHADER_STORAGE_READ_ONLY"};
        case daxa::TaskImageAccess::TESSELLATION_EVALUATION_SHADER_STORAGE_READ_WRITE: return std::string_view{"TESSELLATION_EVALUATION_SHADER_STORAGE_READ_WRITE"};
        case daxa::TaskImageAccess::GEOMETRY_SHADER_SAMPLED: return std::string_view{"GEOMETRY_SHADER_SAMPLED"};
        case daxa::TaskImageAccess::GEOMETRY_SHADER_STORAGE_WRITE_ONLY: return std::string_view{"GEOMETRY_SHADER_STORAGE_WRITE_ONLY"};
        case daxa::TaskImageAccess::GEOMETRY_SHADER_STORAGE_READ_ONLY: return std::string_view{"GEOMETRY_SHADER_STORAGE_READ_ONLY"};
        case daxa::TaskImageAccess::GEOMETRY_SHADER_STORAGE_READ_WRITE: return std::string_view{"GEOMETRY_SHADER_STORAGE_READ_WRITE"};
        case daxa::TaskImageAccess::FRAGMENT_SHADER_SAMPLED: return std::string_view{"FRAGMENT_SHADER_SAMPLED"};
        case daxa::TaskImageAccess::FRAGMENT_SHADER_STORAGE_WRITE_ONLY: return std::string_view{"FRAGMENT_SHADER_STORAGE_WRITE_ONLY"};
        case daxa::TaskImageAccess::FRAGMENT_SHADER_STORAGE_READ_ONLY: return std::string_view{"FRAGMENT_SHADER_STORAGE_READ_ONLY"};
        case daxa::TaskImageAccess::FRAGMENT_SHADER_STORAGE_READ_WRITE: return std::string_view{"FRAGMENT_SHADER_STORAGE_READ_WRITE"};
        case daxa::TaskImageAccess::TRANSFER_READ: return std::string_view{"TRANSFER_READ"};
        case daxa::TaskImageAccess::TRANSFER_WRITE: return std::string_view{"TRANSFER_WRITE"};
        case daxa::TaskImageAccess::COLOR_ATTACHMENT: return std::string_view{"COLOR_ATTACHMENT"};
        case daxa::TaskImageAccess::DEPTH_ATTACHMENT: return std::string_view{"DEPTH_ATTACHMENT"};
        case daxa::TaskImageAccess::STENCIL_ATTACHMENT: return std::string_view{"STENCIL_ATTACHMENT"};
        case daxa::TaskImageAccess::DEPTH_STENCIL_ATTACHMENT: return std::string_view{"DEPTH_STENCIL_ATTACHMENT"};
        case daxa::TaskImageAccess::DEPTH_ATTACHMENT_READ: return std::string_view{"DEPTH_ATTACHMENT_READ"};
        case daxa::TaskImageAccess::STENCIL_ATTACHMENT_READ: return std::string_view{"STENCIL_ATTACHMENT_READ"};
        case daxa::TaskImageAccess::DEPTH_STENCIL_ATTACHMENT_READ: return std::string_view{"DEPTH_STENCIL_ATTACHMENT_READ"};
        case daxa::TaskImageAccess::RESOLVE_WRITE: return std::string_view{"RESOLVE_WRITE"};
        case daxa::TaskImageAccess::PRESENT: return std::string_view{"PRESENT"};
        case daxa::TaskImageAccess::MAX_ENUM: return std::string_view{"MAX_ENUM"};
        default: DAXA_DBG_ASSERT_TRUE_M(false, "unreachable");
        }
        return "invalid";
    }

    ImplPersistentTaskBufferBlasTlas::ImplPersistentTaskBufferBlasTlas(TaskBufferInfo a_info)
        : actual_ids{std::vector<BufferId>{a_info.initial_buffers.buffers.begin(), a_info.initial_buffers.buffers.end()}},
          latest_access{a_info.initial_buffers.latest_access},
          info{std::move(a_info)},
          unique_index{ImplPersistentTaskBufferBlasTlas::exec_unique_next_index++}
    {
    }

    ImplPersistentTaskBufferBlasTlas::ImplPersistentTaskBufferBlasTlas(Device & device, BufferInfo const & a_info)
        : actual_ids{std::vector<BufferId>{device.create_buffer(a_info)}},
          latest_access{},
          info{TaskBufferInfo{.name = a_info.name.c_str().data()}},
          owned_buffer_device{device},
          owned_buffer_info{a_info},
          unique_index{ImplPersistentTaskBufferBlasTlas::exec_unique_next_index++}
    {
    }

    ImplPersistentTaskBufferBlasTlas::ImplPersistentTaskBufferBlasTlas(TaskBlasInfo a_info)
        : actual_ids{std::vector<BlasId>{a_info.initial_blas.blas.begin(), a_info.initial_blas.blas.end()}},
          latest_access{a_info.initial_blas.latest_access},
          info{std::move(a_info)},
          unique_index{ImplPersistentTaskBufferBlasTlas::exec_unique_next_index++}
    {
    }

    ImplPersistentTaskBufferBlasTlas::ImplPersistentTaskBufferBlasTlas(TaskTlasInfo a_info)
        : actual_ids{std::vector<TlasId>{a_info.initial_tlas.tlas.begin(), a_info.initial_tlas.tlas.end()}},
          latest_access{a_info.initial_tlas.latest_access},
          info{std::move(a_info)},
          unique_index{ImplPersistentTaskBufferBlasTlas::exec_unique_next_index++}
    {
    }
    ImplPersistentTaskBufferBlasTlas::~ImplPersistentTaskBufferBlasTlas() = default;

    void ImplPersistentTaskBufferBlasTlas::zero_ref_callback(ImplHandle const * handle)
    {
        auto * self = rc_cast<ImplPersistentTaskBufferBlasTlas *>(handle);
        if (self->owned_buffer_info.has_value())
        {
            BufferId const first_id = std::get<std::vector<BufferId>>(self->actual_ids)[0];
            self->owned_buffer_device.value().destroy_buffer(first_id);
        }
        delete self;
    }

    // --- TaskBuffer ---

    TaskBuffer::TaskBuffer(TaskBufferInfo const & info)
    {
        this->object = new ImplPersistentTaskBufferBlasTlas(info);
    }

    TaskBuffer::TaskBuffer(daxa::Device & device, BufferInfo const & info)
    {
        this->object = new ImplPersistentTaskBufferBlasTlas(device, info);
    }

    auto TaskBuffer::view() const -> TaskBufferView
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        return TaskBufferView{{.task_graph_index = std::numeric_limits<u32>::max(), .index = impl.unique_index}};
    }

    TaskBuffer::operator TaskBufferView() const
    {
        return view();
    }

    auto TaskBuffer::info() const -> TaskBufferInfo const &
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        return std::get<TaskBufferInfo>(impl.info);
    }

    auto TaskBuffer::get_state() const -> TrackedBuffers
    {
        auto const & impl = *r_cast<ImplPersistentTaskBufferBlasTlas const *>(this->object);
        return TrackedBuffers{
            .buffers = {std::get<std::vector<BufferId>>(impl.actual_ids).data(), std::get<std::vector<BufferId>>(impl.actual_ids).size()},
            .latest_access = impl.latest_access,
        };
    }

    auto TaskBuffer::is_owning() const -> bool
    {
        auto const & impl = *r_cast<ImplPersistentTaskBufferBlasTlas const *>(this->object);
        return impl.owned_buffer_device.has_value();
    }

    void TaskBuffer::set_buffers(TrackedBuffers const & buffers)
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        auto & actual_buffers = std::get<std::vector<BufferId>>(impl.actual_ids);
        actual_buffers.clear();
        actual_buffers.insert(actual_buffers.end(), buffers.buffers.begin(), buffers.buffers.end());
        impl.latest_access = buffers.latest_access;
    }

    void TaskBuffer::swap_buffers(TaskBuffer & other)
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        auto & actual_buffers = std::get<std::vector<BufferId>>(impl.actual_ids);
        auto & impl_other = *r_cast<ImplPersistentTaskBufferBlasTlas *>(other.object);
        auto & other_actual_buffers = std::get<std::vector<BufferId>>(impl_other.actual_ids);
        std::swap(actual_buffers, other_actual_buffers);
        std::swap(impl.latest_access, impl_other.latest_access);
    }

    auto TaskBuffer::inc_refcnt(ImplHandle const * object) -> u64
    {
        return object->inc_refcnt();
    }

    auto TaskBuffer::dec_refcnt(ImplHandle const * object) -> u64
    {
        return object->dec_refcnt(
            ImplPersistentTaskBufferBlasTlas::zero_ref_callback,
            nullptr);
    }

    // --- TaskBuffer End ---

    // --- TaskBlas ---

    TaskBlas::TaskBlas(TaskBlasInfo const & info)
    {
        this->object = new ImplPersistentTaskBufferBlasTlas(info);
    }

    auto TaskBlas::view() const -> TaskBlasView
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        return TaskBlasView{{.task_graph_index = std::numeric_limits<u32>::max(), .index = impl.unique_index}};
    }

    TaskBlas::operator TaskBlasView() const
    {
        return view();
    }

    auto TaskBlas::info() const -> TaskBlasInfo const &
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        return std::get<TaskBlasInfo>(impl.info);
    }

    auto TaskBlas::get_state() const -> TrackedBlas
    {
        auto const & impl = *r_cast<ImplPersistentTaskBufferBlasTlas const *>(this->object);
        return TrackedBlas{
            .blas = {std::get<std::vector<BlasId>>(impl.actual_ids).data(), std::get<std::vector<BlasId>>(impl.actual_ids).size()},
            .latest_access = impl.latest_access,
        };
    }

    void TaskBlas::set_blas(TrackedBlas const & other_tracked)
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        auto & actual_ids = std::get<std::vector<BlasId>>(impl.actual_ids);
        actual_ids.clear();
        actual_ids.insert(actual_ids.end(), other_tracked.blas.begin(), other_tracked.blas.end());
        impl.latest_access = other_tracked.latest_access;
    }

    void TaskBlas::swap_blas(TaskBlas & other)
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        auto & actual_buffers = std::get<std::vector<BlasId>>(impl.actual_ids);
        auto & impl_other = *r_cast<ImplPersistentTaskBufferBlasTlas *>(other.object);
        auto & other_actual_buffers = std::get<std::vector<BlasId>>(impl_other.actual_ids);
        std::swap(actual_buffers, other_actual_buffers);
        std::swap(impl.latest_access, impl_other.latest_access);
    }

    auto TaskBlas::inc_refcnt(ImplHandle const * object) -> u64
    {
        return object->inc_refcnt();
    }

    auto TaskBlas::dec_refcnt(ImplHandle const * object) -> u64
    {
        return object->dec_refcnt(
            ImplPersistentTaskBufferBlasTlas::zero_ref_callback,
            nullptr);
    }

    // --- TaskBlas End ---

    // --- TaskTlas ---

    TaskTlas::TaskTlas(TaskTlasInfo const & info)
    {
        this->object = new ImplPersistentTaskBufferBlasTlas(info);
    }

    auto TaskTlas::view() const -> TaskTlasView
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        return TaskTlasView{{.task_graph_index = std::numeric_limits<u32>::max(), .index = impl.unique_index}};
    }

    TaskTlas::operator TaskTlasView() const
    {
        return view();
    }

    auto TaskTlas::info() const -> TaskTlasInfo const &
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        return std::get<TaskTlasInfo>(impl.info);
    }

    auto TaskTlas::get_state() const -> TrackedTlas
    {
        auto const & impl = *r_cast<ImplPersistentTaskBufferBlasTlas const *>(this->object);
        return TrackedTlas{
            .tlas = {std::get<std::vector<TlasId>>(impl.actual_ids).data(), std::get<std::vector<TlasId>>(impl.actual_ids).size()},
            .latest_access = impl.latest_access,
        };
    }

    void TaskTlas::set_tlas(TrackedTlas const & other_tracked)
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        auto & actual_ids = std::get<std::vector<TlasId>>(impl.actual_ids);
        actual_ids.clear();
        actual_ids.insert(actual_ids.end(), other_tracked.tlas.begin(), other_tracked.tlas.end());
        impl.latest_access = other_tracked.latest_access;
    }

    void TaskTlas::swap_tlas(TaskTlas & other)
    {
        auto & impl = *r_cast<ImplPersistentTaskBufferBlasTlas *>(this->object);
        auto & actual_buffers = std::get<std::vector<TlasId>>(impl.actual_ids);
        auto & impl_other = *r_cast<ImplPersistentTaskBufferBlasTlas *>(other.object);
        auto & other_actual_ids = std::get<std::vector<TlasId>>(impl_other.actual_ids);
        std::swap(actual_buffers, other_actual_ids);
        std::swap(impl.latest_access, impl_other.latest_access);
    }

    auto TaskTlas::inc_refcnt(ImplHandle const * object) -> u64
    {
        return object->inc_refcnt();
    }

    auto TaskTlas::dec_refcnt(ImplHandle const * object) -> u64
    {
        return object->dec_refcnt(
            ImplPersistentTaskBufferBlasTlas::zero_ref_callback,
            nullptr);
    }

    // --- TaskTlas End ---

    TaskImage::TaskImage(TaskImageInfo const & a_info)
    {
        this->object = new ImplPersistentTaskImage(a_info);
    }

    ImplPersistentTaskImage::ImplPersistentTaskImage(TaskImageInfo const & a_info)
        : info{a_info},
          actual_images{a_info.initial_images.images.begin(), a_info.initial_images.images.end()},
          latest_slice_states{a_info.initial_images.latest_slice_states.begin(), a_info.initial_images.latest_slice_states.end()},
          unique_index{ImplPersistentTaskImage::exec_unique_next_index++}
    {
    }

    ImplPersistentTaskImage::~ImplPersistentTaskImage() = default;

    void ImplPersistentTaskImage::zero_ref_callback(ImplHandle const * handle)
    {
        auto const * self = r_cast<ImplPersistentTaskImage const *>(handle);
        delete self;
    }

    TaskImage::operator TaskImageView() const
    {
        return view();
    }

    auto TaskImage::view() const -> TaskImageView
    {
        auto & impl = *r_cast<ImplPersistentTaskImage *>(this->object);
        return TaskImageView{{.task_graph_index = std::numeric_limits<u32>::max(), .index = impl.unique_index}};
    }

    auto TaskImage::info() const -> TaskImageInfo const &
    {
        auto & impl = *r_cast<ImplPersistentTaskImage *>(this->object);
        return impl.info;
    }

    auto TaskImage::get_state() const -> TrackedImages
    {
        auto const & impl = *r_cast<ImplPersistentTaskImage const *>(this->object);
        return TrackedImages{
            .images = {impl.actual_images.data(), impl.actual_images.size()},
            .latest_slice_states = {impl.latest_slice_states.data(), impl.latest_slice_states.size()},
        };
    }

    void TaskImage::set_images(TrackedImages const & images)
    {
        auto & impl = *r_cast<ImplPersistentTaskImage *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.info.swapchain_image || (images.images.size() == 1), "swapchain task image can only have at most one runtime image");
        impl.actual_images.clear();
        impl.actual_images.insert(impl.actual_images.end(), images.images.begin(), images.images.end());
        impl.latest_slice_states.clear();
        impl.latest_slice_states.insert(impl.latest_slice_states.end(), images.latest_slice_states.begin(), images.latest_slice_states.end());
        impl.waited_on_acquire = {};
    }

    void TaskImage::swap_images(TaskImage & other)
    {
        auto & impl = *r_cast<ImplPersistentTaskImage *>(this->object);
        auto & impl_other = *r_cast<ImplPersistentTaskImage *>(other.object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.info.swapchain_image || (impl_other.actual_images.size() <= 1), "swapchain task image can only have at most one runtime image");
        std::swap(impl.actual_images, impl_other.actual_images);
        std::swap(impl.latest_slice_states, impl_other.latest_slice_states);
        std::swap(impl.waited_on_acquire, impl_other.waited_on_acquire);
    }

    auto TaskImage::inc_refcnt(ImplHandle const * object) -> u64
    {
        return object->inc_refcnt();
    }

    auto TaskImage::dec_refcnt(ImplHandle const * object) -> u64
    {
        return object->dec_refcnt(
            ImplPersistentTaskImage::zero_ref_callback,
            nullptr);
    }

    TaskGraph::TaskGraph(TaskGraphInfo const & info)
    {
        this->object = new ImplTaskGraph(info);
        auto & impl = *r_cast<ImplTaskGraph *>(this->object);
        impl.permutations.resize(usize{1} << info.permutation_condition_count);
        for (auto & permutation : impl.permutations)
        {
            permutation.batch_submit_scopes.push_back({});
        }
        impl.update_active_permutations();
    }
    TaskGraph::~TaskGraph() = default;

    void TaskGraph::use_persistent_buffer(TaskBuffer const & buffer)
    {
        auto & impl = *reinterpret_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "completed task graphs can not record new tasks");
        DAXA_DBG_ASSERT_TRUE_M(!impl.buffer_name_to_id.contains(buffer.info().name), "task buffer names must be unique");
        TaskBufferView const task_buffer_id{{.task_graph_index = impl.unique_index, .index = static_cast<u32>(impl.global_buffer_infos.size())}};

        for (auto & permutation : impl.permutations)
        {
            permutation.buffer_infos.push_back(PerPermTaskBuffer{
                .valid = false,
            });
        }

        impl.global_buffer_infos.emplace_back(PermIndepTaskBufferInfo{
            .task_buffer_data = PermIndepTaskBufferInfo::Persistent{
                .buffer_blas_tlas = buffer,
            },
        });
        impl.persistent_buffer_index_to_local_index[buffer.view().index] = task_buffer_id.index;
        impl.buffer_name_to_id[buffer.info().name] = task_buffer_id;
    }

    void TaskGraph::use_persistent_blas(TaskBlas const & blas)
    {
        auto & impl = *reinterpret_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "completed task graphs can not record new tasks");
        DAXA_DBG_ASSERT_TRUE_M(!impl.blas_name_to_id.contains(blas.info().name), "task blas names must be unique");
        TaskBlasView const task_blas_id{{.task_graph_index = impl.unique_index, .index = static_cast<u32>(impl.global_buffer_infos.size())}};

        for (auto & permutation : impl.permutations)
        {
            permutation.buffer_infos.push_back(PerPermTaskBuffer{
                .valid = false,
            });
        }

        impl.global_buffer_infos.emplace_back(PermIndepTaskBufferInfo{
            .task_buffer_data = PermIndepTaskBufferInfo::Persistent{
                .buffer_blas_tlas = blas,
            },
        });
        impl.persistent_buffer_index_to_local_index[blas.view().index] = task_blas_id.index;
        impl.blas_name_to_id[blas.info().name] = task_blas_id;
    }

    void TaskGraph::use_persistent_tlas(TaskTlas const & tlas)
    {
        auto & impl = *reinterpret_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "completed task graphs can not record new tasks");
        DAXA_DBG_ASSERT_TRUE_M(!impl.tlas_name_to_id.contains(tlas.info().name), "task tlas names must be unique");
        TaskTlasView const task_tlas_id{{.task_graph_index = impl.unique_index, .index = static_cast<u32>(impl.global_buffer_infos.size())}};

        for (auto & permutation : impl.permutations)
        {
            permutation.buffer_infos.push_back(PerPermTaskBuffer{
                .valid = false,
            });
        }

        impl.global_buffer_infos.emplace_back(PermIndepTaskBufferInfo{
            .task_buffer_data = PermIndepTaskBufferInfo::Persistent{
                .buffer_blas_tlas = tlas,
            },
        });
        impl.persistent_buffer_index_to_local_index[tlas.view().index] = task_tlas_id.index;
        impl.tlas_name_to_id[tlas.info().name] = task_tlas_id;
    }

    void TaskGraph::use_persistent_image(TaskImage const & image)
    {
        auto & impl = *reinterpret_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "completed task graphs can not record new tasks");
        DAXA_DBG_ASSERT_TRUE_M(!impl.image_name_to_id.contains(image.info().name), "task image names must be unique");
        TaskImageView const task_image_id{{.task_graph_index = impl.unique_index, .index = static_cast<u32>(impl.global_image_infos.size())}};

        for (auto & permutation : impl.permutations)
        {
            // For non-persistent resources task graph will synch on the initial to first use every execution.
            permutation.image_infos.emplace_back();
            if (image.info().swapchain_image)
            {
                DAXA_DBG_ASSERT_TRUE_M(permutation.swapchain_image.is_empty(), "can only register one swapchain image per task graph permutation");
                permutation.swapchain_image = task_image_id;
            }
        }

        impl.global_image_infos.emplace_back(PermIndepTaskImageInfo{
            .task_image_data = PermIndepTaskImageInfo::Persistent{
                .image = image,
            }});
        impl.persistent_image_index_to_local_index[image.view().index] = task_image_id.index;
        impl.image_name_to_id[image.info().name] = task_image_id;
    }

    auto TaskGraph::create_transient_buffer(TaskTransientBufferInfo const & info) -> TaskBufferView
    {
        auto & impl = *reinterpret_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "completed task graphs can not record new tasks");
        DAXA_DBG_ASSERT_TRUE_M(!impl.buffer_name_to_id.contains(info.name), "task buffer names must be unique");
        TaskBufferView task_buffer_id{{.task_graph_index = impl.unique_index, .index = static_cast<u32>(impl.global_buffer_infos.size())}};

        for (auto & permutation : impl.permutations)
        {
            permutation.buffer_infos.push_back(PerPermTaskBuffer{
                .valid = permutation.active,
            });
        }
        auto const & info_copy = info; // NOTE: (HACK) we must do this because msvc designated init bugs causing it to not generate copy constructors.
        impl.global_buffer_infos.emplace_back(PermIndepTaskBufferInfo{
            .task_buffer_data = PermIndepTaskBufferInfo::Transient{.info = info_copy}});

        impl.buffer_name_to_id[info.name] = task_buffer_id;
        return task_buffer_id;
    }

    auto TaskGraph::create_transient_image(TaskTransientImageInfo const & info) -> TaskImageView
    {
        auto & impl = *reinterpret_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "completed task graphs can not record new tasks");
        DAXA_DBG_ASSERT_TRUE_M(!impl.image_name_to_id.contains(info.name), "task image names must be unique");
        TaskImageView task_image_view = {{
            .task_graph_index = impl.unique_index,
            .index = static_cast<u32>(impl.global_image_infos.size()),
        }};
        task_image_view.slice = {
            .level_count = info.mip_level_count,
            .layer_count = info.array_layer_count,
        };

        for (auto & permutation : impl.permutations)
        {
            permutation.image_infos.emplace_back(PerPermTaskImage{
                .valid = permutation.active,
                .swapchain_semaphore_waited_upon = false,
            });
        }

        auto info_copy = info; // NOTE: (HACK) we must do this because msvc designated init bugs causing it to not generate copy constructors.
        impl.global_image_infos.emplace_back(PermIndepTaskImageInfo{
            .task_image_data = PermIndepTaskImageInfo::Transient{
                .info = info_copy,
            }});
        impl.image_name_to_id[info.name] = task_image_view;
        return task_image_view;
    }

    // static inline constexpr std::array<BufferId, 64> NULL_BUF_ARRAY = {};
    // auto ImplTaskGraph::get_actual_buffers(TaskBufferView id, TaskGraphPermutation const & perm) const -> std::span<BufferId const>
    // {
    //     if (id.is_null())
    //     {
    //         return NULL_BUF_ARRAY;
    //     }
    //     auto const & global_buffer = global_buffer_infos.at(id.index);
    //     if (global_buffer.is_persistent())
    //     {
    //         return {global_buffer.get_persistent().actual_buffers.data(),
    //                 global_buffer.get_persistent().actual_buffers.size()};
    //     }
    //     else
    //     {
    //         auto const & perm_buffer = perm.buffer_infos.at(id.index);
    //         DAXA_DBG_ASSERT_TRUE_M(perm_buffer.valid, "Can not get actual buffer - buffer is not valid in this permutation");
    //         return {&perm_buffer.actual_id, 1};
    //     }
    // }

    static inline constexpr std::array<ImageId, 64> NULL_IMG_ARRAY = {};

    auto ImplTaskGraph::get_actual_images(TaskImageView id, TaskGraphPermutation const & perm) const -> std::span<ImageId const>
    {
        if (id.is_null())
        {
            return NULL_IMG_ARRAY;
        }
        auto const & global_image = global_image_infos.at(id.index);
        if (global_image.is_persistent())
        {
            return {global_image.get_persistent().actual_images.data(),
                    global_image.get_persistent().actual_images.size()};
        }
        else
        {
            auto const & perm_image = perm.image_infos.at(id.index);
            DAXA_DBG_ASSERT_TRUE_M(perm_image.valid, "Can not get actual image - image is not valid in this permutation");
            return {&perm_image.actual_image, 1};
        }
    }

    auto ImplTaskGraph::id_to_local_id(TaskImageView id) const -> TaskImageView
    {
        if (id.is_null()) return id;
        DAXA_DBG_ASSERT_TRUE_M(!id.is_empty(), "Detected empty task image id. Please make sure to only use initialized task image ids.");
        if (id.is_persistent())
        {
            DAXA_DBG_ASSERT_TRUE_MS(
                persistent_image_index_to_local_index.contains(id.index),
                << "Detected invalid access of persistent task image id "
                << id.index
                << " in task graph \""
                << info.name
                << "\". Please make sure to declare persistent resource use to each task graph that uses this image with the function use_persistent_image!");
            return TaskImageView{{.task_graph_index = this->unique_index, .index = persistent_image_index_to_local_index.at(id.index)}, id.slice};
        }
        else
        {
            DAXA_DBG_ASSERT_TRUE_MS(
                id.task_graph_index == this->unique_index,
                << "Detected invalid access of transient task image id "
                << (id.index)
                << " in task graph \""
                << info.name
                << "\". Please make sure that you only use transient image within the list they are created in!");
            return TaskImageView{{.task_graph_index = this->unique_index, .index = id.index}, id.slice};
        }
    }

    void ImplTaskGraph::update_active_permutations()
    {
        record_active_permutations.clear();

        for (u32 permutation_i = 0; permutation_i < permutations.size(); ++permutation_i)
        {
            bool const active = (record_active_conditional_scopes & permutation_i) == (record_active_conditional_scopes & record_conditional_states);
            permutations[permutation_i].active = active;
            if (active)
            {
                record_active_permutations.push_back(&permutations[permutation_i]);
            }
        }
    }

    void validate_runtime_image_slice(ImplTaskGraph & impl, TaskGraphPermutation const & perm, u32 use_index, u32 task_image_index, ImageMipArraySlice const & access_slice)
    {
        auto const actual_images = impl.get_actual_images(TaskImageView{{.task_graph_index = impl.unique_index, .index = task_image_index}}, perm);
        std::string_view task_name = impl.global_image_infos[task_image_index].get_name();
        for (u32 index = 0; index < actual_images.size(); ++index)
        {
            ImageMipArraySlice const full_slice = impl.info.device.image_view_info(actual_images[index].default_view()).value().slice;
            auto name_sw = impl.info.device.image_info(actual_images[index]).value().name;
            std::string const & name = {name_sw.data(), name_sw.size()};
            [[maybe_unused]] bool const use_within_runtime_image_counts =
                (access_slice.base_mip_level + access_slice.level_count <= full_slice.base_mip_level + full_slice.level_count) &&
                (access_slice.base_array_layer + access_slice.layer_count <= full_slice.base_array_layer + full_slice.layer_count);
            [[maybe_unused]] std::string const error_message =
                fmt::format(R"(task image argument (arg index: {}, task image: "{}", slice: {}) exceeds runtime image (index: {}, name: "{}") dimensions ({})!)",
                            use_index, task_name, to_string(access_slice), index, name, to_string(full_slice));
            DAXA_DBG_ASSERT_TRUE_M(use_within_runtime_image_counts, error_message);
        }
    }

    void validate_image_attachs(ImplTaskGraph & impl, TaskGraphPermutation const & perm, [[maybe_unused]] u32 use_index, u32 task_image_index, TaskImageAccess task_access, [[maybe_unused]] std::string_view task_name)
    {
        ImageUsageFlags const use_flags = access_to_usage(task_access);
        auto const actual_images = impl.get_actual_images(TaskImageView{{.task_graph_index = impl.unique_index, .index = task_image_index}}, perm);
        std::string_view task_image_name = impl.global_image_infos[task_image_index].get_name();
        for (auto image : actual_images)
        {
            [[maybe_unused]] bool const access_valid = (impl.info.device.image_info(image).value().usage & use_flags) != ImageUsageFlagBits::NONE;
            DAXA_DBG_ASSERT_TRUE_M(access_valid, fmt::format("Detected invalid runtime image \"{}\" of task image \"{}\", in use {} of task \"{}\". "
                                                             "The given runtime image does NOT have the image use flag {} set, but the task use requires this use for all runtime images!",
                                                             impl.info.device.image_info(image).value().name.view(), task_image_name, use_index, task_name, daxa::to_string(use_flags)));
        }
    }

    void ImplTaskGraph::update_image_view_cache(ImplTask & task, TaskGraphPermutation const & permutation)
    {
        for_each(
            task.base_task->attachments(),
            [](u32, auto const &) {},
            [&](u32 task_image_attach_index, TaskImageAttachmentInfo const & image_attach)
            {
                // TODO:
                // Replace the validity check with a comparison of last execution actual images vs this frame actual images.
                auto & view_cache = task.image_view_cache[task_image_attach_index];
                auto & imgs_last_exec = task.runtime_images_last_execution[task_image_attach_index];

                if (image_attach.view.is_null())
                {
                    // Initialize the view array once with null ids.
                    if (image_attach.shader_array_size > 0 && view_cache.empty())
                    {
                        for (u32 index = 0; index < image_attach.shader_array_size; ++index)
                        {
                            view_cache.push_back(daxa::ImageViewId{});
                        }
                    }
                    return;
                }
                auto const slice = image_attach.translated_view.slice;
                // The image id here is already the task graph local id.
                // The persistent ids are converted to local ids in the add_task function.
                auto const tid = image_attach.translated_view;
                auto const actual_images = get_actual_images(tid, permutation);

                bool cache_valid = imgs_last_exec.size() == actual_images.size();
                if (imgs_last_exec.size() == actual_images.size())
                {
                    for (u32 i = 0; i < imgs_last_exec.size(); ++i)
                    {
                        cache_valid = cache_valid && (imgs_last_exec[i] == actual_images[i]);
                    }
                }
                if (!cache_valid)
                {
                    imgs_last_exec.clear();
                    // Save current runtime images for the next time tg updates the views.
                    for (auto img : actual_images)
                    {
                        imgs_last_exec.push_back(img);
                    }
                    validate_runtime_image_slice(*this, permutation, task_image_attach_index, tid.index, slice);
                    validate_image_attachs(*this, permutation, task_image_attach_index, tid.index, image_attach.access, task.base_task->name());
                    for (auto & view : view_cache)
                    {
                        if (info.device.is_id_valid(view))
                        {
                            ImageViewId const parent_image_default_view = info.device.image_view_info(view).value().image.default_view();
                            // Can not destroy the default view of an image!!!
                            if (parent_image_default_view != view)
                            {
                                info.device.destroy_image_view(view);
                            }
                        }
                    }
                    view_cache.clear();
                    if (image_attach.shader_array_type == TaskHeadImageArrayType::RUNTIME_IMAGES)
                    {
                        for (auto parent : actual_images)
                        {
                            ImageViewInfo view_info = info.device.image_view_info(parent.default_view()).value();
                            ImageViewType const use_view_type = (image_attach.view_type != ImageViewType::MAX_ENUM) ? image_attach.view_type : view_info.type;
                            if (parent.default_view() == daxa::ImageViewId{})
                            {
                                printf("bruh view\n");
                            }
                            if (parent == daxa::ImageId{})
                            {
                                printf("bruh image\n");
                            }

                            // When the use image view parameters match the default view,
                            // then use the default view id and avoid creating a new id here.
                            bool const is_use_default_slice = view_info.slice == slice;
                            bool const is_use_default_view_type = use_view_type == view_info.type;
                            if (is_use_default_slice && is_use_default_view_type)
                            {
                                view_cache.push_back(parent.default_view());
                            }
                            else
                            {
                                view_info.type = use_view_type;
                                view_info.slice = slice;
                                view_cache.push_back(info.device.create_image_view(view_info));
                            }
                        }
                    }
                    else // image_attach.shader_array_type == TaskHeadImageArrayType::MIP_LEVELS
                    {
                        u32 const base_mip_level = image_attach.translated_view.slice.base_mip_level;
                        view_cache.reserve(image_attach.shader_array_size);
                        auto filled_views = std::min(image_attach.translated_view.slice.level_count, u32(image_attach.shader_array_size));
                        for (u32 index = 0; index < filled_views; ++index)
                        {
                            ImageViewInfo view_info = info.device.image_view_info(actual_images[0].default_view()).value();
                            ImageViewType const use_view_type = (image_attach.view_type != ImageViewType::MAX_ENUM) ? image_attach.view_type : view_info.type;
                            view_info.type = use_view_type;
                            view_info.slice = image_attach.translated_view.slice;
                            view_info.slice.base_mip_level = base_mip_level + index;
                            view_info.slice.level_count = 1;
                            view_cache.push_back(info.device.create_image_view(view_info));
                        }
                        // When the slice is smaller then the array size,
                        // The indices larger then the size are filled with 0 ids.
                        for (u32 index = filled_views; index < image_attach.shader_array_size; ++index)
                        {
                            view_cache.push_back(daxa::ImageViewId{});
                        }
                    }
                }
            });
    }

    void validate_runtime_resources([[maybe_unused]] ImplTaskGraph const & impl, [[maybe_unused]] TaskGraphPermutation const & permutation)
    {
#if DAXA_VALIDATION
        constexpr std::string_view PERSISTENT_RESOURCE_MESSAGE = {
            "when executing a task graph, all used persistent resources must be backed by at least one and exclusively "
            "valid runtime resources"};
        for (u32 local_buffer_i = 0; local_buffer_i < impl.global_buffer_infos.size(); ++local_buffer_i)
        {
            if (!permutation.buffer_infos[local_buffer_i].valid)
            {
                continue;
            }
            if (!impl.global_buffer_infos.at(local_buffer_i).is_persistent())
            {
                continue;
            }
            auto const & runtime_ids = impl.global_buffer_infos.at(local_buffer_i).get_persistent().actual_ids;
            std::visit([&](auto const & runtime_ids) {
                DAXA_DBG_ASSERT_TRUE_M(
                    !runtime_ids.empty(),
                    fmt::format(
                        "Detected persistent task buffer \"{}\" used in task graph \"{}\" with 0 runtime buffers; {}",
                        impl.global_buffer_infos[local_buffer_i].get_name(),
                        impl.info.name,
                        PERSISTENT_RESOURCE_MESSAGE));
                for (usize buffer_index = 0; buffer_index < runtime_ids.size(); ++buffer_index)
                {
                    DAXA_DBG_ASSERT_TRUE_M(
                        impl.info.device.is_id_valid(runtime_ids[buffer_index]),
                        fmt::format(
                            "Detected persistent task buffer \"{}\" used in task graph \"{}\" with invalid buffer id (runtime buffer index: {}); {}",
                            impl.global_buffer_infos[local_buffer_i].get_name(),
                            impl.info.name,
                            buffer_index,
                            PERSISTENT_RESOURCE_MESSAGE));
                }
            }, runtime_ids);
        }
        for (u32 local_image_i = 0; local_image_i < impl.global_image_infos.size(); ++local_image_i)
        {
            if (!permutation.image_infos[local_image_i].valid)
            {
                continue;
            }
            if (!impl.global_image_infos.at(local_image_i).is_persistent())
            {
                continue;
            }
            auto const & runtime_images = impl.global_image_infos.at(local_image_i).get_persistent().actual_images;
            DAXA_DBG_ASSERT_TRUE_M(
                !runtime_images.empty(),
                fmt::format(
                    "Detected persistent task image \"{}\" used in task graph \"{}\" with 0 runtime images; {}",
                    impl.global_image_infos[local_image_i].get_name(),
                    impl.info.name,
                    PERSISTENT_RESOURCE_MESSAGE));
            for (usize image_index = 0; image_index < runtime_images.size(); ++image_index)
            {
                DAXA_DBG_ASSERT_TRUE_M(
                    impl.info.device.is_id_valid(runtime_images[image_index]),
                    fmt::format(
                        "Detected persistent task image \"{}\" used in task graph \"{}\" with invalid image id (runtime image index: {}); {}",
                        impl.global_image_infos[local_image_i].get_name(),
                        impl.info.name,
                        image_index,
                        PERSISTENT_RESOURCE_MESSAGE));
            }
        }
#endif // #if DAXA_VALIDATION
    }

    auto write_attachment_shader_blob(Device device, u32 data_size, std::span<TaskAttachmentInfo const> attachments) -> std::vector<std::byte>
    {
        std::vector<std::byte> attachment_shader_blob = {};
        attachment_shader_blob.resize(data_size);
        usize shader_byte_blob_offset = 0;
        auto upalign = [&](size_t align_size)
        {
            if (align_size == 0)
            {
                return;
            }
            auto current_offset = shader_byte_blob_offset % align_size;
            if (current_offset != 0)
            {
                shader_byte_blob_offset += align_size - current_offset;
            }
        };
        for_each(
            attachments,
            [&](u32, auto const & attach)
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(attach)>, TaskBufferAttachmentInfo>)
                {
                    TaskBufferAttachmentInfo const & buffer_attach = attach;
                    if (buffer_attach.shader_as_address)
                    {
                        upalign(sizeof(DeviceAddress));
                        for (u32 shader_array_i = 0; shader_array_i < buffer_attach.shader_array_size; ++shader_array_i)
                        {
                            BufferId const buf_id = buffer_attach.ids[shader_array_i];
                            DeviceAddress const buf_address = buffer_attach.view.is_null() ? DeviceAddress{} : device.device_address(buf_id).value();
                            auto mini_blob = std::bit_cast<std::array<std::byte, sizeof(DeviceAddress)>>(buf_address);
                            std::memcpy(attachment_shader_blob.data() + shader_byte_blob_offset, &mini_blob, sizeof(DeviceAddress));
                            shader_byte_blob_offset += sizeof(DeviceAddress);
                        }
                    }
                    else
                    {
                        upalign(sizeof(daxa_BufferId));
                        for (u32 shader_array_i = 0; shader_array_i < buffer_attach.shader_array_size; ++shader_array_i)
                        {
                            BufferId const buf_id = buffer_attach.ids[shader_array_i];
                            auto mini_blob = std::bit_cast<std::array<std::byte, sizeof(daxa_BufferId)>>(buf_id);
                            std::memcpy(attachment_shader_blob.data() + shader_byte_blob_offset, &mini_blob, sizeof(daxa_BufferId));
                            shader_byte_blob_offset += sizeof(daxa_BufferId);
                        }
                    }
                }
                if constexpr (std::is_same_v<std::decay_t<decltype(attach)>, TaskTlasAttachmentInfo>)
                {
                    TaskTlasAttachmentInfo const & tlas_attach = attach;
                    if (tlas_attach.shader_as_address)
                    {
                        upalign(sizeof(DeviceAddress));
                        TlasId const tlas_id = attach.ids[0];
                        DeviceAddress const tlas_address = attach.view.is_null() ? DeviceAddress{} : device.device_address(tlas_id).value();
                        auto mini_blob = std::bit_cast<std::array<std::byte, sizeof(DeviceAddress)>>(tlas_address);
                        std::memcpy(attachment_shader_blob.data() + shader_byte_blob_offset, &mini_blob, sizeof(DeviceAddress));
                        shader_byte_blob_offset += sizeof(DeviceAddress);
                    }
                    else
                    {
                        upalign(sizeof(daxa_TlasId));
                        TlasId const tlas_id = tlas_attach.ids[0];
                        auto mini_blob = std::bit_cast<std::array<std::byte, sizeof(daxa_TlasId)>>(tlas_id);
                        std::memcpy(attachment_shader_blob.data() + shader_byte_blob_offset, &mini_blob, sizeof(daxa_TlasId));
                        shader_byte_blob_offset += sizeof(daxa_TlasId);
                    }
                }
            },
            [&](u32, TaskImageAttachmentInfo const & image_attach)
            {
                if (image_attach.shader_as_index)
                {
                    upalign(sizeof(daxa_ImageViewIndex));
                    for (u32 shader_array_i = 0; shader_array_i < image_attach.shader_array_size; ++shader_array_i)
                    {
                        ImageViewId const img_id = image_attach.view_ids[shader_array_i];
                        auto mini_blob = std::bit_cast<std::array<std::byte, sizeof(daxa_ImageViewIndex)>>(static_cast<uint32_t>(img_id.index));
                        std::memcpy(attachment_shader_blob.data() + shader_byte_blob_offset, &mini_blob, sizeof(daxa_ImageViewIndex));
                        shader_byte_blob_offset += sizeof(daxa_ImageViewIndex);
                    }
                }
                else
                {
                    upalign(sizeof(daxa_ImageViewId));
                    for (u32 shader_array_i = 0; shader_array_i < image_attach.shader_array_size; ++shader_array_i)
                    {
                        ImageViewId const img_id = image_attach.view_ids[shader_array_i];
                        auto mini_blob = std::bit_cast<std::array<std::byte, sizeof(daxa_ImageViewId)>>(img_id);
                        std::memcpy(attachment_shader_blob.data() + shader_byte_blob_offset, &mini_blob, sizeof(daxa_ImageViewId));
                        shader_byte_blob_offset += sizeof(daxa_ImageViewId);
                    }
                }
            });
        return attachment_shader_blob;
    }

    void ImplTaskGraph::execute_task(ImplTaskRuntimeInterface & impl_runtime, TaskGraphPermutation & permutation, u32 batch_index, TaskBatchId in_batch_task_index, TaskId task_id)
    {
        // We always allow to reuse the last command list ONCE within the task callback.
        // When the get command list function is called in a task this is set to false.
        // TODO(refactor): create discrete validation functions and call them before doing any work here.
        impl_runtime.reuse_last_command_list = true;
        ImplTask & task = tasks[task_id];
        update_image_view_cache(task, permutation);
        for_each(
            task.base_task->attachments(),
            [&](u32, auto & attach)
            {
                attach.ids = this->get_actual_buffer_blas_tlas(attach.translated_view, permutation);
                validate_task_buffer_blas_tlas_runtime_data(task, attach);
            },
            [&](u32 index, TaskImageAttachmentInfo & attach)
            {
                attach.ids = this->get_actual_images(attach.translated_view, permutation);
                attach.view_ids = std::span{task.image_view_cache[index].data(), task.image_view_cache[index].size()};
                validate_task_image_runtime_data(task, attach);
            });
        std::vector<std::byte> attachment_shader_blob = write_attachment_shader_blob(
            info.device,
            task.base_task->attachment_shader_blob_size(),
            task.base_task->attachments());
        impl_runtime.current_task = &task;
        impl_runtime.recorder.begin_label({
            .label_color = info.task_label_color,
            .name = std::string("batch ") + std::to_string(batch_index) + std::string(" task ") + std::to_string(in_batch_task_index) + std::string(" \"") + std::string(task.base_task->name()) + std::string("\""),
        });
        task.base_task->callback(TaskInterface{
            .device = this->info.device,
            .recorder = impl_runtime.recorder,
            .attachment_infos = task.base_task->attachments(),
            .allocator = this->staging_memory.has_value() ? &this->staging_memory.value() : nullptr,
            .attachment_shader_blob = attachment_shader_blob,
        });
        impl_runtime.recorder.end_label();
    }

    void TaskGraph::conditional(TaskGraphConditionalInfo const & conditional_info)
    {
        auto & impl = *reinterpret_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "completed task graphs can not record new tasks");
        [[maybe_unused]] bool const already_active = ((impl.record_active_conditional_scopes >> conditional_info.condition_index) & 1u) != 0;
        DAXA_DBG_ASSERT_TRUE_M(!already_active, "can not nest scopes of the same condition in itself.");
        DAXA_DBG_ASSERT_TRUE_M(conditional_info.condition_index < impl.info.permutation_condition_count,
                               fmt::format("Detected invalid conditional index {}; conditional indices must all be smaller then the conditional count given in construction", conditional_info.condition_index));
        // Set conditional scope to active.
        impl.record_active_conditional_scopes |= 1u << conditional_info.condition_index;
        impl.update_active_permutations();
        // Set the conditional state to active.
        impl.record_conditional_states |= 1u << conditional_info.condition_index;
        impl.update_active_permutations();
        if (conditional_info.when_true)
        {
            conditional_info.when_true();
        }
        // Set the conditional state to false.
        impl.record_conditional_states &= ~(1u << conditional_info.condition_index);
        impl.update_active_permutations();
        if (conditional_info.when_false)
        {
            conditional_info.when_false();
        }
        // Set conditional scope to inactive.
        impl.record_active_conditional_scopes &= ~(1u << conditional_info.condition_index);
        impl.update_active_permutations();
    }

    template <typename TrackedState>
    struct AccessRelation
    {
        bool is_previous_none = {};
        bool is_previous_read = {};
        bool is_current_read = {};
        bool is_previous_rw_concurrent = {};
        bool is_current_rw_concurrent = {};
        bool is_current_concurrent = {};
        bool are_both_read = {};
        bool are_both_rw_concurrent = {};
        bool are_layouts_identical = {};
        bool are_both_read_and_same_layout = {};
        bool are_both_rw_concurrent_and_same_layout = {};
        bool are_both_concurrent = {};
        bool are_both_concurrent_and_same_layout = {};
        AccessRelation(TrackedState latest, Access new_access, TaskAccessConcurrency new_concurrency, ImageLayout latest_layout = ImageLayout::UNDEFINED, ImageLayout new_layout = ImageLayout::UNDEFINED)
        {
            if constexpr (requires { latest.latest_access; })
            {
                is_previous_none = latest.latest_access.type == AccessTypeFlagBits::NONE;
                is_previous_read = latest.latest_access.type == AccessTypeFlagBits::READ;
                is_previous_rw_concurrent =
                    latest.latest_access.type == AccessTypeFlagBits::READ_WRITE &&
                    latest.latest_access_concurrent == TaskAccessConcurrency::CONCURRENT;
            }
            if constexpr (requires { latest.state.latest_access; })
            {
                is_previous_none = latest.state.latest_access.type == AccessTypeFlagBits::NONE;
                is_previous_read = latest.state.latest_access.type == AccessTypeFlagBits::READ;
                is_previous_rw_concurrent =
                    latest.state.latest_access.type == AccessTypeFlagBits::READ_WRITE &&
                    latest.latest_access_concurrent == TaskAccessConcurrency::CONCURRENT;
            }
            is_current_read = new_access.type == AccessTypeFlagBits::READ;
            is_current_rw_concurrent =
                new_access.type == AccessTypeFlagBits::READ_WRITE &&
                new_concurrency == TaskAccessConcurrency::CONCURRENT;
            is_current_concurrent = is_current_read || is_current_rw_concurrent;
            are_both_read = is_previous_read && is_current_read;
            are_both_rw_concurrent = is_previous_rw_concurrent && is_current_rw_concurrent;
            are_layouts_identical = latest_layout == new_layout;
            are_both_read_and_same_layout = are_both_read && are_layouts_identical;
            are_both_rw_concurrent_and_same_layout = are_both_rw_concurrent && are_layouts_identical;
            are_both_concurrent = (is_previous_read && is_current_read) || (is_previous_rw_concurrent && is_current_rw_concurrent);
            are_both_concurrent_and_same_layout = are_both_concurrent && are_layouts_identical;
        }
    };

    auto schedule_task(
        ImplTaskGraph & impl,
        TaskGraphPermutation & perm,
        TaskBatchSubmitScope & current_submit_scope,
        usize const current_submit_scope_index,
        ITask & task)
        -> usize
    {
        usize first_possible_batch_index = 0;
        if (!impl.info.reorder_tasks)
        {
            first_possible_batch_index = std::max(current_submit_scope.task_batches.size(), static_cast<size_t>(1ull)) - 1ull;
        }

        for_each(
            task.attachments(),
            [&](u32, auto const & attach)
            {
                if (attach.view.is_null()) return;
                PerPermTaskBuffer const & task_buffer = perm.buffer_infos[attach.translated_view.index];
                // If the latest access is in a previous submit scope, the earliest batch we can insert into is
                // the current scopes first batch.
                if (task_buffer.latest_access_submit_scope_index < current_submit_scope_index)
                {
                    return;
                }

                auto [current_buffer_access, current_access_concurrency] = task_buffer_access_to_access(static_cast<TaskBufferAccess>(attach.access));
                // Every other access (NONE, READ_WRITE, WRITE) are interpreted as writes in this context.
                // TODO(msakmary): improve scheduling here to reorder reads in front of each other, respecting the last to read barrier if present!
                // When a buffer has been read in a previous use AND the current task also reads the buffer,
                // we must insert the task at or after the last use batch.
                usize current_buffer_first_possible_batch_index = task_buffer.latest_access_batch_index;
                // When two buffer accesses intersect, we potentially need to insert a ner barrier or modify an existing barrier.
                // If the access is a read on read or a rw_concurrent on rw_concurrent, the task is still allowed within the same batch as the task of the previous access.
                // This means that two tasks reading from buffer X are allowed within the same batch, using the same barrier.
                // If they are not inserted within the same batch due to dependencies of other attachments, daxa will still reuse the barriers.
                // This is only possible for read write concurrent and read access sequences!
                AccessRelation<decltype(task_buffer)> relation{task_buffer, current_buffer_access, current_access_concurrency};
                if (!relation.are_both_read && !relation.are_both_rw_concurrent && !relation.is_previous_none)
                {
                    current_buffer_first_possible_batch_index += 1;
                }
                first_possible_batch_index = std::max(first_possible_batch_index, current_buffer_first_possible_batch_index);
            },
            [&](u32, TaskImageAttachmentInfo const & attach)
            {
                if (attach.view.is_null()) return;
                PerPermTaskImage const & task_image = perm.image_infos[attach.translated_view.index];
                PermIndepTaskImageInfo const & glob_task_image = impl.global_image_infos[attach.translated_view.index];
                DAXA_DBG_ASSERT_TRUE_M(!task_image.swapchain_semaphore_waited_upon, "swapchain image is already presented!");
                if (glob_task_image.is_persistent() && glob_task_image.get_persistent().info.swapchain_image)
                {
                    if (perm.swapchain_image_first_use_submit_scope_index == std::numeric_limits<u64>::max())
                    {
                        perm.swapchain_image_first_use_submit_scope_index = current_submit_scope_index;
                        perm.swapchain_image_last_use_submit_scope_index = current_submit_scope_index;
                    }
                    else
                    {
                        perm.swapchain_image_first_use_submit_scope_index = std::min(current_submit_scope_index, perm.swapchain_image_first_use_submit_scope_index);
                        perm.swapchain_image_last_use_submit_scope_index = std::max(current_submit_scope_index, perm.swapchain_image_last_use_submit_scope_index);
                    }
                }

                auto [this_task_image_layout, this_task_image_access, current_access_concurrent] = task_image_access_to_layout_access(attach.access);
                // As image subresources can be in different layouts and also different synchronization scopes,
                // we need to track these image ranges individually.
                for (ExtendedImageSliceState const & tracked_slice : task_image.last_slice_states)
                {
                    // If the latest access is in a previous submit scope, the earliest batch we can insert into is
                    // the current scopes first batch.
                    // When the slices dont intersect, we dont need to do any sync or execution ordering between them.
                    if (
                        tracked_slice.latest_access_submit_scope_index < current_submit_scope_index ||
                        !tracked_slice.state.slice.intersects(attach.translated_view.slice))
                    {
                        continue;
                    }
                    // Tasks are always shedules after or with the tasks they depend on.
                    usize current_image_first_possible_batch_index = tracked_slice.latest_access_batch_index;
                    // When two image accesses intersect, we potentially need to insert a ner barrier or modify an existing barrier.
                    // If the access is a read on read or a rw_concurrent on rw_concurrent, the task is still allowed within the same batch as the task of the previous access.
                    // This means that two tasks reading from image X are allowed within the same batch, using the same barrier.
                    // If they are not inserted within the same batch due to dependencies of other attachments, daxa will still reuse the barriers.
                    // This is only possible for read write concurrent and read access sequences!
                    AccessRelation<decltype(tracked_slice)> relation{tracked_slice, this_task_image_access, current_access_concurrent, tracked_slice.state.latest_layout, this_task_image_layout};
                    if (!relation.are_both_read_and_same_layout && !relation.are_both_rw_concurrent_and_same_layout)
                    {
                        current_image_first_possible_batch_index += 1;
                    }
                    first_possible_batch_index = std::max(first_possible_batch_index, current_image_first_possible_batch_index);
                }
            });
        // Make sure we have enough batches.
        if (first_possible_batch_index >= current_submit_scope.task_batches.size())
        {
            current_submit_scope.task_batches.resize(first_possible_batch_index + 1);
        }
        return first_possible_batch_index;
    }

    void translate_persistent_ids(ImplTaskGraph const & impl, ITask * task)
    {
        for_each(
            task->attachments(),
            [&](u32 i, auto & attach)
            {
                validate_buffer_blas_tlas_task_view(*task, i, attach);
                attach.translated_view = impl.buffer_blas_tlas_id_to_local_id(attach.view);
            },
            [&](u32 i, TaskImageAttachmentInfo & attach)
            {
                validate_image_task_view(*task, i, attach);
                attach.translated_view = impl.id_to_local_id(attach.view);
            });
    }

    void TaskGraph::add_task(std::unique_ptr<ITask> && task)
    {
        auto & impl = *reinterpret_cast<ImplTaskGraph *>(this->object);
        validate_not_compiled(impl);
        validate_overlapping_attachment_views(impl, task.get());

        TaskId const task_id = impl.tasks.size();

        std::vector<std::vector<ImageViewId>> view_cache = {};
        view_cache.resize(task->attachments().size(), {});
        std::vector<std::vector<ImageId>> id_cache = {};
        id_cache.resize(task->attachments().size(), {});
        auto impl_task = ImplTask{
            .base_task = std::move(task),
            .image_view_cache = std::move(view_cache),
            .runtime_images_last_execution = std::move(id_cache),
        };
        translate_persistent_ids(impl, impl_task.base_task.get());

        for (auto * permutation : impl.record_active_permutations)
        {
            permutation->add_task(impl, impl_task, task_id);
        }

        impl.tasks.emplace_back(std::move(impl_task));
    }

    thread_local std::vector<ImageMipArraySlice> tl_new_access_slices = {};
    void update_image_initial_access_slices(
        PerPermTaskImage & task_image,
        ExtendedImageSliceState new_access_slice)
    {
        // We need to test if a new use adds and or subtracts from initial uses.
        // To do that, we need to test if the new access slice is accessing a subresource BEFORE all other already stored initial access slices.
        // We compare all new use slices with already tracked first uses.
        // We intersect the earlier and later happening slices with the new slice and store the intersection rest or the respective later executing slice access.
        // This list will contain the remainder of the new access ranges after intersections.
        tl_new_access_slices.push_back(new_access_slice.state.slice);
        // Name shortening:
        auto & initial_accesses = task_image.first_slice_states;
        // Note(pahrens):
        // NEVER access new_access_slice.slice in this function past this point.
        // ALWAYS access the new access ImageMipArraySlice's from an iterator or via vector indexing.
        for (usize new_access_slice_i = 0; new_access_slice_i < tl_new_access_slices.size();)
        {
            bool broke_inner_loop = false;
            for (usize initial_access_i = 0; initial_access_i < task_image.first_slice_states.size();)
            {
                bool const slices_disjoint = !tl_new_access_slices[new_access_slice_i].intersects(initial_accesses[initial_access_i].state.slice);
                bool const same_batch =
                    new_access_slice.latest_access_submit_scope_index == initial_accesses[initial_access_i].latest_access_submit_scope_index &&
                    new_access_slice.latest_access_batch_index == initial_accesses[initial_access_i].latest_access_batch_index;
                // We check if the sets are disjoint.
                // If they are we do not need to do anything and advance to the next test.
                // When two accesses are in the same batch and scope, they can not overlap.
                // This is simply forbidden by task graph rules!
                if (same_batch || slices_disjoint)
                {
                    ++initial_access_i;
                    continue;
                }
                // Now that we have this edge case out the way, we now need to test which tracked slice is executed earlier.
                bool const new_use_executes_earlier =
                    new_access_slice.latest_access_submit_scope_index < initial_accesses[initial_access_i].latest_access_submit_scope_index ||
                    (new_access_slice.latest_access_submit_scope_index == initial_accesses[initial_access_i].latest_access_submit_scope_index &&
                     new_access_slice.latest_access_batch_index < initial_accesses[initial_access_i].latest_access_batch_index);
                // When the new use is executing earlier, we subtract from the current initial access slice.
                // We then replace the current initial access slice with the resulting rest.
                if (new_use_executes_earlier)
                {
                    // When we intersect, we remove the old initial access slice and replace it with the rest of the subtraction.
                    // We need a copy of this, as we will erase this value from the vector first.
                    auto const initial_access_slice = initial_accesses[initial_access_i];
                    // Erase value from vector.
                    task_image.first_slice_states.erase(initial_accesses.begin() + isize(initial_access_i));
                    // Subtract ranges.
                    auto const [slice_rest, slice_rest_count] = initial_access_slice.state.slice.subtract(tl_new_access_slices[new_access_slice_i]);
                    // Now construct new sub-ranges from the rest of the subtraction.
                    // We advance the iterator each time.
                    for (usize rest_i = 0; rest_i < slice_rest_count; ++rest_i)
                    {
                        auto rest_tracked_slice = initial_access_slice;
                        rest_tracked_slice.state.slice = slice_rest[rest_i];
                        // We insert into the beginning, so we dont recheck these with the current new use slice.
                        // They are the result of a subtraction therefore disjoint.
                        task_image.first_slice_states.insert(initial_accesses.begin(), rest_tracked_slice);
                    }
                    // We erased, so we implicitly advanced by an element, as erase moves all elements one to the left past the iterator.
                    // But as we inserted into the front, we need to move the index accordingly to "stay in place".
                    initial_access_i += slice_rest_count;
                }
                // When the new use is executing AFTER the current inital access slice, we subtract the current initial access slice from the new slice.
                // We then replace the current new access slice with the resulting rest.
                else
                {
                    // We subtract the initial use from the new use and append the rest.
                    auto const [slice_rest, slice_rest_count] = tl_new_access_slices[new_access_slice_i].subtract(initial_accesses[initial_access_i].state.slice);
                    // We insert the rest of the subtraction into the new use list.
                    tl_new_access_slices.insert(tl_new_access_slices.end(), slice_rest.begin(), slice_rest.begin() + isize(slice_rest_count));
                    // We remove the current new use slice, as it intersects with an initial use slice and is later in the list.
                    tl_new_access_slices.erase(tl_new_access_slices.begin() + isize(new_access_slice_i));
                    // If we advance the new use index, we restart the inner loop over the initial accesses.
                    broke_inner_loop = true;
                    break;
                }
            }
            // When we broke out the inner loop we want to "restart" iteration of the outer loop at the current index.
            if (!broke_inner_loop)
            {
                ++new_access_slice_i;
            }
        }
        // Add the newly found initial access slices to the list of initial access slices.
        for (auto const & new_slice : tl_new_access_slices)
        {
            auto new_tracked_slice = new_access_slice;
            new_tracked_slice.state.slice = new_slice;
            initial_accesses.push_back(new_tracked_slice);
        }
        tl_new_access_slices.clear();
    }

    using ShaderUseIdOffsetTable = std::vector<Variant<std::pair<TaskImageView, usize>, std::pair<TaskBufferView, usize>, Monostate>>;

    void update_buffer_first_access(PerPermTaskBuffer & buffer, usize new_access_batch, usize new_access_submit, Access new_access)
    {
        if (buffer.first_access.type == AccessTypeFlagBits::NONE)
        {
            buffer.first_access = new_access;
            buffer.first_access_batch_index = new_access_batch;
            buffer.first_access_submit_scope_index = new_access_submit;
        }
        else if (buffer.first_access.type == AccessTypeFlagBits::READ && new_access.type == AccessTypeFlagBits::READ)
        {
            buffer.first_access = buffer.first_access | new_access;
            bool const new_is_earlier = new_access_submit < buffer.first_access_submit_scope_index ||
                                        (new_access_submit == buffer.first_access_submit_scope_index && new_access_batch < buffer.first_access_batch_index);
            if (new_is_earlier)
            {
                buffer.first_access_batch_index = new_access_batch;
                buffer.first_access_submit_scope_index = new_access_submit;
            }
        }
    }

    // I hate this function.
    thread_local std::vector<ExtendedImageSliceState> tl_tracked_slice_rests = {};
    thread_local std::vector<ImageMipArraySlice> tl_new_use_slices = {};
    void TaskGraphPermutation::add_task(
        ImplTaskGraph & task_graph_impl,
        ImplTask & impl_task,
        TaskId task_id)
    {
        auto & task = *impl_task.base_task;
        // Set persistent task resources to be valid for the permutation.
        for_each(
            task.attachments(),
            [&](u32, auto const & attach)
            {
                if (attach.view.is_null()) return;
                if (attach.view.is_persistent())
                {
                    buffer_infos[attach.translated_view.index].valid = true;
                }
            },
            [&](u32, TaskImageAttachmentInfo const & attach)
            {
                if (attach.view.is_null()) return;
                if (attach.view.is_persistent())
                {
                    image_infos[attach.translated_view.index].valid = true;
                }
            });

        usize const current_submit_scope_index = this->batch_submit_scopes.size() - 1;
        TaskBatchSubmitScope & current_submit_scope = this->batch_submit_scopes[current_submit_scope_index];

        // All tasks are reordered while recording.
        // Tasks are grouped into "task batches" which are just a group of tasks,
        // that can execute together while overlapping without synchronization between them.
        // Task batches are further grouped into submit scopes.
        // A submit scopes contains a group of batches between two submits.
        // At first, we find the batch we need to insert the new task into.
        // To optimize for optimal overlap and minimal pipeline barriers, we try to insert the task as early as possible.
        usize const batch_index = schedule_task(
            task_graph_impl,
            *this,
            current_submit_scope,
            current_submit_scope_index,
            task);
        TaskBatch & batch = current_submit_scope.task_batches[batch_index];
        // Add the task to the batch.
        batch.tasks.push_back(task_id);

        // Now that we know what batch we need to insert the task into, we need to add synchronization between batches.
        // As stated earlier batches are groups of tasks which can execute together without sync between them.
        // To simplify and optimize the sync placement daxa only synchronizes between batches.
        // This effectively means that all the resource uses, and their memory and execution dependencies in a batch
        // are combined into a single unit which is synchronized against other batches.
        for_each(
            task.attachments(),
            [&](u32, auto const & buffer_attach)
            {
                if (buffer_attach.view.is_null()) return;
                PerPermTaskBuffer & task_buffer = this->buffer_infos[buffer_attach.translated_view.index];
                auto [current_buffer_access, current_access_concurrency] = task_buffer_access_to_access(static_cast<TaskBufferAccess>(buffer_attach.access));
                update_buffer_first_access(task_buffer, batch_index, current_submit_scope_index, current_buffer_access);
                // For transient buffers, we need to record first and last use so that we can later name their allocations.
                // TODO(msakmary, pahrens) We should think about how to combine this with update_buffer_first_access below since
                // they both overlap in what they are doing
                if (!task_graph_impl.global_buffer_infos.at(buffer_attach.translated_view.index).is_persistent())
                {
                    auto & buffer_first_use = task_buffer.lifetime.first_use;
                    auto & buffer_last_use = task_buffer.lifetime.last_use;
                    if (current_submit_scope_index < buffer_first_use.submit_scope_index)
                    {
                        buffer_first_use.submit_scope_index = current_submit_scope_index;
                        buffer_first_use.task_batch_index = batch_index;
                    }
                    else if (current_submit_scope_index == buffer_first_use.submit_scope_index)
                    {
                        buffer_first_use.task_batch_index = std::min(buffer_first_use.task_batch_index, batch_index);
                    }

                    if (current_submit_scope_index > buffer_last_use.submit_scope_index ||
                        buffer_last_use.submit_scope_index == std::numeric_limits<u32>::max())
                    {
                        buffer_last_use.submit_scope_index = current_submit_scope_index;
                        buffer_last_use.task_batch_index = batch_index;
                    }
                    else if (current_submit_scope_index == buffer_last_use.submit_scope_index)
                    {
                        buffer_last_use.task_batch_index = std::max(buffer_last_use.task_batch_index, batch_index);
                    }
                }
                // When the last use was a read AND the new use of the buffer is a read AND,
                // we need to add our stage flags to the existing barrier of the last use.

                AccessRelation<decltype(task_buffer)> relation{task_buffer, current_buffer_access, current_access_concurrency};
                // We only need barriers between two non-concurrent accesses of a resource.
                // If the previous access is none, the current access is the first access.
                // Therefore we do not need to insert any synchronization if the previous access is none.
                // This is buffer specific. Images have a layout that needs to be set from undefined to the current accesses layout.
                // When the latest access  is a read that did not require a barrier before we also do not need a barrier now.
                // So skip, if the latest access is read and there is no latest_concurrent_access_barrer_index present.
                bool const last_access_concurrent_and_external =
                    daxa::holds_alternative<Monostate>(task_buffer.latest_concurrent_access_barrer_index) &&
                    (relation.is_previous_read || relation.is_previous_rw_concurrent);
                if (!relation.is_previous_none && !last_access_concurrent_and_external)
                {
                    if (relation.are_both_concurrent)
                    {
                        // If the last and current access is concurrent of the same type (read or rw concurrent), we can reuse the first barrier in the sequence of concurrent accesses.
                        if (LastConcurrentAccessSplitBarrierIndex const * index0 = daxa::get_if<LastConcurrentAccessSplitBarrierIndex>(&task_buffer.latest_concurrent_access_barrer_index))
                        {
                            auto & last_concurrent_access_split_barrier = this->split_barriers[index0->index];
                            last_concurrent_access_split_barrier.dst_access = last_concurrent_access_split_barrier.dst_access | current_buffer_access;
                        }
                        else if (LastConcurrentAccessBarrierIndex const * index1 = daxa::get_if<LastConcurrentAccessBarrierIndex>(&task_buffer.latest_concurrent_access_barrer_index))
                        {
                            auto & last_concurrent_access_barrier = this->barriers[index1->index];
                            last_concurrent_access_barrier.dst_access = last_concurrent_access_barrier.dst_access | current_buffer_access;
                        }
                    }
                    else
                    {
                        // When the uses are incompatible (no read on read) we need to insert a new barrier.
                        // Host access needs to be handled in a specialized way.
                        bool const src_host_only_access = task_buffer.latest_access.stages == PipelineStageFlagBits::HOST;
                        bool const dst_host_only_access = current_buffer_access.stages == PipelineStageFlagBits::HOST;
                        DAXA_DBG_ASSERT_TRUE_M(!(src_host_only_access && dst_host_only_access), "direct sync between two host accesses on gpu are not allowed");
                        bool const is_host_barrier = src_host_only_access || dst_host_only_access;
                        // When the distance between src and dst batch is one, we can replace the split barrier with a normal barrier.
                        // We also need to make sure we do not use split barriers when the src or dst stage exclusively uses the host stage.
                        // This is because the host stage does not declare an execution dependency on the cpu but only a memory dependency.
                        bool const use_pipeline_barrier =
                            (task_buffer.latest_access_batch_index + 1 == batch_index &&
                             current_submit_scope_index == task_buffer.latest_access_submit_scope_index) ||
                            is_host_barrier;
                        if (use_pipeline_barrier)
                        {
                            usize const barrier_index = this->barriers.size();
                            this->barriers.push_back(TaskBarrier{
                                .image_id = {}, // {} signals that this is not an image barrier.
                                .src_access = task_buffer.latest_access,
                                .dst_access = current_buffer_access,
                            });
                            // And we insert the barrier index into the list of pipeline barriers of the current tasks batch.
                            batch.pipeline_barrier_indices.push_back(barrier_index);
                            if (relation.is_current_concurrent)
                            {
                                // In an uninterrupted sequence of concurrent accesses we need to remember the fist concurrent access and the barrier.
                                task_buffer.latest_concurrent_access_barrer_index = LastConcurrentAccessBarrierIndex{barrier_index};
                            }
                        }
                        else
                        {
                            usize const split_barrier_index = this->split_barriers.size();
                            this->split_barriers.push_back(TaskSplitBarrier{
                                {
                                    .image_id = {}, // {} signals that this is not an image barrier.
                                    .src_access = task_buffer.latest_access,
                                    .dst_access = current_buffer_access,
                                },
                                /* .split_barrier_state = */ task_graph_impl.info.device.create_event({
                                    .name = std::string("tg \"") + task_graph_impl.info.name + "\" sb " + std::to_string(split_barrier_index),
                                }),
                            });
                            // Now we give the src batch the index of this barrier to signal.
                            TaskBatchSubmitScope & src_scope = this->batch_submit_scopes[task_buffer.latest_access_submit_scope_index];
                            TaskBatch & src_batch = src_scope.task_batches[task_buffer.latest_access_batch_index];
                            src_batch.signal_split_barrier_indices.push_back(split_barrier_index);
                            // And we also insert the split barrier index into the waits of the current tasks batch.
                            batch.wait_split_barrier_indices.push_back(split_barrier_index);
                            if (relation.is_current_concurrent)
                            {
                                // In an uninterrupted sequence of concurrent accesses we need to remember the fist concurrent access and the barrier.
                                task_buffer.latest_concurrent_access_barrer_index = LastConcurrentAccessSplitBarrierIndex{split_barrier_index};
                            }
                        }
                    }
                }
                // Now that we inserted/updated the synchronization, we update the latest access.
                task_buffer.latest_access = current_buffer_access;
                task_buffer.latest_access_batch_index = batch_index;
                task_buffer.latest_access_submit_scope_index = current_submit_scope_index;
                task_buffer.latest_access_concurrent = current_access_concurrency;
            },
            [&](u32, TaskImageAttachmentInfo & image_attach)
            {
                if (image_attach.view.is_null()) return;
                auto const & used_image_t_id = image_attach.translated_view;
                auto const & used_image_t_access = image_attach.access;
                auto const & initial_used_image_slice = image_attach.translated_view.slice;
                PerPermTaskImage & task_image = this->image_infos[used_image_t_id.index];
                // For transient images we need to record first and last use so that we can later name their allocations
                // TODO(msakmary, pahrens) We should think about how to combine this with update_image_inital_slices below since
                // they both overlap in what they are doing
                if (!task_graph_impl.global_image_infos.at(used_image_t_id.index).is_persistent())
                {
                    auto & image_first_use = task_image.lifetime.first_use;
                    auto & image_last_use = task_image.lifetime.last_use;
                    if (current_submit_scope_index < image_first_use.submit_scope_index)
                    {
                        image_first_use.submit_scope_index = current_submit_scope_index;
                        image_first_use.task_batch_index = batch_index;
                    }
                    else if (current_submit_scope_index == image_first_use.submit_scope_index)
                    {
                        image_first_use.task_batch_index = std::min(image_first_use.task_batch_index, batch_index);
                    }

                    if (current_submit_scope_index > image_last_use.submit_scope_index ||
                        image_last_use.submit_scope_index == std::numeric_limits<u32>::max())
                    {
                        image_last_use.submit_scope_index = current_submit_scope_index;
                        image_last_use.task_batch_index = batch_index;
                    }
                    else if (current_submit_scope_index == image_last_use.submit_scope_index)
                    {
                        image_last_use.task_batch_index = std::max(image_last_use.task_batch_index, batch_index);
                    }
                }
                task_image.usage |= access_to_usage(used_image_t_access);
                task_image.create_flags |= view_type_to_create_flags(image_attach.view_type);
                auto [current_image_layout, current_image_access, current_access_concurrency] = task_image_access_to_layout_access(used_image_t_access);
                image_attach.layout = current_image_layout;
                // Now this seems strange, why would be need multiple current use slices, as we only have one here.
                // This is because when we intersect this slice with the tracked slices, we get an intersection and a rest.
                // We need to then test the rest against all the remaining tracked uses,
                // as the intersected part is already beeing handled in the following code.
                tl_new_use_slices.push_back(initial_used_image_slice);
                // This is the tracked slice we will insert after we finished analyzing the current used image.
                ExtendedImageSliceState ret_new_use_tracked_slice{
                    .state = {
                        .latest_access = current_image_access,
                        .latest_layout = current_image_layout,
                        .slice = initial_used_image_slice,
                    },
                    .latest_access_concurrent = current_access_concurrency,
                    .latest_access_batch_index = batch_index,
                    .latest_access_submit_scope_index = current_submit_scope_index,
                    .latest_concurrent_access_barrer_index = {}, // This is a dummy value (either set later or ignored entirely).
                };
                // We update the initial access slices.
                update_image_initial_access_slices(task_image, ret_new_use_tracked_slice);
                // As image subresources can be in different layouts and also different synchronization scopes,
                // we need to track these image ranges individually.
                for (
                    auto tracked_slice_iter = task_image.last_slice_states.begin();
                    tracked_slice_iter != task_image.last_slice_states.end();)
                {
                    bool advanced_tracked_slice_iterator = false;
                    for (
                        auto used_image_slice_iter = tl_new_use_slices.begin();
                        used_image_slice_iter != tl_new_use_slices.end();)
                    {
                        // We make a local copy of both slices here.
                        // We can not rely on dereferencing the iterators, as we modify them in this function.
                        // For this inner loop we want to remember the information about these slices,
                        // even after they are removed from their respective vector.
                        ImageMipArraySlice const used_image_slice = *used_image_slice_iter;
                        ExtendedImageSliceState const tracked_slice = *tracked_slice_iter;
                        // We are only interested in intersecting ranges, as use of non intersecting ranges does not need synchronization.
                        if (!used_image_slice.intersects(tracked_slice.state.slice))
                        {
                            // We only need to advance the iterator manually here.
                            // After this if statement there is an unconditional erase that advances the iterator if this is not hit.
                            ++used_image_slice_iter;
                            continue;
                        }
                        // As we found an intersection, part of the old tracked slice must be altered.
                        // Instead of altering it we remove it and add the rest slices back in later.
                        used_image_slice_iter = tl_new_use_slices.erase(used_image_slice_iter);
                        // Now we know that the new use intersects slices with a previous use.
                        // This means that we need to find the intersecting part,
                        // sync the intersecting part from the previous use to the current use
                        // and remove the overlapping part from the tracked slice.
                        // Now we need to split the uses into three groups:
                        // * the slice that is the intersection of tracked image slice and current new use (intersection)
                        // * the part of the tracked image slice that does not intersect with the current new use (tracked_slice_rest)
                        // * the part of the current new use slice that does not intersect with the tracked image (new_use_slice_rest)
                        auto intersection = tracked_slice.state.slice.intersect(used_image_slice);
                        auto [tracked_slice_rest, tracked_slice_rest_count] = tracked_slice.state.slice.subtract(intersection);
                        auto [new_use_slice_rest, new_use_slice_rest_count] = used_image_slice.subtract(intersection);
                        // We now remove the old tracked slice from the list of tracked slices, as we just split it.
                        // We need to know if the iterator was advanced. This erase advances the iterator.
                        // If the iterator got not advanced by this we need to advance it ourself manually later.
                        advanced_tracked_slice_iterator = true;
                        tracked_slice_iter = task_image.last_slice_states.erase(tracked_slice_iter);
                        // Now we remember the left over slice from the original tracked slice.
                        for (usize rest_i = 0; rest_i < tracked_slice_rest_count; ++rest_i)
                        {
                            // The rest tracked slices are the same as the original tracked slice,
                            // except for the slice itself, which is the remainder of the subtraction of the intersection.
                            ExtendedImageSliceState current_rest_tracked_slice = tracked_slice;
                            current_rest_tracked_slice.state.slice = tracked_slice_rest[rest_i];
                            tl_tracked_slice_rests.push_back(current_rest_tracked_slice);
                        }
                        // Now we remember the left over slice from our current used slice.
                        for (usize rest_i = 0; rest_i < new_use_slice_rest_count; ++rest_i)
                        {
                            // We reassign the iterator here as it is getting invalidated by the insert.
                            // The new iterator points to the newly inserted element.
                            // When the next iteration of the outer for loop starts, all these elements get skipped.
                            // This is good as these elements do NOT intersect with the currently inspected tracked slice.
                            used_image_slice_iter = tl_new_use_slices.insert(
                                tl_new_use_slices.end(),
                                new_use_slice_rest[rest_i]);
                        }
                        // Every other access (NONE, READ_WRITE, WRITE) are interpreted as writes in this context.
                        // When the last use was a read AND the new use of the buffer is a read AND,
                        // we need to add our stage flags to the existing barrier of the last use.
                        // To be able to do this the layout of the image slice must also match.
                        // If they differ we need to insert an execution barrier with a layout transition.
                        AccessRelation<decltype(tracked_slice)> relation{tracked_slice, current_image_access, current_access_concurrency, tracked_slice.state.latest_layout, current_image_layout};
                        // Read write concurrent and reads (implicitly concurrent) are reusing the already inserted barriers if there was a previous identical access.
                        if (relation.are_both_concurrent_and_same_layout)
                        {
                            // Reuse first barrier in coherent access sequence.
                            if (auto const * index0 = daxa::get_if<LastConcurrentAccessSplitBarrierIndex>(&tracked_slice.latest_concurrent_access_barrer_index))
                            {
                                auto & last_read_split_barrier = this->split_barriers[index0->index];
                                last_read_split_barrier.dst_access = last_read_split_barrier.dst_access | tracked_slice.state.latest_access;
                            }
                            else if (auto const * index1 = daxa::get_if<LastConcurrentAccessBarrierIndex>(&tracked_slice.latest_concurrent_access_barrer_index))
                            {
                                auto & last_read_barrier = this->barriers[index1->index];
                                last_read_barrier.dst_access = last_read_barrier.dst_access | tracked_slice.state.latest_access;
                            }
                        }
                        else
                        {
                            // When the uses are incompatible (no read on read, or no identical layout) we need to insert a new barrier.
                            // Host access needs to be handled in a specialized way.
                            bool const src_host_only_access = tracked_slice.state.latest_access.stages == PipelineStageFlagBits::HOST;
                            bool const dst_host_only_access = current_image_access.stages == PipelineStageFlagBits::HOST;
                            DAXA_DBG_ASSERT_TRUE_M(!(src_host_only_access && dst_host_only_access), "direct sync between two host accesses on gpu is not allowed");
                            bool const is_host_barrier = src_host_only_access || dst_host_only_access;
                            // When the distance between src and dst batch is one, we can replace the split barrier with a normal barrier.
                            // We also need to make sure we do not use split barriers when the src or dst stage exclusively uses the host stage.
                            // This is because the host stage does not declare an execution dependency on the cpu but only a memory dependency.
                            bool const use_pipeline_barrier =
                                (tracked_slice.latest_access_batch_index + 1 == batch_index &&
                                 current_submit_scope_index == tracked_slice.latest_access_submit_scope_index) ||
                                is_host_barrier;
                            if (use_pipeline_barrier)
                            {
                                usize const barrier_index = this->barriers.size();
                                this->barriers.push_back(TaskBarrier{
                                    .image_id = used_image_t_id,
                                    .slice = intersection,
                                    .layout_before = tracked_slice.state.latest_layout,
                                    .layout_after = current_image_layout,
                                    .src_access = tracked_slice.state.latest_access,
                                    .dst_access = current_image_access,
                                });
                                // And we insert the barrier index into the list of pipeline barriers of the current tasks batch.
                                batch.pipeline_barrier_indices.push_back(barrier_index);
                                if (relation.is_current_concurrent)
                                {
                                    // As the new access is a read we remember our barrier index,
                                    // So that potential future reads after this can reuse this barrier.
                                    ret_new_use_tracked_slice.latest_concurrent_access_barrer_index = LastConcurrentAccessBarrierIndex{barrier_index};
                                }
                            }
                            else
                            {
                                usize const split_barrier_index = this->split_barriers.size();
                                this->split_barriers.push_back(TaskSplitBarrier{
                                    {
                                        .image_id = used_image_t_id,
                                        .slice = intersection,
                                        .layout_before = tracked_slice.state.latest_layout,
                                        .layout_after = current_image_layout,
                                        .src_access = tracked_slice.state.latest_access,
                                        .dst_access = current_image_access,
                                    },
                                    /* .split_barrier_state = */ task_graph_impl.info.device.create_event({
                                        .name = std::string("tg \"") + task_graph_impl.info.name + "\" sbi " + std::to_string(split_barrier_index),
                                    }),
                                });
                                // Now we give the src batch the index of this barrier to signal.
                                TaskBatchSubmitScope & src_scope = this->batch_submit_scopes[tracked_slice.latest_access_submit_scope_index];
                                TaskBatch & src_batch = src_scope.task_batches[tracked_slice.latest_access_batch_index];
                                src_batch.signal_split_barrier_indices.push_back(split_barrier_index);
                                // And we also insert the split barrier index into the waits of the current tasks batch.
                                batch.wait_split_barrier_indices.push_back(split_barrier_index);
                                if (relation.is_current_concurrent)
                                {
                                    // Need to remember the first concurrent access.
                                    // In case of multiple concurrent accesses following on each other, the first barrier in the sequence can be reused for all accesses.
                                    ret_new_use_tracked_slice.latest_concurrent_access_barrer_index = LastConcurrentAccessSplitBarrierIndex{split_barrier_index};
                                }
                            }
                        }
                        // Make sure we have any tracked slices to intersect with left.
                        if (tracked_slice_iter == task_image.last_slice_states.end())
                        {
                            break;
                        }
                    }
                    if (!advanced_tracked_slice_iterator)
                    {
                        // If we didn't find any intersections, we dont remove the tracked slice.
                        // Erasing a tracked slice "advances" iterator. As we did not remove,
                        // we need to advance it manually.
                        ++tracked_slice_iter;
                    }
                }
                tl_new_use_slices.clear();
                // Now we need to add the latest use and tracked range of our current access:
                task_image.last_slice_states.push_back(ret_new_use_tracked_slice);
                // The remainder tracked slices we remembered from earlier are now inserted back into the list of tracked slices.
                // We deferred this step as we dont want to check these in the loop above, as we found them to not intersect with the new use.
                task_image.last_slice_states.insert(
                    task_image.last_slice_states.end(),
                    tl_tracked_slice_rests.begin(),
                    tl_tracked_slice_rests.end());
                tl_tracked_slice_rests.clear();
            });
    }

    void TaskGraph::submit(TaskSubmitInfo const & info)
    {
        auto & impl = *r_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "completed task graphs can not record new tasks");

        for (auto & permutation : impl.record_active_permutations)
        {
            permutation->submit(info);
        }
    }

    void TaskGraphPermutation::submit(TaskSubmitInfo const & info)
    {
        TaskBatchSubmitScope & submit_scope = this->batch_submit_scopes.back();
        submit_scope.submit_info = {};
        // We provide the user submit info to the submit batch.
        submit_scope.user_submit_info = info;
        // Start a new batch.
        this->batch_submit_scopes.emplace_back();
    }

    void TaskGraph::present(TaskPresentInfo const & info)
    {
        auto & impl = *r_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "completed task graphs can not record new tasks");
        DAXA_DBG_ASSERT_TRUE_M(impl.info.swapchain.has_value(), "can only present, when a swapchain was provided in creation");

        for (auto & permutation : impl.record_active_permutations)
        {
            permutation->present(info);
        }
    }

    void TaskGraphPermutation::present(TaskPresentInfo const & info)
    {
        DAXA_DBG_ASSERT_TRUE_M(this->batch_submit_scopes.size() > 1, "can only present if at least one submit was issued before");
        DAXA_DBG_ASSERT_TRUE_M(!this->swapchain_image.is_empty(), "can only present when an image was annotated as swapchain image");
        DAXA_DBG_ASSERT_TRUE_M(!this->image_infos[this->swapchain_image.index].swapchain_semaphore_waited_upon, "Can only present once");
        this->image_infos[this->swapchain_image.index].swapchain_semaphore_waited_upon = true;

        ExtendedImageSliceState const default_slice;
        ExtendedImageSliceState const * tracked_slice = {};
        if (this->image_infos[this->swapchain_image.index].last_slice_states.empty())
        {
            tracked_slice = &default_slice;
        }
        else
        {
            tracked_slice = &this->image_infos[this->swapchain_image.index].last_slice_states.back();
        }
        usize const submit_scope_index = tracked_slice->latest_access_submit_scope_index;
        DAXA_DBG_ASSERT_TRUE_M(submit_scope_index < this->batch_submit_scopes.size() - 1, "the last swapchain image use MUST be before the last submit when presenting");
        TaskBatchSubmitScope & submit_scope = this->batch_submit_scopes[submit_scope_index];
        // We need to insert a pipeline barrier to transition the swapchain image layout to present src optimal.
        usize const barrier_index = this->barriers.size();
        this->barriers.push_back(TaskBarrier{
            .image_id = this->swapchain_image,
            .slice = tracked_slice->state.slice,
            .layout_before = tracked_slice->state.latest_layout,
            .layout_after = ImageLayout::PRESENT_SRC,
            .src_access = tracked_slice->state.latest_access,
            .dst_access = {.stages = PipelineStageFlagBits::BOTTOM_OF_PIPE},
        });
        submit_scope.last_minute_barrier_indices.push_back(barrier_index);
        // Now we need to insert the binary semaphore between submit and present.
        submit_scope.present_info = ImplPresentInfo{
            .additional_binary_semaphores = info.additional_binary_semaphores,
        };
    }

    void ImplTaskGraph::create_transient_runtime_buffers(TaskGraphPermutation & permutation)
    {
        for (u32 buffer_info_idx = 0; buffer_info_idx < u32(global_buffer_infos.size()); buffer_info_idx++)
        {
            auto const & glob_buffer = global_buffer_infos.at(buffer_info_idx);
            auto & perm_buffer = permutation.buffer_infos.at(buffer_info_idx);

            if (!glob_buffer.is_persistent() && perm_buffer.valid)
            {
                auto const & transient_info = daxa::get<PermIndepTaskBufferInfo::Transient>(glob_buffer.task_buffer_data);

                std::get<BufferId>(perm_buffer.actual_id) = info.device.create_buffer_from_memory_block(MemoryBlockBufferInfo{
                    .buffer_info = BufferInfo{
                        .size = transient_info.info.size,
                        .name = transient_info.info.name,
                    },
                    .memory_block = transient_data_memory_block,
                    .offset = perm_buffer.allocation_offset,
                });
            }
        }
    }

    void ImplTaskGraph::create_transient_runtime_images(TaskGraphPermutation & permutation)
    {
        for (u32 image_info_idx = 0; image_info_idx < u32(global_image_infos.size()); image_info_idx++)
        {
            auto const & glob_image = global_image_infos.at(image_info_idx);
            auto & perm_image = permutation.image_infos.at(image_info_idx);

            if (!glob_image.is_persistent() && perm_image.valid)
            {
                DAXA_DBG_ASSERT_TRUE_M(perm_image.usage != ImageUsageFlagBits::NONE,
                                       std::string("Transient image is not used in this permutation but marked as valid either: ") +
                                           std::string("\t- it was used as PRESENT which is not allowed for transient images") +
                                           std::string("\t- it was used as NONE which makes no sense - just don't mark it as used in the task"));
                auto const & transient_image_info = daxa::get<PermIndepTaskImageInfo::Transient>(glob_image.task_image_data).info;
                perm_image.actual_image = info.device.create_image_from_memory_block(
                    MemoryBlockImageInfo{
                        .image_info = ImageInfo{
                            .flags = daxa::ImageCreateFlagBits::ALLOW_ALIAS | perm_image.create_flags,
                            .dimensions = transient_image_info.dimensions,
                            .format = transient_image_info.format,
                            .size = transient_image_info.size,
                            .mip_level_count = transient_image_info.mip_level_count,
                            .array_layer_count = transient_image_info.array_layer_count,
                            .sample_count = transient_image_info.sample_count,
                            .usage = perm_image.usage,
                            .name = transient_image_info.name,
                        },
                        .memory_block = transient_data_memory_block,
                        .offset = perm_image.allocation_offset,
                    });
            }
        }
    }

    void ImplTaskGraph::allocate_transient_resources()
    {
        usize transient_resource_count = 0;
        usize max_alignment_requirement = 0;
        for (auto & permutation : permutations)
        {
            for (u32 image_i = 0; image_i < permutation.image_infos.size(); ++image_i)
            {
                PerPermTaskImage & permut_image = permutation.image_infos[image_i];
                PermIndepTaskImageInfo & global_image = global_image_infos[image_i];
                if (!global_image.is_persistent())
                {
                    transient_resource_count += 1;
                    TaskTransientImageInfo trans_img_info = daxa::get<PermIndepTaskImageInfo::Transient>(global_image.task_image_data).info;
                    ImageInfo image_info{
                        // .flags = trans_img_info.flags,
                        .dimensions = trans_img_info.dimensions,
                        .format = trans_img_info.format,
                        .size = trans_img_info.size,
                        .mip_level_count = trans_img_info.mip_level_count,
                        .array_layer_count = trans_img_info.array_layer_count,
                        .sample_count = trans_img_info.sample_count,
                        .usage = permut_image.usage,
                        .allocate_info = MemoryFlagBits::DEDICATED_MEMORY,
                        .name = "Dummy to figure mem requirements",
                    };
                    permut_image.memory_requirements = info.device.memory_requirements({image_info});
                    max_alignment_requirement = std::max(permut_image.memory_requirements.alignment, max_alignment_requirement);
                }
            }
            for (u32 buffer_i = 0; buffer_i < permutation.buffer_infos.size(); ++buffer_i)
            {
                PerPermTaskBuffer & permut_buffer = permutation.buffer_infos[buffer_i];
                PermIndepTaskBufferInfo & global_buffer = global_buffer_infos[buffer_i];
                if (!global_buffer.is_persistent())
                {
                    transient_resource_count += 1;
                    TaskTransientBufferInfo trans_buf_info = daxa::get<PermIndepTaskBufferInfo::Transient>(global_buffer.task_buffer_data).info;
                    BufferInfo buffer_info{
                        .size = trans_buf_info.size,
                        .allocate_info = MemoryFlagBits::DEDICATED_MEMORY,
                        .name = "Dummy to figure mem requirements",
                    };
                    permut_buffer.memory_requirements = info.device.memory_requirements({buffer_info});
                    max_alignment_requirement = std::max(permut_buffer.memory_requirements.alignment, max_alignment_requirement);
                }
            }
        }
        if (transient_resource_count == 0)
        {
            return;
        }

        // for each permutation figure out the max memory requirements
        for (auto & permutation : permutations)
        {
            usize batches = 0;
            std::vector<usize> submit_batch_offsets(permutation.batch_submit_scopes.size());
            for (u32 submit_scope_idx = 0; submit_scope_idx < permutation.batch_submit_scopes.size(); submit_scope_idx++)
            {
                submit_batch_offsets.at(submit_scope_idx) = batches;
                batches += permutation.batch_submit_scopes.at(submit_scope_idx).task_batches.size();
            }

            struct LifetimeLengthResource
            {
                usize start_batch;
                usize end_batch;
                usize lifetime_length;
                bool is_image;
                u32 resource_idx;
            };

            std::vector<LifetimeLengthResource> lifetime_length_sorted_resources;

            for (u32 perm_image_idx = 0; perm_image_idx < permutation.image_infos.size(); perm_image_idx++)
            {
                if (global_image_infos.at(perm_image_idx).is_persistent() || !permutation.image_infos.at(perm_image_idx).valid)
                {
                    continue;
                }

                auto const & perm_task_image = permutation.image_infos.at(perm_image_idx);

                if (perm_task_image.lifetime.first_use.submit_scope_index == std::numeric_limits<u32>::max() ||
                    perm_task_image.lifetime.last_use.submit_scope_index == std::numeric_limits<u32>::max())
                {
                    // TODO(msakmary) Transient image created but not used - should we somehow warn the user about this?
                    permutation.image_infos.at(perm_image_idx).valid = false;
                    continue;
                }

                usize const start_idx = submit_batch_offsets.at(perm_task_image.lifetime.first_use.submit_scope_index) +
                                        perm_task_image.lifetime.first_use.task_batch_index;
                usize const end_idx = submit_batch_offsets.at(perm_task_image.lifetime.last_use.submit_scope_index) +
                                      perm_task_image.lifetime.last_use.task_batch_index;

                lifetime_length_sorted_resources.emplace_back(LifetimeLengthResource{
                    .start_batch = start_idx,
                    .end_batch = end_idx,
                    .lifetime_length = end_idx - start_idx + 1,
                    .is_image = true,
                    .resource_idx = perm_image_idx,
                });
            }

            for (u32 perm_buffer_idx = 0; perm_buffer_idx < permutation.buffer_infos.size(); perm_buffer_idx++)
            {
                if (global_buffer_infos.at(perm_buffer_idx).is_persistent())
                {
                    continue;
                }

                auto const & perm_task_buffer = permutation.buffer_infos.at(perm_buffer_idx);

                if (perm_task_buffer.lifetime.first_use.submit_scope_index == std::numeric_limits<u32>::max() ||
                    perm_task_buffer.lifetime.last_use.submit_scope_index == std::numeric_limits<u32>::max())
                {
                    // TODO(msakmary) Transient buffer created but not used - should we somehow warn the user about this?
                    permutation.buffer_infos.at(perm_buffer_idx).valid = false;
                    continue;
                }

                usize const start_idx = submit_batch_offsets.at(perm_task_buffer.lifetime.first_use.submit_scope_index) +
                                        perm_task_buffer.lifetime.first_use.task_batch_index;
                usize const end_idx = submit_batch_offsets.at(perm_task_buffer.lifetime.last_use.submit_scope_index) +
                                      perm_task_buffer.lifetime.last_use.task_batch_index;

                lifetime_length_sorted_resources.emplace_back(LifetimeLengthResource{
                    .start_batch = start_idx,
                    .end_batch = end_idx,
                    .lifetime_length = end_idx - start_idx + 1,
                    .is_image = false,
                    .resource_idx = perm_buffer_idx,
                });
            }

            std::sort(lifetime_length_sorted_resources.begin(), lifetime_length_sorted_resources.end(),
                      [](LifetimeLengthResource const & first, LifetimeLengthResource const & second) -> bool
                      {
                          return first.lifetime_length > second.lifetime_length;
                      });

            struct Allocation
            {
                usize offset = {};
                usize size = {};
                usize start_batch = {};
                usize end_batch = {};
                bool is_image = {};
                u32 owning_resource_idx = {};
                u32 memory_type_bits = {};
                ImageMipArraySlice intersection_object = {};
            };
            // Sort allocations in the set in the following way
            //      1) sort by offsets into the memory block
            //  if equal:
            //      2) sort by start batch of the allocation
            //  if equal:
            //      3) sort by owning image index
            struct AllocCompare
            {
                constexpr auto operator()(Allocation const & first, Allocation const & second) const -> bool
                {
                    if (first.offset < second.offset)
                    {
                        return true;
                    }
                    if (first.offset == second.offset)
                    {
                        if (first.start_batch < second.start_batch)
                        {
                            return true;
                        }
                        if (first.start_batch == second.start_batch)
                        {
                            return first.owning_resource_idx < second.owning_resource_idx;
                        }
                        // first.offset == second.offset && first.start_batch > second.start_batch
                        return false;
                    }
                    // first.offset > second.offset
                    return false;
                };
            };
            std::set<Allocation, AllocCompare> allocations = {};
            // Figure out where to allocate each resource
            usize no_alias_back_offset = {};
            for (auto const & resource_lifetime : lifetime_length_sorted_resources)
            {
                MemoryRequirements mem_requirements;
                if (resource_lifetime.is_image)
                {
                    mem_requirements = permutation.image_infos.at(resource_lifetime.resource_idx).memory_requirements;
                }
                else
                {
                    mem_requirements = permutation.buffer_infos.at(resource_lifetime.resource_idx).memory_requirements;
                }
                // Go through all memory block states in which this resource is alive and try to find a spot for it
                u8 const resource_lifetime_duration = static_cast<u8>(resource_lifetime.end_batch - resource_lifetime.start_batch + 1);
                auto new_allocation = Allocation{
                    .offset = 0,
                    .size = mem_requirements.size,
                    .start_batch = resource_lifetime.start_batch,
                    .end_batch = resource_lifetime.end_batch,
                    .is_image = resource_lifetime.is_image,
                    .owning_resource_idx = resource_lifetime.resource_idx,
                    .memory_type_bits = mem_requirements.memory_type_bits,
                    .intersection_object = {
                        .base_mip_level = static_cast<u8>(resource_lifetime.start_batch),
                        .level_count = resource_lifetime_duration,
                        .base_array_layer = static_cast<u16>(0),
                        .layer_count = static_cast<u32>(mem_requirements.size),
                    }};
                usize const align = std::max(mem_requirements.alignment, static_cast<size_t>(1ull));

                if (info.alias_transients)
                {
                    // TODO(msakmary) Fix the intersect functionality so that it is general and does not do hacky stuff like constructing
                    // a mip array slice
                    // Find space in memory and time the new allocation fits into.
                    for (auto const & allocation : allocations)
                    {
                        if (new_allocation.intersection_object.intersects(allocation.intersection_object))
                        {
                            // assign new offset into the memory block - we need to guarantee correct alignment
                            usize const curr_offset = allocation.offset + allocation.size;
                            usize const aligned_curr_offset = (curr_offset + align - 1) / align * align;
                            new_allocation.offset = aligned_curr_offset;
                            new_allocation.intersection_object.base_array_layer = static_cast<u32>(new_allocation.offset);
                        }
                    }
                }
                else
                {
                    usize const aligned_curr_offset = (no_alias_back_offset + align - 1) / align * align;
                    new_allocation.offset = aligned_curr_offset;
                    no_alias_back_offset = new_allocation.offset + new_allocation.size;
                }
                allocations.insert(new_allocation);
            }
            // Once we are done with finding space for all the allocations go through all permutation images and copy over the allocation information
            for (auto const & allocation : allocations)
            {
                if (allocation.is_image)
                {
                    permutation.image_infos.at(allocation.owning_resource_idx).allocation_offset = allocation.offset;
                }
                else
                {
                    permutation.buffer_infos.at(allocation.owning_resource_idx).allocation_offset = allocation.offset;
                }
                // find the amount of memory this permutation requires
                memory_block_size = std::max(memory_block_size, allocation.offset + allocation.size);
                memory_type_bits = memory_type_bits & allocation.memory_type_bits;
            }
        }

        transient_data_memory_block = info.device.create_memory({
            .requirements = {
                .size = memory_block_size,
                .alignment = max_alignment_requirement,
                .memory_type_bits = memory_type_bits,
            },
            .flags = MemoryFlagBits::DEDICATED_MEMORY,
        });
    }

    void TaskGraph::complete(TaskCompleteInfo const & /*unused*/)
    {
        auto & impl = *r_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "task graphs can only be completed once");
        impl.compiled = true;

        impl.allocate_transient_resources();
        // Insert static barriers initializing image layouts.
        for (auto & permutation : impl.permutations)
        {
            impl.create_transient_runtime_buffers(permutation);
            impl.create_transient_runtime_images(permutation);

            // Insert static initialization barriers for non persistent resources:
            // Buffers never need layout initialization, only images.
            for (u32 task_image_index = 0; task_image_index < permutation.image_infos.size(); ++task_image_index)
            {
                TaskImageView const task_image_id = {{impl.unique_index, task_image_index}};
                auto & task_image = permutation.image_infos[task_image_index];
                PermIndepTaskImageInfo const & glob_task_image = impl.global_image_infos[task_image_index];
                if (task_image.valid && !glob_task_image.is_persistent())
                {
                    // Insert barriers, initializing all the initially accesses subresource ranges to the correct layout.
                    for (auto & first_access : task_image.first_slice_states)
                    {
                        usize const new_barrier_index = permutation.barriers.size();
                        permutation.barriers.push_back(TaskBarrier{
                            .image_id = task_image_id,
                            .slice = first_access.state.slice,
                            .layout_before = {},
                            .layout_after = first_access.state.latest_layout,
                            .src_access = {},
                            .dst_access = first_access.state.latest_access,
                        });
                        // Because resources may be aliased we need to insert the barrier into the batch in which the resource is first used
                        // If we just inserted all transitions into the first batch an error as follows might occur:
                        //      Image A lives in batch 1, Image B lives in batch 2
                        //      Image A and B are aliased (share the same/part-of memory)
                        //      Image A is transitioned from UNDEFINED -> TRANSFER_DST in batch 0 BUT
                        //      Image B is also transitioned from UNDEFINED -> TRANSFER_SRT in batch 0
                        // This is an erroneous state - task graph assumes they are separate images and thus,
                        // for example uses Image A thinking it's in TRANSFER_DST which it is not
                        if (impl.info.alias_transients)
                        {
                            // TODO(msakmary) This is only needed when we actually alias two images - should be possible to detect this
                            // and only defer the initialization barrier for these aliased ones instead of all of them
                            auto const submit_scope_index = first_access.latest_access_submit_scope_index;
                            auto const batch_index = first_access.latest_access_batch_index;
                            auto & first_used_batch = permutation.batch_submit_scopes[submit_scope_index].task_batches[batch_index];
                            first_used_batch.pipeline_barrier_indices.push_back(new_barrier_index);
                        }
                        else
                        {
                            auto & first_used_batch = permutation.batch_submit_scopes[0].task_batches[0];
                            first_used_batch.pipeline_barrier_indices.push_back(new_barrier_index);
                        }
                    }
                }
            }
        }
    }

    // auto TaskGraph::get_command_lists() -> std::vector<CommandRecorder>
    // {
    //     auto & impl = *r_cast<ImplTaskGraph *>(this->object);
    //     DAXA_DBG_ASSERT_TRUE_M(impl.compiled, "Can only get command lists of a finished task graph");
    //     DAXA_DBG_ASSERT_TRUE_M(!impl.executed_once, "Can only get command lists of a task graph that has been executed");
    //     auto command_lists = std::move(impl.left_over_command_lists);
    //     impl.left_over_command_lists = {};
    //     return command_lists;
    // }

    auto TaskGraph::get_debug_string() -> std::string
    {
        auto & impl = *r_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(impl.info.record_debug_information,
                               "in order to have debug string you need to set record_debug_information flag to true on task graph creation");
        DAXA_DBG_ASSERT_TRUE_M(impl.executed_once,
                               "in order to have debug string you need to execute the task graph at least once");
        std::string ret = impl.debug_string_stream.str();
        impl.debug_string_stream.str("");
        return ret;
    }

    auto TaskGraph::get_transient_memory_size() -> daxa::usize
    {
        auto & impl = *r_cast<ImplTaskGraph *>(this->object);
        return impl.memory_block_size;
    }

    thread_local std::vector<EventWaitInfo> tl_split_barrier_wait_infos = {};
    thread_local std::vector<ImageMemoryBarrierInfo> tl_image_barrier_infos = {};
    thread_local std::vector<MemoryBarrierInfo> tl_memory_barrier_infos = {};
    void insert_pipeline_barrier(ImplTaskGraph const & impl, TaskGraphPermutation & perm, CommandRecorder & command_list, TaskBarrier & barrier)
    {
        // Check if barrier is image barrier or normal barrier (see TaskBarrier struct comments).
        if (barrier.image_id.is_empty())
        {
            command_list.pipeline_barrier({
                .src_access = barrier.src_access,
                .dst_access = barrier.dst_access,
            });
        }
        else
        {
            auto const actual_images = impl.get_actual_images(barrier.image_id, perm);
            for (usize index = 0; index < actual_images.size(); ++index)
            {
                auto const & image = actual_images[index];
                DAXA_DBG_ASSERT_TRUE_M(
                    impl.info.device.is_id_valid(image),
                    std::string("Detected invalid runtime image id while inserting barriers: the runtime image id at index ") +
                        std::to_string(index) +
                        std::string(" of task image \"") +
                        std::string(impl.global_image_infos[barrier.image_id.index].get_name()) +
                        std::string("\" is invalid"));
                command_list.pipeline_barrier_image_transition({
                    .src_access = barrier.src_access,
                    .dst_access = barrier.dst_access,
                    .src_layout = barrier.layout_before,
                    .dst_layout = barrier.layout_after,
                    .image_slice = barrier.slice,
                    .image_id = image,
                });
            }
        }
    }

    void generate_persistent_resource_synch(
        ImplTaskGraph & impl,
        TaskGraphPermutation & permutation,
        CommandRecorder & recorder)
    {
        // Persistent resources need just in time synch between executions,
        // as pre generating the transitions between all permutations is not manageable.
        std::string out;
        std::string indent;
        if (impl.info.record_debug_information)
        {
            fmt::format_to(std::back_inserter(out), "{}runtime sync memory barriers:\n", indent);
            begin_indent(out, indent, true);
        }
        for (usize task_buffer_index = 0; task_buffer_index < permutation.buffer_infos.size(); ++task_buffer_index)
        {
            auto & task_buffer = permutation.buffer_infos[task_buffer_index];
            auto & glob_buffer_info = impl.global_buffer_infos[task_buffer_index];
            if (task_buffer.valid && glob_buffer_info.is_persistent())
            {
                auto & persistent_data = glob_buffer_info.get_persistent();
                bool const no_prev_access = persistent_data.latest_access == AccessConsts::NONE;
                bool const read_on_read_same_access =
                    persistent_data.latest_access == permutation.buffer_infos[task_buffer_index].first_access &&
                    persistent_data.latest_access.type == AccessTypeFlagBits::READ;
                // TODO(pahrens,msakmary): read on read should only be set to true whenever the access the same and the STAGES are also the same
                // otherwise we need to generate barrier
                //      AS WE CANT MODIFY BARRIERS FROM PREVIOUSLY ALREADY EXECUTED TASK GRAPHS, WE MUST OVER SYNC ON THE LAST WRITE TO READ
                //      WE CAN ONLY SKIP ON THE SAME ACCESS READ ON READ
                //      WE MUST REMEMBER THE LAST WRITE

                // bool const read_on_read =
                //     persistent_data.latest_access.type == AccessTypeFlagBits::READ &&
                //     permutation.buffer_infos[task_buffer_index].first_access.type == AccessTypeFlagBits::READ;
                // For now just over sync on reads.
                if (no_prev_access || read_on_read_same_access)
                {
                    // Skip buffers that have no previous access, as there is nothing to sync on.
                    continue;
                }

                MemoryBarrierInfo const mem_barrier_info{
                    .src_access = persistent_data.latest_access,
                    .dst_access = permutation.buffer_infos[task_buffer_index].first_access,
                };
                recorder.pipeline_barrier(mem_barrier_info);
                if (impl.info.record_debug_information)
                {
                    fmt::format_to(std::back_inserter(out), "{}{}\n", indent, to_string(mem_barrier_info));
                    print_separator_to(out, indent);
                }
                persistent_data.latest_access = {};
            }
        }
        if (impl.info.record_debug_information)
        {
            end_indent(out, indent);
            fmt::format_to(std::back_inserter(out), "{}runtime sync image memory barriers:\n", indent);
            begin_indent(out, indent, true);
        }
        // If parts of the first use slices to not intersect with any previous use,
        // we must synchronize on undefined layout!
        std::vector<ExtendedImageSliceState> remaining_first_accesses = {};
        for (u32 task_image_index = 0; task_image_index < permutation.image_infos.size(); ++task_image_index)
        {
            auto & task_image = permutation.image_infos[task_image_index];
            auto & exec_image = impl.global_image_infos[task_image_index];
            remaining_first_accesses = task_image.first_slice_states;
            // Iterate over all persistent images.
            // Find all intersections between tracked slices of first use and previous use.
            // Synch on the intersection and delete the intersected part from the tracked slice of the previous use.
            if (task_image.valid && exec_image.is_persistent())
            {
                if (impl.info.record_debug_information)
                {
                    fmt::format_to(std::back_inserter(out), "{}sync from previous uses:\n", indent);
                    begin_indent(out, indent, true);
                }
                auto & previous_access_slices = exec_image.get_persistent().latest_slice_states;
                for (u32 previous_access_slice_index = 0; previous_access_slice_index < previous_access_slices.size();)
                {
                    bool broke_inner_loop = false;
                    for (u32 first_access_slice_index = 0; first_access_slice_index < remaining_first_accesses.size(); ++first_access_slice_index)
                    {
                        // Dont sync on disjoint subresource uses.
                        if (!remaining_first_accesses[first_access_slice_index].state.slice.intersects(previous_access_slices[previous_access_slice_index].slice))
                        {
                            // Disjoint subresources or read on read with same layout.
                            continue;
                        }
                        // Intersect previous use and initial use.
                        // Record synchronization for the intersecting part.
                        auto intersection = previous_access_slices[previous_access_slice_index].slice.intersect(remaining_first_accesses[first_access_slice_index].state.slice);
                        // Dont sync on same accesses following each other.
                        bool const both_accesses_read =
                            remaining_first_accesses[first_access_slice_index].state.latest_access.type == AccessTypeFlagBits::READ &&
                            previous_access_slices[previous_access_slice_index].latest_access.type == AccessTypeFlagBits::READ;
                        bool const both_layouts_same =
                            remaining_first_accesses[first_access_slice_index].state.latest_layout ==
                            previous_access_slices[previous_access_slice_index].latest_layout;
                        if (!(both_accesses_read && both_layouts_same))
                        {
                            for (auto execution_image_id : impl.get_actual_images(TaskImageView{{.task_graph_index = impl.unique_index, .index = task_image_index}}, permutation))
                            {
                                ImageMemoryBarrierInfo const img_barrier_info{
                                    .src_access = previous_access_slices[previous_access_slice_index].latest_access,
                                    .dst_access = remaining_first_accesses[first_access_slice_index].state.latest_access,
                                    .src_layout = previous_access_slices[previous_access_slice_index].latest_layout,
                                    .dst_layout = remaining_first_accesses[first_access_slice_index].state.latest_layout,
                                    .image_slice = intersection,
                                    .image_id = execution_image_id,
                                };
                                recorder.pipeline_barrier_image_transition(img_barrier_info);
                                if (impl.info.record_debug_information)
                                {
                                    fmt::format_to(std::back_inserter(out), "{}{}\n", indent, to_string(img_barrier_info));
                                    print_separator_to(out, indent);
                                }
                            }
                        }
                        // Put back the non intersecting rest into the previous use list.
                        auto [previous_use_slice_rest, previous_use_slice_rest_count] = previous_access_slices[previous_access_slice_index].slice.subtract(intersection);
                        auto [first_use_slice_rest, first_use_slice_rest_count] = remaining_first_accesses[first_access_slice_index].state.slice.subtract(intersection);
                        for (usize rest_slice_index = 0; rest_slice_index < previous_use_slice_rest_count; ++rest_slice_index)
                        {
                            auto rest_previous_slice = previous_access_slices[previous_access_slice_index];
                            rest_previous_slice.slice = previous_use_slice_rest[rest_slice_index];
                            previous_access_slices.push_back(rest_previous_slice);
                        }
                        // Append the new rest first uses.nd
                        for (usize rest_slice_index = 0; rest_slice_index < first_use_slice_rest_count; ++rest_slice_index)
                        {
                            auto rest_first_slice = remaining_first_accesses[first_access_slice_index];
                            rest_first_slice.state.slice = first_use_slice_rest[rest_slice_index];
                            remaining_first_accesses.push_back(rest_first_slice);
                        }
                        // Remove the previous use from the list, it is synchronized now.
                        previous_access_slices.erase(std::next(previous_access_slices.begin(), static_cast<ptrdiff_t>(previous_access_slice_index)));
                        // Remove the first use from the remaining first uses, as it was now synchronized from.
                        remaining_first_accesses.erase(std::next(remaining_first_accesses.begin(), static_cast<ptrdiff_t>(first_access_slice_index)));
                        // As we removed an element in this place,
                        // we dont need to advance the iterator as in its place there will be a new element already that we do not want to skip.
                        broke_inner_loop = true;
                        break;
                    }
                    if (!broke_inner_loop)
                    {
                        // We break the loop when we erase the current element.
                        // Erasing moved all elements past the current one one to the left.
                        // This means that the current element is already a new one,
                        // we do not want to skip it so we wont increment the index here.
                        ++previous_access_slice_index;
                    }
                }
                if (impl.info.record_debug_information)
                {
                    end_indent(out, indent);
                    fmt::format_to(std::back_inserter(out), "{}sync from undefined:\n", indent);
                    begin_indent(out, indent, true);
                }
                // For all first uses that did NOT intersect with and previous use,
                // we need to synchronize from an undefined state to initialize the layout of the image.
                for (auto & remaining_first_accesse : remaining_first_accesses)
                {
                    for (auto execution_image_id : impl.get_actual_images(TaskImageView{{.task_graph_index = impl.unique_index, .index = task_image_index}}, permutation))
                    {
                        ImageMemoryBarrierInfo const img_barrier_info{
                            .src_access = AccessConsts::NONE,
                            .dst_access = remaining_first_accesse.state.latest_access,
                            .src_layout = ImageLayout::UNDEFINED,
                            .dst_layout = remaining_first_accesse.state.latest_layout,
                            .image_slice = remaining_first_accesse.state.slice,
                            .image_id = execution_image_id,
                        };
                        recorder.pipeline_barrier_image_transition(img_barrier_info);
                        if (impl.info.record_debug_information)
                        {
                            fmt::format_to(std::back_inserter(out), "{}{}\n", indent, to_string(img_barrier_info));
                            print_separator_to(out, indent);
                        }
                    }
                }
                if (impl.info.record_debug_information)
                {
                    end_indent(out, indent);
                }
            }
        }
        if (impl.info.record_debug_information)
        {
            end_indent(out, indent);
            impl.debug_string_stream << out;
        }
    }

    /// Execution flow:
    /// 1. choose permutation based on conditionals
    /// 2. validate used persistent resources, based on permutation
    /// 3. runtime generate and insert runtime sync for persistent resources.
    /// 4. for every submit scope:
    ///     2.1 for every batch in scope:
    ///         3.1 wait for pipeline and split barriers
    ///         3.2 for every task:
    ///             4.1 validate runtime resources of used resources
    ///             4.2 refresh image view cache.
    ///             4.3 collect shader use handles, allocate gpu local staging memory, copy in handles and bind to constant buffer binding.
    ///             4.4 run task
    ///         3.3 signal split barriers
    ///     2.2 check if submit scope submits work, either submit or collect cmd lists and sync primitives for query
    ///     2.3 check if submit scope presents, present if true.
    void TaskGraph::execute(ExecutionInfo const & info)
    {
        auto & impl = *r_cast<ImplTaskGraph *>(this->object);
        DAXA_DBG_ASSERT_TRUE_M(info.permutation_condition_values.size() >= impl.info.permutation_condition_count, "Detected invalid permutation condition count");
        DAXA_DBG_ASSERT_TRUE_M(impl.compiled, "task graphs must be completed before execution");

        u32 permutation_index = {};
        for (u32 index = 0; index < std::min(usize(32), info.permutation_condition_values.size()); ++index)
        {
            permutation_index |= info.permutation_condition_values[index] ? (1u << index) : 0;
        }
        impl.chosen_permutation_last_execution = permutation_index;
        TaskGraphPermutation & permutation = impl.permutations[permutation_index];

        CommandRecorder recorder = impl.info.device.create_command_recorder({});

        ImplTaskRuntimeInterface impl_runtime{.task_graph = impl, .permutation = permutation, .recorder = recorder};

        validate_runtime_resources(impl, permutation);
        // Generate and insert synchronization for persistent resources:
        generate_persistent_resource_synch(impl, permutation, recorder);

        usize submit_scope_index = 0;
        for (auto & submit_scope : permutation.batch_submit_scopes)
        {
            if (impl.info.enable_command_labels)
            {
                impl_runtime.recorder.begin_label({
                    .label_color = impl.info.task_graph_label_color,
                    .name = impl.info.name + std::string(", submit ") + std::to_string(submit_scope_index),
                });
            }
            usize batch_index = 0;
            for (auto & task_batch : submit_scope.task_batches)
            {
                batch_index += 1;
                // Wait on pipeline barriers before batch execution.
                for (auto barrier_index : task_batch.pipeline_barrier_indices)
                {
                    TaskBarrier & barrier = permutation.barriers[barrier_index];
                    insert_pipeline_barrier(impl, permutation, impl_runtime.recorder, barrier);
                }
                // Wait on split barriers before batch execution.
                if (!impl.info.use_split_barriers)
                {
                    for (auto barrier_index : task_batch.wait_split_barrier_indices)
                    {
                        TaskSplitBarrier const & split_barrier = permutation.split_barriers[barrier_index];
                        // Convert split barrier to normal barrier.
                        TaskBarrier barrier = split_barrier;
                        insert_pipeline_barrier(impl, permutation, impl_runtime.recorder, barrier);
                    }
                }
                else
                {
                    usize needed_image_barriers = 0;
                    for (auto barrier_index : task_batch.wait_split_barrier_indices)
                    {
                        TaskSplitBarrier const & split_barrier = permutation.split_barriers[barrier_index];
                        if (!split_barrier.image_id.is_empty())
                        {
                            needed_image_barriers += impl.get_actual_images(split_barrier.image_id, permutation).size();
                        }
                    }
                    tl_split_barrier_wait_infos.reserve(task_batch.wait_split_barrier_indices.size());
                    tl_memory_barrier_infos.reserve(task_batch.wait_split_barrier_indices.size());
                    tl_image_barrier_infos.reserve(needed_image_barriers);
                    for (auto barrier_index : task_batch.wait_split_barrier_indices)
                    {
                        TaskSplitBarrier & split_barrier = permutation.split_barriers[barrier_index];
                        if (split_barrier.image_id.is_empty())
                        {
                            tl_memory_barrier_infos.push_back(MemoryBarrierInfo{
                                .src_access = split_barrier.src_access,
                                .dst_access = split_barrier.dst_access,
                            });
                            tl_split_barrier_wait_infos.push_back(EventWaitInfo{
                                .memory_barriers = daxa::Span<daxa::MemoryBarrierInfo const>{&tl_memory_barrier_infos.back(), 1},
                                .event = split_barrier.split_barrier_state,
                            });
                        }
                        else
                        {
                            usize const img_bar_vec_start_size = tl_image_barrier_infos.size();
                            for (auto image : impl.get_actual_images(split_barrier.image_id, permutation))
                            {
                                tl_image_barrier_infos.push_back(ImageMemoryBarrierInfo{
                                    .src_access = split_barrier.src_access,
                                    .dst_access = split_barrier.dst_access,
                                    .src_layout = split_barrier.layout_before,
                                    .dst_layout = split_barrier.layout_after,
                                    .image_slice = split_barrier.slice,
                                    .image_id = image,
                                });
                            }
                            usize const img_bar_vec_end_size = tl_image_barrier_infos.size();
                            usize const img_bar_count = img_bar_vec_end_size - img_bar_vec_start_size;
                            tl_split_barrier_wait_infos.push_back(EventWaitInfo{
                                .image_barriers = daxa::Span<daxa::ImageMemoryBarrierInfo const>{tl_image_barrier_infos.data() + img_bar_vec_start_size, img_bar_count},
                                .event = split_barrier.split_barrier_state,
                            });
                        }
                    }
                    if (!tl_split_barrier_wait_infos.empty())
                    {
                        impl_runtime.recorder.wait_events(tl_split_barrier_wait_infos);
                    }
                    tl_split_barrier_wait_infos.clear();
                    tl_image_barrier_infos.clear();
                    tl_memory_barrier_infos.clear();
                }
                // Execute all tasks in the batch.
                usize task_index = 0;
                for (TaskId const task_id : task_batch.tasks)
                {
                    impl.execute_task(impl_runtime, permutation, batch_index, task_index, task_id);
                    task_index += 1;
                }
                if (impl.info.use_split_barriers)
                {
                    // Reset all waited upon split barriers here.
                    for (auto barrier_index : task_batch.wait_split_barrier_indices)
                    {
                        // We wait on the stages, that waited on our split barrier earlier.
                        // This way, we make sure, that the stages that wait on the split barrier
                        // executed and saw the split barrier signaled, before we reset them.
                        impl_runtime.recorder.reset_event({
                            .event = permutation.split_barriers[barrier_index].split_barrier_state,
                            .stage = permutation.split_barriers[barrier_index].dst_access.stages,
                        });
                    }
                    // Signal all signal split barriers after batch execution.
                    for (usize const barrier_index : task_batch.signal_split_barrier_indices)
                    {
                        TaskSplitBarrier & task_split_barrier = permutation.split_barriers[barrier_index];
                        if (task_split_barrier.image_id.is_empty())
                        {
                            MemoryBarrierInfo memory_barrier{
                                .src_access = task_split_barrier.src_access,
                                .dst_access = task_split_barrier.dst_access,
                            };
                            impl_runtime.recorder.signal_event({
                                .memory_barriers = daxa::Span<daxa::MemoryBarrierInfo const>{&memory_barrier, 1},
                                .event = task_split_barrier.split_barrier_state,
                            });
                        }
                        else
                        {
                            for (auto image : impl.get_actual_images(task_split_barrier.image_id, permutation))
                            {
                                tl_image_barrier_infos.push_back({
                                    .src_access = task_split_barrier.src_access,
                                    .dst_access = task_split_barrier.dst_access,
                                    .src_layout = task_split_barrier.layout_before,
                                    .dst_layout = task_split_barrier.layout_after,
                                    .image_slice = task_split_barrier.slice,
                                    .image_id = image,
                                });
                            }
                            impl_runtime.recorder.signal_event({
                                .image_barriers = tl_image_barrier_infos,
                                .event = task_split_barrier.split_barrier_state,
                            });
                            tl_image_barrier_infos.clear();
                        }
                    }
                }
            }
            for (usize const barrier_index : submit_scope.last_minute_barrier_indices)
            {
                TaskBarrier & barrier = permutation.barriers[barrier_index];
                insert_pipeline_barrier(impl, permutation, impl_runtime.recorder, barrier);
            }
            if (impl.info.enable_command_labels)
            {
                impl_runtime.recorder.end_label();
            }

            if (&submit_scope != &permutation.batch_submit_scopes.back())
            {
                PipelineStageFlags const wait_stages = submit_scope.submit_info.wait_stages;
                std::vector<ExecutableCommandList> commands = {submit_scope.submit_info.command_lists.begin(), submit_scope.submit_info.command_lists.end()};
                std::vector<BinarySemaphore> wait_binary_semaphores = {submit_scope.submit_info.wait_binary_semaphores.begin(), submit_scope.submit_info.wait_binary_semaphores.end()};
                std::vector<BinarySemaphore> signal_binary_semaphores = {submit_scope.submit_info.signal_binary_semaphores.begin(), submit_scope.submit_info.signal_binary_semaphores.end()};
                std::vector<std::pair<TimelineSemaphore, u64>> wait_timeline_semaphores = {submit_scope.submit_info.wait_timeline_semaphores.begin(), submit_scope.submit_info.wait_timeline_semaphores.end()};
                std::vector<std::pair<TimelineSemaphore, u64>> signal_timeline_semaphores = {submit_scope.submit_info.signal_timeline_semaphores.begin(), submit_scope.submit_info.signal_timeline_semaphores.end()};
                commands.push_back(recorder.complete_current_commands());
                if (impl.info.swapchain.has_value())
                {
                    Swapchain const & swapchain = impl.info.swapchain.value();
                    if (submit_scope_index == permutation.swapchain_image_first_use_submit_scope_index)
                    {
                        ImplPersistentTaskImage & swapchain_image = impl.global_image_infos.at(permutation.swapchain_image.index).get_persistent();
                        // It can happen, that a previous task graph accessed the swapchain image.
                        // In that case the acquire semaphore is already waited upon and by extension we wait on the previous access and therefore on the acquire.
                        // So we must not wait in the case that the semaphore is already waited upon.
                        if (!swapchain_image.waited_on_acquire)
                        {
                            swapchain_image.waited_on_acquire = true;
                            wait_binary_semaphores.push_back(swapchain.current_acquire_semaphore());
                        }
                    }
                    if (submit_scope_index == permutation.swapchain_image_last_use_submit_scope_index)
                    {
                        signal_binary_semaphores.push_back(swapchain.current_present_semaphore());
                        signal_timeline_semaphores.emplace_back(
                            swapchain.gpu_timeline_semaphore(),
                            swapchain.current_cpu_timeline_value());
                    }
                    if (permutation.swapchain_image_first_use_submit_scope_index == std::numeric_limits<u64>::max() &&
                        submit_scope.present_info.has_value())
                    {
                        // It can be the case, that the only use of the swapchain is the present itself.
                        // If so, simply signal the timeline sema of the swapchain with the latest submit.
                        // TODO: this is a hack until we get timeline semaphores for presenting.
                        // TODO: im not sure about this design, maybe just remove this explicit signaling completely.
                        signal_timeline_semaphores.emplace_back(
                            swapchain.gpu_timeline_semaphore(),
                            swapchain.current_cpu_timeline_value());
                        ImplPersistentTaskImage & swapchain_image = impl.global_image_infos.at(permutation.swapchain_image.index).get_persistent();
                        swapchain_image.waited_on_acquire = true;
                        wait_binary_semaphores.push_back(impl.info.swapchain.value().current_acquire_semaphore());
                        signal_binary_semaphores.push_back(impl.info.swapchain.value().current_present_semaphore());
                    }
                }
                if (submit_scope.user_submit_info.additional_command_lists != nullptr)
                {
                    commands.insert(commands.end(), submit_scope.user_submit_info.additional_command_lists->begin(), submit_scope.user_submit_info.additional_command_lists->end());
                }
                if (submit_scope.user_submit_info.additional_wait_binary_semaphores != nullptr)
                {
                    wait_binary_semaphores.insert(wait_binary_semaphores.end(), submit_scope.user_submit_info.additional_wait_binary_semaphores->begin(), submit_scope.user_submit_info.additional_wait_binary_semaphores->end());
                }
                if (submit_scope.user_submit_info.additional_signal_binary_semaphores != nullptr)
                {
                    signal_binary_semaphores.insert(signal_binary_semaphores.end(), submit_scope.user_submit_info.additional_signal_binary_semaphores->begin(), submit_scope.user_submit_info.additional_signal_binary_semaphores->end());
                }
                if (submit_scope.user_submit_info.additional_wait_timeline_semaphores != nullptr)
                {
                    wait_timeline_semaphores.insert(wait_timeline_semaphores.end(), submit_scope.user_submit_info.additional_wait_timeline_semaphores->begin(), submit_scope.user_submit_info.additional_wait_timeline_semaphores->end());
                }
                if (submit_scope.user_submit_info.additional_signal_timeline_semaphores != nullptr)
                {
                    signal_timeline_semaphores.insert(signal_timeline_semaphores.end(), submit_scope.user_submit_info.additional_signal_timeline_semaphores->begin(), submit_scope.user_submit_info.additional_signal_timeline_semaphores->end());
                }
                signal_timeline_semaphores.emplace_back(impl.staging_memory->timeline_semaphore(), impl.staging_memory->inc_timeline_value());
                daxa::CommandSubmitInfo const submit_info = {
                    .wait_stages = wait_stages,
                    .command_lists = commands,
                    .wait_binary_semaphores = wait_binary_semaphores,
                    .signal_binary_semaphores = signal_binary_semaphores,
                    .wait_timeline_semaphores = wait_timeline_semaphores,
                    .signal_timeline_semaphores = signal_timeline_semaphores,
                };
                impl.info.device.submit_commands(submit_info);

                if (submit_scope.present_info.has_value())
                {
                    ImplPresentInfo & impl_present_info = submit_scope.present_info.value();
                    std::vector<BinarySemaphore> present_wait_semaphores = impl_present_info.binary_semaphores;
                    DAXA_DBG_ASSERT_TRUE_M(impl.info.swapchain.has_value(), "must have swapchain registered in info on creation in order to use present.");
                    present_wait_semaphores.push_back(impl.info.swapchain.value().current_present_semaphore());
                    if (impl_present_info.additional_binary_semaphores != nullptr)
                    {
                        present_wait_semaphores.insert(
                            present_wait_semaphores.end(),
                            impl_present_info.additional_binary_semaphores->begin(),
                            impl_present_info.additional_binary_semaphores->end());
                    }
                    impl.info.device.present_frame(PresentInfo{
                        .wait_binary_semaphores = present_wait_semaphores,
                        .swapchain = impl.info.swapchain.value(),
                    });
                }
            }
            ++submit_scope_index;
        }

        // Insert pervious uses into execution info for tje next executions synch.
        for (usize task_buffer_index = 0; task_buffer_index < permutation.buffer_infos.size(); ++task_buffer_index)
        {
            bool const is_persistent = daxa::holds_alternative<PermIndepTaskBufferInfo::Persistent>(impl.global_buffer_infos[task_buffer_index].task_buffer_data);
            if (permutation.buffer_infos[task_buffer_index].valid && is_persistent)
            {
                daxa::get<PermIndepTaskBufferInfo::Persistent>(impl.global_buffer_infos[task_buffer_index].task_buffer_data).get().latest_access = permutation.buffer_infos[task_buffer_index].latest_access;
            }
        }
        for (usize task_image_index = 0; task_image_index < permutation.image_infos.size(); ++task_image_index)
        {
            if (
                permutation.image_infos[task_image_index].valid &&
                impl.global_image_infos[task_image_index].is_persistent())
            {
                auto & persistent_image = impl.global_image_infos[task_image_index].get_persistent();
                for (auto const & extended_state : permutation.image_infos[task_image_index].last_slice_states)
                {
                    persistent_image.latest_slice_states.push_back(extended_state.state);
                }
            }
        }

        // TODO: reimplement left over commands
        // impl.left_over_command_lists = std::move(impl_runtime.recorder.complete_current_commands());
        impl.executed_once = true;
        impl.prev_frame_permutation_index = permutation_index;

        if (impl.info.record_debug_information)
        {
            impl.debug_print();
        }
    }

    ImplTaskGraph::ImplTaskGraph(TaskGraphInfo a_info)
        : unique_index{ImplTaskGraph::exec_unique_next_index++}, info{std::move(a_info)}
    {
        if (a_info.staging_memory_pool_size != 0)
        {
            this->staging_memory = TransferMemoryPool{TransferMemoryPoolInfo{.device = info.device, .capacity = info.staging_memory_pool_size, .use_bar_memory = true, .name = "Transfer Memory Pool"}};
        }
    }

    ImplTaskGraph::~ImplTaskGraph()
    {
        for (auto & task : tasks)
        {
            for (auto & view_cache : task.image_view_cache)
            {
                for (auto & view : view_cache)
                {
                    if (info.device.is_id_valid(view))
                    {
                        ImageId const parent = info.device.image_view_info(view).value().image;
                        bool const is_default_view = parent.default_view() == view;
                        if (!is_default_view)
                        {
                            info.device.destroy_image_view(view);
                        }
                    }
                }
            }
        }
        for (auto & permutation : permutations)
        {
            // because transient buffers are owned by the task graph, we need to destroy them
            for (u32 buffer_info_idx = 0; buffer_info_idx < static_cast<u32>(global_buffer_infos.size()); buffer_info_idx++)
            {
                auto const & global_buffer = global_buffer_infos.at(buffer_info_idx);
                PerPermTaskBuffer const & perm_buffer = permutation.buffer_infos.at(buffer_info_idx);
                if (!global_buffer.is_persistent() &&
                    perm_buffer.valid)
                {
                    if (auto const * id = std::get_if<BufferId>(&perm_buffer.actual_id))
                    {
                        info.device.destroy_buffer(*id);
                    }
                    if (auto const * id = std::get_if<BlasId>(&perm_buffer.actual_id))
                    {
                        info.device.destroy_blas(*id);
                    }
                    if (auto const * id = std::get_if<TlasId>(&perm_buffer.actual_id))
                    {
                        info.device.destroy_tlas(*id);
                    }
                }
            }
            // because transient images are owned by the task graph, we need to destroy them
            for (u32 image_info_idx = 0; image_info_idx < static_cast<u32>(global_image_infos.size()); image_info_idx++)
            {
                auto const & global_image = global_image_infos.at(image_info_idx);
                auto const & perm_image = permutation.image_infos.at(image_info_idx);
                if (!global_image.is_persistent() && perm_image.valid)
                {
                    info.device.destroy_image(get_actual_images(TaskImageView{{.task_graph_index = unique_index, .index = image_info_idx}}, permutation)[0]);
                }
            }
        }
    }

    void ImplTaskGraph::print_task_image_to(std::string & out, std::string indent, TaskGraphPermutation const & permutation, TaskImageView local_id)
    {
        if (local_id.is_null())
        {
            fmt::format_to(std::back_inserter(out), "{}task image [NULL]\n", indent);
            return;
        }
        auto const & glob_image = global_image_infos[local_id.index];
        std::string persistent_info;
        if (global_image_infos[local_id.index].is_persistent())
        {
            u32 const persistent_index = global_image_infos[local_id.index].get_persistent().unique_index;
            persistent_info = fmt::format(", persistent index: {}", persistent_index);
        }
        fmt::format_to(std::back_inserter(out), "{}task image name: \"{}\", id: ({}){}\n", indent, glob_image.get_name(), to_string(local_id), persistent_info);
        fmt::format_to(std::back_inserter(out), "{}runtime images:\n", indent);
        {
            [[maybe_unused]] FormatIndent const d1{out, indent, true};
            for (u32 child_i = 0; child_i < get_actual_images(local_id, permutation).size(); ++child_i)
            {
                auto const child_id = get_actual_images(local_id, permutation)[child_i];
                auto const & child_info = info.device.image_info(child_id).value();
                fmt::format_to(std::back_inserter(out), "{}name: \"{}\", id: ({})\n", indent, child_info.name.view(), to_string(child_id));
            }
            print_separator_to(out, indent);
        }
    }

    void ImplTaskGraph::print_task_buffer_blas_tlas_to(std::string & out, std::string indent, TaskGraphPermutation const & permutation, TaskGPUResourceView local_id)
    {
        if (local_id.is_null())
        {
            fmt::format_to(std::back_inserter(out), "{}[NULL]\n", indent);
            return;
        }
        auto const & glob_buffer = global_buffer_infos[local_id.index];
        auto const & actual_ids = get_actual_buffer_blas_tlas_generic(local_id, permutation);
        std::string_view type_str = buffer_blas_tlas_str(local_id, permutation);
        std::string persistent_info;
        if (global_buffer_infos[local_id.index].is_persistent())
        {
            u32 const persistent_index = global_buffer_infos[local_id.index].get_persistent().unique_index;
            persistent_info = fmt::format(", persistent index: {}", persistent_index);
        }
        fmt::format_to(std::back_inserter(out), "{}task {} name: \"{}\", id: ({}){}\n", indent, type_str, glob_buffer.get_name(), to_string(local_id), persistent_info);
        fmt::format_to(std::back_inserter(out), "{}runtime {}:\n", indent, type_str);
        std::visit([&](auto const & ids) {
            [[maybe_unused]] FormatIndent const d2{out, indent, true};
            for (u32 child_i = 0; child_i < ids.size(); ++child_i)
            {
                auto const child_id = ids[child_i];
                using ChildIdT = std::decay_t<decltype(child_id)>;
                if constexpr (std::is_same_v<ChildIdT, BufferId>)
                {
                    auto const & child_info = info.device.buffer_info(child_id).value();
                    fmt::format_to(std::back_inserter(out), "{}name: \"{}\", id: ({})\n", indent, child_info.name.view(), to_string(child_id));
                }
                if constexpr (std::is_same_v<ChildIdT, BlasId>)
                {
                    auto const & child_info = info.device.info_blas(child_id).value();
                    fmt::format_to(std::back_inserter(out), "{}name: \"{}\", id: ({})\n", indent, child_info.name.view(), to_string(child_id));
                }
                if constexpr (std::is_same_v<ChildIdT, TlasId>)
                {
                    auto const & child_info = info.device.info_tlas(child_id).value();
                    fmt::format_to(std::back_inserter(out), "{}name: \"{}\", id: ({})\n", indent, child_info.name.view(), to_string(child_id));
                }
            }
            print_separator_to(out, indent);
        }, actual_ids);
    }

    void ImplTaskGraph::print_task_barrier_to(std::string & out, std::string & indent, TaskGraphPermutation const & permutation, usize index, bool const split_barrier)
    {
        TaskBarrier const & barrier = split_barrier ? permutation.split_barriers[index] : permutation.barriers[index];
        if (barrier.image_id.is_empty())
        {
            MemoryBarrierInfo const mem_barrier{
                .src_access = barrier.src_access,
                .dst_access = barrier.dst_access,
            };
            out.append(indent).append(to_string(mem_barrier)).push_back('\n');
        }
        else
        {
            fmt::format_to(std::back_inserter(out), "{}slice: ({})\n", indent, to_string(barrier.slice));
            fmt::format_to(std::back_inserter(out), "{}{}\n", indent, to_string(MemoryBarrierInfo{.src_access = barrier.src_access, .dst_access = barrier.dst_access}));
            fmt::format_to(std::back_inserter(out), "{}layout: ({}) -> ({})\n", indent, to_string(barrier.layout_before), to_string(barrier.layout_after));
            print_task_image_to(out, indent, permutation, barrier.image_id);
        }
    }

    void ImplTaskGraph::print_task_to(std::string & out, std::string & indent, TaskGraphPermutation const & permutation, usize task_id)
    {
        ImplTask const & task = tasks[task_id];
        fmt::format_to(std::back_inserter(out), "{}task name: \"{}\", id: {}\n", indent, task.base_task->name(), task_id);
        fmt::format_to(std::back_inserter(out), "{}task arguments:\n", indent);
        [[maybe_unused]] FormatIndent const d0{out, indent, true};
        for_each(
            task.base_task->attachments(),
            [&](u32, auto const & attach)
            {
                auto [access, is_concurrent] = task_buffer_access_to_access(static_cast<TaskBufferAccess>(attach.access));
                std::string_view type_str = buffer_blas_tlas_str(attach.translated_view, permutation);
                fmt::format_to(std::back_inserter(out), "{}{} argument:\n", indent, type_str);
                fmt::format_to(std::back_inserter(out), "{}access: ({})\n", indent, to_string(access));
                print_task_buffer_blas_tlas_to(out, indent, permutation, attach.translated_view);
                print_separator_to(out, indent);
            },
            [&](u32, TaskImageAttachmentInfo const & img)
            {
                auto [layout, access, is_concurrent] = task_image_access_to_layout_access(img.access);
                fmt::format_to(std::back_inserter(out), "{}image argument:\n", indent);
                fmt::format_to(std::back_inserter(out), "{}access: ({})\n", indent, to_string(access));
                fmt::format_to(std::back_inserter(out), "{}layout: {}\n", indent, to_string(layout));
                fmt::format_to(std::back_inserter(out), "{}slice: {}\n", indent, to_string(img.translated_view.slice));
                print_task_image_to(out, indent, permutation, img.translated_view);
                print_separator_to(out, indent);
            });
    }

    void ImplTaskGraph::print_permutation_aliasing_to(std::string & out, std::string indent, TaskGraphPermutation const & permutation)
    {
        usize batches = 0;
        std::vector<usize> submit_batch_offsets(permutation.batch_submit_scopes.size());
        for (u32 submit_scope_idx = 0; submit_scope_idx < permutation.batch_submit_scopes.size(); submit_scope_idx++)
        {
            submit_batch_offsets.at(submit_scope_idx) = batches;
            batches += permutation.batch_submit_scopes.at(submit_scope_idx).task_batches.size();
        }
        auto print_lifetime = [&](usize start_idx, usize end_idx)
        {
            for (usize i = 0; i < batches; i++)
            {
                if (i >= start_idx && i < end_idx)
                {
                    fmt::format_to(std::back_inserter(out), "{}===", i);
                }
                else if (i == end_idx && end_idx != batches - 1)
                {
                    fmt::format_to(std::back_inserter(out), "{}---", i);
                }
                else if (i != batches - 1)
                {
                    fmt::format_to(std::back_inserter(out), "----", i);
                }
                else
                {
                    if (end_idx == batches - 1)
                    {
                        fmt::format_to(std::back_inserter(out), "{}", i);
                    }
                    else
                    {
                        fmt::format_to(std::back_inserter(out), "-");
                    }
                }
            }
        };
        fmt::format_to(std::back_inserter(out), "{}Resource lifetimes and aliasing:\n", indent);
        for (u32 perm_image_idx = 0; perm_image_idx < permutation.image_infos.size(); perm_image_idx++)
        {
            if (global_image_infos.at(perm_image_idx).is_persistent() || !permutation.image_infos.at(perm_image_idx).valid)
            {
                continue;
            }

            auto const & perm_task_image = permutation.image_infos.at(perm_image_idx);
            usize const start_idx = submit_batch_offsets.at(perm_task_image.lifetime.first_use.submit_scope_index) +
                                    perm_task_image.lifetime.first_use.task_batch_index;
            usize const end_idx = submit_batch_offsets.at(perm_task_image.lifetime.last_use.submit_scope_index) +
                                  perm_task_image.lifetime.last_use.task_batch_index;
            fmt::format_to(std::back_inserter(out), "{}", indent);
            print_lifetime(start_idx, end_idx);
            fmt::format_to(std::back_inserter(out), "  allocation offset: {} allocation size: {} task resource name: {}\n",
                           perm_task_image.allocation_offset,
                           0, // TODO(msakmary)
                           global_image_infos.at(perm_image_idx).get_name());
        }
        for (u32 perm_buffer_idx = 0; perm_buffer_idx < permutation.buffer_infos.size(); perm_buffer_idx++)
        {
            if (global_buffer_infos.at(perm_buffer_idx).is_persistent() || !permutation.buffer_infos.at(perm_buffer_idx).valid)
            {
                continue;
            }

            auto const & perm_task_buffer = permutation.buffer_infos.at(perm_buffer_idx);
            usize const start_idx = submit_batch_offsets.at(perm_task_buffer.lifetime.first_use.submit_scope_index) +
                                    perm_task_buffer.lifetime.first_use.task_batch_index;
            usize const end_idx = submit_batch_offsets.at(perm_task_buffer.lifetime.last_use.submit_scope_index) +
                                  perm_task_buffer.lifetime.last_use.task_batch_index;
            fmt::format_to(std::back_inserter(out), "{}", indent);
            print_lifetime(start_idx, end_idx);
            fmt::format_to(std::back_inserter(out), "  allocation offset: {} allocation size: {} task resource name: {}\n",
                           perm_task_buffer.allocation_offset,
                           0, // TODO(msakmary)
                           global_buffer_infos.at(perm_buffer_idx).get_name());
        }
    }

    void ImplTaskGraph::debug_print()
    {
        std::string out = {};
        std::string indent = {};
        fmt::format_to(std::back_inserter(out), "task graph name: {}, id: {}:\n", info.name, unique_index);
        fmt::format_to(std::back_inserter(out), "device: {}\n", info.device.info().name.view());
        fmt::format_to(std::back_inserter(out), "swapchain: {}\n", (this->info.swapchain.has_value() ? this->info.swapchain.value().info().name.view() : "-"));
        fmt::format_to(std::back_inserter(out), "reorder tasks: {}\n", info.reorder_tasks);
        fmt::format_to(std::back_inserter(out), "use split barriers: {}\n", info.use_split_barriers);
        fmt::format_to(std::back_inserter(out), "permutation_condition_count: {}\n", info.permutation_condition_count);
        fmt::format_to(std::back_inserter(out), "enable_command_labels: {}\n", info.enable_command_labels);
        fmt::format_to(std::back_inserter(out), "task_graph_label_color: ({},{},{},{})\n",
                       info.task_graph_label_color[0],
                       info.task_graph_label_color[1],
                       info.task_graph_label_color[2],
                       info.task_graph_label_color[3]);
        fmt::format_to(std::back_inserter(out), "task_batch_label_color: ({},{},{},{})\n",
                       info.task_batch_label_color[0],
                       info.task_batch_label_color[1],
                       info.task_batch_label_color[2],
                       info.task_batch_label_color[3]);
        fmt::format_to(std::back_inserter(out), "task_label_color: ({},{},{},{})\n",
                       info.task_label_color[0],
                       info.task_label_color[1],
                       info.task_label_color[2],
                       info.task_label_color[3]);
        fmt::format_to(std::back_inserter(out), "record_debug_information: {}\n", info.record_debug_information);
        fmt::format_to(std::back_inserter(out), "staging_memory_pool_size: {}\n", info.staging_memory_pool_size);
        fmt::format_to(std::back_inserter(out), "executed permutation: {}\n", chosen_permutation_last_execution);
        usize permutation_index = this->chosen_permutation_last_execution;
        auto & permutation = this->permutations[permutation_index];
        {
            this->print_permutation_aliasing_to(out, indent, permutation);
            permutation_index += 1;
            fmt::format_to(std::back_inserter(out), "permutations split barriers: {}\n", info.use_split_barriers);
            [[maybe_unused]] FormatIndent const d0{out, indent, true};
            usize submit_scope_index = 0;
            for (auto & submit_scope : permutation.batch_submit_scopes)
            {
                fmt::format_to(std::back_inserter(out), "{}submit scope: {}\n", indent, submit_scope_index);
                [[maybe_unused]] FormatIndent const d1{out, indent, true};
                usize batch_index = 0;
                for (auto & task_batch : submit_scope.task_batches)
                {
                    fmt::format_to(std::back_inserter(out), "{}batch: {}\n", indent, batch_index);
                    batch_index += 1;
                    fmt::format_to(std::back_inserter(out), "{}inserted pipeline barriers:\n", indent);
                    {
                        [[maybe_unused]] FormatIndent const d2{out, indent, true};
                        for (auto barrier_index : task_batch.pipeline_barrier_indices)
                        {
                            this->print_task_barrier_to(out, indent, permutation, barrier_index, false);
                            print_separator_to(out, indent);
                        }
                    }
                    if (!this->info.use_split_barriers)
                    {
                        fmt::format_to(std::back_inserter(out), "{}inserted pipeline barriers (converted from split barrier):\n", indent);
                        [[maybe_unused]] FormatIndent const d2{out, indent, true};
                        for (auto barrier_index : task_batch.wait_split_barrier_indices)
                        {
                            this->print_task_barrier_to(out, indent, permutation, barrier_index, true);
                            print_separator_to(out, indent);
                        }
                    }
                    else
                    {
                        fmt::format_to(std::back_inserter(out), "{}inserted split pipeline barrier waits:\n", indent);
                        [[maybe_unused]] FormatIndent const d2{out, indent, true};
                        print_separator_to(out, indent);
                        for (auto barrier_index : task_batch.wait_split_barrier_indices)
                        {
                            this->print_task_barrier_to(out, indent, permutation, barrier_index, true);
                            print_separator_to(out, indent);
                        }
                    }
                    fmt::format_to(std::back_inserter(out), "{}tasks:\n", indent);
                    {
                        [[maybe_unused]] FormatIndent const d2{out, indent, true};
                        for (TaskId const task_id : task_batch.tasks)
                        {
                            this->print_task_to(out, indent, permutation, task_id);
                            print_separator_to(out, indent);
                        }
                    }
                    if (this->info.use_split_barriers)
                    {
                        fmt::format_to(std::back_inserter(out), "{}inserted split barrier signals:\n", indent);
                        [[maybe_unused]] FormatIndent const d2{out, indent, true};
                        for (usize const barrier_index : task_batch.signal_split_barrier_indices)
                        {
                            this->print_task_barrier_to(out, indent, permutation, barrier_index, true);
                            print_separator_to(out, indent);
                        }
                    }
                    print_separator_to(out, indent);
                }
                if (!submit_scope.last_minute_barrier_indices.empty())
                {
                    fmt::format_to(std::back_inserter(out), "{}inserted last minute pipeline barriers:\n", indent);
                    [[maybe_unused]] FormatIndent const d2{out, indent, true};
                    for (usize const barrier_index : submit_scope.last_minute_barrier_indices)
                    {
                        this->print_task_barrier_to(out, indent, permutation, barrier_index, false);
                        print_separator_to(out, indent);
                    }
                }
                if (&submit_scope != &permutation.batch_submit_scopes.back())
                {
                    fmt::format_to(std::back_inserter(out), "{} -- inserted submit -- \n", indent);
                    if (submit_scope.present_info.has_value())
                    {
                        fmt::format_to(std::back_inserter(out), "{} -- inserted present -- \n", indent);
                    }
                }
                print_separator_to(out, indent);
            }
            print_separator_to(out, indent);
        }
        this->debug_string_stream << out;
    }

    auto TaskGraph::inc_refcnt(ImplHandle const * object) -> u64
    {
        return object->inc_refcnt();
    }

    auto TaskGraph::dec_refcnt(ImplHandle const * object) -> u64
    {
        return object->dec_refcnt(
            ImplTaskGraph::zero_ref_callback,
            nullptr);
    }

    void ImplTaskGraph::zero_ref_callback(ImplHandle const * handle)
    {
        auto const * self = r_cast<ImplTaskGraph const *>(handle);
        delete self;
    }
} // namespace daxa

#endif
