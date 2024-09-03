#include "impl_command_recorder.hpp"

#include <daxa/c/types.h>
#include <utility>

#include "impl_sync.hpp"
#include "impl_device.hpp"
#include "impl_core.hpp"

/// --- Begin Helpers ---

auto get_vk_image_memory_barrier(daxa_ImageMemoryBarrierInfo const & image_barrier, VkImage vk_image, VkImageAspectFlags aspect_flags) -> VkImageMemoryBarrier2
{
    return VkImageMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = image_barrier.src_access.stages,
        .srcAccessMask = image_barrier.src_access.access_type,
        .dstStageMask = image_barrier.dst_access.stages,
        .dstAccessMask = image_barrier.dst_access.access_type,
        .oldLayout = static_cast<VkImageLayout>(image_barrier.src_layout),
        .newLayout = static_cast<VkImageLayout>(image_barrier.dst_layout),
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = vk_image,
        .subresourceRange = make_subresource_range(image_barrier.image_slice, aspect_flags),
    };
}

auto get_vk_memory_barrier(daxa_MemoryBarrierInfo const & memory_barrier) -> VkMemoryBarrier2
{
    return VkMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = memory_barrier.src_access.stages,
        .srcAccessMask = memory_barrier.src_access.access_type,
        .dstStageMask = memory_barrier.dst_access.stages,
        .dstAccessMask = memory_barrier.dst_access.access_type,
    };
}

auto get_vk_dependency_info(
    std::vector<VkImageMemoryBarrier2> const & vk_image_memory_barriers,
    std::vector<VkMemoryBarrier2> const & vk_memory_barriers) -> VkDependencyInfo
{
    return VkDependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = {},
        .memoryBarrierCount = static_cast<u32>(vk_memory_barriers.size()),
        .pMemoryBarriers = vk_memory_barriers.data(),
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = static_cast<u32>(vk_image_memory_barriers.size()),
        .pImageMemoryBarriers = vk_image_memory_barriers.data(),
    };
}

auto CommandPoolPool::get(daxa_Device device) -> VkCommandPool
{
    VkCommandPool pool = {};
    if (pools_and_buffers.empty())
    {
        VkCommandPoolCreateInfo const vk_command_pool_create_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = this->queue_family_index,
        };

        vkCreateCommandPool(device->vk_device, &vk_command_pool_create_info, nullptr, &pool);
    }
    else
    {
        pool = pools_and_buffers.back();
        pools_and_buffers.pop_back();
    }
    return pool;
}

void CommandPoolPool::put_back(VkCommandPool pool)
{
    pools_and_buffers.push_back(pool);
}

void CommandPoolPool::cleanup(daxa_Device device)
{
    for (auto * pool : pools_and_buffers)
    {
        vkDestroyCommandPool(device->vk_device, pool, nullptr);
    }
    pools_and_buffers.clear();
}

template <typename T>
auto only_check_buffer(daxa_CommandRecorder self, T id) -> bool
{
    if constexpr (std::is_same_v<T, daxa_BufferId>)
    {
        return daxa_dvc_is_buffer_valid(self->device, id);
    }
    else
    {
        return true;
    }
}
template <typename T>
auto only_check_image(daxa_CommandRecorder self, T id) -> bool
{
    if constexpr (std::is_same_v<T, daxa_ImageId>)
    {
        return daxa_dvc_is_image_valid(self->device, id);
    }
    else
    {
        return true;
    }
}
template <typename T>
auto only_check_image_view(daxa_CommandRecorder self, T id) -> bool
{
    if constexpr (std::is_same_v<T, daxa_ImageViewId>)
    {
        return daxa_dvc_is_image_view_valid(self->device, id);
    }
    else
    {
        return true;
    }
}
template <typename T>
auto only_check_sampler(daxa_CommandRecorder self, T id) -> bool
{
    if constexpr (std::is_same_v<T, daxa_SamplerId>)
    {
        return daxa_dvc_is_sampler_valid(self->device, id);
    }
    else
    {
        return true;
    }
}
template <typename T>
auto only_check_tlas(daxa_CommandRecorder self, T id) -> bool
{
    if constexpr (std::is_same_v<T, daxa_TlasId>)
    {
        return daxa_dvc_is_tlas_valid(self->device, id);
    }
    else
    {
        return true;
    }
}
template <typename T>
auto only_check_blas(daxa_CommandRecorder self, T id) -> bool
{
    if constexpr (std::is_same_v<T, daxa_BlasId>)
    {
        return daxa_dvc_is_blas_valid(self->device, id);
    }
    else
    {
        return true;
    }
}
template <typename T>
void remember_ids(daxa_CommandRecorder self, T id)
{
    if constexpr (std::is_same_v<daxa_BufferId, T>)
    {
        self->current_command_data.used_buffers.push_back(std::bit_cast<BufferId>(id));
    }
    if constexpr (std::is_same_v<daxa_ImageId, T>)
    {
        self->current_command_data.used_images.push_back(std::bit_cast<ImageId>(id));
    }
    if constexpr (std::is_same_v<daxa_ImageViewId, T>)
    {
        self->current_command_data.used_image_views.push_back(std::bit_cast<ImageViewId>(id));
    }
    if constexpr (std::is_same_v<daxa_SamplerId, T>)
    {
        self->current_command_data.used_samplers.push_back(std::bit_cast<SamplerId>(id));
    }
    if constexpr (std::is_same_v<daxa_TlasId, T>)
    {
        self->current_command_data.used_tlass.push_back(std::bit_cast<TlasId>(id));
    }
    if constexpr (std::is_same_v<daxa_BlasId, T>)
    {
        self->current_command_data.used_blass.push_back(std::bit_cast<BlasId>(id));
    }
}

template <typename... Args>
auto check_ids(daxa_CommandRecorder self, Args... args) -> daxa_Result
{
    if (!(only_check_buffer(self, args) && ...))
    {
        return DAXA_RESULT_INVALID_BUFFER_ID;
    }
    if (!(only_check_image(self, args) && ...))
    {
        return DAXA_RESULT_INVALID_IMAGE_ID;
    }
    if (!(only_check_image_view(self, args) && ...))
    {
        return DAXA_RESULT_INVALID_IMAGE_VIEW_ID;
    }
    if (!(only_check_sampler(self, args) && ...))
    {
        return DAXA_RESULT_INVALID_SAMPLER_ID;
    }
    if (!(only_check_tlas(self, args) && ...))
    {
        return DAXA_RESULT_INVALID_TLAS_ID;
    }
    if (!(only_check_blas(self, args) && ...))
    {
        return DAXA_RESULT_INVALID_BLAS_ID;
    }
    return DAXA_RESULT_SUCCESS;
}

template <typename... Args>
void remember_ids(daxa_CommandRecorder self, Args... args)
{
    (remember_ids(self, args), ...);
}

#define _DAXA_CHECK_IDS(...)                                                  \
    {                                                                         \
        auto _DAXA_CHECK_IDS_RESULT = check_ids(__VA_ARGS__);                 \
        _DAXA_RETURN_IF_ERROR(_DAXA_CHECK_IDS_RESULT, _DAXA_CHECK_IDS_RESULT) \
    }

#define _DAXA_REMEMBER_IDS(...) remember_ids(__VA_ARGS__);

#define DAXA_CHECK_AND_REMEMBER_IDS(...) \
    _DAXA_CHECK_IDS(__VA_ARGS__)         \
    _DAXA_REMEMBER_IDS(__VA_ARGS__)

/// --- End Helpers ---

/// --- Begin API Functions ---

auto daxa_cmd_set_rasterization_samples(daxa_CommandRecorder self, VkSampleCountFlagBits samples) -> daxa_Result
{
    if (self->device->vkCmdSetRasterizationSamplesEXT == nullptr)
    {
        return DAXA_RESULT_ERROR_EXTENSION_NOT_PRESENT;
    }
    daxa_cmd_flush_barriers(self);
    self->device->vkCmdSetRasterizationSamplesEXT(self->current_command_data.vk_cmd_buffer, samples);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_copy_buffer_to_buffer(daxa_CommandRecorder self, daxa_BufferCopyInfo const * info) -> daxa_Result
{
    daxa_cmd_flush_barriers(self);
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->src_buffer, info->dst_buffer)
    auto const * vk_buffer_copy = reinterpret_cast<VkBufferCopy const *>(&info->src_offset);
    ImplBufferSlot const & src_slot = self->device->slot(info->src_buffer);
    ImplBufferSlot const & dst_slot = self->device->slot(info->dst_buffer);
    bool in_bounds = true;
    in_bounds = in_bounds && ((static_cast<u64>(vk_buffer_copy->srcOffset) + static_cast<u64>(vk_buffer_copy->size)) <= static_cast<u64>(src_slot.info.size));
    in_bounds = in_bounds && ((static_cast<u64>(vk_buffer_copy->dstOffset) + static_cast<u64>(vk_buffer_copy->size)) <= static_cast<u64>(dst_slot.info.size));
    if (!in_bounds)
    {
        return DAXA_RESULT_ERROR_COPY_OUT_OF_BOUNDS;
    }
    vkCmdCopyBuffer(
        self->current_command_data.vk_cmd_buffer,
        src_slot.vk_buffer,
        dst_slot.vk_buffer,
        1,
        vk_buffer_copy);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_copy_buffer_to_image(daxa_CommandRecorder self, daxa_BufferImageCopyInfo const * info) -> daxa_Result
{
    daxa_cmd_flush_barriers(self);
    //_DAXA_CHECK_AND_REMEMBER_IDS(self, info->buffer, info->image)
    auto const & img_slot = self->device->slot(info->image);
    VkBufferImageCopy const vk_buffer_image_copy{
        .bufferOffset = info->buffer_offset,
        // TODO(general): make sense of these parameters:
        .bufferRowLength = 0u,   // self->image_extent.x,
        .bufferImageHeight = 0u, // self->image_extent.y,
        .imageSubresource = make_subresource_layers(info->image_slice, img_slot.aspect_flags),
        .imageOffset = info->image_offset,
        .imageExtent = info->image_extent,
    };
    vkCmdCopyBufferToImage(
        self->current_command_data.vk_cmd_buffer,
        self->device->slot(info->buffer).vk_buffer,
        img_slot.vk_image,
        static_cast<VkImageLayout>(info->image_layout),
        1,
        &vk_buffer_image_copy);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_copy_image_to_buffer(daxa_CommandRecorder self, daxa_ImageBufferCopyInfo const * info) -> daxa_Result
{
    daxa_cmd_flush_barriers(self);
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->image, info->buffer)
    auto const & img_slot = self->device->slot(info->image);
    VkBufferImageCopy const vk_buffer_image_copy{
        .bufferOffset = info->buffer_offset,
        // TODO(general): make sense of these parameters:
        .bufferRowLength = 0u,   // info.image_extent.x,
        .bufferImageHeight = 0u, // info.image_extent.y,
        .imageSubresource = make_subresource_layers(info->image_slice, img_slot.aspect_flags),
        .imageOffset = info->image_offset,
        .imageExtent = info->image_extent,
    };
    vkCmdCopyImageToBuffer(
        self->current_command_data.vk_cmd_buffer,
        img_slot.vk_image,
        static_cast<VkImageLayout>(info->image_layout),
        self->device->slot(info->buffer).vk_buffer,
        1,
        &vk_buffer_image_copy);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_copy_image_to_image(daxa_CommandRecorder self, daxa_ImageCopyInfo const * info) -> daxa_Result
{
    daxa_cmd_flush_barriers(self);
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->src_image, info->dst_image)
    auto const & src_slot = self->device->slot(info->src_image);
    auto const & dst_slot = self->device->slot(info->dst_image);
    VkImageCopy const vk_image_copy{
        .srcSubresource = make_subresource_layers(info->src_slice, src_slot.aspect_flags),
        .srcOffset = {*reinterpret_cast<VkOffset3D const *>(&info->src_offset)},
        .dstSubresource = make_subresource_layers(info->dst_slice, dst_slot.aspect_flags),
        .dstOffset = {*reinterpret_cast<VkOffset3D const *>(&info->dst_offset)},
        .extent = {*reinterpret_cast<VkExtent3D const *>(&info->extent)},
    };
    vkCmdCopyImage(
        self->current_command_data.vk_cmd_buffer,
        src_slot.vk_image,
        static_cast<VkImageLayout>(info->src_image_layout),
        dst_slot.vk_image,
        static_cast<VkImageLayout>(info->dst_image_layout),
        1,
        &vk_image_copy);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_blit_image_to_image(daxa_CommandRecorder self, daxa_ImageBlitInfo const * info) -> daxa_Result
{
    daxa_cmd_flush_barriers(self);
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->src_image, info->dst_image)
    auto const & src_slot = self->device->slot(info->src_image);
    auto const & dst_slot = self->device->slot(info->dst_image);
    VkImageBlit const vk_blit{
        .srcSubresource = make_subresource_layers(info->src_slice, src_slot.aspect_flags),
        .srcOffsets = {info->src_offsets[0], info->src_offsets[1]},
        .dstSubresource = make_subresource_layers(info->dst_slice, dst_slot.aspect_flags),
        .dstOffsets = {info->dst_offsets[0], info->dst_offsets[1]},
    };
    vkCmdBlitImage(
        self->current_command_data.vk_cmd_buffer,
        src_slot.vk_image,
        static_cast<VkImageLayout>(info->src_image_layout),
        dst_slot.vk_image,
        static_cast<VkImageLayout>(info->dst_image_layout),
        1,
        &vk_blit,
        static_cast<VkFilter>(info->filter));
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_build_acceleration_structures(daxa_CommandRecorder self, daxa_BuildAccelerationStucturesInfo const * info) -> daxa_Result
{
    daxa_Result result = DAXA_RESULT_SUCCESS;
    if ((self->device->properties.implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_BASIC_RAY_TRACING) == 0)
    {
        result = DAXA_RESULT_INVALID_WITHOUT_ENABLING_RAY_TRACING;
    }
    _DAXA_RETURN_IF_ERROR(result, result)
    daxa_cmd_flush_barriers(self);
    for (auto const & tb_info : std::span{info->tlas_build_infos, info->tlas_build_info_count})
    {
        _DAXA_CHECK_IDS(self, tb_info.dst_tlas)
    }
    for (auto const & bb_info : std::span{info->blas_build_infos, info->blas_build_info_count})
    {
        _DAXA_CHECK_IDS(self, bb_info.dst_blas)
    }
    for (auto const & tb_info : std::span{info->tlas_build_infos, info->tlas_build_info_count})
    {
        _DAXA_REMEMBER_IDS(self, tb_info.dst_tlas)
    }
    for (auto const & bb_info : std::span{info->blas_build_infos, info->blas_build_info_count})
    {
        _DAXA_REMEMBER_IDS(self, bb_info.dst_blas)
    }
    // TODO(Raytracing): properties validation!
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> vk_build_geometry_infos;
    std::vector<VkAccelerationStructureGeometryKHR> vk_geometry_infos;
    std::vector<u32> primitive_counts;
    std::vector<u32 const *> primitive_counts_ptrs;
    daxa_as_build_info_to_vk(
        self->device,
        info->tlas_build_infos,
        info->tlas_build_info_count,
        info->blas_build_infos,
        info->blas_build_info_count,
        vk_build_geometry_infos,
        vk_geometry_infos,
        primitive_counts,
        primitive_counts_ptrs);
    // Convert the primitive count arrays to build range arrays:
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> vk_build_ranges;
    vk_build_ranges.reserve(primitive_counts.size());
    for (auto prim_count : primitive_counts)
    {
        vk_build_ranges.push_back(VkAccelerationStructureBuildRangeInfoKHR{
            .primitiveCount = prim_count,
            .primitiveOffset = {},
            .firstVertex = {},
            .transformOffset = {},
        });
    }
    std::vector<VkAccelerationStructureBuildRangeInfoKHR const *> vk_build_ranges_ptrs;
    vk_build_ranges_ptrs.reserve(primitive_counts_ptrs.size());
    for (auto const * prim_counts_ptr : primitive_counts_ptrs)
    {
        u64 const prim_counts_start_idx = static_cast<u64>(prim_counts_ptr - primitive_counts.data());
        vk_build_ranges_ptrs.push_back(vk_build_ranges.data() + prim_counts_start_idx);
    }
    self->device->vkCmdBuildAccelerationStructuresKHR(
        self->current_command_data.vk_cmd_buffer,
        static_cast<u32>(vk_build_geometry_infos.size()),
        vk_build_geometry_infos.data(),
        vk_build_ranges_ptrs.data());
    return result;
}

auto daxa_cmd_clear_buffer(daxa_CommandRecorder self, daxa_BufferClearInfo const * info) -> daxa_Result
{
    daxa_cmd_flush_barriers(self);
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->buffer)
    vkCmdFillBuffer(
        self->current_command_data.vk_cmd_buffer,
        self->device->slot(info->buffer).vk_buffer,
        static_cast<VkDeviceSize>(info->offset),
        static_cast<VkDeviceSize>(info->size),
        info->clear_value);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_clear_image(daxa_CommandRecorder self, daxa_ImageClearInfo const * info) -> daxa_Result
{
    daxa_cmd_flush_barriers(self);
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->image)
    auto const & img_slot = self->device->slot(info->image);
    bool const is_image_depth_stencil =
        is_depth_format(std::bit_cast<Format>(img_slot.info.format)) ||
        is_stencil_format(std::bit_cast<Format>(img_slot.info.format));
    bool const is_clear_depth_stencil = info->clear_value.index == 3;
    if (is_clear_depth_stencil)
    {
        if (!is_image_depth_stencil)
        {
            return DAXA_RESULT_INVALID_CLEAR_VALUE;
        }
        VkImageSubresourceRange const sub_range = make_subresource_range(info->dst_slice, img_slot.aspect_flags);
        vkCmdClearDepthStencilImage(
            self->current_command_data.vk_cmd_buffer,
            img_slot.vk_image,
            static_cast<VkImageLayout>(info->image_layout),
            &info->clear_value.values.depthStencil,
            1,
            &sub_range);
    }
    else
    {
        if (is_image_depth_stencil)
        {
            return DAXA_RESULT_INVALID_CLEAR_VALUE;
        }
        VkImageSubresourceRange const sub_range = make_subresource_range(info->dst_slice, img_slot.aspect_flags);
        vkCmdClearColorImage(
            self->current_command_data.vk_cmd_buffer,
            img_slot.vk_image,
            static_cast<VkImageLayout>(info->image_layout),
            &info->clear_value.values.color,
            1,
            &sub_range);
    }
    return DAXA_RESULT_SUCCESS;
}

/// @brief  Successive pipeline barrier calls are combined.
///         As soon as a non-pipeline barrier command is recorded, the currently recorded barriers are flushed with a vkCmdPipelineBarrier2 call.
/// @param info parameters.
void daxa_cmd_pipeline_barrier(daxa_CommandRecorder self, daxa_MemoryBarrierInfo const * info)
{
    if (self->memory_barrier_batch_count == COMMAND_LIST_BARRIER_MAX_BATCH_SIZE)
    {
        daxa_cmd_flush_barriers(self);
    }
    self->memory_barrier_batch.at(self->memory_barrier_batch_count++) = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = info->src_access.stages,
        .srcAccessMask = info->src_access.access_type,
        .dstStageMask = info->dst_access.stages,
        .dstAccessMask = info->dst_access.access_type,
    };
}

/// @brief  Successive pipeline barrier calls are combined.
///         As soon as a non-pipeline barrier command is recorded, the currently recorded barriers are flushed with a vkCmdPipelineBarrier2 call.
/// @param info parameters.
auto daxa_cmd_pipeline_barrier_image_transition(daxa_CommandRecorder self, daxa_ImageMemoryBarrierInfo const * info) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->image_id)
    if (self->image_barrier_batch_count == COMMAND_LIST_BARRIER_MAX_BATCH_SIZE)
    {
        daxa_cmd_flush_barriers(self);
    }
    auto const & img_slot = self->device->slot(info->image_id);
    self->image_barrier_batch.at(self->image_barrier_batch_count++) = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = info->src_access.stages,
        .srcAccessMask = info->src_access.access_type,
        .dstStageMask = info->dst_access.stages,
        .dstAccessMask = info->dst_access.access_type,
        .oldLayout = static_cast<VkImageLayout>(info->src_layout),
        .newLayout = static_cast<VkImageLayout>(info->dst_layout),
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = img_slot.vk_image,
        .subresourceRange = make_subresource_range(info->image_slice, img_slot.aspect_flags),
    };
    return DAXA_RESULT_SUCCESS;
}
struct SplitBarrierDependencyInfoBuffer
{
    std::vector<VkImageMemoryBarrier2> vk_image_memory_barriers = {};
    std::vector<VkMemoryBarrier2> vk_memory_barriers = {};
};

inline static thread_local std::vector<SplitBarrierDependencyInfoBuffer> tl_split_barrier_dependency_infos_aux_buffer = {}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
inline static thread_local std::vector<VkDependencyInfo> tl_split_barrier_dependency_infos_buffer = {};                     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
inline static thread_local std::vector<VkEvent> tl_split_barrier_events_buffer = {};

void daxa_cmd_signal_event(daxa_CommandRecorder self, daxa_EventSignalInfo const * info)
{
    daxa_cmd_flush_barriers(self);
    tl_split_barrier_dependency_infos_aux_buffer.push_back({});
    auto & dependency_infos_aux_buffer = tl_split_barrier_dependency_infos_aux_buffer.back();
    for (u64 i = 0; i < info->memory_barrier_count; ++i)
    {
        auto const & memory_barrier = info->memory_barriers[i];
        dependency_infos_aux_buffer.vk_memory_barriers.push_back(get_vk_memory_barrier(memory_barrier));
    }
    for (u64 i = 0; i < info->image_memory_barrier_count; ++i)
    {
        auto const & image_memory_barrier = info->image_memory_barriers[i];
        dependency_infos_aux_buffer.vk_image_memory_barriers.push_back(
            get_vk_image_memory_barrier(
                image_memory_barrier,
                self->device->slot(image_memory_barrier.image_id).vk_image,
                self->device->slot(image_memory_barrier.image_id).aspect_flags));
    }
    VkDependencyInfo const vk_dependency_info = get_vk_dependency_info(
        dependency_infos_aux_buffer.vk_image_memory_barriers,
        dependency_infos_aux_buffer.vk_memory_barriers);
    vkCmdSetEvent2(self->current_command_data.vk_cmd_buffer, (**info->event).vk_event, &vk_dependency_info);
    tl_split_barrier_dependency_infos_aux_buffer.clear();
}

void daxa_cmd_wait_events(daxa_CommandRecorder self, daxa_EventWaitInfo const * infos, size_t info_count)
{
    daxa_cmd_flush_barriers(self);
    for (u64 i = 0; i < info_count; ++i)
    {
        auto const & end_info = infos[i];
        tl_split_barrier_dependency_infos_aux_buffer.push_back({});
        auto & dependency_infos_aux_buffer = tl_split_barrier_dependency_infos_aux_buffer.back();
        for (u64 j = 0; j < end_info.memory_barrier_count; ++j)
        {
            auto const & memory_barrier = end_info.memory_barriers[j];
            dependency_infos_aux_buffer.vk_memory_barriers.push_back(get_vk_memory_barrier(memory_barrier));
        }
        for (u64 j = 0; j < end_info.image_memory_barrier_count; ++j)
        {
            auto const & image_barrier = end_info.image_memory_barriers[j];
            dependency_infos_aux_buffer.vk_image_memory_barriers.push_back(get_vk_image_memory_barrier(
                image_barrier,
                self->device->slot(image_barrier.image_id).vk_image,
                self->device->slot(image_barrier.image_id).aspect_flags));
        }
        tl_split_barrier_dependency_infos_buffer.push_back(get_vk_dependency_info(
            dependency_infos_aux_buffer.vk_image_memory_barriers,
            dependency_infos_aux_buffer.vk_memory_barriers));

        tl_split_barrier_events_buffer.push_back((**end_info.event).vk_event);
    }
    vkCmdWaitEvents2(
        self->current_command_data.vk_cmd_buffer,
        static_cast<u32>(tl_split_barrier_events_buffer.size()),
        tl_split_barrier_events_buffer.data(),
        tl_split_barrier_dependency_infos_buffer.data());
    tl_split_barrier_dependency_infos_aux_buffer.clear();
    tl_split_barrier_dependency_infos_buffer.clear();
    tl_split_barrier_events_buffer.clear();
}

void daxa_cmd_wait_event(daxa_CommandRecorder self, daxa_EventWaitInfo const * info)
{
    daxa_cmd_wait_events(self, info, 1);
}

void daxa_cmd_reset_event(daxa_CommandRecorder self, daxa_ResetEventInfo const * info)
{
    daxa_cmd_flush_barriers(self);
    vkCmdResetEvent2(
        self->current_command_data.vk_cmd_buffer,
        (**info->barrier).vk_event,
        info->stage_masks);
}

auto daxa_cmd_push_constant(daxa_CommandRecorder self, daxa_PushConstantInfo const * info) -> daxa_Result
{
    daxa_cmd_flush_barriers(self);
    if (daxa::holds_alternative<daxa_ImplCommandRecorder::NoPipeline>(self->current_pipeline))
    {
        return DAXA_RESULT_NO_PIPELINE_BOUND;
    }
    VkPipelineLayout vk_pipeline_layout = {};
    u32 current_pipeline_push_constant_size = {};
    if (auto * pipeline = daxa::get_if<daxa_ComputePipeline>(&self->current_pipeline))
    {
        current_pipeline_push_constant_size = (**pipeline).info.push_constant_size;
        vk_pipeline_layout = (**pipeline).vk_pipeline_layout;
    }
    if (auto * pipeline = daxa::get_if<daxa_RasterPipeline>(&self->current_pipeline))
    {
        current_pipeline_push_constant_size = (**pipeline).info.push_constant_size;
        vk_pipeline_layout = (**pipeline).vk_pipeline_layout;
    }
    if (auto * pipeline = daxa::get_if<daxa_RayTracingPipeline>(&self->current_pipeline))
    {
        current_pipeline_push_constant_size = (**pipeline).info.push_constant_size;
        vk_pipeline_layout = (**pipeline).vk_pipeline_layout;
    }
    if (current_pipeline_push_constant_size < (info->offset + info->size))
    {
        return DAXA_RESULT_PUSHCONSTANT_RANGE_EXCEEDED;
    }
    vkCmdPushConstants(self->current_command_data.vk_cmd_buffer, vk_pipeline_layout, VK_SHADER_STAGE_ALL, info->offset, static_cast<u32>(info->size), info->data);
    return DAXA_RESULT_SUCCESS;
}

void daxa_cmd_set_ray_tracing_pipeline(daxa_CommandRecorder self, daxa_RayTracingPipeline pipeline)
{
    daxa_cmd_flush_barriers(self);
    self->current_pipeline = pipeline;
    vkCmdBindDescriptorSets(self->current_command_data.vk_cmd_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline->vk_pipeline_layout, 0, 1, &self->device->gpu_sro_table.vk_descriptor_set, 0, nullptr);
    vkCmdBindPipeline(self->current_command_data.vk_cmd_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline->vk_pipeline);
}

void daxa_cmd_set_compute_pipeline(daxa_CommandRecorder self, daxa_ComputePipeline pipeline)
{
    daxa_cmd_flush_barriers(self);
    self->current_pipeline = pipeline;
    vkCmdBindDescriptorSets(self->current_command_data.vk_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->vk_pipeline_layout, 0, 1, &self->device->gpu_sro_table.vk_descriptor_set, 0, nullptr);
    vkCmdBindPipeline(self->current_command_data.vk_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->vk_pipeline);
}

void daxa_cmd_set_raster_pipeline(daxa_CommandRecorder self, daxa_RasterPipeline pipeline)
{
    daxa_cmd_flush_barriers(self);
    self->current_pipeline = pipeline;
    vkCmdBindDescriptorSets(self->current_command_data.vk_cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->vk_pipeline_layout, 0, 1, &self->device->gpu_sro_table.vk_descriptor_set, 0, nullptr);
    vkCmdBindPipeline(self->current_command_data.vk_cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->vk_pipeline);
}

auto daxa_cmd_trace_rays(daxa_CommandRecorder self, daxa_TraceRaysInfo const * info) -> daxa_Result
{
    // TODO: Check if those offsets are in range?
    if (!daxa::holds_alternative<daxa_RayTracingPipeline>(self->current_pipeline))
    {
        return DAXA_RESULT_NO_RAYTRACING_PIPELINE_BOUND;
    }
    auto const & binding_table = info->shader_binding_table;
    auto raygen_handle = binding_table.raygen_region;
    raygen_handle.deviceAddress += binding_table.raygen_region.stride * info->raygen_handle_offset;
    auto miss_handle = binding_table.miss_region;
    miss_handle.deviceAddress += binding_table.miss_region.stride * info->miss_handle_offset;
    auto hit_handle = binding_table.hit_region;
    hit_handle.deviceAddress += binding_table.hit_region.stride * info->hit_handle_offset;
    auto call_handle = binding_table.callable_region;
    call_handle.deviceAddress += binding_table.callable_region.stride * info->callable_handle_offset;
    self->device->vkCmdTraceRaysKHR(
        self->current_command_data.vk_cmd_buffer,
        &raygen_handle,
        &miss_handle,
        &hit_handle,
        &call_handle,
        info->width, info->height, info->depth);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_trace_rays_indirect(daxa_CommandRecorder self, daxa_TraceRaysIndirectInfo const * info) -> daxa_Result
{
    // TODO: Check if those offsets are in range?
    if (!daxa::holds_alternative<daxa_RayTracingPipeline>(self->current_pipeline))
    {
        return DAXA_RESULT_NO_RAYTRACING_PIPELINE_BOUND;
    }
    auto const & binding_table = info->shader_binding_table;
    auto raygen_handle = binding_table.raygen_region;
    raygen_handle.deviceAddress += binding_table.raygen_region.stride * info->raygen_handle_offset;
    auto miss_handle = binding_table.miss_region;
    miss_handle.deviceAddress += binding_table.miss_region.stride * info->miss_handle_offset;
    auto hit_handle = binding_table.hit_region;
    hit_handle.deviceAddress += binding_table.hit_region.stride * info->hit_handle_offset;
    auto call_handle = binding_table.callable_region;
    call_handle.deviceAddress += binding_table.callable_region.stride * info->callable_handle_offset;
    self->device->vkCmdTraceRaysIndirectKHR(
        self->current_command_data.vk_cmd_buffer,
        &raygen_handle,
        &miss_handle,
        &hit_handle,
        &call_handle,
        info->indirect_device_address);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_dispatch(daxa_CommandRecorder self, daxa_DispatchInfo const * info) -> daxa_Result
{
    // TODO: Check if those offsets are in range?
    if (!daxa::holds_alternative<daxa_ComputePipeline>(self->current_pipeline))
    {
        return DAXA_RESULT_NO_COMPUTE_PIPELINE_BOUND;
    }
    vkCmdDispatch(self->current_command_data.vk_cmd_buffer, info->x, info->y, info->z);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_dispatch_indirect(daxa_CommandRecorder self, daxa_DispatchIndirectInfo const * info) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->indirect_buffer)
    if (!daxa::holds_alternative<daxa_ComputePipeline>(self->current_pipeline))
    {
        return DAXA_RESULT_NO_COMPUTE_PIPELINE_BOUND;
    }
    vkCmdDispatchIndirect(self->current_command_data.vk_cmd_buffer, self->device->slot(info->indirect_buffer).vk_buffer, info->offset);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_destroy_buffer_deferred(daxa_CommandRecorder self, daxa_BufferId id) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, id)
    self->current_command_data.deferred_destructions.emplace_back(std::bit_cast<GPUResourceId>(id), DEFERRED_DESTRUCTION_BUFFER_INDEX);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_destroy_image_deferred(daxa_CommandRecorder self, daxa_ImageId id) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, id)
    self->current_command_data.deferred_destructions.emplace_back(std::bit_cast<GPUResourceId>(id), DEFERRED_DESTRUCTION_IMAGE_INDEX);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_destroy_image_view_deferred(daxa_CommandRecorder self, daxa_ImageViewId id) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, id)
    self->current_command_data.deferred_destructions.emplace_back(std::bit_cast<GPUResourceId>(id), DEFERRED_DESTRUCTION_IMAGE_VIEW_INDEX);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_destroy_sampler_deferred(daxa_CommandRecorder self, daxa_SamplerId id) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, id)
    self->current_command_data.deferred_destructions.emplace_back(std::bit_cast<GPUResourceId>(id), DEFERRED_DESTRUCTION_SAMPLER_INDEX);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_begin_renderpass(daxa_CommandRecorder self, daxa_RenderPassBeginInfo const * info) -> daxa_Result
{
    daxa_cmd_flush_barriers(self);

    auto fill_rendering_attachment_info = [&](daxa_RenderAttachmentInfo const & in, VkRenderingAttachmentInfo & out)
    {
        out = VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = self->device->slot(in.image_view).vk_image_view,
            .imageLayout = std::bit_cast<VkImageLayout>(in.layout),
            .resolveMode = VkResolveModeFlagBits::VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = static_cast<VkAttachmentLoadOp>(in.load_op),
            .storeOp = static_cast<VkAttachmentStoreOp>(in.store_op),
            .clearValue = in.clear_value.values,
        };
        if (in.resolve.has_value)
        {
            out.resolveMode = static_cast<VkResolveModeFlagBits>(in.resolve.value.mode);
            out.resolveImageView = self->device->slot(in.resolve.value.image).vk_image_view;
            out.resolveImageLayout = std::bit_cast<VkImageLayout>(in.resolve.value.layout);
        }
    };

    std::array<VkRenderingAttachmentInfo, COMMAND_LIST_COLOR_ATTACHMENT_MAX> vk_color_attachments = {};
    for (usize i = 0; i < info->color_attachments.size; ++i)
    {
        if (daxa_dvc_is_image_view_valid(self->device, info->color_attachments.data[i].image_view) == 0)
        {
            return DAXA_RESULT_INVALID_IMAGE_VIEW_ID;
        }
        if (daxa_dvc_is_image_valid(self->device, self->device->slot(info->color_attachments.data[i].image_view).info.image) == 0)
        {
            return DAXA_RESULT_INVALID_IMAGE_ID;
        }
        fill_rendering_attachment_info(info->color_attachments.data[i], vk_color_attachments.at(i));
    }
    VkRenderingAttachmentInfo depth_attachment_info = {};
    if (info->depth_attachment.has_value != 0)
    {
        if (daxa_dvc_is_image_view_valid(self->device, info->depth_attachment.value.image_view) == 0)
        {
            return DAXA_RESULT_INVALID_IMAGE_VIEW_ID;
        }
        if (daxa_dvc_is_image_valid(self->device, self->device->slot(info->depth_attachment.value.image_view).info.image) == 0)
        {
            return DAXA_RESULT_INVALID_IMAGE_ID;
        }
        fill_rendering_attachment_info(info->depth_attachment.value, depth_attachment_info);
    };
    VkRenderingAttachmentInfo stencil_attachment_info = {};
    if (info->stencil_attachment.has_value != 0)
    {
        if (daxa_dvc_is_image_view_valid(self->device, info->stencil_attachment.value.image_view) == 0)
        {
            return DAXA_RESULT_INVALID_IMAGE_VIEW_ID;
        }
        if (daxa_dvc_is_image_valid(self->device, self->device->slot(info->stencil_attachment.value.image_view).info.image) == 0)
        {
            return DAXA_RESULT_INVALID_IMAGE_ID;
        }
        fill_rendering_attachment_info(info->stencil_attachment.value, stencil_attachment_info);
    };
    for (usize i = 0; i < info->color_attachments.size; ++i)
    {
        self->current_command_data.used_image_views.push_back(std::bit_cast<ImageViewId>(info->color_attachments.data[i].image_view));
        self->current_command_data.used_images.push_back(std::bit_cast<ImageId>(self->device->slot(info->color_attachments.data[i].image_view).info.image));
    }
    if (info->depth_attachment.has_value != 0)
    {
        self->current_command_data.used_image_views.push_back(std::bit_cast<ImageViewId>(info->depth_attachment.value.image_view));
        self->current_command_data.used_images.push_back(std::bit_cast<ImageId>(self->device->slot(info->depth_attachment.value.image_view).info.image));
    }
    if (info->stencil_attachment.has_value != 0)
    {
        self->current_command_data.used_image_views.push_back(std::bit_cast<ImageViewId>(info->stencil_attachment.value.image_view));
        self->current_command_data.used_images.push_back(std::bit_cast<ImageId>(self->device->slot(info->stencil_attachment.value.image_view).info.image));
    }

    VkRenderingInfo const vk_rendering_info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
        .pNext = nullptr,
        .flags = {},
        .renderArea = info->render_area,
        .layerCount = 1,
        .viewMask = {},
        .colorAttachmentCount = info->color_attachments.size,
        .pColorAttachments = vk_color_attachments.data(),
        .pDepthAttachment = info->depth_attachment.has_value != 0 ? &depth_attachment_info : nullptr,
        .pStencilAttachment = info->stencil_attachment.has_value != 0 ? &stencil_attachment_info : nullptr,
    };
    vkCmdSetScissor(self->current_command_data.vk_cmd_buffer, 0, 1, reinterpret_cast<VkRect2D const *>(&info->render_area));
    VkViewport const vk_viewport = {
        .x = static_cast<f32>(info->render_area.offset.x),
        .y = static_cast<f32>(info->render_area.offset.y),
        .width = static_cast<f32>(info->render_area.extent.width),
        .height = static_cast<f32>(info->render_area.extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(self->current_command_data.vk_cmd_buffer, 0, 1, &vk_viewport);
    vkCmdBeginRendering(self->current_command_data.vk_cmd_buffer, &vk_rendering_info);
    if (self->device->vkCmdSetRasterizationSamplesEXT != nullptr)
    {
        self->device->vkCmdSetRasterizationSamplesEXT(self->current_command_data.vk_cmd_buffer, VK_SAMPLE_COUNT_1_BIT);
    }
    self->in_renderpass = true;
    return DAXA_RESULT_SUCCESS;
}

void daxa_cmd_end_renderpass(daxa_CommandRecorder self)
{
    daxa_cmd_flush_barriers(self);
    vkCmdEndRendering(self->current_command_data.vk_cmd_buffer);
    self->in_renderpass = false;
}

void daxa_cmd_set_viewport(daxa_CommandRecorder self, VkViewport const * info)
{
    daxa_cmd_flush_barriers(self);
    vkCmdSetViewport(self->current_command_data.vk_cmd_buffer, 0, 1, info);
}

void daxa_cmd_set_scissor(daxa_CommandRecorder self, VkRect2D const * info)
{
    daxa_cmd_flush_barriers(self);
    vkCmdSetScissor(self->current_command_data.vk_cmd_buffer, 0, 1, info);
}

void daxa_cmd_set_depth_bias(daxa_CommandRecorder self, daxa_DepthBiasInfo const * info)
{
    daxa_cmd_flush_barriers(self);
    vkCmdSetDepthBias(self->current_command_data.vk_cmd_buffer, info->constant_factor, info->clamp, info->slope_factor);
}

auto daxa_cmd_set_index_buffer(daxa_CommandRecorder self, daxa_SetIndexBufferInfo const * info) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->buffer)
    vkCmdBindIndexBuffer(self->current_command_data.vk_cmd_buffer, self->device->slot(info->buffer).vk_buffer, info->offset, info->index_type);
    return DAXA_RESULT_SUCCESS;
}

void daxa_cmd_draw(daxa_CommandRecorder self, daxa_DrawInfo const * info)
{
    vkCmdDraw(self->current_command_data.vk_cmd_buffer, info->vertex_count, info->instance_count, info->first_vertex, info->first_instance);
}

void daxa_cmd_draw_indexed(daxa_CommandRecorder self, daxa_DrawIndexedInfo const * info)
{
    vkCmdDrawIndexed(self->current_command_data.vk_cmd_buffer, info->index_count, info->instance_count, info->first_index, info->vertex_offset, info->first_instance);
}

auto daxa_cmd_draw_indirect(daxa_CommandRecorder self, daxa_DrawIndirectInfo const * info) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->indirect_buffer)
    if (info->is_indexed != 0)
    {
        vkCmdDrawIndexedIndirect(
            self->current_command_data.vk_cmd_buffer,
            self->device->slot(info->indirect_buffer).vk_buffer,
            info->indirect_buffer_offset,
            info->draw_count,
            info->draw_command_stride);
    }
    else
    {
        vkCmdDrawIndirect(
            self->current_command_data.vk_cmd_buffer,
            self->device->slot(info->indirect_buffer).vk_buffer,
            info->indirect_buffer_offset,
            info->draw_count,
            info->draw_command_stride);
    }
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_draw_indirect_count(daxa_CommandRecorder self, daxa_DrawIndirectCountInfo const * info) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->indirect_buffer, info->count_buffer)
    if (info->is_indexed != 0)
    {
        vkCmdDrawIndexedIndirectCount(
            self->current_command_data.vk_cmd_buffer,
            self->device->slot(info->indirect_buffer).vk_buffer,
            info->indirect_buffer_offset,
            self->device->slot(info->count_buffer).vk_buffer,
            info->count_buffer_offset,
            info->max_draw_count,
            info->draw_command_stride);
    }
    else
    {
        vkCmdDrawIndirectCount(
            self->current_command_data.vk_cmd_buffer,
            self->device->slot(info->indirect_buffer).vk_buffer,
            info->indirect_buffer_offset,
            self->device->slot(info->count_buffer).vk_buffer,
            info->count_buffer_offset,
            info->max_draw_count,
            info->draw_command_stride);
    }
    return DAXA_RESULT_SUCCESS;
}

void daxa_cmd_draw_mesh_tasks(daxa_CommandRecorder self, uint32_t x, uint32_t y, uint32_t z)
{
    if (self->device->properties.implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_MESH_SHADER)
    {
        self->device->vkCmdDrawMeshTasksEXT(self->current_command_data.vk_cmd_buffer, x, y, z);
    }
}

auto daxa_cmd_draw_mesh_tasks_indirect(daxa_CommandRecorder self, daxa_DrawMeshTasksIndirectInfo const * info) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->indirect_buffer)
    if (self->device->properties.implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_MESH_SHADER)
    {
        self->device->vkCmdDrawMeshTasksIndirectEXT(
            self->current_command_data.vk_cmd_buffer,
            self->device->slot(info->indirect_buffer).vk_buffer,
            info->offset,
            info->draw_count,
            info->stride);
    }
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_draw_mesh_tasks_indirect_count(
    daxa_CommandRecorder self,
    daxa_DrawMeshTasksIndirectCountInfo const * info) -> daxa_Result
{
    DAXA_CHECK_AND_REMEMBER_IDS(self, info->indirect_buffer, info->count_buffer)
    if (self->device->properties.implicit_features & DAXA_IMPLICIT_FEATURE_FLAG_MESH_SHADER)
    {
        self->device->vkCmdDrawMeshTasksIndirectCountEXT(
            self->current_command_data.vk_cmd_buffer,
            self->device->slot(info->indirect_buffer).vk_buffer,
            info->offset,
            self->device->slot(info->count_buffer).vk_buffer,
            info->count_offset,
            info->max_count,
            info->stride);
    }
    return DAXA_RESULT_SUCCESS;
}

void daxa_cmd_write_timestamp(daxa_CommandRecorder self, daxa_WriteTimestampInfo const * info)
{
    daxa_cmd_flush_barriers(self);
    vkCmdWriteTimestamp2(
        self->current_command_data.vk_cmd_buffer,
        info->pipeline_stage,
        (**info->query_pool).vk_timeline_query_pool,
        info->query_index);
}

void daxa_cmd_reset_timestamps(daxa_CommandRecorder self, daxa_ResetTimestampsInfo const * info)
{
    daxa_cmd_flush_barriers(self);
    vkCmdResetQueryPool(
        self->current_command_data.vk_cmd_buffer,
        (**info->query_pool).vk_timeline_query_pool,
        info->start_index,
        info->count);
}

void daxa_cmd_begin_label(daxa_CommandRecorder self, daxa_CommandLabelInfo const * info)
{
    daxa_cmd_flush_barriers(self);
    VkDebugUtilsLabelEXT const vk_debug_label_info{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pNext = {},
        .pLabelName = info->name.data,
        .color = {
            info->label_color.x,
            info->label_color.y,
            info->label_color.z,
            info->label_color.w},
    };

    if ((self->device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE)
    {
        self->device->vkCmdBeginDebugUtilsLabelEXT(self->current_command_data.vk_cmd_buffer, &vk_debug_label_info);
    }
}

void daxa_cmd_end_label(daxa_CommandRecorder self)
{
    daxa_cmd_flush_barriers(self);
    if ((self->device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE)
    {
        self->device->vkCmdEndDebugUtilsLabelEXT(self->current_command_data.vk_cmd_buffer);
    }
}

void daxa_cmd_flush_barriers(daxa_CommandRecorder self)
{
    if (self->memory_barrier_batch_count > 0 || self->image_barrier_batch_count > 0)
    {
        VkDependencyInfo const vk_dependency_info{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .dependencyFlags = {},
            .memoryBarrierCount = static_cast<u32>(self->memory_barrier_batch_count),
            .pMemoryBarriers = self->memory_barrier_batch.data(),
            .bufferMemoryBarrierCount = 0,
            .pBufferMemoryBarriers = nullptr,
            .imageMemoryBarrierCount = static_cast<u32>(self->image_barrier_batch_count),
            .pImageMemoryBarriers = self->image_barrier_batch.data(),
        };

        vkCmdPipelineBarrier2(self->current_command_data.vk_cmd_buffer, &vk_dependency_info);

        self->memory_barrier_batch_count = 0;
        self->image_barrier_batch_count = 0;
    }
}

auto daxa_cmd_complete_current_commands(
    daxa_CommandRecorder self,
    daxa_ExecutableCommandList * out_executable_cmds) -> daxa_Result
{
    daxa_cmd_flush_barriers(self);
    auto vk_result = vkEndCommandBuffer(self->current_command_data.vk_cmd_buffer);
    if (vk_result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(vk_result);
    }
    auto cmd_data = std::move(self->current_command_data);
    auto result = self->generate_new_current_command_data();
    if (result != DAXA_RESULT_SUCCESS)
    {
        self->current_command_data = std::move(cmd_data);
        return result;
    }
    *out_executable_cmds = new daxa_ImplExecutableCommandList{
        .cmd_recorder = self,
        .data = std::move(cmd_data),
    };
    self->current_pipeline = daxa_ImplCommandRecorder::NoPipeline{};
    self->inc_refcnt();
    return DAXA_RESULT_SUCCESS;
}

auto daxa_cmd_info(daxa_CommandRecorder self) -> daxa_CommandRecorderInfo const *
{
    return &self->info;
}

auto daxa_cmd_get_vk_command_buffer(daxa_CommandRecorder self) -> VkCommandBuffer
{
    return self->current_command_data.vk_cmd_buffer;
}

auto daxa_cmd_get_vk_command_pool(daxa_CommandRecorder self) -> VkCommandPool
{
    return self->vk_cmd_pool;
}

void daxa_destroy_command_recorder(daxa_CommandRecorder self)
{
    self->device->gpu_sro_table.lifetime_lock.unlock_shared();
    self->dec_refcnt(
        daxa_ImplCommandRecorder::zero_ref_callback,
        self->device->instance);
}

auto daxa_dvc_create_command_recorder(daxa_Device device, daxa_CommandRecorderInfo const * info, daxa_CommandRecorder * out_cmd_list) -> daxa_Result
{
    VkCommandPool vk_cmd_pool = [&]()
    {
        std::unique_lock lock{device->command_pool_pools[info->queue_family].mtx};
        return device->command_pool_pools[info->queue_family].get(device);
    }();
    auto ret = daxa_ImplCommandRecorder{};
    ret.device = device;
    ret.info = *info;
    ret.vk_cmd_pool = vk_cmd_pool;
    auto result = ret.generate_new_current_command_data();
    if (result != DAXA_RESULT_SUCCESS)
    {
        std::unique_lock lock{device->command_pool_pools[info->queue_family].mtx};
        device->command_pool_pools[info->queue_family].put_back(vk_cmd_pool);
        return result;
    }
    if ((ret.device->instance->info.flags & InstanceFlagBits::DEBUG_UTILS) != InstanceFlagBits::NONE && ret.info.name.size != 0)
    {
        auto cmd_pool_name = ret.info.name;
        VkDebugUtilsObjectNameInfoEXT const cmd_pool_name_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = VK_OBJECT_TYPE_COMMAND_POOL,
            .objectHandle = std::bit_cast<uint64_t>(ret.vk_cmd_pool),
            .pObjectName = cmd_pool_name.data,
        };
        ret.device->vkSetDebugUtilsObjectNameEXT(ret.device->vk_device, &cmd_pool_name_info);
    }
    // TODO(lifetime): Maybe we should have a try lock variant?
    ret.device->gpu_sro_table.lifetime_lock.lock_shared();
    ret.strong_count = 1;
    device->inc_weak_refcnt();
    *out_cmd_list = new daxa_ImplCommandRecorder{};
    **out_cmd_list = std::move(ret);
    return DAXA_RESULT_SUCCESS;
}

auto daxa_executable_commands_inc_refcnt(daxa_ExecutableCommandList self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_executable_commands_dec_refcnt(daxa_ExecutableCommandList self) -> u64
{
    return self->dec_refcnt(
        daxa_ImplExecutableCommandList::zero_ref_callback,
        self->cmd_recorder->device->instance);
}

/// --- End API Functions ---

/// --- Begin Internals ---

void executable_cmd_list_execute_deferred_destructions(daxa_Device device, ExecutableCommandListData & cmd_list)
{
    for (auto [id, index] : cmd_list.deferred_destructions)
    {
        // TODO(lifetime): check these and report errors if these were destroyed too early.
        [[maybe_unused]] daxa_Result _ignore = {};
        switch (index)
        {
        case DEFERRED_DESTRUCTION_BUFFER_INDEX: _ignore = daxa_dvc_destroy_buffer(device, std::bit_cast<daxa_BufferId>(id)); break;
        case DEFERRED_DESTRUCTION_IMAGE_INDEX: _ignore = daxa_dvc_destroy_image(device, std::bit_cast<daxa_ImageId>(id)); break;
        case DEFERRED_DESTRUCTION_IMAGE_VIEW_INDEX: _ignore = daxa_dvc_destroy_image_view(device, std::bit_cast<daxa_ImageViewId>(id)); break;
        case DEFERRED_DESTRUCTION_SAMPLER_INDEX:
            _ignore = daxa_dvc_destroy_sampler(device, std::bit_cast<daxa_SamplerId>(id));
            break;
            // TODO(capi): DO NOT THROW FROM A C FUNCTION
            // default: DAXA_DBG_ASSERT_TRUE_M(false, "unreachable");
        }
    }
    cmd_list.deferred_destructions.clear();
}

auto daxa_ImplCommandRecorder::generate_new_current_command_data() -> daxa_Result
{
    VkCommandBufferAllocateInfo const vk_command_buffer_allocate_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = this->vk_cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    auto vk_result = vkAllocateCommandBuffers(this->device->vk_device, &vk_command_buffer_allocate_info, &this->current_command_data.vk_cmd_buffer);
    if (vk_result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(vk_result);
    }
    VkCommandBufferBeginInfo const vk_command_buffer_begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = {},
    };
    vk_result = vkBeginCommandBuffer(this->current_command_data.vk_cmd_buffer, &vk_command_buffer_begin_info);
    if (vk_result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(vk_result);
    }
    this->allocated_command_buffers.push_back(this->current_command_data.vk_cmd_buffer);
    this->current_command_data.used_buffers.reserve(12);
    this->current_command_data.used_images.reserve(12);
    this->current_command_data.used_image_views.reserve(12);
    this->current_command_data.used_samplers.reserve(12);
    return DAXA_RESULT_SUCCESS;
}

void daxa_ImplCommandRecorder::zero_ref_callback(ImplHandle const * handle)
{
    auto * self = rc_cast<daxa_CommandRecorder>(handle);
    u64 const submit_timeline = self->device->global_submit_timeline.load(std::memory_order::relaxed);
    std::unique_lock const lock{self->device->zombies_mtx};
    executable_cmd_list_execute_deferred_destructions(self->device, self->current_command_data);
    self->device->command_list_zombies.emplace_front(
        submit_timeline,
        CommandRecorderZombie{
            .vk_cmd_pool = self->vk_cmd_pool,
            .allocated_command_buffers = std::move(self->allocated_command_buffers),
        });
    self->device->dec_weak_refcnt(
        &daxa_ImplDevice::zero_ref_callback,
        self->device->instance);
    delete self;
}

void daxa_ImplExecutableCommandList::zero_ref_callback(ImplHandle const * handle)
{
    auto * self = rc_cast<daxa_ExecutableCommandList>(handle);
    executable_cmd_list_execute_deferred_destructions(self->cmd_recorder->device, self->data);
    self->cmd_recorder->dec_refcnt(
        daxa_ImplCommandRecorder::zero_ref_callback,
        self->cmd_recorder->device->instance);
    delete self;
}

// --- End Internals ---