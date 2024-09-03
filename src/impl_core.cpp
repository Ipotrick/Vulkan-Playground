#include "impl_core.hpp"

#include "impl_instance.hpp"
#include "impl_device.hpp"

// --- Begin Helpers ---

auto is_depth_format(Format format) -> bool
{
    switch (format)
    {
    case Format::D16_UNORM: return true;
    case Format::X8_D24_UNORM_PACK32: return true;
    case Format::D32_SFLOAT: return true;
    case Format::S8_UINT: return true;
    case Format::D16_UNORM_S8_UINT: return true;
    case Format::D24_UNORM_S8_UINT: return true;
    case Format::D32_SFLOAT_S8_UINT: return true;
    default: return false;
    }
}

auto is_stencil_format(Format format) -> bool
{
    switch (format)
    {
    case Format::S8_UINT: return true;
    case Format::D16_UNORM_S8_UINT: return true;
    case Format::D24_UNORM_S8_UINT: return true;
    case Format::D32_SFLOAT_S8_UINT: return true;
    default: return false;
    }
}

auto infer_aspect_from_format(Format format) -> VkImageAspectFlags
{
    if (is_depth_format(format) || is_stencil_format(format))
    {
        return (is_depth_format(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (is_stencil_format(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    }
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

auto make_subresource_range(ImageMipArraySlice const & slice, VkImageAspectFlags aspect) -> VkImageSubresourceRange
{
    return VkImageSubresourceRange{
        .aspectMask = aspect,
        .baseMipLevel = slice.base_mip_level,
        .levelCount = slice.level_count,
        .baseArrayLayer = slice.base_array_layer,
        .layerCount = slice.layer_count,
    };
}

auto make_subresource_layers(ImageArraySlice const & slice, VkImageAspectFlags aspect) -> VkImageSubresourceLayers
{
    return VkImageSubresourceLayers{
        .aspectMask = aspect,
        .mipLevel = slice.mip_level,
        .baseArrayLayer = slice.base_array_layer,
        .layerCount = slice.layer_count,
    };
}
auto create_surface(daxa_Instance instance, daxa_NativeWindowHandle handle, [[maybe_unused]] daxa_NativeWindowPlatform platform, VkSurfaceKHR * out_surface) -> daxa_Result
{
#if defined(_WIN32)
    VkWin32SurfaceCreateInfoKHR const surface_ci{
        .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .hinstance = GetModuleHandleA(nullptr),
        .hwnd = static_cast<HWND>(handle),
    };
    {
        auto func = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(vkGetInstanceProcAddr(instance->vk_instance, "vkCreateWin32SurfaceKHR"));
        VkResult const vk_result = func(instance->vk_instance, &surface_ci, nullptr, out_surface);
        return std::bit_cast<daxa_Result>(vk_result);
    }
#elif defined(__linux__)
    switch (std::bit_cast<daxa::NativeWindowPlatform>(platform))
    {
#if DAXA_BUILT_WITH_WAYLAND
    case NativeWindowPlatform::WAYLAND_API:
    {
        // TODO(grundlett): figure out how to link Wayland
        VkWaylandSurfaceCreateInfoKHR surface_ci{
            .sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
            .pNext = nullptr,
            .flags = 0,
            .display = wl_display_connect(nullptr),
            .surface = static_cast<wl_surface *>(handle),
        };
        {
            auto func = reinterpret_cast<PFN_vkCreateWaylandSurfaceKHR>(vkGetInstanceProcAddr(instance->vk_instance, "vkCreateWaylandSurfaceKHR"));
            VkResult vk_result = func(instance->vk_instance, &surface_ci, nullptr, out_surface);
            return std::bit_cast<daxa_Result>(vk_result);
        }
    }
    break;
#endif
#if DAXA_BUILT_WITH_X11
    case NativeWindowPlatform::XLIB_API:
    default:
    {
        VkXlibSurfaceCreateInfoKHR surface_ci{
            .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
            .pNext = nullptr,
            .flags = 0,
            .dpy = XOpenDisplay(nullptr),
            .window = reinterpret_cast<Window>(handle),
        };
        {
            auto func = reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(vkGetInstanceProcAddr(instance->vk_instance, "vkCreateXlibSurfaceKHR"));
            VkResult vk_result = func(instance->vk_instance, &surface_ci, nullptr, out_surface);
            return std::bit_cast<daxa_Result>(vk_result);
        }
    }
    break;
    }
#endif
#endif
}

#define DAXA_ASSIGN_ARRAY_3(SRC) \
    & { (SRC)[0], (SRC)[1], (SRC)[3] }

auto construct_daxa_physical_device_properties(VkPhysicalDevice physical_device) -> daxa_DeviceProperties
{
    daxa_DeviceProperties ret = {};

    bool ray_tracing_pipeline_supported = false;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR vk_physical_device_ray_tracing_pipeline_properties_khr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
        .pNext = nullptr,
    };

    bool acceleration_structure_supported = false;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR vk_physical_device_acceleration_structure_properties_khr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
        .pNext = nullptr,
    };

    bool invocation_reorder_supported = false;
    VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV vk_physical_device_ray_tracing_invocation_reorder_properties_nv = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV,
        .pNext = nullptr,
    };

    bool mesh_shader_supported = false;
    VkPhysicalDeviceMeshShaderPropertiesEXT vk_physical_device_mesh_shader_properties_ext = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT,
        .pNext = nullptr,
    };

    void * pNextChain = nullptr;

    u32 count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, extensions.data());
    for (auto & extension : extensions)
    {
        if (std::strcmp(extension.extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0)
        {
            ray_tracing_pipeline_supported = true;
            vk_physical_device_ray_tracing_pipeline_properties_khr.pNext = pNextChain;
            pNextChain = &vk_physical_device_ray_tracing_pipeline_properties_khr;
        }
        if (std::strcmp(extension.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0)
        {
            acceleration_structure_supported = true;
            vk_physical_device_acceleration_structure_properties_khr.pNext = pNextChain;
            pNextChain = &vk_physical_device_acceleration_structure_properties_khr;
        }
        if (std::strcmp(extension.extensionName, VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME) == 0)
        {
            invocation_reorder_supported = true;
            vk_physical_device_ray_tracing_invocation_reorder_properties_nv.pNext = pNextChain;
            pNextChain = &vk_physical_device_ray_tracing_invocation_reorder_properties_nv;
        }
        if (std::strcmp(extension.extensionName, VK_EXT_MESH_SHADER_EXTENSION_NAME) == 0)
        {
            mesh_shader_supported = true;
            vk_physical_device_mesh_shader_properties_ext.pNext = pNextChain;
            pNextChain = &vk_physical_device_mesh_shader_properties_ext;
        }
    }

    VkPhysicalDeviceProperties2 vk_physical_device_properties2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = pNextChain,
    };

    vkGetPhysicalDeviceProperties2(physical_device, &vk_physical_device_properties2);
    // physical device properties are ABI compatible UP TO the mesh_shader_properties field.
    std::memcpy(
        &ret,
        r_cast<std::byte const *>(&vk_physical_device_properties2) + sizeof(void *) * 2 /* skip sType and pNext */,
        offsetof(daxa_DeviceProperties, mesh_shader_properties));
    if (ray_tracing_pipeline_supported)
    {
        ret.ray_tracing_pipeline_properties.has_value = 1;
        std::memcpy(
            &ret.ray_tracing_pipeline_properties.value,
            r_cast<std::byte const *>(&vk_physical_device_ray_tracing_pipeline_properties_khr) + sizeof(void *) * 2, // skip sType and pNext
            sizeof(daxa_RayTracingPipelineProperties));
    }
    if (acceleration_structure_supported)
    {
        ret.acceleration_structure_properties.has_value = 1;
        std::memcpy(
            &ret.acceleration_structure_properties.value,
            r_cast<std::byte const *>(&vk_physical_device_acceleration_structure_properties_khr) + sizeof(void *) * 2, // skip sType and pNext
            sizeof(daxa_AccelerationStructureProperties));
    }
    if (invocation_reorder_supported)
    {
        ret.ray_tracing_invocation_reorder_properties.has_value = 1;
        std::memcpy(
            &ret.ray_tracing_invocation_reorder_properties.value,
            r_cast<std::byte const *>(&vk_physical_device_ray_tracing_invocation_reorder_properties_nv) + sizeof(void *) * 2, // skip sType and pNext
            sizeof(daxa_RayTracingInvocationReorderProperties));
    }
    if (mesh_shader_supported)
    {
        ret.mesh_shader_properties.has_value = 1;
        std::memcpy(
            &ret.mesh_shader_properties.value,
            r_cast<std::byte const *>(&vk_physical_device_mesh_shader_properties_ext) + sizeof(void *) * 2, // skip sType and pNext
            sizeof(daxa_MeshShaderProperties));
        ret.mesh_shader_properties.value.prefers_local_invocation_vertex_output = static_cast<daxa_Bool8>(vk_physical_device_mesh_shader_properties_ext.prefersLocalInvocationVertexOutput);
        ret.mesh_shader_properties.value.prefers_local_invocation_primitive_output = static_cast<daxa_Bool8>(vk_physical_device_mesh_shader_properties_ext.prefersLocalInvocationPrimitiveOutput);
        ret.mesh_shader_properties.value.prefers_compact_vertex_output = static_cast<daxa_Bool8>(vk_physical_device_mesh_shader_properties_ext.prefersCompactVertexOutput);
        ret.mesh_shader_properties.value.prefers_compact_primitive_output = static_cast<daxa_Bool8>(vk_physical_device_mesh_shader_properties_ext.prefersCompactPrimitiveOutput);
    }

    u32 queue_family_props_count = 0;
    std::vector<VkQueueFamilyProperties> queue_props;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_props_count, nullptr);
    queue_props.resize(queue_family_props_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_props_count, queue_props.data());
    std::vector<VkBool32> supports_present;
    supports_present.resize(queue_family_props_count);

    ret.compute_queue_count = ~0u;
    ret.transfer_queue_count = ~0u;
    for (u32 i = 0; i < queue_family_props_count; i++)
    {
        bool const supports_graphics = queue_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;
        bool const supports_compute = queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT;
        bool const supports_transfer = queue_props[i].queueFlags & VK_QUEUE_TRANSFER_BIT;
        if (ret.compute_queue_count == ~0u && !supports_graphics && supports_compute && supports_transfer)
        {
            ret.compute_queue_count = std::min(queue_props[i].queueCount, DAXA_MAX_COMPUTE_QUEUE_COUNT);
        }
        if (ret.transfer_queue_count == ~0u && !supports_graphics && !supports_compute && supports_transfer)
        {
            ret.transfer_queue_count = std::min(queue_props[i].queueCount, DAXA_MAX_TRANSFER_QUEUE_COUNT);
        }
    }
    if (ret.compute_queue_count == ~0u)
    {
        ret.compute_queue_count = 0u;
    }
    if (ret.transfer_queue_count == ~0u)
    {
        ret.transfer_queue_count = 0u;
    }
    
    return ret;
}

auto mask_from_bit_count(u64 bits) -> u64
{
    return (1ull << bits) - 1;
}

void daxa_as_build_info_to_vk(
    daxa_Device device,
    daxa_TlasBuildInfo const * tlas_infos,
    usize tlas_count,
    daxa_BlasBuildInfo const * blas_infos,
    usize blas_count,
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> & vk_build_geometry_infos,
    std::vector<VkAccelerationStructureGeometryKHR> & vk_geometry_infos,
    std::vector<u32> & primitive_counts,
    std::vector<u32 const *> & primitive_counts_ptrs)
{
    vk_build_geometry_infos.reserve(tlas_count + blas_count);
    primitive_counts.reserve(tlas_count + blas_count);
    size_t geo_infos_count = 0;
    for (u32 tlas_i = 0; tlas_i < tlas_count; ++tlas_i)
    {
        geo_infos_count += tlas_infos[tlas_i].instance_count;
    }
    for (u32 blas_i = 0; blas_i < blas_count; ++blas_i)
    {
        // As both variants are spans and ABI compatible, we can just unconditionaally read one of the variants here.
        geo_infos_count += blas_infos[blas_i].geometries.values.triangles.count;
    }
    vk_geometry_infos.reserve(geo_infos_count);
    primitive_counts.reserve(geo_infos_count);
    for (u32 tlas_i = 0; tlas_i < tlas_count; ++tlas_i)
    {
        daxa_TlasBuildInfo const & info = tlas_infos[tlas_i];
        VkAccelerationStructureGeometryKHR const * vk_geo_array_ptr = vk_geometry_infos.data() + vk_geometry_infos.size();
        u32 const * primitive_counts_ptr = primitive_counts.data() + primitive_counts.size();
        for (u32 inst_i = 0; inst_i < info.instance_count; ++inst_i)
        {
            daxa_TlasInstanceInfo const & inst_info = info.instances[inst_i];
            VkAccelerationStructureGeometryInstancesDataKHR const vk_inst_data = {
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                .pNext = nullptr,
                .arrayOfPointers = static_cast<VkBool32>(inst_info.is_data_array_of_pointers),
                .data = std::bit_cast<VkDeviceOrHostAddressConstKHR>(inst_info.data),
            };
            vk_geometry_infos.push_back(VkAccelerationStructureGeometryKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .pNext = nullptr,
                .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                .geometry = VkAccelerationStructureGeometryDataKHR{
                    .instances = vk_inst_data,
                },
                .flags = std::bit_cast<VkGeometryFlagsKHR>(inst_info.flags),
            });
            primitive_counts.push_back(inst_info.count);
        }
        vk_build_geometry_infos.push_back({
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .pNext = nullptr,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            .flags = static_cast<VkBuildAccelerationStructureFlagsKHR>(info.flags),
            .mode = info.update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR  : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .srcAccelerationStructure = info.src_tlas.value != 0
                    ? device->slot(info.src_tlas).vk_acceleration_structure
                    : nullptr,
            .dstAccelerationStructure =
                info.dst_tlas.value != 0
                    ? device->slot(info.dst_tlas).vk_acceleration_structure
                    : nullptr,
            .geometryCount = info.instance_count,
            .pGeometries = vk_geo_array_ptr,
            .ppGeometries = nullptr,
            .scratchData = std::bit_cast<VkDeviceOrHostAddressKHR>(info.scratch_data),
        });
        primitive_counts_ptrs.push_back(primitive_counts_ptr);
    }
    for (u32 blas_i = 0; blas_i < blas_count; ++blas_i)
    {
        daxa_BlasBuildInfo const & info = blas_infos[blas_i];
        VkAccelerationStructureGeometryKHR const * vk_geo_array_ptr = vk_geometry_infos.data() + vk_geometry_infos.size();
        u32 const * primitive_counts_ptr = primitive_counts.data() + primitive_counts.size();
        // As both variants are spans and ABI compatible, we can just unconditionally read one of the variants here.
        u32 const geo_count = static_cast<u32>(info.geometries.values.triangles.count);
        for (u32 geo_i = 0; geo_i < geo_count; ++geo_i)
        {
            auto geo_info = VkAccelerationStructureGeometryKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .pNext = nullptr,
            };
            if (info.geometries.index == 0) // triangles
            {
                geo_info.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                geo_info.flags = std::bit_cast<VkGeometryTypeKHR>(info.geometries.values.triangles.triangles[geo_i].flags);
                geo_info.geometry.triangles = VkAccelerationStructureGeometryTrianglesDataKHR{
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                    .pNext = nullptr,
                    .vertexFormat = info.geometries.values.triangles.triangles[geo_i].vertex_format,
                    .vertexData = std::bit_cast<VkDeviceOrHostAddressConstKHR>(info.geometries.values.triangles.triangles[geo_i].vertex_data),
                    .vertexStride = info.geometries.values.triangles.triangles[geo_i].vertex_stride,
                    .maxVertex = info.geometries.values.triangles.triangles[geo_i].max_vertex,
                    .indexType = info.geometries.values.triangles.triangles[geo_i].index_type,
                    .indexData = std::bit_cast<VkDeviceOrHostAddressConstKHR>(info.geometries.values.triangles.triangles[geo_i].index_data),
                    .transformData = std::bit_cast<VkDeviceOrHostAddressConstKHR>(info.geometries.values.triangles.triangles[geo_i].transform_data),
                };
                primitive_counts.push_back(info.geometries.values.triangles.triangles[geo_i].count);
            }
            else // aabbs
            {
                geo_info.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
                geo_info.flags = static_cast<VkGeometryFlagsKHR>(info.geometries.values.aabbs.aabbs[geo_i].flags);
                geo_info.geometry.aabbs = VkAccelerationStructureGeometryAabbsDataKHR{
                    .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
                    .pNext = nullptr,
                    .data = std::bit_cast<VkDeviceOrHostAddressConstKHR>(info.geometries.values.aabbs.aabbs[geo_i].data),
                    .stride = info.geometries.values.aabbs.aabbs[geo_i].stride,
                };
                primitive_counts.push_back(info.geometries.values.aabbs.aabbs[geo_i].count);
            }
            vk_geometry_infos.push_back(geo_info);
        }
        vk_build_geometry_infos.push_back({
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .pNext = nullptr,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .flags = static_cast<VkBuildAccelerationStructureFlagsKHR>(info.flags),
            .mode = info.update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR  : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .srcAccelerationStructure = info.src_blas.value != 0
                    ? device->slot(info.src_blas).vk_acceleration_structure
                    : nullptr,
            .dstAccelerationStructure =
                info.dst_blas.value != 0
                    ? device->slot(info.dst_blas).vk_acceleration_structure
                    : nullptr,
            .geometryCount = geo_count,
            .pGeometries = vk_geo_array_ptr,
            .ppGeometries = nullptr,
            .scratchData = std::bit_cast<VkDeviceOrHostAddressKHR>(info.scratch_data),
        });
        primitive_counts_ptrs.push_back(primitive_counts_ptr);
    }
}

// --- End Helpers ---

// --- Begin API Functions

auto daxa_default_view(daxa_ImageId id) -> daxa_ImageViewId
{
    return daxa_ImageViewId{.value = id.value};
}

auto daxa_index_of_buffer(daxa_BufferId id) -> u32
{
    return static_cast<u32>(id.value & mask_from_bit_count(DAXA_ID_INDEX_BITS));
}

auto daxa_index_of_image(daxa_ImageId id) -> u32
{
    return static_cast<u32>(id.value & mask_from_bit_count(DAXA_ID_INDEX_BITS));
}

auto daxa_index_of_image_view(daxa_ImageViewId id) -> u32
{
    return static_cast<u32>(id.value & mask_from_bit_count(DAXA_ID_INDEX_BITS));
}

auto daxa_index_of_sampler(daxa_SamplerId id) -> u32
{
    return static_cast<u32>(id.value & mask_from_bit_count(DAXA_ID_INDEX_BITS));
}

auto daxa_version_of_buffer(daxa_BufferId id) -> u64
{
    return (id.value >> DAXA_ID_VERSION_OFFSET) & mask_from_bit_count(DAXA_ID_VERSION_BITS);
}

auto daxa_version_of_image(daxa_ImageId id) -> u64
{
    return (id.value >> DAXA_ID_VERSION_OFFSET) & mask_from_bit_count(DAXA_ID_VERSION_BITS);
}

auto daxa_version_of_image_view(daxa_ImageViewId id) -> u64
{
    return (id.value >> DAXA_ID_VERSION_OFFSET) & mask_from_bit_count(DAXA_ID_VERSION_BITS);
}

auto daxa_version_of_sampler(daxa_SamplerId id) -> u64
{
    return (id.value >> DAXA_ID_VERSION_OFFSET) & mask_from_bit_count(DAXA_ID_VERSION_BITS);
}

// --- End API Functions

// --- Begin ImplHandle ---

auto ImplHandle::inc_refcnt() const -> u64
{
    auto & mut_strong_ref = *rc_cast<u64 *>(&this->strong_count);
    return std::atomic_ref{mut_strong_ref}.fetch_add(1, std::memory_order::relaxed);
}

auto ImplHandle::dec_refcnt(void (*zero_ref_callback)(ImplHandle const *), daxa_Instance instance) const -> u64
{
    auto & mut_strong_ref = *rc_cast<u64 *>(&this->strong_count);
    auto prev = std::atomic_ref{mut_strong_ref}.fetch_sub(1, std::memory_order::relaxed);
    if (prev == 1)
    {
        auto weak = this->get_weak_refcnt();
        if (weak == 0)
        {
            zero_ref_callback(this);
        }
        else if (instance != nullptr && (instance->info.flags & InstanceFlagBits::PARENT_MUST_OUTLIVE_CHILD) != InstanceFlagBits::NONE)
        {
            DAXA_DBG_ASSERT_TRUE_M(false, "not all children have been destroyed prior to destroying object");
        }
    }
    return prev;
}

auto ImplHandle::get_refcnt() const -> u64
{
    return std::atomic_ref{this->strong_count}.load(std::memory_order::relaxed);
}

auto ImplHandle::impl_inc_weak_refcnt([[maybe_unused]] char const * callsite) const -> u64
{
    _DAXA_TEST_PRINT("called \"inc_weak_refcnt\" in \"%s\"\n", callsite);
    auto & mut_weak_ref = *rc_cast<u64 *>(&this->weak_count);
    return std::atomic_ref{mut_weak_ref}.fetch_add(1, std::memory_order::relaxed);
}

auto ImplHandle::impl_dec_weak_refcnt(void (*zero_ref_callback)(ImplHandle const *), daxa_Instance /*unused*/, [[maybe_unused]] char const * callsite) const -> u64
{
    _DAXA_TEST_PRINT("called \"dec_weak_refcnt\" in \"%s\"\n", callsite);
    auto & mut_weak_ref = *rc_cast<u64 *>(&this->weak_count);
    auto prev = std::atomic_ref{mut_weak_ref}.fetch_sub(1, std::memory_order::relaxed);
    if (prev == 1)
    {
        auto strong = this->get_refcnt();
        if (strong == 0)
        {
            zero_ref_callback(this);
        }
    }
    return prev;
}

auto ImplHandle::get_weak_refcnt() const -> u64
{
    return std::atomic_ref{this->weak_count}.load(std::memory_order::relaxed);
}

// --- End ImplHandle ---

// --- Begin daxa_ImplMemoryBlock ---

auto daxa_dvc_create_memory(daxa_Device self, daxa_MemoryBlockInfo const * info, daxa_MemoryBlock * out_memory_block) -> daxa_Result
{
    daxa_ImplMemoryBlock ret = {};
    ret.device = self;
    ret.info = std::bit_cast<daxa::MemoryBlockInfo>(*info);

    if (info->requirements.memoryTypeBits == 0)
    {
        // TODO(capi): This should not be here, the point is to return an error!
        // DAXA_DBG_ASSERT_TRUE_M(false, "memory_type_bits must be non zero");
        return DAXA_RESULT_ERROR_UNKNOWN;
    }

    VmaAllocationCreateInfo const create_info{
        .flags = info->flags,
        .usage = VMA_MEMORY_USAGE_GPU_ONLY,
        .requiredFlags = {}, // TODO: idk what this is...
        .preferredFlags = {},
        .memoryTypeBits = {}, // TODO: idk what this is....
        .pool = {},
        .pUserData = {},
        .priority = 0.5f,
    };
    auto result = vmaAllocateMemory(self->vma_allocator, &info->requirements, &create_info, &ret.allocation, &ret.alloc_info);
    if (result != VK_SUCCESS)
    {
        return std::bit_cast<daxa_Result>(result);
    }

    ret.strong_count = 1;
    self->inc_weak_refcnt();
    *out_memory_block = new daxa_ImplMemoryBlock{};
    // TODO(general): memory block is missing a name.
    **out_memory_block = ret;
    return DAXA_RESULT_SUCCESS;
}

auto daxa_memory_block_info(daxa_MemoryBlock self) -> daxa_MemoryBlockInfo const *
{
    return r_cast<daxa_MemoryBlockInfo const *>(&self->info);
}

auto daxa_memory_block_inc_refcnt(daxa_MemoryBlock self) -> u64
{
    return self->inc_refcnt();
}

auto daxa_memory_block_dec_refcnt(daxa_MemoryBlock self) -> u64
{
    return self->dec_refcnt(
        &daxa_ImplMemoryBlock::zero_ref_callback,
        self->device->instance);
}

void daxa_ImplMemoryBlock::zero_ref_callback(ImplHandle const * handle)
{
    auto * self = rc_cast<daxa_ImplMemoryBlock *>(handle);
    std::unique_lock const lock{self->device->zombies_mtx};
    u64 const submit_timeline_value = self->device->global_submit_timeline.load(std::memory_order::relaxed);
    self->device->memory_block_zombies.emplace_front(
        submit_timeline_value,
        MemoryBlockZombie{
            .allocation = self->allocation,
        });
    self->device->dec_weak_refcnt(
        daxa_ImplDevice::zero_ref_callback,
        self->device->instance);
    delete self;
}

// --- End daxa_ImplMemoryBlock ---
