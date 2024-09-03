#ifndef __DAXA_GPU_RESOURCES_H__
#define __DAXA_GPU_RESOURCES_H__

#include "types.h"
#include <vk_mem_alloc.h>

#include <daxa/c/types.h>

DAXA_EXPORT daxa_ImageViewId
daxa_default_view(daxa_ImageId image);

DAXA_EXPORT uint32_t
daxa_index_of_buffer(daxa_BufferId id);
DAXA_EXPORT uint32_t
daxa_index_of_image(daxa_ImageId id);
DAXA_EXPORT uint32_t
daxa_index_of_image_view(daxa_ImageViewId id);
DAXA_EXPORT uint32_t
daxa_index_of_sampler(daxa_SamplerId id);

DAXA_EXPORT uint64_t
daxa_version_of_buffer(daxa_BufferId id);
DAXA_EXPORT uint64_t
daxa_version_of_image(daxa_ImageId id);
DAXA_EXPORT uint64_t
daxa_version_of_image_view(daxa_ImageViewId id);
DAXA_EXPORT uint64_t
daxa_version_of_sampler(daxa_SamplerId id);
typedef struct
{
    uint64_t address;
} daxa_DeviceAddress;

typedef struct
{
    size_t size;
    // Ignored when allocating with a memory block.
    daxa_MemoryFlags allocate_info;
    daxa_SmallString name;
} daxa_BufferInfo;

typedef uint32_t daxa_ImageFlags;
static daxa_ImageFlags const DAXA_IMAGE_FLAG_NONE = 0x00000000;
static daxa_ImageFlags const DAXA_IMAGE_FLAG_ALLOW_MUTABLE_FORMAT = 0x00000008;
static daxa_ImageFlags const DAXA_IMAGE_FLAG_COMPATIBLE_CUBE = 0x00000010;
static daxa_ImageFlags const DAXA_IMAGE_FLAG_COMPATIBLE_2D_ARRAY = 0x00000020;
static daxa_ImageFlags const DAXA_IMAGE_FLAG_ALLOW_ALIAS = 0x00000400;

typedef uint32_t daxa_ImageUsageFlags;
static daxa_ImageUsageFlags const DAXA_IMAGE_USE_FLAG_NONE = 0x00000000;
static daxa_ImageUsageFlags const DAXA_IMAGE_USE_FLAG_TRANSFER_SRC = 0x00000001;
static daxa_ImageUsageFlags const DAXA_IMAGE_USE_FLAG_TRANSFER_DST = 0x00000002;
static daxa_ImageUsageFlags const DAXA_IMAGE_USE_FLAG_SHADER_SAMPLED = 0x00000004;
static daxa_ImageUsageFlags const DAXA_IMAGE_USE_FLAG_SHADER_STORAGE = 0x00000008;
static daxa_ImageUsageFlags const DAXA_IMAGE_USE_FLAG_COLOR_ATTACHMENT = 0x00000010;
static daxa_ImageUsageFlags const DAXA_IMAGE_USE_FLAG_DEPTH_STENCIL_ATTACHMENT = 0x00000020;
static daxa_ImageUsageFlags const DAXA_IMAGE_USE_FLAG_TRANSIENT_ATTACHMENT = 0x00000040;
static daxa_ImageUsageFlags const DAXA_IMAGE_USE_FLAG_FRAGMENT_DENSITY_MAP = 0x00000200;
static daxa_ImageUsageFlags const DAXA_IMAGE_USE_FLAG_FRAGMENT_SHADING_RATE_ATTACHMENT = 0x00000100;

typedef enum
{
    DAXA_SHARING_MODE_EXCLUSIVE,
    DAXA_SHARING_MODE_CONCURRENT,
} daxa_SharingMode;

typedef struct
{
    daxa_ImageFlags flags;
    uint32_t dimensions;
    VkFormat format;
    VkExtent3D size;
    uint32_t mip_level_count;
    uint32_t array_layer_count;
    uint32_t sample_count;
    daxa_ImageUsageFlags usage;
    daxa_SharingMode sharing_mode;
    // Ignored when allocating with a memory block.
    daxa_MemoryFlags allocate_info;
    daxa_SmallString name;
} daxa_ImageInfo;

typedef struct
{
    VkImageViewType type;
    VkFormat format;
    daxa_ImageId image;
    daxa_ImageMipArraySlice slice;
    daxa_SmallString name;
} daxa_ImageViewInfo;

typedef struct
{
    VkFilter magnification_filter;
    VkFilter minification_filter;
    VkFilter mipmap_filter;
    VkSamplerReductionMode reduction_mode;
    VkSamplerAddressMode address_mode_u;
    VkSamplerAddressMode address_mode_v;
    VkSamplerAddressMode address_mode_w;
    float mip_lod_bias;
    daxa_Bool8 enable_anisotropy;
    float max_anisotropy;
    daxa_Bool8 enable_compare;
    VkCompareOp compare_op;
    float min_lod;
    float max_lod;
    VkBorderColor border_color;
    daxa_Bool8 enable_unnormalized_coordinates;
    daxa_SmallString name;
} daxa_SamplerInfo;

static daxa_BufferInfo const DAXA_DEFAULT_BUFFER_INFO = {
    .size = 0,
    .allocate_info = DAXA_MEMORY_FLAG_NONE,
    .name = {.data = DAXA_ZERO_INIT, .size = 0},
};
static daxa_ImageInfo const DAXA_DEFAULT_IMAGE_INFO = {
    .flags = 0,
    .dimensions = 2,
    .format = VK_FORMAT_R8G8B8A8_SRGB,
    .size = {0, 0, 0},
    .mip_level_count = 1,
    .array_layer_count = 1,
    .sample_count = 1,
    .usage = 0,
    .sharing_mode = DAXA_SHARING_MODE_EXCLUSIVE,
    .allocate_info = DAXA_MEMORY_FLAG_NONE,
    .name = {.data = DAXA_ZERO_INIT, .size = 0},
};
static daxa_ImageViewInfo const DAXA_DEFAULT_IMAGE_VIEW_INFO = {
    .type = VK_IMAGE_VIEW_TYPE_2D,
    .format = VK_FORMAT_R8G8B8A8_SRGB,
    .name = {.data = DAXA_ZERO_INIT, .size = 0},
};
static daxa_SamplerInfo const DAXA_DEFAULT_SAMPLER_INFO = {
    .magnification_filter = VK_FILTER_LINEAR,
    .minification_filter = VK_FILTER_LINEAR,
    .mipmap_filter = VK_FILTER_LINEAR,
    .reduction_mode = VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
    .address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    .address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    .address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    .mip_lod_bias = 0.5f,
    .enable_anisotropy = 0,
    .max_anisotropy = 0.0f,
    .enable_compare = 0,
    .compare_op = VK_COMPARE_OP_ALWAYS,
    .min_lod = 0.0f,
    .max_lod = 1000.0f,
    .border_color = VK_BORDER_COLOR_INT_TRANSPARENT_BLACK,
    .enable_unnormalized_coordinates = 0,
    .name = {.data = DAXA_ZERO_INIT, .size = 0},
};

DAXA_EXPORT VmaAllocation
daxa_memory_block_get_vma_allocation(daxa_MemoryBlock memory_block);

typedef enum
{
    DAXA_GEOMETRY_OPAQUE = 0x1 << 0,
    DAXA_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION = 0x1 << 1,
} daxa_GeometryFlagBits;

typedef int32_t daxa_GeometryFlags;

typedef struct
{
    VkFormat vertex_format;
    daxa_DeviceAddress vertex_data;
    uint64_t vertex_stride;
    uint32_t max_vertex;
    VkIndexType index_type;
    daxa_DeviceAddress index_data;
    daxa_DeviceAddress transform_data;
    uint32_t count;
    daxa_GeometryFlags flags;
} daxa_BlasTriangleGeometryInfo;

static daxa_BlasTriangleGeometryInfo const DAXA_DEFAULT_BLAS_TRIANGLE_GEPMETRY_INFO = {
    .vertex_format = VK_FORMAT_R32G32B32_SFLOAT,
    .vertex_data = DAXA_ZERO_INIT,
    .vertex_stride = 24,
    .max_vertex = 0,
    .index_type = VK_INDEX_TYPE_UINT32,
    .index_data = DAXA_ZERO_INIT,
    .transform_data = DAXA_ZERO_INIT,
    .count = 0,
    .flags = DAXA_GEOMETRY_OPAQUE,
};

typedef struct
{
    daxa_DeviceAddress data;
    uint64_t stride;
    uint32_t count;
    daxa_GeometryFlags flags;
} daxa_BlasAabbGeometryInfo;

static daxa_BlasAabbGeometryInfo const DAXA_DEFAULT_BLAS_AABB_GEOMETRY_INFO = {
    .data = DAXA_ZERO_INIT,
    .stride = 24,
    .count = 0,
    .flags = DAXA_GEOMETRY_OPAQUE,
};

/// Instances are defines as VkAccelerationStructureInstanceKHR;
typedef struct
{
    daxa_DeviceAddress data;
    uint32_t count;
    daxa_Bool8 is_data_array_of_pointers;
    daxa_GeometryFlags flags;
} daxa_TlasInstanceInfo;

static daxa_TlasInstanceInfo const DAXA_DEFAULT_TLAS_INSTANCE_INFO = {
    .data = DAXA_ZERO_INIT,
    .count = 0,
    .is_data_array_of_pointers = 0,
    .flags = DAXA_GEOMETRY_OPAQUE,
};

typedef struct
{
    daxa_BlasTriangleGeometryInfo const * triangles;
    size_t count;
} daxa_BlasTriangleGeometryInfoSpan;

typedef struct
{
    daxa_BlasAabbGeometryInfo const * aabbs;
    size_t count;
} daxa_BlasAabbsGeometryInfoSpan;

typedef union
{
    daxa_BlasTriangleGeometryInfoSpan triangles;
    daxa_BlasAabbsGeometryInfoSpan aabbs;
} daxa_BlasGeometryInfoSpansUnion;

typedef enum
{
    DAXA_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE = 0x00000001,
    DAXA_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION = 0x00000002,
    DAXA_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE = 0x00000004,
    DAXA_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD = 0x00000008,
    DAXA_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY = 0x00000010,
} daxa_BuildAccelerationStructureFlagBits;
typedef uint32_t daxa_BuildAcclelerationStructureFlags;

typedef struct
{
    daxa_BuildAcclelerationStructureFlags flags;
    daxa_Bool8 update;
    daxa_TlasId src_tlas;
    daxa_TlasId dst_tlas;
    daxa_TlasInstanceInfo const * instances;
    uint32_t instance_count;
    daxa_DeviceAddress scratch_data;
} daxa_TlasBuildInfo;

static daxa_TlasBuildInfo const DAXA_DEFAULT_TLAS_BUILD_INFO = {
    .flags = DAXA_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE,
    .update = 0,
    .src_tlas = DAXA_ZERO_INIT,
    .dst_tlas = DAXA_ZERO_INIT,
    .instances = DAXA_ZERO_INIT,
    .instance_count = DAXA_ZERO_INIT,
    .scratch_data = DAXA_ZERO_INIT,
};

typedef struct
{
    daxa_BuildAcclelerationStructureFlags flags;
    daxa_Bool8 update;
    daxa_BlasId src_blas;
    daxa_BlasId dst_blas;
    daxa_Variant(daxa_BlasGeometryInfoSpansUnion) geometries;
    daxa_DeviceAddress scratch_data;
} daxa_BlasBuildInfo;

static daxa_BlasBuildInfo const DAXA_DEFAULT_BLAS_BUILD_INFO = {
    .flags = DAXA_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE,
    .update = 0,
    .src_blas = DAXA_ZERO_INIT,
    .dst_blas = DAXA_ZERO_INIT,
    .geometries = DAXA_ZERO_INIT,
    .scratch_data = DAXA_ZERO_INIT,
};

typedef struct
{
    uint64_t size;
    daxa_SmallString name;
} daxa_TlasInfo;

static daxa_TlasInfo const DAXA_DEFAULT_TLAS_INFO = {
    .size = DAXA_ZERO_INIT,
    .name = {
        .data = DAXA_ZERO_INIT,
        .size = 0,
    },
};

typedef struct
{
    uint64_t size;
    daxa_SmallString name;
} daxa_BlasInfo;

static daxa_BlasInfo const DAXA_DEFAULT_BLAS_INFO = {
    .size = DAXA_ZERO_INIT,
    .name = {
        .data = DAXA_ZERO_INIT,
        .size = 0,
    },
};

#endif // #ifndef __DAXA_GPU_RESOURCES_H__
