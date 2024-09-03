#pragma once

#if !defined(DAXA_GPU_TABLE_SET_BINDING)
#define DAXA_GPU_TABLE_SET_BINDING 0
#define DAXA_STORAGE_BUFFER_BINDING 0
#define DAXA_STORAGE_IMAGE_BINDING 1
#define DAXA_SAMPLED_IMAGE_BINDING 2
#define DAXA_SAMPLER_BINDING 3
#define DAXA_BUFFER_DEVICE_ADDRESS_BUFFER_BINDING 4
#define DAXA_ACCELERATION_STRUCTURE_BINDING 5
#endif

#define DAXA_DECL_STORAGE_BUFFERS [[vk::binding(DAXA_STORAGE_BUFFER_BINDING, 0)]]
#define DAXA_DECL_STORAGE_IMAGES [[vk::binding(DAXA_STORAGE_IMAGE_BINDING, 0)]]
#define DAXA_DECL_SAMPLED_IMAGES [[vk::binding(DAXA_SAMPLED_IMAGE_BINDING, 0)]]
#define DAXA_DECL_SAMPLERS [[vk::binding(DAXA_SAMPLER_BINDING, 0)]]
#define DAXA_DECL_ACCELERATION_STRUCTURES [[vk::binding(DAXA_ACCELERATION_STRUCTURE_BINDING, 0)]]

#define DAXA_ID_INDEX_BITS 20
#define DAXA_ID_INDEX_MASK ((uint64_t(1) << DAXA_ID_INDEX_BITS) - uint64_t(1))
#define DAXA_ID_INDEX_OFFSTET 0
#define DAXA_ID_VERSION_BITS 44
#define DAXA_ID_VERSION_MASK ((uint64_t(1) << DAXA_ID_VERSION_BITS) - uint64_t(1))
#define DAXA_ID_VERSION_OFFSTET DAXA_ID_INDEX_BITS

#define _DAXA_DECL_VEC_TYPES(SLANG_TYPE, DAXA_TYPE) \
    typedef SLANG_TYPE DAXA_TYPE;                   \
    typedef SLANG_TYPE ## 1 DAXA_TYPE ## vec1;      \
    typedef SLANG_TYPE ## 2 DAXA_TYPE ## vec2;      \
    typedef SLANG_TYPE ## 3 DAXA_TYPE ## vec3;      \
    typedef SLANG_TYPE ## 4 DAXA_TYPE ## vec4;

#define _DAXA_DECL_MAT_TYPES(SLANG_TYPE, DAXA_TYPE)      \
    typedef matrix<SLANG_TYPE, 1, 1> DAXA_TYPE ## mat1x1 \
    typedef matrix<SLANG_TYPE, 2, 1> DAXA_TYPE ## mat1x2 \
    typedef matrix<SLANG_TYPE, 3, 1> DAXA_TYPE ## mat1x3 \
    typedef matrix<SLANG_TYPE, 4, 1> DAXA_TYPE ## mat1x4 \
    typedef matrix<SLANG_TYPE, 1, 2> DAXA_TYPE ## mat2x1 \
    typedef matrix<SLANG_TYPE, 2, 2> DAXA_TYPE ## mat2x2 \
    typedef matrix<SLANG_TYPE, 3, 2> DAXA_TYPE ## mat2x3 \
    typedef matrix<SLANG_TYPE, 4, 2> DAXA_TYPE ## mat2x4 \
    typedef matrix<SLANG_TYPE, 1, 3> DAXA_TYPE ## mat3x1 \
    typedef matrix<SLANG_TYPE, 2, 3> DAXA_TYPE ## mat3x2 \
    typedef matrix<SLANG_TYPE, 3, 3> DAXA_TYPE ## mat3x3 \
    typedef matrix<SLANG_TYPE, 4, 3> DAXA_TYPE ## mat3x4 \
    typedef matrix<SLANG_TYPE, 1, 4> DAXA_TYPE ## mat4x1 \
    typedef matrix<SLANG_TYPE, 2, 4> DAXA_TYPE ## mat4x2 \
    typedef matrix<SLANG_TYPE, 3, 4> DAXA_TYPE ## mat4x3 \
    typedef matrix<SLANG_TYPE, 4, 4> DAXA_TYPE ## mat4x4

#define _DAXA_DECL_VEC_MAT_8_16_32_64_TYPES(SLANG_TYPE, DAXA_TYPE)\
    _DAXA_DECL_VEC_TYPES(SLANG_TYPE ## 8_t, DAXA_TYPE ## 8)\
    _DAXA_DECL_VEC_TYPES(SLANG_TYPE ## 16_t, DAXA_TYPE ## 16)\
    _DAXA_DECL_VEC_TYPES(SLANG_TYPE ## 32_t, DAXA_TYPE ## 32)\
    _DAXA_DECL_VEC_TYPES(SLANG_TYPE ## 64_t, DAXA_TYPE ## 64)\
    _DAXA_DECL_MAT_TYPES(SLANG_TYPE ## 8_t, DAXA_TYPE ## 8)\
    _DAXA_DECL_MAT_TYPES(SLANG_TYPE ## 16_t, DAXA_TYPE ## 16)\
    _DAXA_DECL_MAT_TYPES(SLANG_TYPE ## 32_t, DAXA_TYPE ## 32)\
    _DAXA_DECL_MAT_TYPES(SLANG_TYPE ## 64_t, DAXA_TYPE ## 64)

namespace daxa
{
    _DAXA_DECL_VEC_MAT_8_16_32_64_TYPES(uint, u)
    _DAXA_DECL_VEC_MAT_8_16_32_64_TYPES(int, i)
    _DAXA_DECL_VEC_TYPES(half, f16)
    _DAXA_DECL_VEC_TYPES(float, f32)
    _DAXA_DECL_VEC_TYPES(double, f64)
    _DAXA_DECL_MAT_TYPES(half, f16)
    _DAXA_DECL_MAT_TYPES(float, f32)
    _DAXA_DECL_MAT_TYPES(double, f64)
    _DAXA_DECL_VEC_TYPES(bool, b32)
    _DAXA_DECL_MAT_TYPES(bool, b32)

    DAXA_DECL_STORAGE_BUFFERS __DynamicResource buffers[];
    DAXA_DECL_STORAGE_BUFFERS coherent __DynamicResource coherent_buffers[];
    DAXA_DECL_STORAGE_IMAGES __DynamicResource storage_images[];
    DAXA_DECL_STORAGE_IMAGES coherent __DynamicResource coherent_storage_images[];
    DAXA_DECL_SAMPLED_IMAGES __DynamicResource sampled_images[];
    DAXA_DECL_SAMPLED_IMAGES coherent __DynamicResource coherent_sampled_images[];
    DAXA_DECL_SAMPLERS SamplerState samplers[];
    DAXA_DECL_ACCELERATION_STRUCTURES RaytracingAccelerationStructure acceleration_structures[];
    
    [[vk::binding(DAXA_BUFFER_DEVICE_ADDRESS_BUFFER_BINDING, 0)]] StructuredBuffer<daxa_u64> buffer_addresses;
    struct BufferId
    {
        daxa_u64 value;
        daxa_u32 index()
        {
            return daxa_u32(DAXA_ID_INDEX_MASK & value);
        }
        daxa_u64 version()
        {
            return (value >> DAXA_ID_VERSION_OFFSTET);
        }
        daxa_b32 is_empty()
        {
            return value == 0;
        }
        daxa_u64 device_address()
        {
            return buffer_addresses[index()];
        }
    };

    struct ImageViewId
    {
        daxa_u64 value;
        daxa_u32 index()
        {
            return daxa_u32(DAXA_ID_INDEX_MASK & value);
        }
        daxa_u64 version()
        {
            return (value >> DAXA_ID_VERSION_OFFSTET);
        }
        daxa_b32 is_empty()
        {
            return value == 0;
        }
    };

    struct ImageViewIndex
    {
        daxa_u32 value;
        daxa_u32 index()
        {
            return value;
        }
        daxa_b32 is_empty()
        {
            return value == 0;
        }
    };

    struct SamplerId
    {
        daxa_u64 value;
        daxa_u32 index()
        {
            return daxa_u32(DAXA_ID_INDEX_MASK & value);
        }
        daxa_u64 version()
        {
            return (value >> DAXA_ID_VERSION_OFFSTET);
        }
        daxa_b32 is_empty()
        {
            return value == 0;
        }
    };

#if defined(DAXA_RAY_TRACING)
    struct TlasId
    {
        daxa_u64 value;
        daxa_u32 index()
        {
            return daxa_u32(DAXA_ID_INDEX_MASK & value);
        }
        daxa_u64 version()
        {
            return (value >> DAXA_ID_VERSION_OFFSTET);
        }
        daxa_b32 is_empty()
        {
            return value == 0;
        }
    };

    [[vk::binding(DAXA_ACCELERATION_STRUCTURE_BINDING, 0)]] RaytracingAccelerationStructure RayTracingAccelerationStructureTable[];
    extension RaytracingAccelerationStructure
    {
        static RaytracingAccelerationStructure get(daxa::TlasId i) { return RayTracingAccelerationStructureTable[i.index()]; }
    }
#endif // DAXA_RAY_TRACING

    extension ByteAddressBuffer
    {
        static ByteAddressBuffer get(daxa::BufferId i) { return buffers[i.index()]; }
        static ByteAddressBuffer get_coherent(daxa::BufferId i) { return buffers[i.index()]; }
    };    

    extension RWByteAddressBuffer
    {
        static RWByteAddressBuffer get(daxa::BufferId i) { return buffers[i.index()]; }
        static RWByteAddressBuffer get_coherent(daxa::BufferId i) { return buffers[i.index()]; }
    };    

    extension SamplerState
    {
        static SamplerState get(daxa::SamplerId i) { return samplers[i.index()]; }
    }

} // namespace daxa

#define _DAXA_DECL_TEX_EXTENSION(TYPE, TABLE)\
__generic<T> extension TYPE<T>\
{\
    static TYPE<T> get(daxa::ImageViewId i) { return TABLE[i.index()]; }\
    static TYPE<T> get_coherent(daxa::ImageViewId i) { return TABLE[i.index()]; }\
    static TYPE<T> get(daxa::ImageViewIndex i) { return TABLE[i.index()]; }\
    static TYPE<T> get_coherent(daxa::ImageViewIndex i) { return TABLE[i.index()]; }\
}\
namespace daxa\
{\
    __generic<T>\
    struct TYPE ## Id\
    {\
        ImageViewId id;\
        TYPE<T> get() { return TYPE<T>::get(id); }\
        TYPE<T> get_coherent() { return TYPE<T>::get_coherent(id); }\
    };    \
    __generic<T>\
    struct TYPE ## Idx\
    {\
        ImageViewIndex index;\
        TYPE<T> get() { return TYPE<T>::get(index); }\
        TYPE<T> get_coherent() { return TYPE<T>::get_coherent(index); }\
    };\
}

_DAXA_DECL_TEX_EXTENSION(RWTexture1D, daxa::storage_images)
_DAXA_DECL_TEX_EXTENSION(RWTexture2D, daxa::storage_images)
_DAXA_DECL_TEX_EXTENSION(RWTexture3D, daxa::storage_images)
_DAXA_DECL_TEX_EXTENSION(RWTexture1DArray, daxa::storage_images)
_DAXA_DECL_TEX_EXTENSION(RWTexture2DArray, daxa::storage_images)
    
_DAXA_DECL_TEX_EXTENSION(Texture1D, daxa::sampled_images)
_DAXA_DECL_TEX_EXTENSION(Texture2D, daxa::sampled_images)
_DAXA_DECL_TEX_EXTENSION(Texture3D, daxa::sampled_images)
_DAXA_DECL_TEX_EXTENSION(Texture1DArray, daxa::sampled_images)
_DAXA_DECL_TEX_EXTENSION(Texture2DArray, daxa::sampled_images)
_DAXA_DECL_TEX_EXTENSION(TextureCube, daxa::sampled_images)
_DAXA_DECL_TEX_EXTENSION(TextureCubeArray, daxa::sampled_images)
_DAXA_DECL_TEX_EXTENSION(Texture2DMS, daxa::sampled_images)


// These are hopefully only temporary global namespace declarations:
_DAXA_DECL_VEC_MAT_8_16_32_64_TYPES(uint, daxa_u)
_DAXA_DECL_VEC_MAT_8_16_32_64_TYPES(int, daxa_i)
_DAXA_DECL_VEC_TYPES(half, daxa_f16)
_DAXA_DECL_VEC_TYPES(float, daxa_f32)
_DAXA_DECL_VEC_TYPES(double, daxa_f64)
_DAXA_DECL_MAT_TYPES(half, daxa_f16)
_DAXA_DECL_MAT_TYPES(float, daxa_f32)
_DAXA_DECL_MAT_TYPES(double, daxa_f64)
_DAXA_DECL_VEC_TYPES(bool, daxa_b32)
_DAXA_DECL_MAT_TYPES(bool, daxa_b32)
typedef daxa::BufferId daxa_BufferId
typedef daxa::ImageViewId daxa_ImageViewId
typedef daxa::SamplerId daxa_SamplerId
#if defined(DAXA_RAY_TRACING)
typedef daxa::TlasId daxa_TlasId
#endif // DAXA_RAY_TRACING
#define daxa_RWBufferPtr(STRUCT_TYPE) Ptr<STRUCT_TYPE>
#define daxa_BufferPtr(STRUCT_TYPE) Ptr<STRUCT_TYPE>
#define DAXA_DECL_BUFFER_PTR_ALIGN(STRUCT_TYPE, ALIGN)
#define DAXA_DECL_BUFFER_PTR(STRUCT_TYPE)
#define deref(PTR) (*PTR)
#define deref_i(PTR, INDEX) PTR[INDEX]
#define as_address(PTR) uint64_t(PTR)