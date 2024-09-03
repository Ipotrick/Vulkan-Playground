#pragma once
#include "../daxa.inl"

/**
 * DAXA_TH_IMAGE:
 *   Declares an image attachment.
 *   The attachment is NOT represented at all within the shader blob.
 *
 * DAXA_TH_IMAGE_ID:
 *   Declares an image attachment.
 *   The first runtime image will be represented within the blob.
 *
 * DAXA_TH_IMAGE_ID_ARRAY:
 *   Declares an image attachment.
 *   A partial array of the attachments runtime ids are represented within an id array in the blob.
 *
 * DAXA_TH_IMAGE_ID_MIP_ARRAY:
 *   Declares an image attachment.
 *   A partial array of image views pointing to separate mip levels of the first runtime image are represented within the blob.
 *
 * DAXA_TH_BUFFER:
 *   Declares a buffer attachment.
 *   The attachment is NOT represented at all within the shader blob.
 *
 * DAXA_TH_BUFFER_ID:
 *   Declares a buffer attachment.
 *   The first runtime buffer will be represented within the blob.
 *
 * DAXA_TH_BUFFER_PTR:
 *   Declares a buffer attachment.
 *   The first runtime buffer will be represented with its buffer device address within the blob.
 *
 * DAXA_TH_BUFFER_ID_ARRAY:
 *   Declares a buffer attachment.
 *   A partial array of the attachments runtime buffers are represented within an id array in the blob.
 *
 * DAXA_TH_BUFFER_PTR_ARRAY:
 *   Declares a buffer attachment.
 *   A partial array of the attachments runtime buffers are represented with their buffer device addresses within an id array in the blob.
 *
 */

// DAXA 3.0 Task-Head-Shader interface:
#if DAXA_SHADER
#if DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
#define DAXA_DECL_TASK_HEAD_BEGIN(HEAD_NAME) \
    namespace HEAD_NAME                      \
    {                                        \
        struct AttachmentShaderBlob          \
        {
#define DAXA_TH_IMAGE(TASK_ACCESS, VIEW_TYPE, NAME)
#define DAXA_TH_IMAGE_ID(TASK_ACCESS, VIEW_TYPE, NAME) daxa::ImageViewId NAME;
#define DAXA_TH_IMAGE_ID_ARRAY(TASK_ACCESS, VIEW_TYPE, NAME, SIZE) daxa::ImageViewId NAME[SIZE];
#define DAXA_TH_IMAGE_ID_MIP_ARRAY(TASK_ACCESS, VIEW_TYPE, NAME, SIZE) daxa::ImageViewId NAME[SIZE];
#define DAXA_TH_IMAGE_INDEX(TASK_ACCESS, VIEW_TYPE, NAME) daxa::ImageViewIndex NAME;
#define DAXA_TH_IMAGE_INDEX_ARRAY(TASK_ACCESS, VIEW_TYPE, NAME, SIZE) daxa::ImageViewIndex NAME[SIZE];
#define DAXA_TH_IMAGE_INDEX_MIP_ARRAY(TASK_ACCESS, VIEW_TYPE, NAME, SIZE) daxa::ImageViewIndex NAME[SIZE];
#define DAXA_TH_IMAGE_TYPED(TASK_ACCESS, TEX_TYPE, NAME) TEX_TYPE NAME;
#define DAXA_TH_IMAGE_TYPED_ARRAY(TASK_ACCESS, TEX_TYPE, NAME, SIZE) TEX_TYPE NAME[SIZE];
#define DAXA_TH_IMAGE_TYPED_MIP_ARRAY(TASK_ACCESS, TEX_TYPE, NAME, SIZE) TEX_TYPE NAME[SIZE];
#define DAXA_TH_BUFFER(TASK_ACCESS, NAME)
#define DAXA_TH_BUFFER_ID(TASK_ACCESS, NAME) daxa::BufferId NAME;
#define DAXA_TH_BUFFER_PTR(TASK_ACCESS, PTR_TYPE, NAME) PTR_TYPE NAME;
#define DAXA_TH_BUFFER_ID_ARRAY(TASK_ACCESS, NAME, SIZE) daxa::BufferId NAME[SIZE];
#define DAXA_TH_BUFFER_PTR_ARRAY(TASK_ACCESS, PTR_TYPE, NAME, SIZE) PTR_TYPE NAME[SIZE];
#define DAXA_TH_BLAS(TASK_ACCESS, NAME)
#define DAXA_TH_TLAS(TASK_ACCESS, NAME)
#define DAXA_TH_TLAS_PTR(TASK_ACCESS, NAME) daxa::u64 NAME;
#define DAXA_TH_TLAS_ID(TASK_ACCESS, NAME) daxa::TlasId NAME;
#define DAXA_DECL_TASK_HEAD_END \
    }                           \
    ;                           \
    }                           \
    ;
#define DAXA_TH_BLOB(HEAD_NAME, field_name) HEAD_NAME::AttachmentShaderBlob field_name;
#else // glsl
#define DAXA_DECL_TASK_HEAD_BEGIN(HEAD_NAME) \
    struct HEAD_NAME                         \
    {
#define DAXA_TH_IMAGE(TASK_ACCESS, VIEW_TYPE, NAME)
#define DAXA_TH_IMAGE_ID(TASK_ACCESS, VIEW_TYPE, NAME) daxa_ImageViewId NAME;
#define DAXA_TH_IMAGE_ID_ARRAY(TASK_ACCESS, VIEW_TYPE, NAME, SIZE) daxa_ImageViewId NAME[SIZE];
#define DAXA_TH_IMAGE_ID_MIP_ARRAY(TASK_ACCESS, VIEW_TYPE, NAME, SIZE) daxa_ImageViewId NAME[SIZE];
#define DAXA_TH_IMAGE_INDEX(TASK_ACCESS, VIEW_TYPE, NAME) daxa_ImageViewIndex NAME;
#define DAXA_TH_IMAGE_INDEX_ARRAY(TASK_ACCESS, VIEW_TYPE, NAME, SIZE) daxa_ImageViewIndex NAME[SIZE];
#define DAXA_TH_IMAGE_INDEX_MIP_ARRAY(TASK_ACCESS, VIEW_TYPE, NAME, SIZE) daxa_ImageViewIndex NAME[SIZE];
#define DAXA_TH_IMAGE_TYPED(TASK_ACCESS, TEX_TYPE, NAME) ERROR_TYPED_HANDLES_ARE_SLANG_ONLY
#define DAXA_TH_IMAGE_TYPED_ARRAY(TASK_ACCESS, TEX_TYPE, NAME, SIZE) ERROR_TYPED_HANDLES_ARE_SLANG_ONLY
#define DAXA_TH_IMAGE_TYPED_MIP_ARRAY(TASK_ACCESS, TEX_TYPE, NAME, SIZE) ERROR_TYPED_HANDLES_ARE_SLANG_ONLY
#define DAXA_TH_BUFFER(TASK_ACCESS, NAME)
#define DAXA_TH_BUFFER_ID(TASK_ACCESS, NAME) daxa_BufferId NAME;
#define DAXA_TH_BUFFER_PTR(TASK_ACCESS, PTR_TYPE, NAME) PTR_TYPE NAME;
#define DAXA_TH_BUFFER_ID_ARRAY(TASK_ACCESS, NAME, SIZE) daxa_BufferId NAME[SIZE];
#define DAXA_TH_BUFFER_PTR_ARRAY(TASK_ACCESS, PTR_TYPE, NAME, SIZE) PTR_TYPE NAME[SIZE];
#define DAXA_TH_BLAS(TASK_ACCESS, NAME)
#define DAXA_TH_TLAS(TASK_ACCESS, NAME)
#define DAXA_TH_TLAS_PTR(TASK_ACCESS, NAME) daxa_u64 NAME;
#define DAXA_TH_TLAS_ID(TASK_ACCESS, NAME) daxa_TlasId NAME;
#define DAXA_DECL_TASK_HEAD_END \
    }                           \
    ;
#define DAXA_TH_BLOB(HEAD_NAME, field_name) HEAD_NAME field_name;
#endif
#elif __cplusplus
#include "task_graph_types.hpp"
#else // C
#error TASK GRAPH INL ONLY SUPPORTED IN SHADERS AND C++!
#endif