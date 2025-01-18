#pragma once

#include "daxa/daxa.inl"

struct GpuInput
{
    daxa_f32 time;
    daxa_f32 delta_time;
};
DAXA_DECL_BUFFER_PTR(GpuInput)

struct ComputePush
{
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
    daxa_ImageViewId image_id;
    daxa_BufferId input_buffer_id;
    daxa_BufferPtr(GpuInput) ptr;
    daxa_u32vec2 frame_dim;
#else
    daxa::RWTexture2DId<daxa_f32vec4> image_id;
    daxa::BufferId input_buffer_id;
    daxa_BufferPtr(GpuInput) ptr;
    daxa_u32vec2 frame_dim;
#endif
};
