#pragma once

#include "daxa/daxa.inl"

struct GpuInput
{
    daxa::f32 time;
    daxa::f32 delta_time;
};

struct ComputePush
{
    daxa::RWTexture2DId<daxa_f32vec4> image_id;
    daxa::BufferId input_buffer_id;
    daxa_BufferPtr(GpuInput) ptr;
    daxa_u32vec2 frame_dim;
};
