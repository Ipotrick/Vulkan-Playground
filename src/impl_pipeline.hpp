#pragma once

#include <daxa/pipeline.hpp>

#include "impl_core.hpp"

using namespace daxa;

static inline constexpr usize pipeline_manager_MAX_ATTACHMENTS = 16;

struct PipelineZombie
{
    VkPipeline vk_pipeline = {};
};

struct ImplPipeline : ImplHandle
{
    daxa_Device device = {};
    VkPipeline vk_pipeline = {};
    VkPipelineLayout vk_pipeline_layout = {};

    static void zero_ref_callback(ImplHandle const * handle);
};

struct daxa_ImplRasterPipeline final : ImplPipeline
{
    RasterPipelineInfo info = {};
};

struct daxa_ImplComputePipeline final : ImplPipeline
{
    ComputePipelineInfo info = {};
};

struct daxa_ImplRayTracingPipeline final : ImplPipeline
{
    // NOTE(grundlett): This is bogus. The state passed to info is in spans which means it does not persist...
    std::vector<RayTracingShaderGroupInfo> shader_groups = {};
    RayTracingPipelineInfo info = {};
};