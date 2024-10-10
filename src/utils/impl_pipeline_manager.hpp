#pragma once

#include <daxa/utils/pipeline_manager.hpp>

#undef Bool

#if DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_SLANG
#include <slang.h>
#include <slang-com-ptr.h>
#endif

#if DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_GLSLANG
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <glslang/Include/ResourceLimits.h>
#endif

#if DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_SPIRV_VALIDATION
#include <spirv-tools/libspirv.hpp>
#endif

namespace daxa
{
    struct ImplDevice;

    using ShaderFileTimeSet = std::map<std::filesystem::path, std::chrono::file_clock::time_point>;

    struct VirtualFileState
    {
        std::string contents;
        std::chrono::file_clock::time_point timestamp;
    };

    using VirtualFileSet = std::map<std::string, VirtualFileState>;

    struct ImplPipelineManager final : ImplHandle
    {
        enum class ShaderStage
        {
            COMP,
            VERT,
            FRAG,
            TESS_CONTROL,
            TESS_EVAL,
            TASK,
            MESH,
            RAY_GEN,
            RAY_INTERSECT,
            RAY_ANY_HIT,
            RAY_CLOSEST_HIT,
            RAY_MISS,
            RAY_CALLABLE,
        };

        PipelineManagerInfo info = {};

        std::vector<std::filesystem::path> current_seen_shader_files = {};
        ShaderFileTimeSet * current_observed_hotload_files = nullptr;

        VirtualFileSet virtual_files = {};

        template <typename PipeT, typename InfoT>
        struct PipelineState
        {
            std::shared_ptr<PipeT> pipeline_ptr;
            InfoT info;
            std::chrono::file_clock::time_point last_hotload_time = {};
            ShaderFileTimeSet observed_hotload_files = {};
        };

        using ComputePipelineState = PipelineState<ComputePipeline, ComputePipelineCompileInfo>;
        using RasterPipelineState = PipelineState<RasterPipeline, RasterPipelineCompileInfo>;
        using RayTracingPipelineState = PipelineState<RayTracingPipeline, RayTracingPipelineCompileInfo>;

        std::vector<ComputePipelineState> compute_pipelines;
        std::vector<RasterPipelineState> raster_pipelines;
        std::vector<RayTracingPipelineState> ray_tracing_pipelines;

        // TODO(grundlett): Maybe make the pipeline compiler *internally* thread-safe!
        // This variable is accessed by the includer, which makes that not thread-safe
        // PipelineManager is still externally thread-safe. You can create as many
        // PipelineManagers from as many threads as you'd like!
        ShaderCompileInfo const * current_shader_info = nullptr;

#if DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_GLSLANG
        struct GlslangBackend
        {
        };
        GlslangBackend glslang_backend = {};
#endif

#if DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_SLANG
        struct SlangBackend
        {
            // NOTE(grundlett): literally global session. Not sure if this is reasonable or not.
            static inline Slang::ComPtr<slang::IGlobalSession> global_session;
            static inline std::mutex session_mtx = {};
        };
        SlangBackend slang_backend = {};
#endif

#if DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_SPIRV_VALIDATION
        spvtools::SpirvTools spirv_tools = spvtools::SpirvTools{SPV_ENV_VULKAN_1_3};
#endif

        ImplPipelineManager(PipelineManagerInfo && a_info);
        ~ImplPipelineManager();

        auto create_ray_tracing_pipeline(RayTracingPipelineCompileInfo const & a_info) -> Result<RayTracingPipelineState>;
        auto create_compute_pipeline(ComputePipelineCompileInfo const & a_info) -> Result<ComputePipelineState>;
        auto create_raster_pipeline(RasterPipelineCompileInfo const & a_info) -> Result<RasterPipelineState>;
        auto add_ray_tracing_pipeline(RayTracingPipelineCompileInfo const & a_info) -> Result<std::shared_ptr<RayTracingPipeline>>;
        auto add_compute_pipeline(ComputePipelineCompileInfo const & a_info) -> Result<std::shared_ptr<ComputePipeline>>;
        auto add_raster_pipeline(RasterPipelineCompileInfo const & a_info) -> Result<std::shared_ptr<RasterPipeline>>;
        void remove_ray_tracing_pipeline(std::shared_ptr<RayTracingPipeline> const & pipeline);
        void remove_compute_pipeline(std::shared_ptr<ComputePipeline> const & pipeline);
        void remove_raster_pipeline(std::shared_ptr<RasterPipeline> const & pipeline);
        void add_virtual_file(VirtualFileInfo const & virtual_info);
        auto reload_all() -> PipelineReloadResult;
        auto all_pipelines_valid() const -> bool;

        auto try_load_shader_cache(std::filesystem::path const & cache_folder, uint64_t shader_info_hash) -> Result<std::vector<u32>>;
        void save_shader_cache(std::filesystem::path const & out_folder, uint64_t shader_info_hash, std::vector<u32> const & spirv);
        auto full_path_to_file(std::filesystem::path const & path) -> Result<std::filesystem::path>;
        auto load_shader_source_from_file(std::filesystem::path const & path) -> Result<ShaderCode>;

        auto get_spirv(ShaderCompileInfo const & shader_info, std::string const & debug_name_opt, ShaderStage shader_stage) -> Result<std::vector<u32>>;
        auto get_spirv_glslang(ShaderCompileInfo const & shader_info, std::string const & debug_name_opt, ShaderStage shader_stage, ShaderCode const & code) -> Result<std::vector<u32>>;
        auto get_spirv_slang(ShaderCompileInfo const & shader_info, ShaderStage shader_stage, ShaderCode const & code) -> Result<std::vector<u32>>;

        static auto zero_ref_callback(ImplHandle const * handle);
    };
} // namespace daxa
