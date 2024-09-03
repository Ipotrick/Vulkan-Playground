#pragma once

#if !DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_GLSLANG && !DAXA_BUILT_WITH_UTILS_PIPELINE_MANAGER_SLANG
#error "[package management error] You must build Daxa with the DAXA_ENABLE_UTILS_PIPELINE_MANAGER_(GLSLANG|SLANG) CMake option enabled, or request the utils-pipeline-manager-(glslang|slang) feature in vcpkg"
#endif

#include <daxa/device.hpp>

#include <filesystem>
#include <functional>

namespace daxa
{
    struct ShaderFile
    {
        std::filesystem::path path;
    };

    // This string will only work if it is valid GLSL/SLANG
    struct ShaderCode
    {
        std::string string;
    };

    using ShaderSource = Variant<Monostate, ShaderFile, ShaderCode>;

    struct ShaderDefine
    {
        std::string name = {};
        std::string value = {};
    };

    enum struct ShaderLanguage
    {
        GLSL,
        SLANG,
        MAX_ENUM = 0x7fffffff,
    };

    struct ShaderModel
    {
        u32 major, minor;
    };

    struct ShaderCompileOptions
    {
        std::optional<std::string> entry_point = {};
        std::vector<std::filesystem::path> root_paths = {};
        std::optional<std::filesystem::path> write_out_preprocessed_code = {};
        std::optional<std::filesystem::path> write_out_shader_binary = {};
        std::optional<std::filesystem::path> spirv_cache_folder = {};
        std::optional<ShaderLanguage> language = {};
        std::vector<ShaderDefine> defines = {};
        std::optional<bool> enable_debug_info = {};
        std::optional<ShaderCreateFlags> create_flags = {};
        std::optional<u32> required_subgroup_size = {};

        void inherit(ShaderCompileOptions const & other);
    };

    struct ShaderCompileInfo
    {
        ShaderSource source = Monostate{};
        ShaderCompileOptions compile_options = {};
    };

    struct RayTracingPipelineCompileInfo
    {
        std::vector<ShaderCompileInfo> ray_gen_infos = {};
        std::vector<ShaderCompileInfo> intersection_infos = {};
        std::vector<ShaderCompileInfo> any_hit_infos = {};
        std::vector<ShaderCompileInfo> callable_infos = {};
        std::vector<ShaderCompileInfo> closest_hit_infos = {};
        std::vector<ShaderCompileInfo> miss_hit_infos = {};
        std::vector<RayTracingShaderGroupInfo> shader_groups_infos = {};
        u32 max_ray_recursion_depth = {};
        u32 push_constant_size = {};
        std::string name = {};
    };

    struct ComputePipelineCompileInfo
    {
        ShaderCompileInfo shader_info = {};
        u32 push_constant_size = {};
        std::string name = {};
    };

    struct RasterPipelineCompileInfo
    {
        Optional<ShaderCompileInfo> mesh_shader_info = {};
        Optional<ShaderCompileInfo> vertex_shader_info = {};
        Optional<ShaderCompileInfo> tesselation_control_shader_info = {};
        Optional<ShaderCompileInfo> tesselation_evaluation_shader_info = {};
        Optional<ShaderCompileInfo> fragment_shader_info = {};
        Optional<ShaderCompileInfo> task_shader_info = {};
        std::vector<RenderAttachment> color_attachments = {};
        Optional<DepthTestInfo> depth_test = {};
        RasterizerInfo raster = {};
        TesselationInfo tesselation = {};
        u32 push_constant_size = {};
        std::string name = {};
    };

    struct PipelineManagerInfo
    {
        Device device;
        ShaderCompileOptions shader_compile_options = {};
        bool register_null_pipelines_when_first_compile_fails = false;
        std::function<void(std::string &, std::filesystem::path const & path)> custom_preprocessor = {};
        std::string name = {};
    };

    struct VirtualFileInfo
    {
        std::string name = {};
        std::string contents = {};
    };

    struct PipelineReloadSuccess
    {
    };
    struct PipelineReloadError
    {
        std::string message;
    };
    using NoPipelineChanged = Monostate;

    using PipelineReloadResult = Variant<NoPipelineChanged, PipelineReloadSuccess, PipelineReloadError>;

    struct ImplPipelineManager;
    struct DAXA_EXPORT_CXX PipelineManager : ManagedPtr<PipelineManager, ImplPipelineManager *>
    {
        PipelineManager() = default;

        PipelineManager(PipelineManagerInfo info);

        auto add_ray_tracing_pipeline(RayTracingPipelineCompileInfo const & info) -> Result<std::shared_ptr<RayTracingPipeline>>;
        auto add_compute_pipeline(ComputePipelineCompileInfo const & info) -> Result<std::shared_ptr<ComputePipeline>>;
        auto add_raster_pipeline(RasterPipelineCompileInfo const & info) -> Result<std::shared_ptr<RasterPipeline>>;
        void remove_ray_tracing_pipeline(std::shared_ptr<RayTracingPipeline> const & pipeline);
        void remove_compute_pipeline(std::shared_ptr<ComputePipeline> const & pipeline);
        void remove_raster_pipeline(std::shared_ptr<RasterPipeline> const & pipeline);
        void add_virtual_file(VirtualFileInfo const & info);
        auto reload_all() -> PipelineReloadResult;
        auto all_pipelines_valid() const -> bool;

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };
} // namespace daxa
