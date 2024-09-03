#pragma once

#include <daxa/gpu_resources.hpp>
#include <daxa/core.hpp>
#include <daxa/types.hpp>

namespace daxa
{
    struct ShaderCreateFlagsProperties
    {
        using Data = u32;
    };
    using ShaderCreateFlags = Flags<ShaderCreateFlagsProperties>;
    struct ShaderCreateFlagBits
    {
        static inline constexpr ShaderCreateFlags NONE = {0x00000000};
        static inline constexpr ShaderCreateFlags ALLOW_VARYING_SUBGROUP_SIZE  = {0x00000001};
        static inline constexpr ShaderCreateFlags REQUIRE_FULL_SUBGROUPS  = {0x00000002};
    };

    struct ShaderInfo
    {
        u32 const * byte_code = {};
        u32 byte_code_size = {};
        ShaderCreateFlags create_flags = {};
        Optional<u32> required_subgroup_size = {};
        SmallString entry_point = "main";
    };

    // TODO: find a better way to link shader groups to shaders than by index
    struct RayTracingShaderGroupInfo
    {
        ShaderGroup type;
        u32 general_shader_index = (~0U);
        u32 closest_hit_shader_index = (~0U);
        u32 any_hit_shader_index = (~0U);
        u32 intersection_shader_index = (~0U);
    };

    struct StridedDeviceAddressRegion
    {
        DeviceAddress address {};
        u64 stride {};
        u64 size {};
    };

    struct RayTracingShaderBindingTable
    {
        StridedDeviceAddressRegion raygen_region = {};
        StridedDeviceAddressRegion miss_region = {};
        StridedDeviceAddressRegion hit_region = {};
        StridedDeviceAddressRegion callable_region = {};
    };

    struct RayTracingPipelineInfo
    {
        Span<ShaderInfo const> ray_gen_shaders = {};
        Span<ShaderInfo const> intersection_shaders = {};
        Span<ShaderInfo const> any_hit_shaders = {};
        Span<ShaderInfo const> callable_shaders = {};
        Span<ShaderInfo const> closest_hit_shaders = {};
        Span<ShaderInfo const> miss_hit_shaders = {};
        Span<RayTracingShaderGroupInfo const> shader_groups = {};
        u32 max_ray_recursion_depth;
        u32 push_constant_size = {};
        SmallString name = {};
    };

    /**
     * @brief   Represents a pipeline state object, usable in recording commands.
     *
     * THREADSAFETY:
     * * is internally synchronized
     * * may be passed to different threads
     * * may be used by multiple threads at the same time.
     */
    struct DAXA_EXPORT_CXX RayTracingPipeline final : ManagedPtr<RayTracingPipeline, daxa_RayTracingPipeline>
    {
        RayTracingPipeline() = default;

        /// THREADSAFETY:
        /// * reference MUST NOT be read after the object is destroyed.
        /// @return reference to info of object.
        [[nodiscard]] auto info() const -> RayTracingPipelineInfo const &;

        struct SbtPair { daxa::BufferId buffer; RayTracingShaderBindingTable table; };
        [[nodiscard]] auto create_default_sbt() const -> SbtPair;
        void get_shader_group_handles(void *out_blob) const;

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };

    struct ComputePipelineInfo
    {
        ShaderInfo shader_info = {};
        u32 push_constant_size = {};
        SmallString name = {};
    };

    /**
     * @brief   Represents a pipeline state object, usable in recording commands.
     *
     * THREADSAFETY:
     * * is internally synchronized
     * * may be passed to different threads
     * * may be used by multiple threads at the same time.
     */
    struct DAXA_EXPORT_CXX ComputePipeline final : ManagedPtr<ComputePipeline, daxa_ComputePipeline>
    {
        ComputePipeline() = default;

        /// THREADSAFETY:
        /// * reference MUST NOT be read after the object is destroyed.
        /// @return reference to info of object.
        [[nodiscard]] auto info() const -> ComputePipelineInfo const &;

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };

    struct DepthTestInfo
    {
        Format depth_attachment_format = Format::UNDEFINED;
        bool enable_depth_write = {};
        CompareOp depth_test_compare_op = CompareOp::LESS_OR_EQUAL;
        f32 min_depth_bounds = 0.0f;
        f32 max_depth_bounds = 1.0f;
    };

    struct ConservativeRasterInfo
    {
        ConservativeRasterizationMode mode = {};
        float size = {};
    };

    enum struct RasterizationSamples : u32
    {
        E1 = 0x00000001,
        E2 = 0x00000002,
        E4 = 0x00000004,
        E8 = 0x00000008,
    };

    struct RasterizerInfo
    {
        PrimitiveTopology primitive_topology = PrimitiveTopology::TRIANGLE_LIST;
        bool primitive_restart_enable = {};
        PolygonMode polygon_mode = PolygonMode::FILL;
        FaceCullFlags face_culling = FaceCullFlagBits::NONE;
        FrontFaceWinding front_face_winding = FrontFaceWinding::CLOCKWISE;
        bool depth_clamp_enable = {};
        bool rasterizer_discard_enable = {};
        bool depth_bias_enable = {};
        f32 depth_bias_constant_factor = 0.0f;
        f32 depth_bias_clamp = 0.0f;
        f32 depth_bias_slope_factor = 0.0f;
        f32 line_width = 1.0f;
        Optional<ConservativeRasterInfo> conservative_raster_info = {};
        // When left as none, the pipeline will use the current msaa value of the command recorder.
        Optional<RasterizationSamples> static_state_sample_count = {};
    };

    struct RenderAttachment
    {
        Format format = {};
        Optional<BlendInfo> blend = {};
    };

    struct TesselationInfo
    {
        u32 control_points = 3;
        TesselationDomainOrigin origin = {};
    };

    struct RasterPipelineInfo
    {
        Optional<ShaderInfo> mesh_shader_info = {};
        Optional<ShaderInfo> vertex_shader_info = {};
        Optional<ShaderInfo> tesselation_control_shader_info = {};
        Optional<ShaderInfo> tesselation_evaluation_shader_info = {};
        Optional<ShaderInfo> fragment_shader_info = {};
        Optional<ShaderInfo> task_shader_info = {};
        FixedList<RenderAttachment, 8> color_attachments = {};
        Optional<DepthTestInfo> depth_test = {};
        Optional<TesselationInfo> tesselation = {};
        RasterizerInfo raster = {};
        u32 push_constant_size = {};
        SmallString name = {};
    };

    /**
     * @brief   Represents a pipeline state object, usable in recording commands.
     *
     * THREADSAFETY:
     * * is internally synchronized
     * * may be passed to different threads
     * * may be used by multiple threads at the same time.
     */
    struct DAXA_EXPORT_CXX RasterPipeline final : ManagedPtr<RasterPipeline, daxa_RasterPipeline>
    {
        RasterPipeline() = default;

        /// THREADSAFETY:
        /// * reference MUST NOT be read after the object is destroyed.
        /// @return reference to info of object.
        [[nodiscard]] auto info() const -> RasterPipelineInfo const &;

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };
} // namespace daxa
