#include <shared.inl>

[[vk::push_constant]] const ComputePush p;

// clang-format off
[numthreads(8, 8, 1)]
void main(uint3 pixel_i : SV_DispatchThreadID)
// clang-format on
{
    RWTexture2D<float4> render_image = RWTexture2D<float4>.get(p.image);
    if (pixel_i.x >= p.frame_dim.x || pixel_i.y >= p.frame_dim.y)
        return;

    float2 uv = float2(pixel_i.xy) / float2(p.frame_dim.xy);
    uv = (uv - 0.5) * float2(float(p.frame_dim.x) / float(p.frame_dim.y), 1);
    uv = uv * 2;

    float3 col = float3(0, 0, 0);

    float2 points[3] = {
        float2(-0.5, +0.5),
        float2(+0.5, +0.5),
        float2(+0.0, -0.5),
    };

    float3 point_colors[3] = {
        float3(1, 0, 0),
        float3(0, 1, 0),
        float3(0, 0, 1),
    };

    float2 points_del[3] = {
        points[1] - points[0],
        points[2] - points[1],
        points[0] - points[2],
    };

    float slopes[3] = {
        points_del[0].y / points_del[0].x,
        points_del[1].y / points_del[1].x,
        points_del[2].y / points_del[2].x,
    };

    if (slopes[0] * (uv.x - points[0].x) > (uv.y - points[0].y) &&
        slopes[1] * (uv.x - points[1].x) < (uv.y - points[1].y) &&
        slopes[2] * (uv.x - points[2].x) < (uv.y - points[2].y))
    {
        float p0 = clamp(dot(points_del[0], uv - points[0]) / dot(points_del[0], points_del[0]), 0, 1);
        float p1 = clamp(dot(points_del[1], uv - points[1]) / dot(points_del[1], points_del[1]), 0, 1);
        float p2 = clamp(dot(points_del[2], uv - points[2]) / dot(points_del[2], points_del[2]), 0, 1);

        col = lerp(col, point_colors[0], 1);
        col = lerp(col, point_colors[1], p0);
        col = lerp(col, point_colors[2], clamp((p1 - p2 + 0.5) / 1.5, 0, 1));
    }

    render_image[pixel_i.xy] = float4(col, 1.0);
}
