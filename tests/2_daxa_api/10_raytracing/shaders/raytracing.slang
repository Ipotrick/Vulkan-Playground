#include "shared.inl"

[[vk::push_constant]] PushConstant p;

struct ShadowRayPayload
{
    daxa_f32vec4 shadow;
};

// struct MyAttributes
// {
//     daxa_f32vec3 normal;
//     daxa_f32vec3 position;
// };

struct MyParams
{
    daxa_f32vec4 color;
};

[shader("raygeneration")] void primary_raygen_shader()
{
    const daxa_u32vec2 threadIdx = DispatchRaysIndex().xy;
    const daxa_u32vec2 rt_size = DispatchRaysDimensions().xy;

    // Camera setup
    daxa_f32mat4x4 inv_view = deref(p.camera_buffer).inv_view;
    daxa_f32mat4x4 inv_proj = deref(p.camera_buffer).inv_proj;

    const daxa_f32vec2 pixelCenter = daxa_f32vec2(threadIdx) + daxa_f32vec2(0.5);
    const daxa_f32vec2 inUV = pixelCenter / daxa_f32vec2(rt_size);
    daxa_f32vec2 d = inUV * 2.0 - 1.0;

    daxa_f32vec4 origin = mul(inv_view, daxa_f32vec4(0, 0, 0, 1));
    daxa_f32vec4 target = mul(inv_proj, daxa_f32vec4(d.x, d.y, 1, 1));
    daxa_f32vec4 direction = mul(inv_view, daxa_f32vec4(normalize(target.xyz), 0));
    daxa_f32 tMin = 0.0001;
    daxa_f32 tMax = 10000.0;
    RayDesc ray = {origin.xyz, tMin, direction.xyz, tMax};
    
    daxa_u32 frame_number = p.frame;
    daxa_u32 seed = tea(threadIdx.y * rt_size.x + threadIdx.x, frame_number);

    hitPayload payload = {daxa_f32vec3(0.0, 0.0, 0.0), seed};

    // credits: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html

    // Instantiate ray query object.
    // Template parameter allows driver to generate a specialized
    // implementation.
    RayQuery<RAY_FLAG_NONE> q;

    // Set up a trace.  No work is done yet.
    q.TraceRayInline(
        RaytracingAccelerationStructure::get(p.tlas), 
        RAY_FLAG_NONE, ~0, ray);
    
    daxa_f32 THit = tMax;

    // Proceed() is where behind-the-scenes traversal happens,
    // including the heaviest of any driver inlined code.
    // Returns TRUE if there's a task for the shader to perform
    // as part of traversal
    while(q.Proceed())
    {
        switch(q.CandidateType())
        {
            case CANDIDATE_PROCEDURAL_PRIMITIVE:
            {
                daxa_u32 prim_index = q.CandidatePrimitiveIndex() +  q.CandidateGeometryIndex();
                Aabb aabb = deref(p.aabb_buffer).aabbs[prim_index];

                THit = aabb_get_hit(aabb, ray.Origin, ray.Direction, q.CandidateWorldToObject4x3());
                if(THit > 0.0){
                    q.CommitProceduralPrimitiveHit(THit);
                    break;
                }
            }
            case CANDIDATE_NON_OPAQUE_TRIANGLE:
            {
                q.CommitNonOpaqueTriangleHit();
                break;
            }
        }
    }
    switch(q.CommittedStatus())
    {
    case COMMITTED_TRIANGLE_HIT:
    {
        payload.hitValue = 1.0 / q.CommittedRayT() * daxa_f32vec3(1.0, 1.0, 1.0);
        break;
    }
    case COMMITTED_PROCEDURAL_PRIMITIVE_HIT:
    {
        payload.hitValue = 1.0 / THit * daxa_f32vec3(1.0, 1.0, 1.0);
        break;
    }
    case COMMITTED_NOTHING:
    {
        payload.hitValue = daxa_f32vec3(0.0, 0.0, 0.0);
        break;
    }
    }

    p.swapchain.get()[threadIdx.xy] = daxa_f32vec4(payload.hitValue, 1.0);
}

[shader("raygeneration")] void raygen_shader()
{
    const daxa_u32vec2 threadIdx = DispatchRaysIndex().xy;
    const daxa_u32vec2 rt_size = DispatchRaysDimensions().xy;

    // Camera setup
    daxa_f32mat4x4 inv_view = deref(p.camera_buffer).inv_view;
    daxa_f32mat4x4 inv_proj = deref(p.camera_buffer).inv_proj;

    const daxa_f32vec2 pixelCenter = daxa_f32vec2(threadIdx) + daxa_f32vec2(0.5);
    const daxa_f32vec2 inUV = pixelCenter / daxa_f32vec2(rt_size);
    daxa_f32vec2 d = inUV * 2.0 - 1.0;

    daxa_f32vec4 origin = mul(inv_view, daxa_f32vec4(0, 0, 0, 1));
    daxa_f32vec4 target = mul(inv_proj, daxa_f32vec4(d.x, d.y, 1, 1));
    daxa_f32vec4 direction = mul(inv_view, daxa_f32vec4(normalize(target.xyz), 0));
    daxa_f32 tMin = 0.0001;
    daxa_f32 tMax = 10000.0;
    RayDesc ray = {origin.xyz, tMin, direction.xyz, tMax};
    
    daxa_u32 frame_number = p.frame;
    daxa_u32 seed = tea(threadIdx.y * rt_size.x + threadIdx.x, frame_number);

    hitPayload payload = {daxa_f32vec3(0.0, 0.0, 0.0), seed};


    TraceRay(RaytracingAccelerationStructure::get(p.tlas), RAY_FLAG_NONE, ~0, 0, 0, 0, ray, payload);


    p.swapchain.get()[threadIdx.xy] = daxa_f32vec4(payload.hitValue, 1.0);
}

[shader("miss")] void miss_shader(inout hitPayload payload)
{
    payload.hitValue = daxa_f32vec3(0.5, 0.7, 1.0);
}

[shader("miss")] void shadow_miss_shader(inout ShadowRayPayload payload)
{
    payload.shadow.x = 1.0;
}

[shader("closesthit")] void closest_hit_shader(inout hitPayload payload,
                                                 in BuiltInTriangleIntersectionAttributes attr)
{
    // Barycentric coordinates of the hit point
    const daxa_f32vec3 barycentrics = daxa_f32vec3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    // Color vertices (adjust these colors as desired)
    daxa_f32vec3 color0 = daxa_f32vec3(1.0, 0.0, 0.0); // Red
    daxa_f32vec3 color1 = daxa_f32vec3(0.0, 1.0, 0.0); // Green
    daxa_f32vec3 color2 = daxa_f32vec3(0.0, 0.0, 1.0); // Blue

    // Interpolate colors using barycentric coordinates
    daxa_f32vec3 interpolatedColor = barycentrics.x * color0 + barycentrics.y * color1 + barycentrics.z * color2;
    
    daxa_f32mat4x4 objectToWorld = transpose(Convert3x4To4x4(ObjectToWorld4x3()));
    
    daxa_f32vec3 v0 = deref(p.vertex_buffer).vertices[0];
    daxa_f32vec3 v1 = deref(p.vertex_buffer).vertices[1];
    daxa_f32vec3 v2 = deref(p.vertex_buffer).vertices[2];

    v0 = (mul(objectToWorld, daxa_f32vec4(v0, 1))).xyz;
    v1 = (mul(objectToWorld, daxa_f32vec4(v1, 1))).xyz;
    v2 = (mul(objectToWorld, daxa_f32vec4(v2, 1))).xyz;

    daxa_f32vec3 u = v1 - v0;
    daxa_f32vec3 v = v2 - v0;

    daxa_f32vec3 world_nrm = normalize(cross(u, v));

    const daxa_f32vec3 diffuse = interpolatedColor;

    daxa_f32vec3 hitLocation = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    rayLight cLight = {hitLocation, 0.0, daxa_f32vec3(0.0, 0.0, 0.0),   0.0};

    if (p.callable_index == 0)
        CallShader(0, cLight);
    else if (p.callable_index == 1)
        CallShader(1, cLight);
    else
    {
        daxa_f32vec3 lDir = daxa_f32vec3(1.0, 2.0, -0.4) - cLight.inHitPosition;
        cLight.outLightDistance = length(lDir);
        cLight.outIntensity = 5.0 / (cLight.outLightDistance * cLight.outLightDistance);
        cLight.outLightDir = normalize(lDir);
    }

    daxa_f32 attenuation = 0.3;

    // Tracing shadow ray only if the light is visible from the surface
    if (dot(world_nrm, cLight.outLightDir) > 0)
    {
        RayDesc ray = {hitLocation, 0.001, cLight.outLightDir, cLight.outLightDistance};
        daxa_u32 flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE  | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
        ShadowRayPayload shadowPayload = { daxa_f32vec4(0.0, 0.0, 0.0, 0.0) };

        TraceRay(RaytracingAccelerationStructure::get(p.tlas), flags, ~0, 1, 0, 0, ray, shadowPayload);

        attenuation =  1.0 - (0.7 * shadowPayload.shadow.x);
    }

    payload.hitValue = daxa_f32vec3(cLight.outIntensity * attenuation * (diffuse));
}

// #define DEBUG_NORMALS

[shader("closesthit")] void procedural_closest_hit_shader(inout hitPayload payload,
                                                       in MyParams attr)
{
    daxa_f32vec3 hitLocation = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    rayLight cLight = {hitLocation, 0.0, daxa_f32vec3(0.0, 0.0, 0.0),   0.0};

    if (p.callable_index == 0)
        CallShader(0, cLight);
    else if (p.callable_index == 1)
        CallShader(1, cLight);
    else
    {
        daxa_f32vec3 lDir = daxa_f32vec3(1.0, 2.0, 0.4) - cLight.inHitPosition;
        cLight.outLightDistance = length(lDir);
        cLight.outIntensity = 100.0 / (cLight.outLightDistance * cLight.outLightDistance);
        cLight.outLightDir = normalize(lDir);
    }

    daxa_u32 prim_index = PrimitiveIndex() +  GeometryIndex();

    Aabb aabb = deref(p.aabb_buffer).aabbs[prim_index];
    
    daxa_f32mat4x4 objectToWorld = transpose(Convert3x4To4x4(ObjectToWorld4x3()));

    daxa_f32vec3 center = (aabb.minimum + aabb.maximum) * 0.5;
    center = (mul(objectToWorld, daxa_f32vec4(center, 1))).xyz;

    // Computing the normal at hit position
    daxa_f32vec3 world_nrm = normalize(hitLocation - center);

    {
        daxa_f32vec3 absN = abs(world_nrm);
        daxa_f32 maxC = max(max(absN.x, absN.y), absN.z);
        world_nrm = (maxC == absN.x) ? daxa_f32vec3(sign(world_nrm.x), 0, 0) : (maxC == absN.y) ? daxa_f32vec3(0, sign(world_nrm.y), 0)
                                                                                        : daxa_f32vec3(0, 0, sign(world_nrm.z));
    }

    const daxa_f32vec3 diffuse = world_nrm * 0.5 + 0.5;
#ifdef DEBUG_NORMALS
    payload.hitValue = diffuse;
#else
    daxa_f32 attenuation = 0.3;

    // Tracing shadow ray only if the light is visible from the surface
    if (dot(world_nrm, cLight.outLightDir) > 0)
    {
        RayDesc ray = {hitLocation, 0.001, cLight.outLightDir, cLight.outLightDistance};
        daxa_u32 flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE  | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
        ShadowRayPayload shadowPayload = { daxa_f32vec4(0.0, 0.0, 0.0, 0.0) };

        TraceRay(RaytracingAccelerationStructure::get(p.tlas), flags, ~0, 1, 0, 0, ray, shadowPayload);

        attenuation =  1.0 - (0.7 * shadowPayload.shadow.x);
    }

    payload.hitValue = daxa_f32vec3(cLight.outIntensity * attenuation * (diffuse));
#endif // DEBUG_NORMALS
}

// Shader de intersección
[shader("intersection")] 
void intersection_shader()
{
    daxa_f32 THit = RayTCurrent();
    MyParams params = {daxa_f32vec4(1.0, 0.0, 0.0, 1.0)};

    daxa_u32 prim_index = PrimitiveIndex() +  GeometryIndex();

    // Obtener el AABB del objeto
    Aabb aabb = deref(p.aabb_buffer).aabbs[prim_index];

    THit = aabb_get_hit(aabb, WorldRayOrigin(), WorldRayDirection(), WorldToObject4x3());

    if(THit > 0.0)
    {
        ReportHit(THit, /*hitKind*/ 0, params);
    }
}

// hardcoded dissolve

[shader("anyhit")]
void anyhit_shader( inout hitPayload payload, in MyParams attr )
{
    const daxa_f32 DISSOLVE = 0.3;
    
    daxa_u32 seed = payload.seed; // We don't want to modify the PRD
    if (rnd(seed) > DISSOLVE)
        IgnoreHit();

    AcceptHitAndEndSearch(); // aborts function
}

[shader("callable")]
void callable_shader(inout rayLight cLight)
{
    // Harcoded light direction
    const daxa_f32vec3 lightDirection = daxa_f32vec3(0.0, 1.0, 0.0);
    
    cLight.outLightDistance = 1000000;
    cLight.outIntensity = 3.0;
    cLight.outLightDir = normalize(lightDirection);
}

[shader("callable")]
void spot_light_callable_shader(inout rayLight cLight)
{
    // Hardcoded light position
    const daxa_f32vec3 lightPos = daxa_f32vec3(1.0, 2.0, 0.4);
    // Hardcoded light intensity
    const daxa_f32 lightIntensity = 10.0;

    daxa_f32vec3 lDir = lightPos - cLight.inHitPosition;
    cLight.outLightDistance = length(lDir);
    cLight.outIntensity = lightIntensity / (cLight.outLightDistance * cLight.outLightDistance);
    cLight.outLightDir = normalize(lDir);
    
}
