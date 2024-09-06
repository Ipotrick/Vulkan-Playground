#define DAXA_RAY_TRACING 1
#extension GL_EXT_ray_tracing : enable
#if defined(ACTIVATE_ATOMIC_FLOAT)
#extension GL_EXT_shader_atomic_float : enable
#endif
#include <daxa/daxa.inl>

#include "shared.inl"

#if SER_ON == 1
#extension GL_NV_shader_invocation_reorder : enable
layout(location = 0) hitObjectAttributeNV vec3 hitValue;
#endif

DAXA_DECL_PUSH_CONSTANT(PushConstant, p)

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN

layout(location = 0) rayPayloadEXT hitPayload prd;


#if PRIMARY_RAYS == 1
    void main()
{
    const ivec2 index = ivec2(gl_LaunchIDEXT.xy);
    daxa_u32vec2 rt_size = daxa_u32vec2(gl_LaunchSizeEXT.xy);

    uint cull_mask = 0xff;

    daxa_f32vec3 color = daxa_f32vec3(0);

    // Camera setup
    daxa_f32mat4x4 inv_view = deref(p.camera_buffer).inv_view;
    daxa_f32mat4x4 inv_proj = deref(p.camera_buffer).inv_proj;

    const vec2 pixelCenter = vec2(index) + vec2(0.5);
    const vec2 inUV = pixelCenter / vec2(rt_size);
    vec2 d = inUV * 2.0 - 1.0;

    vec4 origin = inv_view * vec4(0, 0, 0, 1);
    vec4 target = inv_proj * vec4(d.x, d.y, 1, 1);
    vec4 direction = inv_view * vec4(normalize(target.xyz), 0);

    uint rayFlags = gl_RayFlagsNoneEXT;
    float tMin = 0.0001;
    float tMax = 10000.0;
    uint cullMask = 0xFF;

    daxa_u32 frame_number = p.frame;
    daxa_u32 seed = tea(index.y * rt_size.x + index.x, frame_number);
    prd.seed = seed;

#if SER_ON == 1
    hitObjectNV hitObject;
    //Initialize to an empty hit object
    hitObjectRecordEmptyNV(hitObject);

    // Trace the ray
    hitObjectTraceRayNV(hitObject,
                        daxa_accelerationStructureEXT(p.tlas), // topLevelAccelerationStructure
                        rayFlags,      // rayFlags
                        cullMask,      // cullMask
                        0,             // sbtRecordOffset
                        0,             // sbtRecordStride
                        0,             // missIndex
                        origin.xyz,    // ray origin
                        tMin,          // ray min range
                        direction.xyz, // ray direction
                        tMax,          // ray max range
                        0              // payload (location = 0)
    );
    
    if(hitObjectIsHitNV(hitObject))
    { 
        daxa_f32 tHit = hitObjectGetRayTMaxNV(hitObject);
        color = 1.0 / tHit * vec3(1.0, 1.0, 1.0);
#if defined(ACTIVATE_ATOMIC_FLOAT)
        atomicAdd(deref(p.camera_buffer).hit_count, 1);
#endif
    }

#else
    
#extension GL_EXT_ray_query : enable

    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, daxa_accelerationStructureEXT(p.tlas),
        rayFlags,
        cullMask, // cullMask
        origin.xyz, tMin, direction.xyz, tMax);

    while (rayQueryProceedEXT(rayQuery))
    {
        uint type = rayQueryGetIntersectionTypeEXT(rayQuery, false);
        if (type == gl_RayQueryCandidateIntersectionTriangleEXT)
        {
            rayQueryConfirmIntersectionEXT(rayQuery);
        }
        else if (type == gl_RayQueryCandidateIntersectionAABBEXT)
        {
            rayQueryGenerateIntersectionEXT(rayQuery, tMax);
        }
    }

    uint type = rayQueryGetIntersectionTypeEXT(rayQuery, true);

    if (type == gl_RayQueryCommittedIntersectionTriangleEXT)
    {
        color = vec3(1.0, 1.0, 1.0);
#if defined(ACTIVATE_ATOMIC_FLOAT)
        atomicAdd(deref(p.camera_buffer).hit_count, 1);
#endif
    }
    else if (type == gl_RayQueryCommittedIntersectionGeneratedEXT)
    {
        color = vec3(1.0, 1.0, 1.0);
#if defined(ACTIVATE_ATOMIC_FLOAT)
        atomicAdd(deref(p.camera_buffer).hit_count, 1);
#endif
    }
#endif

    // imageStore(daxa_image2D(p.swapchain), index, fromLinear(vec4(prd.hitValue, 1.0)));
    imageStore(daxa_image2D(p.swapchain), index, vec4(color, 1.0));
    // imageStore(daxa_image2D(p.swapchain), index, vec4(1.0, 1.0, 1.0, 1.0));
}

#else 

const uint NBSAMPLES = 1;

void main()
{
    const ivec2 index = ivec2(gl_LaunchIDEXT.xy);
    daxa_u32vec2 rt_size = daxa_u32vec2(gl_LaunchSizeEXT.xy);

    uint cull_mask = 0xff;

    // Camera setup
    daxa_f32mat4x4 inv_view = deref(p.camera_buffer).inv_view;
    daxa_f32mat4x4 inv_proj = deref(p.camera_buffer).inv_proj;

    const vec2 pixelCenter = vec2(index) + vec2(0.5);
    const vec2 inUV = pixelCenter / vec2(rt_size);
    vec2 d = inUV * 2.0 - 1.0;

    vec4 origin = inv_view * vec4(0, 0, 0, 1);
    vec4 target = inv_proj * vec4(d.x, d.y, 1, 1);
    vec4 direction = inv_view * vec4(normalize(target.xyz), 0);

    uint rayFlags = gl_RayFlagsNoneEXT;
    float tMin = 0.0001;
    float tMax = 10000.0;
    uint cullMask = 0xFF;

    daxa_u32 frame_number = p.frame;
    daxa_u32 seed = tea(index.y * rt_size.x + index.x, frame_number);
    prd.seed = seed;

#if SER_ON == 1
    hitObjectNV hitObject;
    //Initialize to an empty hit object
    hitObjectRecordEmptyNV(hitObject);

    // Trace the ray
    hitObjectTraceRayNV(hitObject,
                        daxa_accelerationStructureEXT(p.tlas), // topLevelAccelerationStructure
                        rayFlags,      // rayFlags
                        cullMask,      // cullMask
                        0,             // sbtRecordOffset
                        0,             // sbtRecordStride
                        0,             // missIndex
                        origin.xyz,    // ray origin
                        tMin,          // ray min range
                        direction.xyz, // ray direction
                        tMax,          // ray max range
                        0              // payload (location = 0)
    );


    int mesh_id = 0;
    
    if(hitObjectIsHitNV(hitObject))
    { 
        mesh_id = hitObjectGetInstanceCustomIndexNV(hitObject);
    }
        

    // Reorder the ray (based on the hit object or whatever else user wants)
    reorderThreadNV(hitObject, mesh_id, 2);
    // reorderThreadNV(hitObject);
    // reorderThreadNV(mesh_id, 2);

    //Get Attributes
    hitObjectGetAttributesNV(hitObject, 0); // hitObjectAttributeNV hit_value
    // Execute either the closest hit or the miss shader
    hitObjectExecuteShaderNV(hitObject, 0); // hitObjectAttributeNV hit_value
    

#else
#if defined(ACTIVATE_ATOMIC_FLOAT)
    prd.is_hit = false;
#endif
    traceRayEXT(
        daxa_accelerationStructureEXT(p.tlas), // topLevelAccelerationStructure
        rayFlags,      // rayFlags
        cullMask,      // cullMask
        0,             // sbtRecordOffset
        0,             // sbtRecordStride
        0,             // missIndex
        origin.xyz,    // ray origin
        tMin,          // ray min range
        direction.xyz, // ray direction
        tMax,          // ray max range
        0              // payload (location = 0)
    );
#endif

#if defined(ACTIVATE_ATOMIC_FLOAT)
    if(prd.is_hit)
    {
        atomicAdd(deref(p.camera_buffer).hit_count, 1);
    }
#endif

    imageStore(daxa_image2D(p.swapchain), index, vec4(prd.hitValue, 1.0));
}

#endif // PRIMARY_RAYS

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION

void main()
{
    Ray ray;
    ray.origin = gl_ObjectRayOriginEXT;
    ray.direction = gl_ObjectRayDirectionEXT;

    mat4 inv_model = mat4(
        gl_ObjectToWorld3x4EXT[0][0], gl_ObjectToWorld3x4EXT[0][1], gl_ObjectToWorld3x4EXT[0][2], gl_ObjectToWorld3x4EXT[0][3],
        gl_ObjectToWorld3x4EXT[0][1], gl_ObjectToWorld3x4EXT[1][1], gl_ObjectToWorld3x4EXT[1][2], gl_ObjectToWorld3x4EXT[1][3],
        gl_ObjectToWorld3x4EXT[2][0], gl_ObjectToWorld3x4EXT[2][1], gl_ObjectToWorld3x4EXT[2][2], gl_ObjectToWorld3x4EXT[2][3],
        0, 0, 0, 1.0);

    ray.origin = (inv_model * vec4(ray.origin, 1)).xyz;
    ray.direction = (inv_model * vec4(ray.direction, 0)).xyz;

    float tHit = -1;

    uint i = gl_PrimitiveID + gl_GeometryIndexEXT + gl_InstanceCustomIndexEXT;

    Aabb aabb = deref(p.aabb_buffer).aabbs[i];
    // aabb.minimum = (model * vec4(aabb.minimum, 1)).xyz;
    // aabb.maximum = (model * vec4(aabb.maximum, 1)).xyz;

    tHit = hitAabb(aabb, ray);

    // Report hit point
    if (tHit > 0)
        reportIntersectionEXT(tHit, 0); // 0 is the hit kind (hit group index)
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_ANY_HIT

layout(location = 0) rayPayloadInEXT hitPayload prd;

// hardcoded dissolve
const float dissolve = 0.3;

void main()
{

    uint seed = prd.seed; // We don't want to modify the PRD
    if (rnd(seed) > dissolve)
        ignoreIntersectionEXT;
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CALLABLE

layout(location = 3) callableDataInEXT rayLight cLight;

#if defined(SPOT_LIGHT)

// Hardcoded light position
vec3 lightPos = vec3(1.0, 2.0, 0.4);

// Hardcoded light intensity
float lightIntensity = 10.0;

void main()
{
    vec3 lDir = lightPos - cLight.inHitPosition;
    cLight.outLightDistance = length(lDir);
    cLight.outIntensity = lightIntensity / (cLight.outLightDistance * cLight.outLightDistance);
    cLight.outLightDir = normalize(lDir);
}

#else

// Harcoded light direction
vec3 lightDirection = vec3(0.0, 1.0, 0.0);

void main()
{
    cLight.outLightDistance = 1000000;
    cLight.outIntensity = 3.0;
    cLight.outLightDir = normalize(lightDirection);
}

#endif // SPOT_LIGHT

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CLOSEST_HIT

layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;
layout(location = 3) callableDataEXT rayLight cLight;

#if defined(HIT_TRIANGLE)

#extension GL_EXT_ray_tracing_position_fetch : require

hitAttributeEXT vec2 attribs;

void main()
{
    daxa_f32vec3 world_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    
    cLight.inHitPosition = world_pos;

    if (p.callable_index == 0)
        executeCallableEXT(0, 3);
    else if(p.callable_index == 1)
        executeCallableEXT(1, 3);
    else
    {
        vec3 lDir = vec3(1.0, 2.0, -0.4) - cLight.inHitPosition;
        cLight.outLightDistance = length(lDir);
        cLight.outIntensity = 5.0 / (cLight.outLightDistance * cLight.outLightDistance);
        cLight.outLightDir = normalize(lDir);
    }

    const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    mat4 model = mat4(
        gl_ObjectToWorldEXT[0][0], gl_ObjectToWorldEXT[0][1], gl_ObjectToWorldEXT[0][2], 0,
        gl_ObjectToWorldEXT[0][1], gl_ObjectToWorldEXT[1][1], gl_ObjectToWorldEXT[1][2], 0,
        gl_ObjectToWorldEXT[2][0], gl_ObjectToWorldEXT[2][1], gl_ObjectToWorldEXT[2][2], 0,
        gl_ObjectToWorldEXT[3][0], gl_ObjectToWorldEXT[3][1], gl_ObjectToWorldEXT[3][2], 1.0);

    
    vec3 v0 = deref(p.vertex_buffer).vertices[0];
    vec3 v1 = deref(p.vertex_buffer).vertices[1];
    vec3 v2 = deref(p.vertex_buffer).vertices[2];

    v0 = (model * vec4(v0, 1)).xyz;
    v1 = (model * vec4(v1, 1)).xyz;
    v2 = (model * vec4(v2, 1)).xyz;
    
    daxa_f32vec3 u = v1 - v0;
    daxa_f32vec3 v = v2 - v0;

    daxa_f32vec3 world_nrm = normalize(cross(u, v));

    // Color vertices (adjust these colors as desired)
    vec3 color0 = vec3(1.0, 0.0, 0.0); // Red
    vec3 color1 = vec3(0.0, 1.0, 0.0); // Green
    vec3 color2 = vec3(0.0, 0.0, 1.0); // Blue

    // Interpolate colors using barycentric coordinates
    vec3 interpolatedColor = barycentrics.x * color0 + barycentrics.y * color1 + barycentrics.z * color2;


    float attenuation = 0.3;

    // Tracing shadow ray only if the light is visible from the surface
    if (dot(world_nrm, cLight.outLightDir) > 0)
    {
        float tMin = 0.001;
        float tMax = cLight.outLightDistance;
        vec3 origin = world_pos;
        vec3 rayDir = cLight.outLightDir;
        uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
        isShadowed = true;
        traceRayEXT(
            daxa_accelerationStructureEXT(p.tlas),
            flags,  // rayFlags
            0xFF,   // cullMask
            0,      // sbtRecordOffset
            0,      // sbtRecordStride
            1,      // missIndex
            origin, // ray origin
            tMin,   // ray min range
            rayDir, // ray direction
            tMax,   // ray max range
            1       // payload (location = 1)
        );

        if (isShadowed)
        {
            attenuation = 0.3;
        }
        else
        {
            attenuation = 1.0;
        }
    }

    prd.hitValue = vec3(cLight.outIntensity * attenuation * (interpolatedColor));
#if defined(ACTIVATE_ATOMIC_FLOAT)
    prd.is_hit = true;
#endif // HIT_TRIANGLE
}

#else // HIT_TRIANGLE

// #define DEBUG_NORMALS

void main()
{
    vec3 world_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    cLight.inHitPosition = world_pos;

    if (p.callable_index == 0)
        executeCallableEXT(0, 3);
    else if(p.callable_index == 1)
        executeCallableEXT(1, 3);
    else
    {
        vec3 lDir = vec3(1.0, 2.0, 0.4) - cLight.inHitPosition;
        cLight.outLightDistance = length(lDir);
        cLight.outIntensity = 100.0 / (cLight.outLightDistance * cLight.outLightDistance);
        cLight.outLightDir = normalize(lDir);
    }

    mat4 model = mat4(
        gl_ObjectToWorldEXT[0][0], gl_ObjectToWorldEXT[0][1], gl_ObjectToWorldEXT[0][2], 0,
        gl_ObjectToWorldEXT[0][1], gl_ObjectToWorldEXT[1][1], gl_ObjectToWorldEXT[1][2], 0,
        gl_ObjectToWorldEXT[2][0], gl_ObjectToWorldEXT[2][1], gl_ObjectToWorldEXT[2][2], 0,
        gl_ObjectToWorldEXT[3][0], gl_ObjectToWorldEXT[3][1], gl_ObjectToWorldEXT[3][2], 1.0);

    uint prim_index = gl_PrimitiveID + gl_GeometryIndexEXT + gl_InstanceCustomIndexEXT;

    Aabb aabb = deref(p.aabb_buffer).aabbs[prim_index];
    vec3 center = (aabb.minimum + aabb.maximum) * 0.5;
    center = (model * vec4(center, 1)).xyz;

    // Computing the normal at hit position
    vec3 world_nrm = normalize(world_pos - center);

    {
        vec3 absN = abs(world_nrm);
        float maxC = max(max(absN.x, absN.y), absN.z);
        world_nrm = (maxC == absN.x) ? vec3(sign(world_nrm.x), 0, 0) : (maxC == absN.y) ? vec3(0, sign(world_nrm.y), 0)
                                                                                        : vec3(0, 0, sign(world_nrm.z));
    }

    const vec3 diffuse = world_nrm * 0.5 + 0.5;

#ifdef DEBUG_NORMALS
    prd.hitValue = diffuse;
#else
    float attenuation = 0.3;

    // Tracing shadow ray only if the light is visible from the surface
    if (dot(world_nrm, cLight.outLightDir) > 0)
    {
        float tMin = 0.001;
        float tMax = cLight.outLightDistance;
        vec3 origin = world_pos;
        vec3 rayDir = cLight.outLightDir;
        uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
        isShadowed = true;
        traceRayEXT(
            daxa_accelerationStructureEXT(p.tlas),
            flags,  // rayFlags
            0xFF,   // cullMask
            0,      // sbtRecordOffset
            0,      // sbtRecordStride
            1,      // missIndex
            origin, // ray origin
            tMin,   // ray min range
            rayDir, // ray direction
            tMax,   // ray max range
            1       // payload (location = 1)
        );

        if (isShadowed)
        {
            attenuation = 0.3;
        }
        else
        {
            attenuation = 1.0;
        }
    }

    prd.hitValue = vec3(cLight.outIntensity * attenuation * (diffuse));
#if defined(ACTIVATE_ATOMIC_FLOAT)
    prd.is_hit = true;
#endif // ACTIVATE_ATOMIC_FLOAT
#endif // DEBUG_NORMALS
}

#endif // HIT_TRIANGLE

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_MISS

#if defined(MISS_SHADOW)

layout(location = 1) rayPayloadInEXT bool isShadowed;

void main()
{
    isShadowed = false;
}

#else

layout(location = 0) rayPayloadInEXT hitPayload prd;

void main()
{
    prd.hitValue = vec3(0.5, 0.7, 1.0);
}

#endif // MISS_SHADOW

#endif // DAXA_SHADER_STAGE