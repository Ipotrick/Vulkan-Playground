#include "shared.inl"

struct MyAttributes
{
    daxa_f32vec3 normal;
    daxa_f32vec3 position;
};

struct MyParams
{
    daxa_f32vec4 color;
};

[shader("raygeneration")]
void ray_generation()
{
  const daxa_u32vec2 threadIdx = DispatchRaysIndex().xy;
  const daxa_u32vec2 rt_size = DispatchRaysDimensions().xy;

  // Camera setup
  daxa_f32mat4x4 inv_view = p.camera->inv_view;
  daxa_f32mat4x4 inv_proj = p.camera->inv_proj;
  
  GpuInput* config = p.input_ptr;

  Ray _ray =
      get_ray_from_current_pixel(daxa_f32vec2(threadIdx), daxa_f32vec2(rt_size), inv_view, inv_proj);

  daxa_f32 tMin = 0.0001;
  daxa_f32 tMax = 10000.0;
  RayDesc ray = {_ray.origin, tMin, _ray.direction, tMax};
  
  daxa_u32 frame_number = daxa_u32(config->frame_number);
  daxa_u32 seed = tea(threadIdx.y * rt_size.x + threadIdx.x, frame_number);

  hitPayload payload = {daxa_f32vec3(0.0), seed, daxa_f32vec3(MAX_DIST), -1, -1};

  TraceRay(RaytracingAccelerationStructure::get(p.tlas), RAY_FLAG_NONE, ~0, 0, 0, 0, ray, payload);

  p.swapchain.get()[threadIdx.xy] = daxa_f32vec4(payload.hit_value, 1.0);

  Ptr<GpuStatus> status = p.status_ptr;

  check_mouse_input(threadIdx, config->mouse_pos, Ray(ray.Origin, ray.Direction), payload, status);
}

[shader("miss")]
void miss(inout hitPayload payload)
{
    payload.hit_value = daxa_f32vec3(0.0, 0.0, 0.05); // Background color
}
[shader("miss")]
void miss_shadows(inout ShadowRayPayload payload)
{
    payload.shadow = 1.0f;
}

CONST_STATIC_VARIABLE daxa_f32vec3 light_position = daxa_f32vec3(2, 5, 3);
CONST_STATIC_VARIABLE daxa_f32vec3 light_intensity = daxa_f32vec3(2.5);
  
[shader("closesthit")]
void closest_hit(inout hitPayload payload, 
in MyAttributes attr)
{
  daxa_f32vec3 hit_location = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

  payload.hit_pos = hit_location;

  daxa_u32 prim_index = PrimitiveIndex();

  daxa_f32mat4x4 objectToWorld = transpose(Convert3x4To4x4(ObjectToWorld4x3()));

  Aabb aabb;
  Particle particle;
  if(InstanceID() == 0) {
    aabb = p.aabbs[prim_index];
    particle = p.particles[prim_index];
  } 
#if defined(DAXA_RIGID_BODY_FLAG)
  else {
    RigidParticle rigid_particle = get_rigid_particle_by_index(prim_index);
      
    aabb.min = mat4_vec4_mul(objectToWorld, daxa_f32vec4(rigid_particle.min, 1)).xyz;
    aabb.max = mat4_vec4_mul(objectToWorld, daxa_f32vec4(rigid_particle.max, 1)).xyz;
    particle.type = MAT_RIGID;
    particle.v = daxa_f32vec3(0);
  }
#endif // DAXA_RIGID_BODY_FLAG

  daxa_f32vec3 center = (aabb.min + aabb.max) * 0.5;

  // Computing the normal at hit position
  daxa_f32vec3 normal = normalize(hit_location - center);

  // Vector toward the light
  daxa_f32vec3 L = normalize(light_position - daxa_f32vec3(0));
  
  GpuInput* config = p.input_ptr;
  daxa_f32 max_v = config->max_velocity;

  // Diffuse
  daxa_f32 dotNL = max(dot(normal, L), 0.0);
  daxa_f32 gradient = clamp(pow(length(particle.v) / (max_v/10.f), 0.5f), 0.0, 1.0);
  daxa_f32vec3 material_color = daxa_f32vec3(1.0);
  if(particle.type == MAT_WATER) {
    material_color = lerp(WATER_LOW_SPEED_COLOR, WATER_HIGH_SPEED_COLOR,  gradient);
  } else if(particle.type == MAT_SNOW) {
    material_color = lerp(SNOW_LOW_SPEED_COLOR, SNOW_HIGH_SPEED_COLOR, gradient);
  } else if(particle.type == MAT_JELLY) {
    material_color = lerp(JELLY_LOW_SPEED_COLOR, JELLY_HIGH_SPEED_COLOR, gradient);
  } else if(particle.type == MAT_RIGID) {
    material_color = RIGID_BODY_PARTICLE_COLOR;
  }
    
  daxa_f32vec3 diffuse = dotNL * material_color;
  daxa_f32vec3 specular = daxa_f32vec3(0);
  daxa_f32 attenuation = 0.5;

  // Tracing shadow ray only if the light is visible from the surface
  if (dot(normal, L) > 0)
  {
    daxa_f32vec3 ray_dir = L;
    RayDesc ray = {hit_location, MIN_DIST, ray_dir, MAX_DIST};
    daxa_u32 flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE  | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
    ShadowRayPayload shadowPayload = { 0.0f };

    TraceRay(RaytracingAccelerationStructure::get(p.tlas), flags, ~0, 1, 0, 0, ray, shadowPayload);

    let shadow_result = (0.5 * shadowPayload.shadow);

    attenuation =  1.0 - shadow_result;

    if (shadow_result > 0)
    {
        attenuation = 1;
        // Specular
        // specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, normal);
    }
  }

  payload.hit_value = daxa_f32vec3(light_intensity * attenuation * (diffuse + specular));
}

#if defined(DAXA_RIGID_BODY_FLAG)
[shader("closesthit")]
void closest_hit_rigids(inout hitPayload payload, 
in MyAttributes attr)
{
  daxa_f32vec3 hit_location = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

  daxa_u32 instance_index = InstanceID() - (1 + Ptr<GpuInput>(p.input_ptr).rigid_body_count);
  daxa_u32 prim_index = PrimitiveIndex();

  payload.hit_pos = hit_location;
  payload.rigid_body_index = instance_index;
  payload.rigid_element_index = prim_index;

  daxa_f32mat4x4 objectToWorld = transpose(Convert3x4To4x4(ObjectToWorld4x3()));
  
  daxa_f32mat3x3 vertices = get_vertices_by_triangle_index(prim_index);

  daxa_f32vec3 v0 = vertices[0];
  daxa_f32vec3 v1 = vertices[1];
  daxa_f32vec3 v2 = vertices[2];

  daxa_f32vec3 vertices_world[3];
  vertices_world[0] = mat4_vec4_mul(objectToWorld, daxa_f32vec4(v0, 1)).xyz;
  vertices_world[1] = mat4_vec4_mul(objectToWorld, daxa_f32vec4(v1, 1)).xyz;
  vertices_world[2] = mat4_vec4_mul(objectToWorld, daxa_f32vec4(v2, 1)).xyz;

  daxa_f32vec3 u = vertices_world[1] - vertices_world[0];
  daxa_f32vec3 v = vertices_world[2] - vertices_world[0];

#if TRIANGLE_ORIENTATION == CLOCKWISE
  daxa_f32vec3 normal = normalize(cross(u, v));
#else
  daxa_f32vec3 normal = normalize(cross(v, u));
#endif

  daxa_f32vec3 L = normalize(light_position - daxa_f32vec3(0));

  float dotNL = max(dot(normal, L), 0.0);
  daxa_f32vec3 material_color = get_rigid_body_color_by_index(instance_index);

  daxa_f32vec3 diffuse = dotNL * material_color;
  daxa_f32vec3 specular = daxa_f32vec3(0);
  float attenuation = 0.3;

  // Tracing shadow ray only if the light is visible from the surface
  if (dot(normal, L) > 0)
  {
    daxa_f32vec3 ray_dir = L;
    RayDesc ray = {hit_location, MIN_DIST, ray_dir, MAX_DIST};
    daxa_u32 flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE  | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
    ShadowRayPayload shadowPayload = { 0.0f };

    TraceRay(RaytracingAccelerationStructure::get(p.tlas), flags, ~0, 1, 0, 0, ray, shadowPayload);

    let shadow_result = (0.5 * shadowPayload.shadow);

    attenuation =  1.0 - shadow_result;

    if (shadow_result > 0)
    {
        attenuation = 1;
        // Specular
        // specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, normal);
    }
  }

  payload.hit_value = daxa_f32vec3(light_intensity * attenuation * (diffuse + specular));
}
#endif // DAXA_RIGID_BODY_FLAG

[shader("intersection")]
void intersectionShader()
{
  daxa_f32 THit = RayTCurrent();
  MyParams params = {daxa_f32vec4(1.0, 0.0, 0.0, 1.0)};

  daxa_u32 prim_index = PrimitiveIndex();

  // Get Aabb
  Aabb aabb = p.aabbs[prim_index];
  Particle particle;
  if(InstanceID() == 0) {
    p.aabbs[prim_index];
    particle = p.particles[prim_index];
  } 
#if defined(DAXA_RIGID_BODY_FLAG)
  else {
    RigidParticle rigid_particle = get_rigid_particle_by_index(prim_index);
    daxa_f32mat4x4 objectToWorld = transpose(Convert3x4To4x4(ObjectToWorld4x3()));


    aabb.min = mat4_vec4_mul(objectToWorld, daxa_f32vec4(rigid_particle.min, 1)).xyz;
    aabb.max = mat4_vec4_mul(objectToWorld, daxa_f32vec4(rigid_particle.max, 1)).xyz;
  }
#endif

  daxa_f32vec3 center = (aabb.min + aabb.max) * 0.5;
  // radius inside the AABB
  daxa_f32 radius = (aabb.max.x - aabb.min.x) * 0.5 * 0.9;
  THit = hitSphere(center, radius, {WorldRayOrigin(), WorldRayDirection()});

  if(THit > 0.0)
  {
      ReportHit(THit, /*hitKind*/ 0, params);
  }
}