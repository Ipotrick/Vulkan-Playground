#include <shared.inl>

struct hitPayload
{
  daxa_f32vec3 hit_value;
  daxa_u32 seed;
  daxa_f32vec3 hit_pos;
};

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN

layout(location = 0) rayPayloadEXT hitPayload prd;
void main()
{
  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));
  
  daxa_BufferPtr(GpuStatus) status = daxa_BufferPtr(GpuStatus)(daxa_id_to_address(p.status_buffer_id));

  prd.hit_value = daxa_f32vec3(0.0);
  prd.seed = 0;
  prd.hit_pos = daxa_f32vec3(MAX_DIST);

  const daxa_u32 cull_mask = 0xff;
  const daxa_u32 ray_flags = gl_RayFlagsNoneEXT;

  const daxa_i32vec2 rt_size = daxa_i32vec2(gl_LaunchSizeEXT.xy);

  // Camera setup
  daxa_f32mat4x4 inv_view = deref(p.camera).inv_view;
  daxa_f32mat4x4 inv_proj = deref(p.camera).inv_proj;

  Ray ray =
      get_ray_from_current_pixel(daxa_f32vec2(gl_LaunchIDEXT.xy), daxa_f32vec2(rt_size), inv_view, inv_proj);

  traceRayEXT(
      daxa_accelerationStructureEXT(p.tlas), // topLevelAccelerationStructure
      ray_flags,                             // rayFlags
      cull_mask,                             // cullMask
      0,                                     // sbtRecordOffset
      0,                                     // sbtRecordStride
      0,                                     // missIndex
      ray.origin.xyz,                        // ray origin
      MIN_DIST,                              // ray min range
      ray.direction.xyz,                     // ray direction
      MAX_DIST,                              // ray max range
      0                                      // payload (location = 0)
  );

  daxa_f32vec3 color = prd.hit_value;

  imageStore(daxa_image2D(p.image_id), daxa_i32vec2(gl_LaunchIDEXT.xy), vec4(color, 1.0));

  if ((deref(status).flags & MOUSE_DOWN_FLAG) == MOUSE_DOWN_FLAG) {
    if (gl_LaunchIDEXT.x == uint(deref(config).mouse_pos.x) && gl_LaunchIDEXT.y == uint(deref(config).mouse_pos.y))
    {
        if (prd.hit_pos != daxa_f32vec3(MAX_DIST))
        {
          deref(status).flags |= MOUSE_TARGET_FLAG;
            deref(status).mouse_target = prd.hit_pos;
        } 
    }
  }
  else {
    deref(status).flags &= ~MOUSE_TARGET_FLAG;
  }

}
#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CLOSEST_HIT

layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool is_shadowed;

vec3 light_position = vec3(2, 5, 3);
vec3 light_intensity = vec3(2.5);

#if defined(HIT_TRIANGLE)

hitAttributeEXT vec2 attribs;

void main()
{

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  vec3 world_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

  prd.hit_pos = world_pos;

  uint o = gl_InstanceCustomIndexEXT - (1 + deref(config).rigid_body_count);
  uint i = gl_PrimitiveID;

  mat3 vertices = get_vertices_by_triangle_index(i);

  vec3 v0 = vertices[0];
  vec3 v1 = vertices[1];
  vec3 v2 = vertices[2];

  mat4 model = mat4(
      gl_ObjectToWorldEXT[0].x, gl_ObjectToWorldEXT[1].x, gl_ObjectToWorldEXT[2].x, gl_ObjectToWorldEXT[3].x,
      gl_ObjectToWorldEXT[0].y, gl_ObjectToWorldEXT[1].y, gl_ObjectToWorldEXT[2].y, gl_ObjectToWorldEXT[3].y,
      gl_ObjectToWorldEXT[0].z, gl_ObjectToWorldEXT[1].z, gl_ObjectToWorldEXT[2].z, gl_ObjectToWorldEXT[3].z,
      0, 0, 0, 1);

  vec3 vertices_world[3];
  vertices_world[0] = (model * vec4(v0, 1)).xyz;
  vertices_world[1] = (model * vec4(v1, 1)).xyz;
  vertices_world[2] = (model * vec4(v2, 1)).xyz;

  vec3 u = vertices_world[1] - vertices_world[0];
  vec3 v = vertices_world[2] - vertices_world[0];

#if TRIANGLE_ORIENTATION == CLOCKWISE
  vec3 normal = normalize(cross(u, v));
#else
  vec3 normal = normalize(cross(v, u));
#endif

  // const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // vec3 normal = normalize(barycentrics.x * vertices[0] + barycentrics.y * vertices[1] + barycentrics.z * vertices[2]);

  vec3 L = normalize(light_position - vec3(0));

  float dotNL = max(dot(normal, L), 0.0);
  vec3 material_color = get_rigid_body_color_by_index(o);

  vec3 diffuse = dotNL * material_color;
  vec3 specular = vec3(0);
  float attenuation = 0.3;

  // Tracing shadow ray only if the light is visible from the surface
  if (dot(normal, L) > 0)
  {
      vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT * 1.0001f;
      vec3 ray_dir = L;
      const uint ray_flags =
          gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
      const daxa_u32 cull_mask = 0xff;
      is_shadowed = true;

      traceRayEXT(
        daxa_accelerationStructureEXT(p.tlas), // topLevelAccelerationStructure
        ray_flags,                             // rayFlags
        cull_mask,                             // cullMask
        0,                                     // sbtRecordOffset
        0,                                     // sbtRecordStride
        1,                                     // missIndex
        origin,                        // ray origin
        MIN_DIST,                              // ray min range
        ray_dir,                     // ray direction
        MAX_DIST,                              // ray max range
        1                                      // payload (location = 0)
    );

      if (is_shadowed)
      {
          attenuation = 0.5;
      }
      else
      {
          attenuation = 1;
          // Specular
          // specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, normal);
      }
  }

  prd.hit_value = vec3(light_intensity * attenuation * (diffuse + specular));
}


#else // HIT_TRIANGLE
void main()
{
  vec3 world_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

  prd.hit_pos = world_pos;

  uint i = gl_PrimitiveID;

  Aabb aabb;
  Particle particle;
  if(gl_InstanceCustomIndexEXT == 0) {
    aabb = get_aabb_by_index(i);
    particle = get_particle_by_index(i);
  } 
#if defined(CHECK_RIGID_BODY_FLAG)
  else {
    RigidParticle rigid_particle = get_rigid_particle_by_index(i);
    aabb.min = rigid_particle.min;
    aabb.max = rigid_particle.max;
    particle.type = MAT_RIGID;
    particle.v = vec3(0);
  }
#endif // CHECK_RIGID_BODY_FLAG

  vec3 center = (aabb.min + aabb.max) * 0.5;

  // Computing the normal at hit position
  vec3 normal = normalize(world_pos - center);

#if defined(VOXEL_PARTICLES)
  vec3 absN = abs(normal);
  float maxC = max(max(absN.x, absN.y), absN.z);
  normal     = (maxC == absN.x) ?
                 vec3(sign(normal.x), 0, 0) :
                 (maxC == absN.y) ? vec3(0, sign(normal.y), 0) : vec3(0, 0, sign(normal.z));
#endif // VOXEL_PARTICLES 

  // Vector toward the light
  vec3 L = normalize(light_position - vec3(0));
  
  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));
  float max_v = deref(config).max_velocity;

  // Diffuse
  float dotNL = max(dot(normal, L), 0.0);
  float gradient = clamp(pow(length(particle.v) / (max_v/10.f), 0.5f), 0.0, 1.0);
  vec3 material_color = vec3(1.0);
  if(particle.type == MAT_WATER) {
    material_color = mix(WATER_LOW_SPEED_COLOR, WATER_HIGH_SPEED_COLOR,  gradient);
  } else if(particle.type == MAT_SNOW) {
    material_color = mix(SNOW_LOW_SPEED_COLOR, SNOW_HIGH_SPEED_COLOR, gradient);
  } else if(particle.type == MAT_JELLY) {
    material_color = mix(JELLY_LOW_SPEED_COLOR, JELLY_HIGH_SPEED_COLOR, gradient);
  } else if(particle.type == MAT_RIGID) {
    material_color = RIGID_BODY_PARTICLE_COLOR;
  }
    
  vec3 diffuse = dotNL * material_color;
  vec3 specular = vec3(0);
  float attenuation = 0.3;

  // Tracing shadow ray only if the light is visible from the surface
  if (dot(normal, L) > 0)
  {
      vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT * 1.0001f;
      vec3 ray_dir = L;
      const uint ray_flags =
          gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
      const daxa_u32 cull_mask = 0xff;
      is_shadowed = true;

      traceRayEXT(
        daxa_accelerationStructureEXT(p.tlas), // topLevelAccelerationStructure
        ray_flags,                             // rayFlags
        cull_mask,                             // cullMask
        0,                                     // sbtRecordOffset
        0,                                     // sbtRecordStride
        1,                                     // missIndex
        origin,                        // ray origin
        MIN_DIST,                              // ray min range
        ray_dir,                     // ray direction
        MAX_DIST,                              // ray max range
        1                                      // payload (location = 0)
    );

      if (is_shadowed)
      {
          attenuation = 0.5;
      }
      else
      {
          attenuation = 1;
          // Specular
          // specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, normal);
      }
  }

  prd.hit_value = vec3(light_intensity * attenuation * (diffuse + specular));
}
#endif // HIT_TRIANGLE

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_MISS

#if DAXA_SHADOW_RAY_FLAG == 1

layout(location = 1) rayPayloadInEXT bool is_shadowed;
void main()
{
  is_shadowed = false;
}

#else

layout(location = 0) rayPayloadInEXT hitPayload prd;

void main()
{
  prd.hit_value = vec3(0.0, 0.0, 0.05); // Background color
}

#endif // DAXA_SHADOW_RAY_FLAG

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

  uint i = gl_PrimitiveID;

  Aabb aabb;
  Particle particle;
  if(gl_InstanceCustomIndexEXT == 0) {
    aabb = get_aabb_by_index(i);
    particle = get_particle_by_index(i);
  } 
#if defined(CHECK_RIGID_BODY_FLAG)
  else {
    RigidParticle rigid_particle = get_rigid_particle_by_index(i);
    aabb.min = rigid_particle.min;
    aabb.max = rigid_particle.max;
    particle.type = MAT_RIGID;
    particle.v = vec3(0);
  }
#endif

#if defined(VOXEL_PARTICLES)
  tHit = hitAabb(aabb, ray);
#else
  vec3 center = (aabb.min + aabb.max) * 0.5;
  // radius inside the AABB
  float radius = (aabb.max.x - aabb.min.x) * 0.5 * 0.9;
  tHit = hitSphere(center, radius, ray);
#endif

  // Report hit point
  if (tHit > 0)
      reportIntersectionEXT(tHit, 0); // 0 is the hit kind (hit group index)
}

#else

void main() {}

#endif