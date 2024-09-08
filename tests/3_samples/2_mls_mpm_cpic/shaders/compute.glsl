#include <shared.inl>

#if RESET_RIGID_GRID_COMPUTE_FLAG == 1
layout(local_size_x = MPM_GRID_COMPUTE_X, local_size_y = MPM_GRID_COMPUTE_Y, local_size_z = MPM_GRID_COMPUTE_Z) in;
void main()
{
  uvec3 pixel_i = gl_GlobalInvocationID.xyz;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i.x >= deref(config).grid_dim.x || pixel_i.y >= deref(config).grid_dim.y || pixel_i.z >= deref(config).grid_dim.z)
  {
      return;
  }

  uint cell_index = pixel_i.x + pixel_i.y * deref(config).grid_dim.x + pixel_i.z * deref(config).grid_dim.x * deref(config).grid_dim.y;

  zeroed_out_node_cdf_by_index(cell_index);
}
#elif RIGID_BODY_CHECK_BOUNDARIES_COMPUTE_FLAG == 1
layout(local_size_x = MPM_CPIC_COMPUTE_X, local_size_y = 1, local_size_z = 1) in;
void main()
{
  uint pixel_i_x = gl_GlobalInvocationID.x;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i_x >= deref(config).rigid_body_count)
  {
      return;
  }
  daxa_f32 dt = deref(config).dt;
  daxa_f32 dx = deref(config).dx;

  daxa_BufferPtr(GpuStatus) status = daxa_BufferPtr(GpuStatus)(daxa_id_to_address(p.status_buffer_id));

  rigid_body_check_boundaries(dt, dx, status, config, pixel_i_x);
}
#elif RASTER_RIGID_BOUND_COMPUTE_FLAG == 1
// Main compute shader
layout(local_size_x = MPM_P2G_COMPUTE_X, local_size_y = 1, local_size_z = 1) in;
void main()
{
  daxa_u32 pixel_i_x = gl_GlobalInvocationID.x;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i_x >= deref(config).r_p_count)
  {
      return;
  }

  daxa_f32 dx = deref(config).dx;
  daxa_f32 inv_dx = deref(config).inv_dx;

  rigid_body_raster_rigid_bound(dx, inv_dx, pixel_i_x, daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id)));
}

#elif P2G_WATER_COMPUTE_FLAG == 1
// Main compute shader
layout(local_size_x = MPM_P2G_COMPUTE_X, local_size_y = 1, local_size_z = 1) in;
void main()
{
  uint pixel_i_x = gl_GlobalInvocationID.x;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i_x >= deref(config).p_count)
  {
      return;
  }

  float dx = deref(config).dx;
  float inv_dx = deref(config).inv_dx;
  float dt = deref(config).dt;
  float p_mass = 1.0f;

  Particle particle = get_particle_by_index(pixel_i_x);

  Aabb aabb = get_aabb_by_index(pixel_i_x);
  
  daxa_f32vec3 center = (aabb.min + aabb.max) * 0.5f;
  
  if(any(lessThan(center, vec3(0))) || any(greaterThanEqual(center, vec3(1)))) {
      return;
  }

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  daxa_f32mat3x3 affine = particle.C;

  uvec3 array_grid = uvec3(base_coord);

  // Scatter to grid
  for (uint i = 0; i < 3; ++i)
  {
      for (uint j = 0; j < 3; ++j)
      {
          for (uint k = 0; k < 3; ++k)
          {
              uvec3 coord = array_grid + uvec3(i, j, k);
              if (coord.x >= deref(config).grid_dim.x || coord.y >= deref(config).grid_dim.y || coord.z >= deref(config).grid_dim.z)
              {
                  continue;
              }

              vec3 dpos = (vec3(i, j, k) - fx);
              float weight = w[i].x * w[j].y * w[k].z;
              uint index = (coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y);

              float m = weight * p_mass;
              vec3 velocity_mass = m * (particle.v + affine * dpos);

              set_atomic_cell_vel_x_by_index(index, velocity_mass.x);
              set_atomic_cell_vel_y_by_index(index, velocity_mass.y);
              set_atomic_cell_vel_z_by_index(index, velocity_mass.z);
              set_atomic_cell_mass_by_index(index, m);
          }
      }
  }
}
#elif P2G_WATER_SECOND_COMPUTE_FLAG == 1
// Main compute shader
layout(local_size_x = MPM_P2G_COMPUTE_X, local_size_y = 1, local_size_z = 1) in;
void main()
{
  uint pixel_i_x = gl_GlobalInvocationID.x;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i_x >= deref(config).p_count)
  {
      return;
  }

  // float dx = deref(config).dx;
  float inv_dx = deref(config).inv_dx;
  float dt = deref(config).dt;
  float p_mass = 1.0f;

  // fluid parameters
  const float rest_density = 1.0f;
  const float dynamic_viscosity = 0.1f;
  // equation of state
  const float eos_stiffness = 25.0f;
  const float eos_power = 4;

  Particle particle = get_particle_by_index(pixel_i_x);

  Aabb aabb = get_aabb_by_index(pixel_i_x);
  
  daxa_f32vec3 center = (aabb.min + aabb.max) * 0.5f;
  
  if(any(lessThan(center, vec3(0))) || any(greaterThanEqual(center, vec3(1)))) {
      return;
  }

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  uvec3 array_grid = uvec3(base_coord);

  // estimating particle volume by summing up neighbourhood's weighted mass contribution
  // MPM course, equation 152
  float density = 0.0f;
  for (uint i = 0; i < 3; ++i)
  {
      for (uint j = 0; j < 3; ++j)
      {
          for (uint k = 0; k < 3; ++k)
          {
              uvec3 coord = array_grid + uvec3(i, j, k);
              if (coord.x >= deref(config).grid_dim.x || coord.y >= deref(config).grid_dim.y || coord.z >= deref(config).grid_dim.z)
              {
                  continue;
              }

              float weight = w[i].x * w[j].y * w[k].z;
              uint index = (coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y);

              float mass = get_cell_mass_by_index(index);
              float m = weight * mass;
              density += m;
          }
      }
  }

  float p_vol = p_mass / density;

  // end goal, constitutive equation for isotropic fluid:
  // stress = -pressure * I + viscosity * (velocity_gradient + velocity_gradient_transposed)

  // Tait equation of state. i clamped it as a bit of a hack.
  // clamping helps prevent particles absorbing into each other with negative pressures
  float pressure = max(-0.1f, eos_stiffness * (pow(density / rest_density, eos_power) - 1));

  // velocity gradient - CPIC eq. 17, where deriv of quadratic polynomial is linear
  daxa_f32mat3x3 stress = daxa_f32mat3x3(-pressure) + dynamic_viscosity * (particle.C + transpose(particle.C));

  daxa_f32mat3x3 eq_16_term_0 = -p_vol * 4 * stress * dt;

  for (uint i = 0; i < 3; ++i)
  {
      for (uint j = 0; j < 3; ++j)
      {
          for (uint k = 0; k < 3; ++k)
          {
              uvec3 coord = array_grid + uvec3(i, j, k);
              if (coord.x >= deref(config).grid_dim.x || coord.y >= deref(config).grid_dim.y || coord.z >= deref(config).grid_dim.z)
              {
                  continue;
              }

              vec3 dpos = (vec3(i, j, k) - fx);
              float weight = w[i].x * w[j].y * w[k].z;
              uint index = (coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y);

              // fused force + momentum contribution from MLS-MPM
              vec3 momentum = (eq_16_term_0 * weight) * dpos;

              set_atomic_cell_vel_x_by_index(index, momentum.x);
              set_atomic_cell_vel_y_by_index(index, momentum.y);
              set_atomic_cell_vel_z_by_index(index, momentum.z);
          }
      }
  }
}
#elif P2G_COMPUTE_FLAG == 1
// Main compute shader
layout(local_size_x = MPM_P2G_COMPUTE_X, local_size_y = 1, local_size_z = 1) in;
void main()
{
  uint pixel_i_x = gl_GlobalInvocationID.x;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i_x >= deref(config).p_count)
  {
      return;
  }

  float dx = deref(config).dx;
  float inv_dx = deref(config).inv_dx;
  float dt = deref(config).dt;
  float p_rho = 1;
  float p_vol = (dx * 0.5f) * (dx * 0.5f) * (dx * 0.5f); // Particle volume (cube)
  float p_mass = p_vol * p_rho;
  float E = 1000;
  float nu = 0.2f; //  Poisson's ratio
  float mu_0 = E / (2 * (1 + nu));
  float lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu)); // Lame parameters

  Particle particle = get_particle_by_index(pixel_i_x);

  Aabb aabb = get_aabb_by_index(pixel_i_x);
  
  daxa_f32vec3 center = (aabb.min + aabb.max) * 0.5f;
  
  if(any(lessThan(center, vec3(0))) || any(greaterThanEqual(center, vec3(1)))) {
      return;
  }

#if defined(DAXA_RIGID_BODY_FLAG)
  gather_CDF_compute(pixel_i_x, aabb, daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id)));
  ParticleCDF particle_CDF = get_rigid_particle_CDF_by_index(pixel_i_x);
#endif // DAXA_RIGID_BODY_FLAG

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  daxa_f32mat3x3 affine; // Affine matrix
  daxa_f32vec3 mv = calculate_p2g(particle, dt, p_vol, mu_0, lambda_0, inv_dx, p_mass, affine);

  uvec3 array_grid = uvec3(base_coord);

  // Scatter to grid
  for (uint i = 0; i < 3; ++i)
  {
      for (uint j = 0; j < 3; ++j)
      {
          for (uint k = 0; k < 3; ++k)
          {
              uvec3 coord = array_grid + uvec3(i, j, k);
              if (coord.x >= deref(config).grid_dim.x || coord.y >= deref(config).grid_dim.y || coord.z >= deref(config).grid_dim.z)
              {
                  continue;
              }

              uint index = (coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y);

#if defined(DAXA_RIGID_BODY_FLAG)
              daxa_u32 grid_color = get_node_cdf_color_by_index(index);

              daxa_u32 particle_color = get_rigid_particle_CDF_color_by_index(pixel_i_x);

              // only update compatible particles
              if(!cdf_is_compatible(grid_color, particle_color)) {
                  continue;
              }
#endif // DAXA_RIGID_BODY_FLAG

              vec3 dpos = (vec3(i, j, k) - fx) * dx;

              float weight = w[i].x * w[j].y * w[k].z;

              vec3 velocity_mass = calculate_weighted_p2g_velocity(dpos, weight, mv, affine);
              float m = weight * p_mass;

              set_atomic_cell_vel_x_by_index(index, velocity_mass.x);
              set_atomic_cell_vel_y_by_index(index, velocity_mass.y);
              set_atomic_cell_vel_z_by_index(index, velocity_mass.z);
              set_atomic_cell_mass_by_index(index, m);
          }
      }
  }

  particle_set_F_by_index(pixel_i_x, particle.F);
}
#elif GRID_COMPUTE_FLAG == 1
layout(local_size_x = MPM_GRID_COMPUTE_X, local_size_y = MPM_GRID_COMPUTE_Y, local_size_z = MPM_GRID_COMPUTE_Z) in;
void main()
{
  uvec3 pixel_i = gl_GlobalInvocationID.xyz;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i.x >= deref(config).grid_dim.x || pixel_i.y >= deref(config).grid_dim.y || pixel_i.z >= deref(config).grid_dim.z)
  {
      return;
  }

  uint cell_index = pixel_i.x + pixel_i.y * deref(config).grid_dim.x + pixel_i.z * deref(config).grid_dim.x * deref(config).grid_dim.y;

  float dt = deref(config).dt;
  float gravity = deref(config).gravity;
  float inv_dx = deref(config).inv_dx;
  uint bound = BOUNDARY;

  Cell cell = get_cell_by_index(cell_index);

  if (cell.m != 0)
  {
      cell.v /= cell.m; // Normalize by mass
      // if cell velocity less than 0 and pixel_i.xyz < bound, set to 0
      bool bound_x =
          (pixel_i.x < bound) && (cell.v.x < 0) || (pixel_i.x > deref(config).grid_dim.x - bound) && (cell.v.x > 0);
      bool bound_y =
          (pixel_i.y < bound) && (cell.v.y < 0) || (pixel_i.y > deref(config).grid_dim.y - bound) && (cell.v.y > 0);
      bool bound_z =
          (pixel_i.z < bound) && (cell.v.z < 0) || (pixel_i.z > deref(config).grid_dim.z - bound) && (cell.v.z > 0);
      // cell.v += dt * (vec3(0, gravity, 0) + cell.f / cell.m);
      cell.v += dt * vec3(0, gravity, 0);
      if (bound_x)
      {
          cell.v.x = 0;
      }
      if (bound_y)
      {
          cell.v.y = 0;
      }
      if (bound_z)
      {
          cell.v.z = 0;
      }

      set_cell_by_index(cell_index, cell);
  }
}
#elif G2P_WATER_COMPUTE_FLAG == 1
// Main compute shader
layout(local_size_x = MPM_P2G_COMPUTE_X, local_size_y = 1, local_size_z = 1) in;
void main()
{
  uint pixel_i_x = gl_GlobalInvocationID.x;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i_x >= deref(config).p_count)
  {
      return;
  }

  float dx = deref(config).dx;
  float inv_dx = deref(config).inv_dx;
  float dt = deref(config).dt;
  uint64_t frame_number = deref(config).frame_number;
  float p_mass = 1.0f;

  Particle particle = get_particle_by_index(pixel_i_x);
  Aabb aabb = get_aabb_by_index(pixel_i_x);

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  particle.C = daxa_f32mat3x3(0);
  particle.v = vec3(0.f);

  uvec3 array_grid = uvec3(base_coord);

  for (uint i = 0; i < 3; ++i)
  {
      for (uint j = 0; j < 3; ++j)
      {
          for (uint k = 0; k < 3; ++k)
          {
              uvec3 coord = array_grid + uvec3(i, j, k);

              if (coord.x >= deref(config).grid_dim.x || coord.y >= deref(config).grid_dim.y || coord.z >= deref(config).grid_dim.z)
              {
                  continue;
              }

              uint index = coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y;

              vec3 dpos = (vec3(i, j, k) - fx);
              float weight = w[i].x * w[j].y * w[k].z;

              vec3 vel_value = get_cell_by_index(index).v;

              vec3 w_grid = vec3(vel_value * weight);

              particle.v += w_grid; // Velocity
              particle.C += 4 * weight * outer_product(vel_value, dpos);
          }
      }
  }

  aabb.min += dt * particle.v;
  aabb.max += dt * particle.v;
  
  vec3 pos = (aabb.min + aabb.max) * 0.5f;
  const float wall_min = 3 * dx;
  const float wall_max = (float(deref(config).grid_dim.x) - 3) * dx;

  check_boundaries(pos, particle, wall_min, wall_max);
  
  daxa_BufferPtr(GpuStatus) status = daxa_BufferPtr(GpuStatus)(daxa_id_to_address(p.status_buffer_id));

  daxa_u32 flags = deref(status).flags;
  daxa_f32 mouse_radius = deref(config).mouse_radius;
  daxa_f32vec3 mouse_target = deref(status).mouse_target;

  // Repulsion force
  particle_apply_external_force(particle, pos, wall_min, wall_max, mouse_target, mouse_radius, flags);

  set_aabb_by_index(pixel_i_x, aabb);

  particle_set_velocity_by_index(pixel_i_x, particle.v);
  particle_set_C_by_index(pixel_i_x, particle.C);
}
#elif G2P_COMPUTE_FLAG == 1
layout(local_size_x = MPM_P2G_COMPUTE_X, local_size_y = 1, local_size_z = 1) in;
void main()
{
  uint pixel_i_x = gl_GlobalInvocationID.x;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i_x >= deref(config).p_count)
  {
      return;
  }

  Particle particle = get_particle_by_index(pixel_i_x);
  Aabb aabb = get_aabb_by_index(pixel_i_x);

  vec3 pos_x = (aabb.min + aabb.max) * 0.5f;
  
  if(any(lessThan(pos_x, vec3(0))) || any(greaterThanEqual(pos_x, vec3(1)))) {
      return;
  }

  daxa_BufferPtr(GpuStatus) status = daxa_BufferPtr(GpuStatus)(daxa_id_to_address(p.status_buffer_id));

  float dx = deref(config).dx;
  float inv_dx = deref(config).inv_dx;
  float dt = deref(config).dt;
  float p_mass = 1.0f;

#if defined(DAXA_RIGID_BODY_FLAG)
  ParticleCDF particle_CDF = get_rigid_particle_CDF_by_index(pixel_i_x);
#endif // DAXA_RIGID_BODY_FLAG

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  daxa_f32mat3x3 particle_C = daxa_f32mat3x3(0);
  daxa_f32vec3 particle_velocity = daxa_f32vec3(0);

  uvec3 array_grid = uvec3(base_coord);

  for (daxa_u32 i = 0; i < 3; ++i)
  {
      for (daxa_u32 j = 0; j < 3; ++j)
      {
          for (daxa_u32 k = 0; k < 3; ++k)
          {
              uvec3 coord = array_grid + uvec3(i, j, k);

              if (coord.x >= deref(config).grid_dim.x || coord.y >= deref(config).grid_dim.y || coord.z >= deref(config).grid_dim.z)
              {
                  continue;
              }

              daxa_u32 index = coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y;

              vec3 dpos = (vec3(i, j, k) - fx) * dx;
              float weight = w[i].x * w[j].y * w[k].z;

              vec3 vel_value;
#if defined(DAXA_RIGID_BODY_FLAG)
              rigid_body_g2p_check_particle_interaction(vel_value, index, particle_CDF.color, pos_x, particle.v, particle_CDF.normal, weight, p_mass, dt, dx);
#else
              vel_value = get_cell_by_index(index).v;
#endif // DAXA_RIGID_BODY_FLAG
              vec3 w_grid = vec3(vel_value * weight);

              particle_velocity += w_grid; // Velocity
              particle_C += calculate_weighted_g2p_deformation(dpos, weight, vel_value); // Deformation gradient
          }
      }
  }

  particle.v = particle_velocity;
  particle.C = 4 * inv_dx * inv_dx * particle_C;

  // Apply penalty force to particle if it is in collision with a rigid body
#if defined(DAXA_RIGID_BODY_FLAG)
  if(particle_CDF.difference != 0) {
      daxa_f32vec3 f_penalty = abs(particle_CDF.distance) * particle_CDF.normal * PENALTY_FORCE;
      particle.v += dt * f_penalty / p_mass;
  }
#endif // DAXA_RIGID_BODY_FLAG

  aabb.min += dt * particle.v;
  aabb.max += dt * particle.v;

  daxa_f32vec3 pos = (aabb.min + aabb.max) * 0.5f;
  const float wall_min = BOUNDARY * dx;
  const float wall_max = (float(deref(config).grid_dim.x) - BOUNDARY) * dx;

  check_boundaries(pos, particle, wall_min, wall_max);

  daxa_u32 flags = deref(status).flags;
  daxa_f32 mouse_radius = deref(config).mouse_radius;
  daxa_f32vec3 mouse_target = deref(status).mouse_target;

  // Repulsion force
  particle_apply_external_force(particle, pos, wall_min, wall_max, mouse_target, mouse_radius, flags);

  float max_v = deref(config).max_velocity;

  // cap velocity
  if (length(particle.v) > max_v)
  {
      particle.v = normalize(particle.v) * max_v;
  }

  set_aabb_by_index(pixel_i_x, aabb);

  particle_set_velocity_by_index(pixel_i_x, particle.v);
  particle_set_C_by_index(pixel_i_x, particle.C);
}
#elif ADVECT_RIGID_BODIES_FLAG == 1
layout(local_size_x = MPM_CPIC_COMPUTE_X, local_size_y = 1, local_size_z = 1) in;
void main()
{
  uint pixel_i_x = gl_GlobalInvocationID.x;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i_x >= deref(config).rigid_body_count)
  {
      return;
  }
  daxa_f32 dt = deref(config).dt;
  daxa_f32 gravity = deref(config).gravity;

  daxa_BufferPtr(GpuStatus) status = daxa_BufferPtr(GpuStatus)(daxa_id_to_address(p.status_buffer_id));

  rigid_body_advect(dt, gravity, deref(status).flags, pixel_i_x);
}
#else
// Main compute shader
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{
}
#endif
