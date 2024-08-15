#include <shared.inl>
// #include <custom file!!>


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

  zeroed_out_rigid_cell_by_index(cell_index);

}

#elif RASTER_RIGID_BOUND_COMPUTE_FLAG == 1
// Main compute shader
layout(local_size_x = MPM_P2G_COMPUTE_X, local_size_y = 1, local_size_z = 1) in;
void main()
{
  uint pixel_i_x = gl_GlobalInvocationID.x;

  daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

  if (pixel_i_x >= deref(config).r_p_count)
  {
      return;
  }

  float dx = deref(config).dx;
  float inv_dx = deref(config).inv_dx;

  RigidParticle particle = get_rigid_particle_by_index(pixel_i_x);

  if(particle.rigid_id > MAX_RIGID_BODY_COUNT) {
    return;
  }

  Aabb aabb = Aabb(particle.min, particle.max);

  daxa_i32vec3 base_coord = calculate_particle_grid_pos(aabb, inv_dx);

  uvec3 array_grid = uvec3(base_coord);

  // get 
  vec3 p0 = get_first_vertex_by_triangle_index(particle.triangle_id);
  vec3 p1 = get_second_vertex_by_triangle_index(particle.triangle_id);
  vec3 p2 = get_third_vertex_by_triangle_index(particle.triangle_id);

  mat3 world_to_elem = get_world_to_object_matrix(p0, p1, p2);

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

              vec3 grid_pos = vec3(coord) * dx;
              vec3 diff = world_to_elem * (grid_pos - p0);
              bool negative = diff[2] < 0;
              daxa_f32 dist = abs(diff[2]);
              bool in_range = false;

              if(0 <= diff[0] && 0 <= diff[1] && diff[0] + diff[1] <= 1) {
                  in_range = true;
              }

              if(!in_range) {
                  continue;
              }

              uint index = (coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y);

              dist *= inv_dx;

              if(set_atomic_rigid_cell_distance_by_index(index, dist) > dist) {
                if(set_atomic_rigid_cell_distance_by_index(index, dist) == dist) {
                  set_atomic_rigid_cell_rigid_id_by_index(index, particle.rigid_id);
                  set_atomic_rigid_cell_rigid_particle_index_by_index(index, pixel_i_x);
                }
              }

              set_atomic_rigid_cell_status_by_index(index, particle.rigid_id, negative);

          }
      }
  }
}

#elif GATHER_CDF_COMPUTE_FLAG == 1
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

  // zeroed_out_rigid_particle_status_by_index(pixel_i_x);

  ParticleCDF particle_status = ParticleCDF(MAX_DIST, 0, false, vec3(0));

  Aabb aabb = get_aabb_by_index(pixel_i_x);

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  daxa_f32 Tpr = 0.f;

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

              daxa_u32 index = coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y;

              daxa_f32vec3 dpos = (daxa_f32vec3(i, j, k) - fx) * dx;

              RigidCell rigid_cell = get_rigid_cell_by_index(index);

              if(rigid_cell.rigid_id == -1) {
                continue;
              }

              particle_status.status |= (rigid_cell.status >> (rigid_cell.rigid_id * 2)) & 0x2;

              float weight = w[i].x * w[j].y * w[k].z;

              float T = ((rigid_cell.status >> (rigid_cell.rigid_id * 2)) & 0x1) == 1 ? -1 : 1;

              float dist = from_emulated_float(rigid_cell.d);

              Tpr += weight * T * dist;
          }
      }
  }


  if(particle_status.status != 0) {
    particle_status.d = abs(Tpr);
    particle_status.negative = sign(Tpr) < 0;
  }

  set_rigid_particle_status_by_index(pixel_i_x, particle_status);

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

  // float dx = deref(config).dx;
  float inv_dx = deref(config).inv_dx;
  float dt = deref(config).dt;
  float p_mass = 1.0f;

  Particle particle = get_particle_by_index(pixel_i_x);

  Aabb aabb = get_aabb_by_index(pixel_i_x);

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  mat3 affine = particle.C;

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
  const float rest_density = 0.2f;
  const float dynamic_viscosity = 0.1f;
  // equation of state
  const float eos_stiffness = 4.0f;
  const float eos_power = 4;

  Particle particle = get_particle_by_index(pixel_i_x);

  Aabb aabb = get_aabb_by_index(pixel_i_x);

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
  float pressure = max(-0.1f, eos_stiffness * (pow(density /rest_density, eos_power) - 1));

  // velocity gradient - CPIC eq. 17, where deriv of quadratic polynomial is linear
  mat3 stress = mat3(-pressure) + dynamic_viscosity * (particle.C + transpose(particle.C));

  mat3 eq_16_term_0 = -p_vol * 4 * stress * dt;

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

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  mat3 stress = calculate_p2g(particle, dt, p_vol, mu_0, lambda_0, inv_dx);

  mat3 affine = stress + p_mass * particle.C;

  // float w_k = 50.f;
  // float w_gamma = 5.0f;

  // float stress = calculate_p2g_water(particle, p_vol, w_k, w_gamma, inv_dx);

  // mat3 affine = p_mass * particle.C;

  // Transactional momentum
  vec3 mv = vec3(p_mass * particle.v);

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

              vec3 dpos = (vec3(i, j, k) - fx) * dx;

              float weight = w[i].x * w[j].y * w[k].z;

              vec3 velocity_mass = weight * (mv + affine * dpos);
              float m = weight * p_mass;

              uint index = (coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y);

              // vec3 force_stress = stress * dpos;

              set_atomic_cell_vel_x_by_index(index, velocity_mass.x);
              set_atomic_cell_vel_y_by_index(index, velocity_mass.y);
              set_atomic_cell_vel_z_by_index(index, velocity_mass.z);
              set_atomic_cell_mass_by_index(index, m);
              // set_atomic_cell_force_x_by_index(index, force_stress.x);
              // set_atomic_cell_force_y_by_index(index, force_stress.y);
              // set_atomic_cell_force_z_by_index(index, force_stress.z);
          }
      }
  }

  // TODO: optimize this write
  set_particle_by_index(pixel_i_x, particle);
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
  uint bound = 3;

  Cell cell = get_cell_by_index(cell_index);

  if(cell.m != 0) {
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
    if(bound_x) {
      cell.v.x = 0;
    }
    if(bound_y) {
      cell.v.y = 0;
    }
    if(bound_z) {
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

  Particle particle = get_particle_by_index(pixel_i_x);
  Aabb aabb = get_aabb_by_index(pixel_i_x);

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  particle.C = mat3(0);
  particle.v = vec3(0.f);
  
  uvec3 array_grid = uvec3(base_coord);

  for(uint i = 0; i < 3; ++i) {
    for(uint j = 0; j < 3; ++j) {
      for(uint k = 0; k < 3; ++k) {
        uvec3 coord = array_grid + uvec3(i, j, k);
        
        if(coord.x >= deref(config).grid_dim.x || coord.y >= deref(config).grid_dim.y || coord.z >= deref(config).grid_dim.z) {
          continue;
        }

        vec3 dpos = (vec3(i, j, k) - fx);
        float weight = w[i].x * w[j].y * w[k].z;
        uint index = coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y;

        vec3 grid_value = get_cell_by_index(index).v;

        vec3 w_grid = vec3(grid_value * weight);

        particle.v += w_grid; // Velocity
        particle.C += 4 * weight * outer_product(grid_value, dpos);
      }
    }
  }

  aabb.min += dt * particle.v;
  aabb.max += dt * particle.v;

  // int bounds = 4;
  // const float tiny_displacement = 1e-9f;

  // uint seed = tea(pixel_i_x, uint(frame_number));

  // // Add a tiny bit displacement to avoid particles sticking together
  // // check if particle is near the boundary using dx
  // vec3 particle_displacement = vec3(0);
  // if(aabb.min.x < (bounds * dx)) {
  //   particle_displacement.x = tiny_displacement;
  // } else if(aabb.max.x > (deref(config).grid_dim.x - bounds) * dx) {
  //   particle_displacement.x = -tiny_displacement;
  // } else {
  //   particle_displacement.x = rnd_interval(seed, -tiny_displacement, tiny_displacement);
  // }

  // if(aabb.min.y < bounds * dx) {
  //   particle_displacement.y = tiny_displacement;
  // } else if(aabb.max.y > (deref(config).grid_dim.y - bounds) * dx) {
  //   particle_displacement.y = -tiny_displacement;
  // } else {
  //   particle_displacement.y = rnd_interval(seed, -tiny_displacement, tiny_displacement);
  // }

  // if(aabb.min.z < bounds * dx) {
  //   particle_displacement.z = tiny_displacement;
  // } else if(aabb.max.z > (deref(config).grid_dim.z - bounds) * dx) {
  //   particle_displacement.z = -tiny_displacement;
  // } else {
  //   particle_displacement.z = rnd_interval(seed, -tiny_displacement, tiny_displacement);
  // }

  // aabb.min += particle_displacement;
  // aabb.max += particle_displacement;
  

  set_aabb_by_index(pixel_i_x, aabb);
  

  // TODO: optimize this write
  set_particle_by_index(pixel_i_x, particle);
}
#elif G2P_COMPUTE_FLAG == 1
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

  daxa_BufferPtr(GpuStatus) status = daxa_BufferPtr(GpuStatus)(daxa_id_to_address(p.status_buffer_id));

  float dx = deref(config).dx;
  float inv_dx = deref(config).inv_dx;
  float dt = deref(config).dt;

  Particle particle = get_particle_by_index(pixel_i_x);
  Aabb aabb = get_aabb_by_index(pixel_i_x);

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  particle.C = mat3(0);
  particle.v = vec3(0.f);
  
  uvec3 array_grid = uvec3(base_coord);

  for(uint i = 0; i < 3; ++i) {
    for(uint j = 0; j < 3; ++j) {
      for(uint k = 0; k < 3; ++k) {
        uvec3 coord = array_grid + uvec3(i, j, k);
        
        if(coord.x >= deref(config).grid_dim.x || coord.y >= deref(config).grid_dim.y || coord.z >= deref(config).grid_dim.z) {
          continue;
        }

        vec3 dpos = (vec3(i, j, k) - fx) * dx;

        float weight = w[i].x * w[j].y * w[k].z;

        uint index = coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y;

        vec3 grid_value = get_cell_by_index(index).v;

        vec3 w_grid = vec3(grid_value * weight);

        particle.v += w_grid; // Velocity
        // particle.C += 4 * inv_dx * weight * outer_product(grid_value, dpos);
        particle.C += 4 * inv_dx * inv_dx * weight * outer_product(grid_value, dpos);
      }
    }
  }

  // particle.J *= 1.0f + dt * trace(particle.C);

  aabb.min += dt * particle.v;
  aabb.max += dt * particle.v;

  vec3 pos = (aabb.min + aabb.max) * 0.5f;
  const float wall_min = 3 * dx;
  float wall_max = (float(deref(config).grid_dim.x) - 3) * dx;

  // Repulsion force
  if ((deref(status).flags & MOUSE_TARGET_FLAG) == MOUSE_TARGET_FLAG) {
    if(all(greaterThan(deref(status).mouse_target, vec3(wall_min))) &&
       all(lessThan(deref(status).mouse_target, vec3(wall_max)))){
      vec3 dist = pos - deref(status).mouse_target;
      if (dot(dist, dist) < deref(config).mouse_radius * deref(config).mouse_radius) {
          vec3 force = normalize(dist) * 0.05f;
          particle.v += force;
      }
    }
  }

  float max_v = deref(config).max_velocity;

  // cap velocity
  if (length(particle.v) > max_v) {
    particle.v = normalize(particle.v) * max_v;
  }


  set_aabb_by_index(pixel_i_x, aabb);

  // TODO: optimize this write
  set_particle_by_index(pixel_i_x, particle);
}
#elif SPHERE_TRACING_FLAG == 1

// Main compute shader
layout(local_size_x = MPM_SHADING_COMPUTE_X, local_size_y = MPM_SHADING_COMPUTE_Y, local_size_z = 1) in;
void main()
{
    uvec2 pixel_i = gl_GlobalInvocationID.xy;

    uvec2 frame_dim = uvec2(deref(p.camera).frame_dim);

    if (pixel_i.x >= deref(p.camera).frame_dim.x || pixel_i.y >= deref(p.camera).frame_dim.y)
        return;
    // Camera setup
    daxa_f32mat4x4 inv_view = deref(p.camera).inv_view;
    daxa_f32mat4x4 inv_proj = deref(p.camera).inv_proj;

    Ray ray =
        get_ray_from_current_pixel(daxa_f32vec2(pixel_i.xy), daxa_f32vec2(frame_dim), inv_view, inv_proj);

    vec3 col = vec3(0, 0, 0); // Background color
    daxa_f32 t = 0.0f;
    daxa_f32 dist = 0.0f;
    
    bool hit = false;
    daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

    while (t < MAX_DIST) {
        vec3 pos = ray.origin + t * ray.direction;
        daxa_f32 min_dist = MAX_DIST;
        for (uint i = 0; i < deref(config).p_count; ++i) {
            Particle particle = get_particle_by_index(i);
            Aabb aabb = get_aabb_by_index(i);
            daxa_f32vec3 p_center = (aabb.min + aabb.max) * 0.5f;
            daxa_f32 hit_dist = compute_sphere_distance(pos, p_center, PARTICLE_RADIUS);
            if (hit_dist < min_dist) {
                min_dist = hit_dist;
            }
        }

        if (min_dist < MIN_DIST) {
            hit = true;
            break;
        }

        t += min_dist;
        if (t >= MAX_DIST) {
            break;
        }
    }

    if (hit) {
        col = vec3(0.3, 0.8, 1); // Sphere color
    }

    imageStore(daxa_image2D(p.image_id), ivec2(pixel_i.xy), vec4(col, 1.0));
}

#else 
// Main compute shader
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
}
#endif