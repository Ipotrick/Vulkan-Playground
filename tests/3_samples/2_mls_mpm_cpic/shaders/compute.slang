#include "shared.inl"

struct Test
{
    GpuInput* value;
    Particle* particles;
    Cell* cells;
};


[shader("compute")]
    [numthreads(MPM_GRID_COMPUTE_X, MPM_GRID_COMPUTE_Y, MPM_GRID_COMPUTE_Z)] void
    entry_MPM_reset_rigid_boundary(daxa_u32vec3 pixel_i : SV_DispatchThreadID)
{
    GpuInput* config = p.input_ptr;

  if (pixel_i.x >= config->grid_dim.x || pixel_i.y >= config->grid_dim.y || pixel_i.z >= config->grid_dim.z)
  {
    return;
  }

  daxa_u32 index = pixel_i.x + pixel_i.y * config->grid_dim.x + pixel_i.z * config->grid_dim.x * config->grid_dim.y;

  zeroed_out_node_cdf_by_index(index);
}

[shader("compute")]
    [numthreads(MPM_CPIC_COMPUTE_X, 1, 1)] void
    entry_MPM_rigid_body_check_boundaries(daxa_u32vec3 pixel_i : SV_DispatchThreadID)
{
  GpuInput* config = p.input_ptr;

  if (pixel_i.x >= config->rigid_body_count)
  {
    return;
  }

  daxa_f32 dx = config->dx;
  daxa_f32 dt = config->dt;

  GpuStatus* status = p.status_ptr;

  rigid_body_check_boundaries(dt, dx, status, config, pixel_i.x);
}


[shader("compute")]
    [numthreads(MPM_P2G_COMPUTE_X, 1, 1)] void
    entry_MPM_raster_rigid_boundary(daxa_u32vec3 pixel_i : SV_DispatchThreadID)
{
  GpuInput* config = p.input_ptr;

  if (pixel_i.x >= config->r_p_count)
  {
    return;
  }
  daxa_f32 dx = config->dx;
  daxa_f32 inv_dx = config->inv_dx;

  rigid_body_raster_rigid_bound(dx, inv_dx, pixel_i.x, config);
}
        
[shader("compute")]
    [numthreads(MPM_P2G_COMPUTE_X, 1, 1)] void
    entry_MPM_P2G(daxa_u32vec3 pixel_i : SV_DispatchThreadID)
{
  GpuInput* config = p.input_ptr;

  if (pixel_i.x >= config->p_count)
  {
    return;
  }

  float dx = config->dx;
  float inv_dx = config->inv_dx;
  float dt = config->dt;
  float p_rho = 1;
  float p_vol = (dx * 0.5f) * (dx * 0.5f) * (dx * 0.5f); // Particle volume (cube)
  float p_mass = p_vol * p_rho;
  float E = 1000;
  float nu = 0.2f; //  Poisson's ratio
  float mu_0 = E / (2 * (1 + nu));
  float lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));  // Lame parameters

  Particle* particles = p.particles;
  Cell* cells = p.cells;
  Aabb* aabbs = p.aabbs;

  Particle particle = particles[pixel_i.x];
  Aabb aabb = aabbs[pixel_i.x];
  
  daxa_f32vec3 center = (aabb.min + aabb.max) * 0.5f;
  
  if(any(center < daxa_f32vec3(0)) || any(center >= daxa_f32vec3(1))) {
      return;
  }

#if defined(DAXA_RIGID_BODY_FLAG)
  gather_CDF_compute(pixel_i.x, aabb, config);
  ParticleCDF particle_CDF = get_rigid_particle_CDF_by_index(pixel_i.x);
#endif // DAXA_RIGID_BODY_FLAG

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  daxa_f32mat3x3 affine; // Affine matrix
  daxa_f32vec3 mv = calculate_p2g(particle, dt, p_vol, mu_0, lambda_0, inv_dx, p_mass, affine);

  daxa_u32vec3 array_grid = daxa_u32vec3(base_coord);

  // Scatter to grid
  for(daxa_u32 i = 0; i < 3; ++i) {
    for(daxa_u32 j = 0; j < 3; ++j) {
      for(daxa_u32 k = 0; k < 3; ++k) {
        daxa_u32vec3 coord = array_grid + daxa_u32vec3(i, j, k);
        if(coord.x >= config->grid_dim.x || coord.y >= config->grid_dim.y || coord.z >= config->grid_dim.z) {
          continue;
        }

        daxa_u32 index = (coord.x + coord.y * config->grid_dim.x + coord.z * config->grid_dim.x * config->grid_dim.y);
        
#if defined(DAXA_RIGID_BODY_FLAG)
        daxa_u32 grid_color = get_node_cdf_color_by_index(index);

        daxa_u32 particle_color = get_rigid_particle_CDF_color_by_index(pixel_i.x);

        // only update compatible particles
        if(!cdf_is_compatible(grid_color, particle_color)) {
            continue;
        }
#endif // DAXA_RIGID_BODY_FLAG

        daxa_f32vec3 dpos = (daxa_f32vec3(i, j, k) - fx) * dx;

        float weight = w[i].x * w[j].y * w[k].z;

        float m = weight * p_mass;
        daxa_f32vec3 velocity_mass = calculate_weighted_p2g_velocity(dpos, weight, mv, affine);

        atomicAdd(cells[index].v.x, velocity_mass.x);
        atomicAdd(cells[index].v.y, velocity_mass.y);
        atomicAdd(cells[index].v.z, velocity_mass.z);
        atomicAdd(cells[index].m, m);
      }
    }
  }

  particles[pixel_i.x].F = particle.F;
}


[shader("compute")]
    [numthreads(MPM_GRID_COMPUTE_X, MPM_GRID_COMPUTE_Y, MPM_GRID_COMPUTE_Z)] void
    entry_MPM_grid(daxa_u32vec3 pixel_i : SV_DispatchThreadID)
{
    GpuInput* config = p.input_ptr;

  if (pixel_i.x >= config->grid_dim.x || pixel_i.y >= config->grid_dim.y || pixel_i.z >= config->grid_dim.z)
  {
    return;
  }

  Cell* cells = p.cells;

  float dt = config->dt;
  float gravity = config->gravity;
  daxa_u32 bound = BOUNDARY;

  daxa_u32 cell_index = pixel_i.x + pixel_i.y * config->grid_dim.x + pixel_i.z * config->grid_dim.x * config->grid_dim.y;

  Ptr<Cell> cell = &Ptr<Cell>(cells)[cell_index];

  if(cell->m != 0) {
    cell->v /= cell->m; // Normalize by mass
    // if cell velocity less than 0 and pixel_i.xyz < bound, set to 0
    bool bound_x =
        (pixel_i.x < bound) & (cell.v.x < 0) | (pixel_i.x > config->grid_dim.x - bound) & (cell.v.x > 0);
    bool bound_y = 
        (pixel_i.y < bound) & (cell.v.y < 0) | (pixel_i.y > config->grid_dim.y - bound) & (cell.v.y > 0);
    bool bound_z = 
        (pixel_i.z < bound) & (cell.v.z < 0) | (pixel_i.z > config->grid_dim.z - bound) & (cell.v.z > 0);
    cell->v += dt * daxa_f32vec3(0, gravity, 0); // Gravity
    if(bound_x) {
      cell.v.x = 0;
    }
    if(bound_y) {
      cell.v.y = 0;
    }
    if(bound_z) {
      cell.v.z = 0;
    }
  }
}


[shader("compute")]
    [numthreads(MPM_P2G_COMPUTE_X, 1, 1)] void
    entry_MPM_G2P(daxa_u32vec3 pixel_i : SV_DispatchThreadID)
{

  Ptr<GpuInput> config = p.input_ptr;

  if (pixel_i.x >= config->p_count)
  {
    return;
  }


  float dx = config->dx;
  float inv_dx = config->inv_dx;
  float dt = config->dt;
  // float p_rho = 1;
  // float p_vol = (dx * 0.5f) * (dx * 0.5f) * (dx * 0.5f); // Particle volume (cube)
  // float p_mass = p_vol * p_rho;
  float p_mass = 1.0;

  Particle* particles = p.particles;
  Cell* cells = p.cells;
  Aabb* aabbs = p.aabbs;

  Particle particle = particles[pixel_i.x];
  Ptr<Aabb> aabb = &aabbs[pixel_i.x];

  daxa_f32vec3 center = (aabb->min + aabb->max) * 0.5f;
  
  if(any(center < daxa_f32vec3(0)) || any(center >= daxa_f32vec3(1))) {
      return;
  }

#if defined(DAXA_RIGID_BODY_FLAG)
  ParticleCDF particle_CDF = get_rigid_particle_CDF_by_index(pixel_i.x);
#endif // DAXA_RIGID_BODY_FLAG

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(*aabb, inv_dx, fx, w);


  daxa_f32mat3x3 particle_C = daxa_f32mat3x3(0);
  daxa_f32vec3 particle_velocity = daxa_f32vec3(0);
  
  daxa_u32vec3 array_grid = daxa_u32vec3(base_coord);

  for(daxa_u32 i = 0; i < 3; ++i) {
    for(daxa_u32 j = 0; j < 3; ++j) {
      for(daxa_u32 k = 0; k < 3; ++k) {
        daxa_u32vec3 coord = array_grid + daxa_u32vec3(i, j, k);
        
        if(coord.x >= config->grid_dim.x || coord.y >= config->grid_dim.y || coord.z >= config->grid_dim.z) {
          continue;
        }

        daxa_u32 index = coord.x + coord.y * config->grid_dim.x + coord.z * config->grid_dim.x * config->grid_dim.y;

        daxa_f32vec3 dpos = (daxa_f32vec3(i, j, k) - fx) * dx;

        float weight = w[i].x * w[j].y * w[k].z;

        daxa_f32vec3 vel_value;
#if defined(DAXA_RIGID_BODY_FLAG)
              rigid_body_g2p_check_particle_interaction(vel_value, index, particle_CDF.color, center, particle.v, particle_CDF.normal, weight, p_mass, dt, dx);
#else
              vel_value = get_cell_by_index(index).v;
#endif // DAXA_RIGID_BODY_FLAG

        daxa_f32vec3 w_grid = daxa_f32vec3(vel_value * weight);

        particle_velocity += w_grid; // Velocity
        particle_C += calculate_weighted_g2p_deformation(dpos, weight, vel_value); // Deformation gradient
      }
    }
  }

  particle.C = 4 * inv_dx * inv_dx * particle_C;
  particle.v = particle_velocity;
  
    // Apply penalty force to particle if it is in collision with a rigid body
#if defined(DAXA_RIGID_BODY_FLAG)
  if(particle_CDF.difference != 0) {
      daxa_f32vec3 f_penalty = abs(particle_CDF.distance) * particle_CDF.normal * PENALTY_FORCE;
      particle.v += dt * f_penalty / p_mass;
  }
#endif // DAXA_RIGID_BODY_FLAG

  aabb->min += particle.v * dt;
  aabb->max += particle.v * dt;

  daxa_f32vec3 pos = (aabb.min + aabb.max) * 0.5f;
  const daxa_f32 wall_min = BOUNDARY * dx;
  const daxa_f32 wall_max = ((config->grid_dim.x) - BOUNDARY) * dx;

  check_boundaries(pos, particle, wall_min, wall_max);

  GpuStatus* status = p.status_ptr;

  daxa_u32 flags = status->flags;
  daxa_f32 mouse_radius = config->mouse_radius;
  daxa_f32vec3 mouse_target = status->mouse_target;

  // Repulsion force
  particle_apply_external_force(particle, pos, wall_min, wall_max, mouse_target, mouse_radius, flags);

  daxa_f32 max_v = config->max_velocity;

  // cap velocity
  if (length(particle.v) > max_v)
  {
      particle.v = normalize(particle.v) * max_v;
  }

  particles[pixel_i.x].v = particle.v;
  particles[pixel_i.x].C = particle.C;
}


[shader("compute")]
    [numthreads(MPM_CPIC_COMPUTE_X, 1, 1)] void
    entry_MPM_advect_rigid_bodies(daxa_u32vec3 pixel_i : SV_DispatchThreadID)
{
  GpuInput* config = p.input_ptr;

  if (pixel_i.x >= config->rigid_body_count)
  {
    return;
  }

  daxa_f32 dt = config->dt;
  daxa_f32 gravity = config->gravity;

  GpuStatus* status = p.status_ptr;

  rigid_body_advect(dt, gravity, status->flags, pixel_i.x);
}