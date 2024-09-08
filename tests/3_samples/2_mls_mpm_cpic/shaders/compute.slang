#include "shared.inl"
import "custom file!!";

[[vk::push_constant]] ComputePush p;

struct Test
{
    GpuInput* value;
    Particle* particles;
    Cell* cells;
};

[shader("compute")]
    [numthreads(MPM_GRID_COMPUTE_X, MPM_GRID_COMPUTE_Y, MPM_GRID_COMPUTE_Z)] void
    entry_MPM_reset_grid(daxa_u32vec3 pixel_i : SV_DispatchThreadID)
{
    GpuInput* config = p.input_ptr;

  if (pixel_i.x >= config->grid_dim.x || pixel_i.y >= config->grid_dim.y || pixel_i.z >= config->grid_dim.z)
  {
    return;
  }

  Cell* cells = p.cells;

  uint cell_index = pixel_i.x + pixel_i.y * config->grid_dim.x + pixel_i.z * config->grid_dim.x * config->grid_dim.y;

  Ptr<Cell> cell = &Ptr<Cell>(cells)[cell_index];

  cell->v = daxa_f32vec3(0, 0, 0);
  cell->m = 0;
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

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  daxa_f32mat3x3 affine; // Affine matrix
  daxa_f32vec3 mv = calculate_p2g(particle, dt, p_vol, mu_0, lambda_0, inv_dx, p_mass, affine);

  daxa_u32vec3 array_grid = daxa_u32vec3(base_coord);

  // Scatter to grid
  for(uint i = 0; i < 3; ++i) {
    for(uint j = 0; j < 3; ++j) {
      for(uint k = 0; k < 3; ++k) {
        daxa_u32vec3 coord = array_grid + daxa_u32vec3(i, j, k);
        if(coord.x >= config->grid_dim.x || coord.y >= config->grid_dim.y || coord.z >= config->grid_dim.z) {
          continue;
        }

        daxa_f32vec3 dpos = (daxa_f32vec3(i, j, k) - fx) * dx;

        float weight = w[i].x * w[j].y * w[k].z;

        float m = weight * p_mass;
        daxa_f32vec3 velocity_mass = calculate_weighted_p2g_velocity(dpos, weight, mv, affine);

        uint index = (coord.x + coord.y * config->grid_dim.x + coord.z * config->grid_dim.x * config->grid_dim.y);

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
  uint bound = 3;

  uint cell_index = pixel_i.x + pixel_i.y * config->grid_dim.x + pixel_i.z * config->grid_dim.x * config->grid_dim.y;

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

  GpuInput* config = p.input_ptr;

  if (pixel_i.x >= config->p_count)
  {
    return;
  }


  float dx = config->dx;
  float inv_dx = config->inv_dx;
  float dt = config->dt;

  Particle* particles = p.particles;
  Cell* cells = p.cells;
  Aabb* aabbs = p.aabbs;

  Particle particle = particles[pixel_i.x];
  Ptr<Aabb> aabb = &aabbs[pixel_i.x];

  daxa_f32vec3 center = (aabb->min + aabb->max) * 0.5f;
  
  if(any(center < daxa_f32vec3(0)) || any(center >= daxa_f32vec3(1))) {
      return;
  }

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(*aabb, inv_dx, fx, w);


  daxa_f32mat3x3 particle_C = daxa_f32mat3x3(0);
  daxa_f32vec3 particle_velocity = daxa_f32vec3(0);
  
  daxa_u32vec3 array_grid = daxa_u32vec3(base_coord);

  for(uint i = 0; i < 3; ++i) {
    for(uint j = 0; j < 3; ++j) {
      for(uint k = 0; k < 3; ++k) {
        int3 coord = array_grid + int3(i, j, k);
        
        if(coord.x >= config->grid_dim.x || coord.y >= config->grid_dim.y || coord.z >= config->grid_dim.z) {
          continue;
        }

        daxa_f32vec3 dpos = (daxa_f32vec3(i, j, k) - fx) * dx;

        float weight = w[i].x * w[j].y * w[k].z;

        uint index = coord.x + coord.y * config->grid_dim.x + coord.z * config->grid_dim.x * config->grid_dim.y;

        daxa_f32vec3 vel_value = cells[index].v;

        daxa_f32vec3 w_grid = daxa_f32vec3(vel_value * weight);

        particle_velocity += w_grid; // Velocity
        particle_C += calculate_weighted_g2p_deformation(dpos, weight, vel_value); // Deformation gradient
      }
    }
  }

  particle.C = 4 * inv_dx * inv_dx * particle_C;
  particle.v = particle_velocity;

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