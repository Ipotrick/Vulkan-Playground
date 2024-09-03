#include "shared.inl"
import "custom file!!";


// _glsl_extension(GL_EXT_shader_atomic_float)
float atomicAdd(__ref float value, float amount)
{
    __target_switch
    {
    case cpp:
        __requirePrelude("#include <atomic>");
        __intrinsic_asm "std::atomic_ref(*$0).fetch_add($1)";
    case spirv:
        // _glsl_extension(GL_EXT_shader_atomic_float)
        return __atomicAdd(value, amount);
    }
    return 0;
}

[[vk::push_constant]] ComputePush p;

struct Test
{
    GpuInput* value;
    Particle* particles;
    Cell* cells;
};

[shader("compute")]
    [numthreads(MPM_GRID_COMPUTE_X, MPM_GRID_COMPUTE_Y, MPM_GRID_COMPUTE_Z)] void
    entry_MPM_reset_grid(uint3 pixel_i : SV_DispatchThreadID)
{
    GpuInput* config = p.input_ptr;

  if (pixel_i.x >= config->grid_dim.x || pixel_i.y >= config->grid_dim.y || pixel_i.z >= config->grid_dim.z)
  {
    return;
  }

  Cell* cells = p.cells;

  uint cell_index = pixel_i.x + pixel_i.y * config->grid_dim.x + pixel_i.z * config->grid_dim.x * config->grid_dim.y;

  Ptr<Cell> cell = &Ptr<Cell>(cells)[cell_index];

  cell->info = float4(0, 0, 0, 0);
}

        
[shader("compute")]
    [numthreads(MPM_P2G_COMPUTE_X, 1, 1)] void
    entry_MPM_P2G(uint3 pixel_i : SV_DispatchThreadID)
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

  float3 w[3];
  float3 fx;
  int3 base_coord;

  float3x3 stress = calculate_p2g(particle, aabb, dt, p_vol, mu_0, lambda_0, inv_dx, base_coord, fx, w);

  float3x3 affine = stress + p_mass * particle.C;

  // Transactional momentum
  float3 mv = float3(p_mass * particle.v);

  uint3 array_grid = uint3(base_coord);

  // Scatter to grid
  for(uint i = 0; i < 3; ++i) {
    for(uint j = 0; j < 3; ++j) {
      for(uint k = 0; k < 3; ++k) {
        uint3 coord = array_grid + uint3(i, j, k);
        if(coord.x >= config->grid_dim.x || coord.y >= config->grid_dim.y || coord.z >= config->grid_dim.z) {
          continue;
        }

        float3 dpos = (float3(i, j, k) - fx) * dx;

        float weight = w[i].x * w[j].y * w[k].z;

        float3 velocity_mass = weight * (mv + mul(affine, dpos));
        float m = weight * p_mass;

        uint index = (coord.x + coord.y * config->grid_dim.x + coord.z * config->grid_dim.x * config->grid_dim.y);

        float4 old_vel_mass_value;
        Ptr<float> grid_x = &Ptr<float>(cells)[(index*4) + 0];
        Ptr<float> grid_y = &Ptr<float>(cells)[(index*4) + 1];
        Ptr<float> grid_z = &Ptr<float>(cells)[(index*4) + 2];
        Ptr<float> grid_w = &Ptr<float>(cells)[(index*4) + 3];

        old_vel_mass_value.x = atomicAdd(*grid_x, velocity_mass.x);
        old_vel_mass_value.y = atomicAdd(*grid_y, velocity_mass.y);
        old_vel_mass_value.z = atomicAdd(*grid_z, velocity_mass.z);
        old_vel_mass_value.w = atomicAdd(*grid_w, m);

        // Ptr<Cell> grid_cell = &cells[index];
        // grid_cell.info.x += velocity_mass.x;
        // grid_cell.info.y += velocity_mass.y;
        // grid_cell.info.z += velocity_mass.z;
        // grid_cell.info.w += m;
      }
    }
  }

  // TODO: optimize this write
  particles[pixel_i.x] = particle;
}


[shader("compute")]
    [numthreads(MPM_GRID_COMPUTE_X, MPM_GRID_COMPUTE_Y, MPM_GRID_COMPUTE_Z)] void
    entry_MPM_grid(uint3 pixel_i : SV_DispatchThreadID)
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

  if(cell->info.w != 0) {
    cell->info.xyz /= cell->info.w; // Normalize by mass
    // if cell velocity less than 0 and pixel_i.xyz < bound, set to 0
    bool bound_x =
        (pixel_i.x < bound) & (cell->info.x < 0) | (pixel_i.x > config->grid_dim.x - bound) & (cell->info.x > 0);
    bool bound_y = 
        (pixel_i.y < bound) & (cell->info.y < 0) | (pixel_i.y > config->grid_dim.y - bound) & (cell->info.y > 0);
    bool bound_z = 
        (pixel_i.z < bound) & (cell->info.z < 0) | (pixel_i.z > config->grid_dim.z - bound) & (cell->info.z > 0);
    cell->info += dt * float4(0, gravity, 0, 0); // Gravity
    if(bound_x) {
      cell->info.x = 0;
    }
    if(bound_y) {
      cell->info.y = 0;
    }
    if(bound_z) {
      cell->info.z = 0;
    }
  }
}


[shader("compute")]
    [numthreads(MPM_P2G_COMPUTE_X, 1, 1)] void
    entry_MPM_G2P(uint3 pixel_i : SV_DispatchThreadID)
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
  
  float3 particle_center = (aabb->min + aabb->max) * 0.5f * inv_dx;
  float3 particle_center_dx = particle_center - float3(0.5f, 0.5f, 0.5f);

  int3 base_coord = int3(particle_center_dx); // Floor

  float3 fx = particle_center - float3(base_coord); // Fractional

  // Quadratic kernels Eqn. 123, with x=fx, fx-1,fx-2]
  float3 x = float3(1.5) - fx;
  float3 y = fx - float3(1.0);
  float3 z = fx - float3(0.5);

  float3 w[3] = {float3(0.5) * (x * x),
                 float3(0.75) - y * y,
                 float3(0.5) * (z * z)};

  particle.C = float3x3(0);
  particle.v = float3(0.f);
  
  uint3 array_grid = uint3(base_coord);

  for(uint i = 0; i < 3; ++i) {
    for(uint j = 0; j < 3; ++j) {
      for(uint k = 0; k < 3; ++k) {
        int3 coord = array_grid + int3(i, j, k);
        
        if(coord.x >= config->grid_dim.x || coord.y >= config->grid_dim.y || coord.z >= config->grid_dim.z) {
          continue;
        }

        float3 dpos = (float3(i, j, k) - fx) * dx;

        float weight = w[i].x * w[j].y * w[k].z;

        uint index = coord.x + coord.y * config->grid_dim.x + coord.z * config->grid_dim.x * config->grid_dim.y;

        float4 grid_value = cells[index].info;

        float3 w_grid = float3(grid_value.xyz * weight);

        particle.v += w_grid; // Velocity
        particle.C += 4 * inv_dx * inv_dx * weight * outer_product(grid_value.xyz, dpos);
      }
    }
  }

  aabb->min += particle.v * dt;
  aabb->max += particle.v * dt;

  // TODO: optimize this write
  particles[pixel_i.x] = particle;
}

[shader("compute")]
[numthreads(MPM_SHADING_COMPUTE_X, MPM_SHADING_COMPUTE_Y, 1)]
void entry_sphere_tracing(uint3 pixel_i : SV_DispatchThreadID)
{
    Camera* cam = p.camera;

    if (pixel_i.x >= cam->frame_dim.x || pixel_i.y >= cam->frame_dim.y)
        return;
    // Camera setup
    daxa_f32mat4x4 inv_view = cam->inv_view;
    daxa_f32mat4x4 inv_proj = cam->inv_proj;

    Ray ray =
        get_ray_from_current_pixel(daxa_f32vec2(pixel_i.xy), daxa_f32vec2(cam->frame_dim), inv_view, inv_proj);

    float3 col = float3(0, 0, 0); // Background color
    daxa::f32 t = 0.0f;
    daxa::f32 dist = 0.0f;
    
    bool hit = false;
    Particle* particles = p.particles;
    Aabb* aabbs = p.aabbs;
    GpuInput* config = p.input_ptr;

    while (t < MAX_DIST) {
        float3 pos = ray.origin + t * ray.direction;
        daxa::f32 min_dist = MAX_DIST;
        for (uint i = 0; i < config->p_count; ++i) {
            Particle particle = particles[i];
            Aabb aabb = aabbs[i];
            float3 p_center = (aabb.min + aabb.max) * 0.5f;
            daxa::f32 hit_dist = compute_sphere_distance(pos, p_center, PARTICLE_RADIUS);
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
        col = float3(0.3, 0.8, 1); // Sphere color
    }

    RWTexture2D<float3> tex = daxa::storage_images[p.image_id.index()];
    tex[pixel_i.xy] = col;
}