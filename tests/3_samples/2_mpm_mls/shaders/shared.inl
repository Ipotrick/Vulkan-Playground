#pragma once

#define DAXA_RAY_TRACING 1
#if defined(GL_core_profile) // GLSL
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_ARB_gpu_shader_int64 : enable
#endif // GL_core_profile


#include "daxa/daxa.inl"

#define GRID_DIM 128
#define GRID_SIZE (GRID_DIM * GRID_DIM * GRID_DIM)
#define QUALITY 2
#define SIM_LOOP_COUNT 30
#define CHECK_RIGID_BODY_FLAG 1
// #define NUM_PARTICLES 8192 * QUALITY * QUALITY * QUALITY
#define NUM_PARTICLES 16384 * QUALITY * QUALITY * QUALITY
// #define NUM_PARTICLES 32768 * QUALITY * QUALITY * QUALITY
// #define NUM_PARTICLES 65536 * QUALITY * QUALITY * QUALITY
// #define NUM_PARTICLES 512
// #define NUM_PARTICLES 64
#define NUM_RIGID_BOXES 1
#define TOTAL_AABB_COUNT (NUM_PARTICLES + NUM_RIGID_BOXES)

#define MPM_P2G_COMPUTE_X 64
#define MPM_GRID_COMPUTE_X 4 
#define MPM_GRID_COMPUTE_Y 4
#define MPM_GRID_COMPUTE_Z 4

#define MPM_SHADING_COMPUTE_X 8
#define MPM_SHADING_COMPUTE_Y 8

#define PARTICLE_RADIUS 0.0025f
#define MIN_DIST 1e-6f
#define MAX_DIST 1e10f
#define EULER 2.71828
// #define VOXEL_PARTICLES 1
#define MOUSE_DOWN_FLAG (1u << 0)
#define MOUSE_TARGET_FLAG (1u << 1)

#define MAT_WATER 0
#define MAT_SNOW 1
#define MAT_JELLY 2
// #define MAT_SAND 3
#define MAT_RIGID_BOX 4
#define MAT_COUNT (MAT_JELLY + 1)

struct Camera {
  daxa_f32mat4x4 inv_view;
  daxa_f32mat4x4 inv_proj;
  daxa_u32vec2 frame_dim;
};

struct GpuInput
{
  daxa_u32 p_count;
  daxa_u32 p_count_rigid_boxes;
  daxa_u32vec3 grid_dim;
  daxa_f32 dt;
  daxa_f32 dx;
  daxa_f32 inv_dx;
  daxa_f32 gravity;
  daxa_u64 frame_number;
  daxa_f32vec2 mouse_pos;
  daxa_f32 mouse_radius;
  daxa_f32 max_velocity;
};

struct GpuStatus 
{
  daxa_u32 flags;
  daxa_f32vec3 mouse_target;
};

struct Particle {
  daxa_u32 type;
  daxa_f32vec3 v;
  daxa_f32mat3x3 F;
  daxa_f32mat3x3 C;
  daxa_f32 J;
};


struct Cell {
  daxa_f32vec3 v;
  daxa_f32 m;
  // daxa_f32vec3 f;
};

struct Aabb {
  daxa_f32vec3 min;
  daxa_f32vec3 max;
};

DAXA_DECL_BUFFER_PTR(GpuInput)
DAXA_DECL_BUFFER_PTR(GpuStatus)
DAXA_DECL_BUFFER_PTR(Particle)
DAXA_DECL_BUFFER_PTR(Cell)
DAXA_DECL_BUFFER_PTR(Camera)
DAXA_DECL_BUFFER_PTR(Aabb)

struct ComputePush
{
    daxa_ImageViewId image_id;
    daxa_BufferId input_buffer_id;
    daxa_RWBufferPtr(GpuInput) input_ptr;
    daxa_BufferId status_buffer_id;
    daxa_RWBufferPtr(Particle) particles;
    daxa_RWBufferPtr(Cell) cells;
    daxa_RWBufferPtr(Aabb) aabbs;
    daxa_BufferPtr(Camera) camera;
    daxa_TlasId tlas;
};


struct Ray
{
  daxa_f32vec3 origin;
  daxa_f32vec3 direction;
};



#define Four_Gamma_Squared 5.82842712474619f
#define Sine_Pi_Over_Eight 0.3826834323650897f
#define Cosine_Pi_Over_Eight 0.9238795325112867f
#define One_Half 0.5f
#define One 1.0f
#define Tiny_Number 1.e-20f
#define Small_Number 1.e-12f


#if !defined(__cplusplus)

#if defined(GL_core_profile) // GLSL
#extension GL_EXT_shader_atomic_float : enable

DAXA_DECL_PUSH_CONSTANT(ComputePush, p)

layout(buffer_reference, scalar) buffer PARTICLE_BUFFER {Particle particles[]; }; // Particle buffer
layout(buffer_reference, scalar) buffer CELL_BUFFER {Cell cells[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer AABB_BUFFER {Aabb aabbs[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer RIGID_BODY_AABB_IDS {uint rigid_box_ids[]; }; // Positions of an object


Particle get_particle_by_index(daxa_u32 particle_index) {
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  return particle_buffer.particles[particle_index];
}

Cell get_cell_by_index(daxa_u32 cell_index) {
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return cell_buffer.cells[cell_index];
}

daxa_f32vec3 get_cell_vel_by_index(daxa_u32 cell_index) {
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return cell_buffer.cells[cell_index].v;
}

daxa_f32 get_cell_mass_by_index(daxa_u32 cell_index) {
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return cell_buffer.cells[cell_index].m;
}

Aabb get_aabb_by_index(daxa_u32 aabb_index) {
  AABB_BUFFER aabb_buffer = AABB_BUFFER(p.aabbs);
  return aabb_buffer.aabbs[aabb_index];
}

void set_particle_by_index(daxa_u32 particle_index, Particle particle) {
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  particle_buffer.particles[particle_index] = particle;
}

void zeroed_out_cell_by_index(daxa_u32 cell_index) {
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  cell_buffer.cells[cell_index].v = vec3(0, 0, 0);
  cell_buffer.cells[cell_index].m = 0;
}

void set_cell_by_index(daxa_u32 cell_index, Cell cell) {
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  cell_buffer.cells[cell_index] = cell;
}

daxa_f32 set_atomic_cell_vel_x_by_index(daxa_u32 cell_index, daxa_f32 x) {
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return atomicAdd(cell_buffer.cells[cell_index].v.x, x);
}

daxa_f32 set_atomic_cell_vel_y_by_index(daxa_u32 cell_index, daxa_f32 y) {
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return atomicAdd(cell_buffer.cells[cell_index].v.y, y);
}

daxa_f32 set_atomic_cell_vel_z_by_index(daxa_u32 cell_index, daxa_f32 z) {
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return atomicAdd(cell_buffer.cells[cell_index].v.z, z);
}

daxa_f32 set_atomic_cell_mass_by_index(daxa_u32 cell_index, daxa_f32 w) {
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return atomicAdd(cell_buffer.cells[cell_index].m, w);
}

// daxa_f32 set_atomic_cell_force_x_by_index(daxa_u32 cell_index, daxa_f32 f) {
//   CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
//   return atomicAdd(cell_buffer.cells[cell_index].f.x, f);
// }

// daxa_f32 set_atomic_cell_force_y_by_index(daxa_u32 cell_index, daxa_f32 f) {
//   CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
//   return atomicAdd(cell_buffer.cells[cell_index].f.y, f);
// }

// daxa_f32 set_atomic_cell_force_z_by_index(daxa_u32 cell_index, daxa_f32 f) {
//   CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
//   return atomicAdd(cell_buffer.cells[cell_index].f.z, f);
// }

void set_aabb_by_index(daxa_u32 aabb_index, Aabb aabb) {
  AABB_BUFFER aabb_buffer = AABB_BUFFER(p.aabbs);
  aabb_buffer.aabbs[aabb_index] = aabb;
}
#endif // GL_core_profile



// Generate a random unsigned int from two unsigned int values, using 16 pairs
// of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"
daxa_u32 tea(daxa_u32 val0, daxa_u32 val1)
{
  daxa_u32 v0 = val0;
  daxa_u32 v1 = val1;
  daxa_u32 s0 = 0;

  for(daxa_u32 n = 0; n < 16; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}


// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
daxa_u32 lcg(inout daxa_u32 prev)
{
  daxa_u32 LCG_A = 1664525u;
  daxa_u32 LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate a random float in [0, 1) given the previous RNG state
daxa_f32 rnd(inout daxa_u32 prev)
{
  return (daxa_f32(lcg(prev)) / daxa_f32(0x01000000));
}

daxa_f32 rnd_interval(inout daxa_u32 prev, daxa_f32 min, daxa_f32 max)
{
  return min + rnd(prev) * (max - min);
}




daxa_f32mat3x3 outer_product(daxa_f32vec3 a, daxa_f32vec3 b)
{
    return daxa_f32mat3x3(a.x * b, a.y * b, a.z * b);
}

daxa_f32 trace(daxa_f32mat3x3 m)
{
    return m[0][0] + m[1][1] + m[2][2];
}

daxa_f32mat3x3 matmul(daxa_f32mat3x3 a, daxa_f32mat3x3 b) {
    return daxa_f32mat3x3(
        dot(a[0], daxa_f32vec3(b[0][0], b[1][0], b[2][0])),
        dot(a[0], daxa_f32vec3(b[0][1], b[1][1], b[2][1])),
        dot(a[0], daxa_f32vec3(b[0][2], b[1][2], b[2][2])),
        dot(a[1], daxa_f32vec3(b[0][0], b[1][0], b[2][0])),
        dot(a[1], daxa_f32vec3(b[0][1], b[1][1], b[2][1])),
        dot(a[1], daxa_f32vec3(b[0][2], b[1][2], b[2][2])),
        dot(a[2], daxa_f32vec3(b[0][0], b[1][0], b[2][0])),
        dot(a[2], daxa_f32vec3(b[0][1], b[1][1], b[2][1])),
        dot(a[2], daxa_f32vec3(b[0][2], b[1][2], b[2][2]))
    );
}


daxa_f32 rsqrt(daxa_f32 f)
{
  return 1.0f / sqrt(f);
}

void swap(inout daxa_f32 a, inout daxa_f32 b)
{
  daxa_f32 temp = a;
  a = b;
  b = temp;
}

void swapColumns(inout daxa_f32mat3x3 mat, daxa_i32 col1, daxa_i32 col2)
{
  daxa_f32vec3 temp = daxa_f32vec3(mat[0][col1], mat[1][col1], mat[2][col1]);
  mat[0][col1] = mat[0][col2];
  mat[1][col1] = mat[1][col2];
  mat[2][col1] = mat[2][col2];
  mat[0][col2] = temp.x;
  mat[1][col2] = temp.y;
  mat[2][col2] = temp.z;
}

// Function to normalize a vector and handle small magnitude cases
daxa_f32vec3 normalizeSafe(daxa_f32vec3 v, daxa_f32 epsilon)
{
  daxa_f32 len = length(v);
  return len > epsilon ? v / len : daxa_f32vec3(1, 0, 0);
}

// Main SVD function
void svd(daxa_f32mat3x3 A, out daxa_f32mat3x3 U, out daxa_f32mat3x3 S, out daxa_f32mat3x3 V, int iters)
{
  // Initialize U, V as identity matrices
  U = daxa_f32mat3x3(1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f);
  V = daxa_f32mat3x3(1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f);
  S = A;

  // Perform Jacobi iterations
  for (int sweep = 0; sweep < iters; sweep++)
  {
      daxa_f32 Sch, Ssh, Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;

      // First rotation (zero out Ss21)
      Ssh = S[1][0] * One_Half;
      Stmp5 = S[0][0] - S[1][1];
      Stmp2 = Ssh * Ssh;
      Stmp1 = (Stmp2 >= Tiny_Number) ? 1.0f : 0.0f;
      Ssh = Stmp1 * Ssh;
      Sch = Stmp1 * Stmp5 + (1.0f - Stmp1);
      Stmp1 = Ssh * Ssh;
      Stmp2 = Sch * Sch;
      Stmp3 = Stmp1 + Stmp2;
      Stmp4 = rsqrt(Stmp3);
      Ssh *= Stmp4;
      Sch *= Stmp4;
      Stmp1 = Four_Gamma_Squared * Stmp1;
      Stmp1 = (Stmp2 <= Stmp1) ? 1.0f : 0.0f;
      Ssh = Ssh * (1.0f - Stmp1) + Stmp1 * Sine_Pi_Over_Eight;
      Sch = Sch * (1.0f - Stmp1) + Stmp1 * Cosine_Pi_Over_Eight;
      Stmp1 = Ssh * Ssh;
      Stmp2 = Sch * Sch;
      daxa_f32 Sc = Stmp2 - Stmp1;
      daxa_f32 Ss = 2.0f * Sch * Ssh;

      Stmp1 = Ss * S[2][0];
      Stmp2 = Ss * S[2][1];
      S[2][0] = Sc * S[2][0] + Stmp2;
      S[2][1] = Sc * S[2][1] - Stmp1;

      Stmp2 = Ss * Ss * S[0][0];
      Stmp3 = Ss * Ss * S[1][1];
      Stmp4 = Sch * Sch;
      S[0][0] = S[0][0] * Stmp4 + Stmp3;
      S[1][1] = S[1][1] * Stmp4 + Stmp2;
      Stmp4 = Stmp4 - Ss * Ss;
      S[1][0] = S[1][0] * Stmp4 - Stmp5 * Ss * Ss;
      S[0][0] += S[1][0] * 2.0f * Sch * Ssh;
      S[1][1] -= S[1][0] * 2.0f * Sch * Ssh;
      S[1][0] *= Sch * Sch - Ss * Ss;

      daxa_f32 Sqvs = 1.0f, Sqvvx = 0.0f, Sqvvy = 0.0f, Sqvvz = 0.0f;

      Stmp1 = Ssh * Sqvvx;
      Stmp2 = Ssh * Sqvvy;
      Stmp3 = Ssh * Sqvvz;
      Ssh *= Sqvs;

      Sqvs = Sch * Sqvs - Stmp3;
      Sqvvx = Sch * Sqvvx + Stmp2;
      Sqvvy = Sch * Sqvvy - Stmp1;
      Sqvvz = Sch * Sqvvz + Ssh;

      // Second rotation (zero out Ss32)
      Ssh = S[2][1] * One_Half;
      Stmp5 = S[1][1] - S[2][2];
      Stmp2 = Ssh * Ssh;
      Stmp1 = (Stmp2 >= Tiny_Number) ? 1.0f : 0.0f;
      Ssh = Stmp1 * Ssh;
      Sch = Stmp1 * Stmp5 + (1.0f - Stmp1);
      Stmp1 = Ssh * Ssh;
      Stmp2 = Sch * Sch;
      Stmp3 = Stmp1 + Stmp2;
      Stmp4 = rsqrt(Stmp3);
      Ssh *= Stmp4;
      Sch *= Stmp4;
      Stmp1 = Four_Gamma_Squared * Stmp1;
      Stmp1 = (Stmp2 <= Stmp1) ? 1.0f : 0.0f;
      Ssh = Ssh * (1.0f - Stmp1) + Stmp1 * Sine_Pi_Over_Eight;
      Sch = Sch * (1.0f - Stmp1) + Stmp1 * Cosine_Pi_Over_Eight;
      Stmp1 = Ssh * Ssh;
      Stmp2 = Sch * Sch;
      Sc = Stmp2 - Stmp1;
      Ss = 2.0f * Sch * Ssh;

      Stmp1 = Ss * S[1][0];
      Stmp2 = Ss * S[2][0];
      S[1][0] = Sc * S[1][0] + Stmp2;
      S[2][0] = Sc * S[2][0] - Stmp1;

      Stmp2 = Ss * Ss * S[1][1];
      Stmp3 = Ss * Ss * S[2][2];
      Stmp4 = Sch * Sch;
      S[1][1] = S[1][1] * Stmp4 + Stmp3;
      S[2][2] = S[2][2] * Stmp4 + Stmp2;
      Stmp4 = Stmp4 - Ss * Ss;
      S[2][1] = S[2][1] * Stmp4 - Stmp5 * Ss * Ss;
      S[1][1] += S[2][1] * 2.0f * Sch * Ssh;
      S[2][2] -= S[2][1] * 2.0f * Sch * Ssh;
      S[2][1] *= Sch * Sch - Ss * Ss;

      Stmp1 = Ssh * Sqvvx;
      Stmp2 = Ssh * Sqvvy;
      Stmp3 = Ssh * Sqvvz;
      Ssh *= Sqvs;

      Sqvs = Sch * Sqvs - Stmp3;
      Sqvvx = Sch * Sqvvx + Stmp2;
      Sqvvy = Sch * Sqvvy - Stmp1;
      Sqvvz = Sch * Sqvvz + Ssh;

      // Third rotation (zero out Ss31)
      Ssh = S[2][0] * One_Half;
      Stmp5 = S[0][0] - S[2][2];
      Stmp2 = Ssh * Ssh;
      Stmp1 = (Stmp2 >= Tiny_Number) ? 1.0f : 0.0f;
      Ssh = Stmp1 * Ssh;
      Sch = Stmp1 * Stmp5 + (1.0f - Stmp1);
      Stmp1 = Ssh * Ssh;
      Stmp2 = Sch * Sch;
      Stmp3 = Stmp1 + Stmp2;
      Stmp4 = rsqrt(Stmp3);
      Ssh *= Stmp4;
      Sch *= Stmp4;
      Stmp1 = Four_Gamma_Squared * Stmp1;
      Stmp1 = (Stmp2 <= Stmp1) ? 1.0f : 0.0f;
      Ssh = Ssh * (1.0f - Stmp1) + Stmp1 * Sine_Pi_Over_Eight;
      Sch = Sch * (1.0f - Stmp1) + Stmp1 * Cosine_Pi_Over_Eight;
      Stmp1 = Ssh * Ssh;
      Stmp2 = Sch * Sch;
      Sc = Stmp2 - Stmp1;
      Ss = 2.0f * Sch * Ssh;

      Stmp1 = Ss * S[1][0];
      Stmp2 = Ss * S[2][0];
      S[1][0] = Sc * S[1][0] + Stmp2;
      S[2][0] = Sc * S[2][0] - Stmp1;

      Stmp2 = Ss * Ss * S[0][0];
      Stmp3 = Ss * Ss * S[2][2];
      Stmp4 = Sch * Sch;
      S[0][0] = S[0][0] * Stmp4 + Stmp3;
      S[2][2] = S[2][2] * Stmp4 + Stmp2;
      Stmp4 = Stmp4 - Ss * Ss;
      S[2][0] = S[2][0] * Stmp4 - Stmp5 * Ss * Ss;
      S[0][0] += S[2][0] * 2.0f * Sch * Ssh;
      S[2][2] -= S[2][0] * 2.0f * Sch * Ssh;
      S[2][0] *= Sch * Sch - Ss * Ss;

      Stmp1 = Ssh * Sqvvx;
      Stmp2 = Ssh * Sqvvy;
      Stmp3 = Ssh * Sqvvz;
      Ssh *= Sqvs;

      Sqvs = Sch * Sqvs - Stmp3;
      Sqvvx = Sch * Sqvvx + Stmp2;
      Sqvvy = Sch * Sqvvy - Stmp1;
      Sqvvz = Sch * Sqvvz + Ssh;
  }

  // Sorting singular values and ensuring non-negative values
  daxa_f32 sigma1 = S[0][0], sigma2 = S[1][1], sigma3 = S[2][2];
  if (sigma1 < sigma2)
  {
      swap(sigma1, sigma2);
      swapColumns(U, 0, 1);
      swapColumns(V, 0, 1);
  }
  if (sigma1 < sigma3)
  {
      swap(sigma1, sigma3);
      swapColumns(U, 0, 2);
      swapColumns(V, 0, 2);
  }
  if (sigma2 < sigma3)
  {
      swap(sigma2, sigma3);
      swapColumns(U, 1, 2);
      swapColumns(V, 1, 2);
  }
  if (sigma1 < 0.0f)
  {
      sigma1 = -sigma1;
      U[0] = -U[0];
  }
  if (sigma2 < 0.0f)
  {
      sigma2 = -sigma2;
      U[1] = -U[1];
  }
  if (sigma3 < 0.0f)
  {
      sigma3 = -sigma3;
      U[2] = -U[2];
  }

  // Construct diagonal matrix S
  S = daxa_f32mat3x3(sigma1, 0.0f, 0.0f,
                     0.0f, sigma2, 0.0f,
                     0.0f, 0.0f, sigma3);
}

daxa_f32mat3x3 element_wise_mul(daxa_f32mat3x3 a, daxa_f32mat3x3 b)
{
  return daxa_f32mat3x3(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

daxa_f32mat3x3 mat3_mul(daxa_f32mat3x3 a, daxa_f32mat3x3 b) {
    daxa_f32mat3x3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    return result;
}

daxa_f32vec4 mat4_vec4_mul(daxa_f32mat4x4 m, daxa_f32vec4 v) {
    return 
#if defined(GL_core_profile) // GLSL
      (m * v);
#else // HLSL
      mul(m, v);
#endif
}

daxa_f32mat3x3 calculate_stress(daxa_f32mat3x3 F, daxa_f32mat3x3 U, daxa_f32mat3x3 V, daxa_f32 mu, daxa_f32 la, daxa_f32 J) {
    daxa_f32mat3x3 V_T = transpose(V); // Transpuesta de V
    daxa_f32mat3x3 U_V_T = mat3_mul(U, V_T); // U @ V.transpose()
    daxa_f32mat3x3 F_T = transpose(F); // Transpuesta de F_dg[p]

    daxa_f32mat3x3 term1 = 2.0 * mu * mat3_mul((F - U_V_T), F_T); // 2 * mu * (F_dg[p] - U @ V.transpose()) @ F_dg[p].transpose()
    daxa_f32mat3x3 identity = daxa_f32mat3x3(1.0); // Matriz de identidad
    daxa_f32mat3x3 term2 = identity * la * J * (J - 1.0); // ti.Matrix.identity(daxa_f32, 3) * la * J * (J - 1)

    return term1 + term2;
}

daxa_f32mat3x3 update_deformation_gradient(daxa_f32mat3x3 F, daxa_f32mat3x3 C, daxa_f32 dt) {
    daxa_f32mat3x3 identity = daxa_f32mat3x3(1.0); // Matriz de identidad
    return element_wise_mul(identity + dt * C, F); // deformation gradient update
}



daxa_i32vec3 calculate_particle_status(Aabb aabb, daxa_f32 inv_dx, out daxa_f32vec3 fx, out daxa_f32vec3 w[3]) {
  
  daxa_f32vec3 particle_center = (aabb.min + aabb.max) * 0.5f * inv_dx;
  daxa_f32vec3 particle_center_dx = particle_center - daxa_f32vec3(0.5f, 0.5f, 0.5f);

  daxa_i32vec3 base_coord = daxa_i32vec3(particle_center_dx); // Floor

  fx = particle_center - daxa_f32vec3(base_coord); // Fractional

  // Quadratic kernels Eqn. 123, with x=fx, fx-1,fx-2]
  daxa_f32vec3 x = daxa_f32vec3(1.5) - fx;
  daxa_f32vec3 y = fx - daxa_f32vec3(1.0);
  daxa_f32vec3 z = fx - daxa_f32vec3(0.5);

  w[0] = daxa_f32vec3(0.5) * (x * x);
  w[1] = daxa_f32vec3(0.75) - (y * y);
  w[2] = daxa_f32vec3(0.5) * (z * z);

  return base_coord;
}



daxa_f32mat3x3 calculate_p2g(inout Particle particle, daxa_f32 dt, daxa_f32 p_vol, daxa_f32 mu_0, daxa_f32 lambda_0, daxa_f32 inv_dx) {

  daxa_f32mat3x3 identity_matrix = daxa_f32mat3x3(1); // Identity matrix

  particle.F = update_deformation_gradient(particle.F, particle.C, dt); // deformation gradient update

  // Hardening coefficient: snow gets harder when compressed
  daxa_f32 h = pow(EULER, 10 * (1 - particle.J));
  if(particle.type == MAT_JELLY)
      h = 1.0f;



  daxa_f32 mu = mu_0 * h;
  daxa_f32 la = lambda_0 * h;
  // WATER
  if (particle.type == MAT_WATER)
      mu = 0.0f;

  daxa_f32mat3x3 U, sig, V;
  svd(particle.F, U, sig, V, 5);
  daxa_f32 J = 1.0f;
  // Calculate J
  for (uint i = 0; i < 3; ++i)
  {
      daxa_f32 new_sigma = sig[i][i];
      if (particle.type == MAT_SNOW)
      {
          new_sigma = min(max(sig[i][i], 1 - 2.5e-2), 1 + 4.5e-3);
      }
      particle.J *= sig[i][i] / new_sigma;
      sig[i][i] = new_sigma;
      J *= sig[i][i];
  }

  // WATER
  if (particle.type == MAT_WATER)
  {
      daxa_f32mat3x3 new_F = identity_matrix;
      new_F[0][0] = J;
      particle.F = new_F;
  }
  else if (particle.type == MAT_SNOW)
  {
      particle.F = U * sig * transpose(V);
  }

  // Fixed Corotated
  // APIC C (Mp = 1/4 * ∆x^2 * I for quadratic Ni(x))
  // S = ∆t * Vp * Mp-1 * @I/@F * (Fp)
  daxa_f32mat3x3 stress = calculate_stress(particle.F, U, V, mu, la, J); // Stress tensor

  return (-dt * p_vol * (4 * inv_dx * inv_dx)) * stress;
}


daxa_f32 calculate_p2g_water(inout Particle particle, daxa_f32 p_vol, daxa_f32 w_k, daxa_f32 w_gamma, daxa_f32 inv_dx) {
  daxa_f32 stress = w_k * (1 - 1 / pow(particle.J, w_gamma));
  return -p_vol * (4 * inv_dx * inv_dx) * stress;
}


Ray get_ray_from_current_pixel(daxa_f32vec2 index, daxa_f32vec2 rt_size,
                               daxa_f32mat4x4 inv_view, daxa_f32mat4x4 inv_proj) {

  const daxa_f32vec2 pixel_center = index + 0.5;
  const daxa_f32vec2 inv_UV = pixel_center / rt_size;
  daxa_f32vec2 d = inv_UV * 2.0 - 1.0;

  // Ray setup
  Ray ray;

  daxa_f32vec4 origin = mat4_vec4_mul(inv_view, daxa_f32vec4(0, 0, 0, 1));
  ray.origin = origin.xyz;

  daxa_f32vec4 target = mat4_vec4_mul(inv_proj, daxa_f32vec4(d.x, d.y, 1, 1));
  daxa_f32vec4 direction = mat4_vec4_mul(inv_view, daxa_f32vec4(normalize(target.xyz), 0));

  ray.direction = direction.xyz;

  return ray;
}

daxa_f32 compute_sphere_distance(daxa_f32vec3 p, daxa_f32vec3 center, daxa_f32 radius) {
    return length(p - center) - radius;
}

daxa_f32 hitSphere(daxa_f32vec3 center, daxa_f32 radius, Ray r)
{
  daxa_f32vec3  oc           = r.origin - center;
  daxa_f32 a            = dot(r.direction, r.direction);
  daxa_f32 b            = 2.0 * dot(oc, r.direction);
  daxa_f32 c            = dot(oc, oc) - radius * radius;
  daxa_f32 discriminant = b * b - 4 * a * c;
  if(discriminant < 0)
  {
    return -1.0;
  }
  else
  {
    return (-b - sqrt(discriminant)) / (2.0 * a);
  }
}

daxa_f32 hitAabb(const Aabb aabb, const Ray r)
{
  daxa_f32vec3  invDir = 1.0 / r.direction;
  daxa_f32vec3  tbot   = invDir * (aabb.min - r.origin);
  daxa_f32vec3  ttop   = invDir * (aabb.max - r.origin);
  daxa_f32vec3  tmin   = min(ttop, tbot);
  daxa_f32vec3  tmax   = max(ttop, tbot);
  daxa_f32 t0     = max(tmin.x, max(tmin.y, tmin.z));
  daxa_f32 t1     = min(tmax.x, min(tmax.y, tmax.z));
  return t1 > max(t0, 0.0) ? t0 : -1.0;
}

#endif // GLSL & HLSL