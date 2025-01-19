#pragma once

#define DAXA_RAY_TRACING 1
#if defined(GL_core_profile) // GLSL
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_ARB_gpu_shader_int64 : enable
#endif // GL_core_profile

#include "daxa/daxa.inl"

#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
#define DAXA_DEREF(X) deref(X).
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
#define DAXA_DEREF(X) X->
#endif // DAXA_SHADERLANG

#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
#define DAXA_PTR(X) daxa_BufferPtr(X)
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
#define DAXA_PTR(X) Ptr<X>
#endif // DAXA_SHADERLANG

#if defined(GL_core_profile) // GLSL
#define CONST_STATIC_VARIABLE const
#else // SLANG
#define CONST_STATIC_VARIABLE static const
#endif // GL_core_profile




// #define DAXA_SIMULATION_WATER_MPM_MLS
#if !defined(DAXA_SIMULATION_WATER_MPM_MLS)
#define DAXA_RIGID_BODY_FLAG
// #define DAXA_SIMULATION_MANY_MATERIALS
#if defined(DAXA_RIGID_BODY_FLAG)
// WARN: rigid bodies don't collide with each other
#define DAXA_SIMULATION_MANY_RIGID_BODIES
#endif // DAXA_RIGID_BODY_FLAG
#endif // DAXA_SIMULATION_WATER_MPM_MLS

#define GRID_DIM 128
#define GRID_SIZE (GRID_DIM * GRID_DIM * GRID_DIM)
#define QUALITY 2
#define SIM_LOOP_COUNT 30
// #define NUM_PARTICLES 8192 * QUALITY * QUALITY * QUALITY
#define NUM_PARTICLES 16384 * QUALITY * QUALITY * QUALITY
// #define NUM_PARTICLES 32768 * QUALITY * QUALITY * QUALITY
// #define NUM_PARTICLES 65536 * QUALITY * QUALITY * QUALITY
// #define NUM_PARTICLES 131072 * QUALITY * QUALITY * QUALITY
// #define NUM_PARTICLES 262144 * QUALITY * QUALITY * QUALITY
// #define NUM_PARTICLES 524288 * QUALITY * QUALITY * QUALITY
// #define NUM_PARTICLES 512
// #define NUM_PARTICLES 64
// #define TOTAL_AABB_COUNT (NUM_PARTICLES + NUM_RIGID_BOX_COUNT)
#define TOTAL_AABB_COUNT NUM_PARTICLES
#define BOUNDARY 3U
#define GRAVITY -9.8f

#define MPM_P2G_COMPUTE_X 64
#define MPM_GRID_COMPUTE_X 4 
#define MPM_GRID_COMPUTE_Y 4
#define MPM_GRID_COMPUTE_Z 4
#define MPM_CPIC_COMPUTE_X 64

#define MPM_SHADING_COMPUTE_X 8
#define MPM_SHADING_COMPUTE_Y 8

#define PARTICLE_RADIUS 0.0025f
#define MIN_DIST 1e-6f
#define MAX_DIST 1e10f
#define EULER 2.71828
#define MOUSE_DOWN_FLAG (1u << 0)
#define MOUSE_TARGET_FLAG (1u << 1)
#define PARTICLE_FORCE_ENABLED_FLAG (1u << 2)
#if defined(DAXA_RIGID_BODY_FLAG)
#define RIGID_BODY_ADD_GRAVITY_FLAG (1u << 3)
#define RIGID_BODY_PICK_UP_ENABLED_FLAG (1u << 4)
#define RIGID_BODY_PUSHING_ENABLED_FLAG (1u << 5)
#define RIGID_BODY_REST_POS_ENABLED_FLAG (1u << 6)
#define RIGID_BODY_FORCE_ENABLED_FLAG RIGID_BODY_PICK_UP_ENABLED_FLAG | RIGID_BODY_PUSHING_ENABLED_FLAG | RIGID_BODY_REST_POS_ENABLED_FLAG
#endif // DAXA_RIGID_BODY_FLAG

#define MAT_WATER 0
#define MAT_SNOW 1
#define MAT_JELLY 2
// #define MAT_SAND 3
#define MAT_RIGID 4
#define MAT_COUNT (MAT_JELLY + 1)


#if defined(DAXA_RIGID_BODY_FLAG)
#define MAX_RIGID_BODY_COUNT 16U
#define RIGID_BODY_BOX 0
#define RIGID_BODY_MAX_ENUM (RIGID_BODY_BOX + 1)

#if defined(DAXA_SIMULATION_MANY_RIGID_BODIES)
#define NUM_RIGID_BOX_COUNT 3
const daxa_f32 rigid_body_densities[NUM_RIGID_BOX_COUNT] = {10000.0f, 800.0f, 200.0f};
const daxa_f32 rigid_body_frictions[NUM_RIGID_BOX_COUNT] = {0.1f, 0.3f, -0.1f};
const daxa_f32 rigid_body_restitutions[NUM_RIGID_BOX_COUNT] = {0.1f, 0.3f, 0.5f};
#else 
#define NUM_RIGID_BOX_COUNT 1
const daxa_f32 rigid_body_densities[NUM_RIGID_BOX_COUNT] = {5000.0f};
const daxa_f32 rigid_body_frictions[NUM_RIGID_BOX_COUNT] = {0.1f};
const daxa_f32 rigid_body_restitutions[NUM_RIGID_BOX_COUNT] = {0.1f};
#endif
#define NUM_RIGID_PARTICLES 32768

#define BOX_VOLUME 0.47684f // dim.x * dim.y * dim.z

#define BOX_VERTEX_COUNT 8
#define BOX_INDEX_COUNT 36
#define BOX_TRIANGLE_COUNT 12

// #define STATE_MASK 0xAAAAAAAAU
// #define SIGN_MASK 0x55555555U
#define TAG_DISPLACEMENT MAX_RIGID_BODY_COUNT
#define RECONSTRUCTION_GUARD 1e-20f
#define COLLISION_GUARD 1e-7f
#define EPSILON 1e-6f

#define COUNTER_CLOCKWISE 0
#define CLOCKWISE 1
#define TRIANGLE_ORIENTATION COUNTER_CLOCKWISE

#define PENALTY_FORCE 1e1f
#define FRICTION -0.2f
// #define PUSHING_FORCE 2000.0f
#define PUSHING_FORCE 0.0f
#define APPLIED_FORCE_RIGID_BODY 100.0f
#define BOUNDARY_FRICTION 0.1f
#define BASE_VELOCITY 0.1f
#else // DAXA_RIGID_BODY_FLAG

#define NUM_RIGID_BOX_COUNT 0

#endif // DAXA_RIGID_BODY_FLAG

struct Camera {
  daxa_f32mat4x4 inv_view;
  daxa_f32mat4x4 inv_proj;
  daxa_u32vec2 frame_dim;
};

struct GpuInput
{
  daxa_u32 p_count;
#if defined(DAXA_RIGID_BODY_FLAG)
  daxa_u32 rigid_body_count;
  daxa_u32 r_p_count;
#endif  // DAXA_RIGID_BODY_FLAG
  daxa_u32vec3 grid_dim;
  daxa_f32 dt;
  daxa_f32 dx;
  daxa_f32 inv_dx;
  daxa_f32 gravity;
  daxa_u64 frame_number;
  daxa_f32vec2 mouse_pos;
  daxa_f32 mouse_radius;
  daxa_f32 max_velocity;
#if defined(DAXA_RIGID_BODY_FLAG)
  daxa_f32 applied_force;
  daxa_f32 max_rigid_velocity;
#endif  // DAXA_RIGID_BODY_FLAG
  };

struct GpuStatus 
{
  daxa_u32 flags;
  daxa_f32vec3 hit_origin;
  daxa_f32vec3 hit_direction;
  daxa_f32 hit_distance;
  daxa_f32vec3 mouse_target;
  daxa_f32vec3 local_hit_position;
  daxa_u32 rigid_body_index;
  daxa_u32 rigid_element_index;
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

#if defined(DAXA_RIGID_BODY_FLAG)
struct RigidBody  {
  daxa_u32 type;
  daxa_f32vec3 min;
  daxa_f32vec3 max;
  daxa_u32 p_count;
  daxa_u32 p_offset;
  daxa_u32 triangle_count;
  daxa_u32 triangle_offset;
  daxa_f32vec3 color;
  daxa_f32 friction;
  daxa_f32 pushing_force;
  daxa_f32vec3 position;
  daxa_f32vec3 velocity;
  daxa_f32vec3 omega;
  daxa_f32vec3 velocity_delta;
  daxa_f32vec3 omega_delta;
  daxa_f32 mass;
  daxa_f32 inv_mass;
  daxa_f32mat3x3 inertia;
  daxa_f32mat3x3 inv_inertia;
  daxa_f32vec4 rotation;
  daxa_f32vec3 rotation_axis;
  daxa_f32 linear_damping;
  daxa_f32 angular_damping;
  daxa_f32 restitution;
};

struct RigidParticle  {
  daxa_f32vec3 min;
  daxa_f32vec3 max;
  daxa_u32 rigid_id;
  daxa_u32 triangle_id;
};

struct ParticleCDF  {
  daxa_f32 distance;
  daxa_u32 color;
  daxa_u32 difference;
  daxa_f32vec3 normal;
};

struct NodeCDF {
  daxa_i32 unsigned_distance;
  // daxa_f32 d;
  daxa_u32 color;
  daxa_u32 rigid_id;
  daxa_u32 rigid_particle_index;
};
#endif // DAXA_RIGID_BODY_FLAG

DAXA_DECL_BUFFER_PTR(GpuInput)
DAXA_DECL_BUFFER_PTR(GpuStatus)
DAXA_DECL_BUFFER_PTR(Particle)
#if defined(DAXA_RIGID_BODY_FLAG)
DAXA_DECL_BUFFER_PTR(RigidBody)
DAXA_DECL_BUFFER_PTR(RigidParticle)
DAXA_DECL_BUFFER_PTR(NodeCDF)
DAXA_DECL_BUFFER_PTR(ParticleCDF)
#endif // DAXA_RIGID_BODY_FLAG
DAXA_DECL_BUFFER_PTR(Cell)
DAXA_DECL_BUFFER_PTR(Camera)
DAXA_DECL_BUFFER_PTR(Aabb)

struct ComputePush
{
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
    daxa_ImageViewId swapchain;
    daxa_BufferId input_buffer_id;
    daxa_BufferId status_buffer_id;
    daxa_RWBufferPtr(GpuInput) input_ptr;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
    daxa::RWTexture2DId<daxa_f32vec4> swapchain;
    daxa_RWBufferPtr(GpuInput) input_ptr;
    daxa_RWBufferPtr(GpuStatus) status_ptr;
#endif // DAXA_SHADERLANG
    daxa_RWBufferPtr(Particle) particles;
#if defined(DAXA_RIGID_BODY_FLAG)
    daxa_BufferPtr(RigidBody) rigid_bodies;
    daxa_BufferPtr(daxa_u32) indices;
    daxa_BufferPtr(daxa_f32vec3) vertices;
    daxa_RWBufferPtr(RigidParticle) rigid_particles;
    daxa_RWBufferPtr(NodeCDF) rigid_cells;
    daxa_RWBufferPtr(ParticleCDF) rigid_particle_color;
#endif // DAXA_RIGID_BODY_FLAG
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

const daxa_f32vec3 RIGID_BODY_GREEN_COLOR = daxa_f32vec3(0.5f, 0.8f, 0.3f); // green
const daxa_f32vec3 RIGID_BODY_RED_COLOR = daxa_f32vec3(0.8f, 0.3f, 0.3f); // red
const daxa_f32vec3 RIGID_BODY_YELLOW_COLOR = daxa_f32vec3(0.8f, 0.8f, 0.3f); // yellow
const daxa_f32vec3 RIGID_BODY_PURPLE_COLOR = daxa_f32vec3(0.8f, 0.3f, 0.8f); // purple



#if !defined(__cplusplus)

#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
#define FLOAT_TO_INT(f) floatBitsToInt(f)
#define INT_TO_FLOAT(i) intBitsToFloat(i)
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
#define FLOAT_TO_INT(f) asint(f)
#define INT_TO_FLOAT(i) asfloat(i)
#endif // DAXA_SHADERLANG

daxa_i32
to_emulated_float(daxa_f32 f)
{
   daxa_i32 bits = FLOAT_TO_INT(f);
   return f < 0 ? -2147483648 - bits : bits;
}

daxa_f32
from_emulated_float(daxa_i32 bits)
{
   return INT_TO_FLOAT(bits < 0 ? -2147483648 - bits : bits);
}

daxa_i32
to_emulated_positive_float(daxa_f32 f)
{
   return FLOAT_TO_INT(f);
}

daxa_f32
from_emulated_positive_float(daxa_i32 bits)
{
   return INT_TO_FLOAT(bits);
}

daxa_f32 inverse_f32(daxa_f32 f) {
  // check for divide by zero
  return f == 0.0f ? 0.0f : 1.0f / f;
}


#if defined(GL_core_profile) // GLSL
#extension GL_EXT_shader_atomic_float : enable

DAXA_DECL_PUSH_CONSTANT(ComputePush, p)

layout(buffer_reference, scalar) buffer PARTICLE_BUFFER {Particle particles[]; }; // Particle buffer
layout(buffer_reference, scalar) buffer CELL_BUFFER {Cell cells[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer AABB_BUFFER {Aabb aabbs[]; }; // Particle positions
#if defined(DAXA_RIGID_BODY_FLAG)
layout(buffer_reference, scalar) buffer RIGID_BODY_BUFFER {RigidBody rigid_bodies[]; }; // Rigid body information
layout(buffer_reference, scalar) buffer INDEX_BUFFER {daxa_u32 indices[]; }; // Rigid body indices info
layout(buffer_reference, scalar) buffer VERTEX_BUFFER {vec3 vertices[]; }; // Rigid body vertices info
layout(buffer_reference, scalar) buffer RIGID_PARTICLE_BUFFER {RigidParticle particles[]; }; // Rigid particle buffer
layout(buffer_reference, scalar) buffer RIGID_CELL_BUFFER {NodeCDF cells[]; }; // Rigid cell buffer
layout(buffer_reference, scalar) buffer RIGID_PARTICLE_STATUS_BUFFER {ParticleCDF particles[]; }; // Rigid  Particle color buffer
#endif  // DAXA_RIGID_BODY_FLAG
#else // SLANG

float atomicAdd(__ref float value, float amount)
{
    __target_switch
    {
    case cpp:
        __requirePrelude("#include <atomic>");
        __intrinsic_asm "std::atomic_ref(*$0).fetch_add($1)";
    case spirv:
        return __atomicAdd(value, amount);
    }
    return 0;
}

[[vk::push_constant]] ComputePush p;



daxa_f32mat4x4 inverse(const daxa_f32mat4x4 m)
{
  daxa_f32 a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3];
  daxa_f32 a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3];
  daxa_f32 a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3];

  daxa_f32 b01 = a22 * a11 - a12 * a21;
  daxa_f32 b11 = -a22 * a01 + a02 * a21;
  daxa_f32 b21 = a12 * a01 - a02 * a11;

  daxa_f32 det = a00 * b01 + a10 * b11 + a20 * b21;

  if (det == 0)
    return daxa_f32mat4x4(0.0f);

  det = 1.0 / det;

  daxa_f32 a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3];
  daxa_f32 b02 = a32 * a11 - a12 * a31;
  daxa_f32 b12 = -a32 * a01 + a02 * a31;
  daxa_f32 b22 = a12 * a01 - a02 * a11;
  daxa_f32 b03 = a21 * a10 - a20 * a11;
  daxa_f32 b13 = -a21 * a00 + a01 * a20;
  daxa_f32 b23 = a11 * a00 - a01 * a10;

  return daxa_f32mat4x4(
    b01 * det, b02 * det, b03 * det, 0.0f,
    b11 * det, b12 * det, b13 * det, 0.0f,
    b21 * det, b22 * det, b23 * det, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );
}

#endif // GL_core_profile


// GLOBAL SHADER REGION



#if defined(DAXA_RIGID_BODY_FLAG)
daxa_f32vec3 rigid_body_add_atomic_velocity_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 velocity) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  daxa_f32 x = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity.x, velocity.x);
  daxa_f32 y = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity.y, velocity.y);
  daxa_f32 z = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity.z, velocity.z);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  daxa_f32 x = atomicAdd(p.rigid_bodies[rigid_body_index].velocity.x, velocity.x);
  daxa_f32 y = atomicAdd(p.rigid_bodies[rigid_body_index].velocity.y, velocity.y);
  daxa_f32 z = atomicAdd(p.rigid_bodies[rigid_body_index].velocity.z, velocity.z);
#endif // DAXA_SHADERLANG
  return daxa_f32vec3(x, y, z);
}

daxa_f32vec3 rigid_body_add_atomic_omega_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 omega) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  daxa_f32 x = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega.x, omega.x);
  daxa_f32 y = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega.y, omega.y);
  daxa_f32 z = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega.z, omega.z);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  daxa_f32 x = atomicAdd(p.rigid_bodies[rigid_body_index].omega.x, omega.x);
  daxa_f32 y = atomicAdd(p.rigid_bodies[rigid_body_index].omega.y, omega.y);
  daxa_f32 z = atomicAdd(p.rigid_bodies[rigid_body_index].omega.z, omega.z);
#endif // DAXA_SHADERLANG
  return daxa_f32vec3(x, y, z);
}

void rigid_body_add_atomic_velocity_delta_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 velocity_delta) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity_delta.x, velocity_delta.x);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity_delta.y, velocity_delta.y);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity_delta.z, velocity_delta.z);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  atomicAdd(p.rigid_bodies[rigid_body_index].velocity_delta.x, velocity_delta.x);
  atomicAdd(p.rigid_bodies[rigid_body_index].velocity_delta.y, velocity_delta.y);
  atomicAdd(p.rigid_bodies[rigid_body_index].velocity_delta.z, velocity_delta.z);
#endif // DAXA_SHADERLANG
}

void rigid_body_add_atomic_omega_delta_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 omega_delta) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega_delta.x, omega_delta.x);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega_delta.y, omega_delta.y);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega_delta.z, omega_delta.z);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  atomicAdd(p.rigid_bodies[rigid_body_index].omega_delta.x, omega_delta.x);
  atomicAdd(p.rigid_bodies[rigid_body_index].omega_delta.y, omega_delta.y);
  atomicAdd(p.rigid_bodies[rigid_body_index].omega_delta.z, omega_delta.z);
#endif // DAXA_SHADERLANG
}

daxa_f32vec3 rigid_body_get_position_by_index(daxa_u32 rigid_body_index) {
# if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].position;
# elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_bodies[rigid_body_index].position;
# endif // DAXA_SHADERLANG
}

void rigid_body_set_position_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 position) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].position = position;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.rigid_bodies[rigid_body_index].position = position;
#endif // DAXA_SHADERLANG
}

void rigid_body_set_rotation_by_index(daxa_u32 rigid_body_index, daxa_f32vec4 rotation) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].rotation = rotation;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.rigid_bodies[rigid_body_index].rotation = rotation;
#endif // DAXA_SHADERLANG
}

daxa_f32vec3 rigid_body_get_velocity_by_index(daxa_u32 rigid_body_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].velocity;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_bodies[rigid_body_index].velocity;
#endif // DAXA_SHADERLANG
}

void rigid_body_set_velocity_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 velocity) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].velocity = velocity;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.rigid_bodies[rigid_body_index].velocity = velocity;
#endif // DAXA_SHADERLANG
}

daxa_f32vec3 rigid_body_get_omega_by_index(daxa_u32 rigid_body_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].omega;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_bodies[rigid_body_index].omega;
#endif // DAXA_SHADERLANG
}

void rigid_body_set_omega_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 omega) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].omega = omega;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.rigid_bodies[rigid_body_index].omega = omega;
#endif // DAXA_SHADERLANG
}

daxa_f32vec3 rigid_body_get_velocity_delta_by_index(daxa_u32 rigid_body_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].velocity_delta;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_bodies[rigid_body_index].velocity_delta;
#endif // DAXA_SHADERLANG
}

void rigid_body_reset_velocity_delta_by_index(daxa_u32 rigid_body_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].velocity_delta = daxa_f32vec3(0, 0, 0);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.rigid_bodies[rigid_body_index].velocity_delta = daxa_f32vec3(0, 0, 0);
#endif // DAXA_SHADERLANG
}

daxa_f32vec3 rigid_body_get_omega_delta_by_index(daxa_u32 rigid_body_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].omega_delta;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_bodies[rigid_body_index].omega_delta;
#endif // DAXA_SHADERLANG
}

void rigid_body_reset_omega_delta_by_index(daxa_u32 rigid_body_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].omega_delta = daxa_f32vec3(0, 0, 0);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.rigid_bodies[rigid_body_index].omega_delta = daxa_f32vec3(0, 0, 0);
#endif // DAXA_SHADERLANG
}

daxa_f32vec4 rigid_body_get_rotation_by_index(daxa_u32 rigid_body_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].rotation;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_bodies[rigid_body_index].rotation;
#endif // DAXA_SHADERLANG
}

RigidParticle get_rigid_particle_by_index(daxa_u32 particle_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_PARTICLE_BUFFER rigid_particle_buffer = RIGID_PARTICLE_BUFFER(p.rigid_particles);
  return rigid_particle_buffer.particles[particle_index];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_particles[particle_index];
#endif // DAXA_SHADERLANG
}

NodeCDF get_node_cdf_by_index(daxa_u32 cell_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  return rigid_cell_buffer.cells[cell_index];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_cells[cell_index];
#endif // DAXA_SHADERLANG
}

void node_cdf_set_by_index(daxa_u32 cell_index, NodeCDF cell) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  rigid_cell_buffer.cells[cell_index] = cell;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.rigid_cells[cell_index] = cell;
#endif // DAXA_SHADERLANG
}


daxa_u32 get_node_cdf_color_by_index(daxa_u32 cell_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  return rigid_cell_buffer.cells[cell_index].color;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_cells[cell_index].color;
#endif // DAXA_SHADERLANG
}

void zeroed_out_node_cdf_by_index(daxa_u32 cell_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  rigid_cell_buffer.cells[cell_index].unsigned_distance = to_emulated_positive_float(MAX_DIST);
  rigid_cell_buffer.cells[cell_index].color = 0;
  rigid_cell_buffer.cells[cell_index].rigid_id = -1;
  rigid_cell_buffer.cells[cell_index].rigid_particle_index = -1;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.rigid_cells[cell_index].unsigned_distance = to_emulated_positive_float(MAX_DIST);
  p.rigid_cells[cell_index].color = 0;
  p.rigid_cells[cell_index].rigid_id = -1;
  p.rigid_cells[cell_index].rigid_particle_index = -1;
#endif // DAXA_SHADERLANG
}

daxa_f32 set_atomic_rigid_cell_distance_by_index(daxa_u32 cell_index, daxa_f32 unsigned_distance) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  daxa_i32 bits = to_emulated_positive_float(unsigned_distance);
  daxa_i32 result = atomicMin(rigid_cell_buffer.cells[cell_index].unsigned_distance, bits);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  daxa_i32 bits = to_emulated_positive_float(unsigned_distance);
  daxa_i32 result; 
  InterlockedMin(p.rigid_cells[cell_index].unsigned_distance, bits, result);
#endif // DAXA_SHADERLANG
  return from_emulated_positive_float(result);
}

daxa_u32 set_atomic_rigid_cell_color_by_index(daxa_u32 cell_index, daxa_u32 rigid_id, bool negative) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  daxa_u32 negative_flag = negative ? 1u : 0u;
  daxa_u32 flags = (negative_flag << (TAG_DISPLACEMENT + rigid_id)) | (1u << rigid_id);
  return atomicOr(rigid_cell_buffer.cells[cell_index].color, flags);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  daxa_u32 negative_flag = negative ? 1u : 0u;
  daxa_u32 flags = (negative_flag << (TAG_DISPLACEMENT + rigid_id)) | (1u << rigid_id);
  daxa_u32 result;
  InterlockedOr(p.rigid_cells[cell_index].color, flags, result);
  return result;
#endif // DAXA_SHADERLANG
}

daxa_u32 set_atomic_rigid_cell_rigid_id_by_index(daxa_u32 cell_index, daxa_u32 rigid_id) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  return atomicExchange(rigid_cell_buffer.cells[cell_index].rigid_id, rigid_id);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  daxa_u32 result;
  InterlockedExchange(p.rigid_cells[cell_index].rigid_id, rigid_id, result);
  return result;
#endif // DAXA_SHADERLANG
}

daxa_u32 set_atomic_rigid_cell_rigid_particle_index_by_index(daxa_u32 cell_index, daxa_u32 rigid_particle_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  return atomicExchange(rigid_cell_buffer.cells[cell_index].rigid_particle_index, rigid_particle_index);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  daxa_u32 result;
  InterlockedExchange(p.rigid_cells[cell_index].rigid_particle_index, rigid_particle_index, result);
  return result;
#endif // DAXA_SHADERLANG
}

void particle_CDF_init(inout ParticleCDF particle_CDF) {
  particle_CDF.distance = MAX_DIST;
  particle_CDF.color = 0;
  particle_CDF.difference = 0;
  particle_CDF.normal = daxa_f32vec3(0, 0, 0);
}

ParticleCDF get_rigid_particle_CDF_by_index(daxa_u32 particle_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_PARTICLE_STATUS_BUFFER rigid_particle_color_buffer = RIGID_PARTICLE_STATUS_BUFFER(p.rigid_particle_color);
  return rigid_particle_color_buffer.particles[particle_index];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_particle_color[particle_index];
#endif // DAXA_SHADERLANG
}

daxa_u32 get_rigid_particle_CDF_color_by_index(daxa_u32 particle_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_PARTICLE_STATUS_BUFFER rigid_particle_color_buffer = RIGID_PARTICLE_STATUS_BUFFER(p.rigid_particle_color);
  return rigid_particle_color_buffer.particles[particle_index].color;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_particle_color[particle_index].color;
#endif // DAXA_SHADERLANG
}

void set_rigid_particle_CDF_by_index(daxa_u32 particle_index, ParticleCDF color) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_PARTICLE_STATUS_BUFFER rigid_particle_color_buffer = RIGID_PARTICLE_STATUS_BUFFER(p.rigid_particle_color);
  rigid_particle_color_buffer.particles[particle_index] = color;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.rigid_particle_color[particle_index] = color;
#endif // DAXA_SHADERLANG
}
#endif // DAXA_RIGID_BODY_FLAG

Cell get_cell_by_index(daxa_u32 cell_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return cell_buffer.cells[cell_index];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.cells[cell_index];
#endif // DAXA_SHADERLANG
}

daxa_f32vec3 get_cell_vel_by_index(daxa_u32 cell_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return cell_buffer.cells[cell_index].v;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.cells[cell_index].v;
#endif // DAXA_SHADERLANG
}

daxa_f32 get_cell_mass_by_index(daxa_u32 cell_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return cell_buffer.cells[cell_index].m;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.cells[cell_index].m;
#endif // DAXA_SHADERLANG
}

Aabb get_aabb_by_index(daxa_u32 aabb_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  AABB_BUFFER aabb_buffer = AABB_BUFFER(p.aabbs);
  return aabb_buffer.aabbs[aabb_index];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.aabbs[aabb_index];
#endif // DAXA_SHADERLANG
}

void set_particle_by_index(daxa_u32 particle_index, Particle particle) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  particle_buffer.particles[particle_index] = particle;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.particles[particle_index] = particle;
#endif // DAXA_SHADERLANG
}

void particle_set_velocity_by_index(daxa_u32 particle_index, daxa_f32vec3 v) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  particle_buffer.particles[particle_index].v = v;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.particles[particle_index].v = v;
#endif // DAXA_SHADERLANG
}

void particle_set_F_by_index(daxa_u32 particle_index, daxa_f32mat3x3 F) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  particle_buffer.particles[particle_index].F = F;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.particles[particle_index].F = F;
#endif // DAXA_SHADERLANG
}

void particle_set_C_by_index(daxa_u32 particle_index, daxa_f32mat3x3 C) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  particle_buffer.particles[particle_index].C = C;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.particles[particle_index].C = C;
#endif // DAXA_SHADERLANG
}

void zeroed_out_cell_by_index(daxa_u32 cell_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  cell_buffer.cells[cell_index].v = daxa_f32vec3(0, 0, 0);
  cell_buffer.cells[cell_index].m = 0;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.cells[cell_index].v = daxa_f32vec3(0, 0, 0);
  p.cells[cell_index].m = 0;
#endif // DAXA_SHADERLANG
}

void set_cell_by_index(daxa_u32 cell_index, Cell cell) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  cell_buffer.cells[cell_index] = cell;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.cells[cell_index] = cell;
#endif // DAXA_SHADERLANG
}

daxa_f32 set_atomic_cell_vel_x_by_index(daxa_u32 cell_index, daxa_f32 x) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return atomicAdd(cell_buffer.cells[cell_index].v.x, x);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return atomicAdd(p.cells[cell_index].v.x, x);
#endif // DAXA_SHADERLANG
}

daxa_f32 set_atomic_cell_vel_y_by_index(daxa_u32 cell_index, daxa_f32 y) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return atomicAdd(cell_buffer.cells[cell_index].v.y, y);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return atomicAdd(p.cells[cell_index].v.y, y);
#endif // DAXA_SHADERLANG
}

daxa_f32 set_atomic_cell_vel_z_by_index(daxa_u32 cell_index, daxa_f32 z) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return atomicAdd(cell_buffer.cells[cell_index].v.z, z);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return atomicAdd(p.cells[cell_index].v.z, z);
#endif // DAXA_SHADERLANG
}

daxa_f32 set_atomic_cell_mass_by_index(daxa_u32 cell_index, daxa_f32 w) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  CELL_BUFFER cell_buffer = CELL_BUFFER(p.cells);
  return atomicAdd(cell_buffer.cells[cell_index].m, w);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return atomicAdd(p.cells[cell_index].m, w);
#endif // DAXA_SHADERLANG
}

void set_aabb_by_index(daxa_u32 aabb_index, Aabb aabb) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  AABB_BUFFER aabb_buffer = AABB_BUFFER(p.aabbs);
  aabb_buffer.aabbs[aabb_index] = aabb;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  p.aabbs[aabb_index] = aabb;
#endif // DAXA_SHADERLANG
}

Particle get_particle_by_index(daxa_u32 particle_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  return particle_buffer.particles[particle_index];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.particles[particle_index];
#endif // DAXA_SHADERLANG
}

#if defined(DAXA_RIGID_BODY_FLAG)
RigidBody get_rigid_body_by_index(daxa_u32 rigid_body_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer =
      RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_bodies[rigid_body_index];
#endif // DAXA_SHADERLANG
}

daxa_f32vec3 get_rigid_body_color_by_index(daxa_u32 rigid_body_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  RIGID_BODY_BUFFER rigid_body_buffer =
      RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].color;
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.rigid_bodies[rigid_body_index].color;
#endif // DAXA_SHADERLANG
}

daxa_u32 get_first_index_by_triangle_index(daxa_u32 triangle_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  INDEX_BUFFER index_buffer = INDEX_BUFFER(p.indices);
  return index_buffer.indices[triangle_index * 3];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.indices[triangle_index * 3];
#endif // DAXA_SHADERLANG
}

daxa_u32 get_second_index_by_triangle_index(daxa_u32 triangle_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  INDEX_BUFFER index_buffer = INDEX_BUFFER(p.indices);
  return index_buffer.indices[triangle_index * 3 + 1];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.indices[triangle_index * 3 + 1];
#endif // DAXA_SHADERLANG
}

daxa_u32 get_third_index_by_triangle_index(daxa_u32 triangle_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  INDEX_BUFFER index_buffer = INDEX_BUFFER(p.indices);
  return index_buffer.indices[triangle_index * 3 + 2];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.indices[triangle_index * 3 + 2];
#endif // DAXA_SHADERLANG
}


daxa_u32vec3 get_indices_by_triangle_index(daxa_u32 triangle_index) {
  return daxa_u32vec3(get_first_index_by_triangle_index(triangle_index), get_second_index_by_triangle_index(triangle_index), get_third_index_by_triangle_index(triangle_index));

}

daxa_f32vec3 get_vertex_by_index(daxa_u32 vertex_index) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  VERTEX_BUFFER vertex_buffer = VERTEX_BUFFER(p.vertices);
  return vertex_buffer.vertices[vertex_index];
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return p.vertices[vertex_index];
#endif // DAXA_SHADERLANG
}

daxa_f32vec3 get_first_vertex_by_triangle_index(daxa_u32 triangle_index) {
  return get_vertex_by_index(get_first_index_by_triangle_index(triangle_index));
}

daxa_f32vec3 get_second_vertex_by_triangle_index(daxa_u32 triangle_index) {
  return get_vertex_by_index(get_second_index_by_triangle_index(triangle_index));
}

daxa_f32vec3 get_third_vertex_by_triangle_index(daxa_u32 triangle_index) {
  return get_vertex_by_index(get_third_index_by_triangle_index(triangle_index));
}

daxa_f32mat3x3 get_vertices_by_triangle_index(daxa_u32 triangle_index) {
  daxa_u32vec3 indices = get_indices_by_triangle_index(triangle_index);
  return daxa_f32mat3x3(get_vertex_by_index(indices.x), get_vertex_by_index(indices.y), get_vertex_by_index(indices.z));
}
#endif // DAXA_RIGID_BODY_FLAG




// CONSTANTS
CONST_STATIC_VARIABLE daxa_f32vec3 RIGID_BODY_PARTICLE_COLOR = daxa_f32vec3(0.6f, 0.4f, 0.2f);
CONST_STATIC_VARIABLE daxa_f32vec3 WATER_HIGH_SPEED_COLOR = daxa_f32vec3(0.3f, 0.5f, 1.0f);
CONST_STATIC_VARIABLE daxa_f32vec3 WATER_LOW_SPEED_COLOR = daxa_f32vec3(0.1f, 0.2f, 0.4f);
CONST_STATIC_VARIABLE daxa_f32vec3 SNOW_HIGH_SPEED_COLOR = daxa_f32vec3(0.9f, 0.9f, 1.0f);
CONST_STATIC_VARIABLE daxa_f32vec3 SNOW_LOW_SPEED_COLOR = daxa_f32vec3(0.5f, 0.5f, 0.6f);
CONST_STATIC_VARIABLE daxa_f32vec3 JELLY_HIGH_SPEED_COLOR = daxa_f32vec3(1.0f, 0.5f, 0.5f);
CONST_STATIC_VARIABLE daxa_f32vec3 JELLY_LOW_SPEED_COLOR = daxa_f32vec3(0.7f, 0.2f, 0.2f);



// RT STRUCTS
struct hitPayload
{
  daxa_f32vec3 hit_value;
  daxa_u32 seed;
  daxa_f32vec3 hit_pos;
  daxa_u32 rigid_body_index;
  daxa_u32 rigid_element_index;
};

struct ShadowRayPayload
{
  daxa_f32 shadow;
};


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

daxa_f32mat4x4 Convert3x4To4x4(
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  daxa_f32mat4x3
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  daxa_f32mat3x4 
#endif // DAXA_SHADERLANG
  objectToWorld4x3)
{
    daxa_f32mat4x4 objectToWorld4x4;

    objectToWorld4x4[0] = daxa_f32vec4(objectToWorld4x3[0], 0.0f);
    objectToWorld4x4[1] = daxa_f32vec4(objectToWorld4x3[1], 0.0f);
    objectToWorld4x4[2] = daxa_f32vec4(objectToWorld4x3[2], 0.0f);
    objectToWorld4x4[3] = daxa_f32vec4(objectToWorld4x3[3], 1.0f);

    return objectToWorld4x4;
}


daxa_f32mat4x4 mat4_from_mat3(daxa_f32mat3x3 m) {
    return daxa_f32mat4x4(
        daxa_f32vec4(m[0], 0.0f),
        daxa_f32vec4(m[1], 0.0f),
        daxa_f32vec4(m[2], 0.0f),
        daxa_f32vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

daxa_f32vec4 mat4_vec4_mul(daxa_f32mat4x4 m, daxa_f32vec4 v) {
    return 
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
      (m * v);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
      mul(m, v);
#endif
}

daxa_f32mat4x4 mat4_mul(daxa_f32mat4x4 a, daxa_f32mat4x4 b) {
    return 
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
      (a * b);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
      mul(a, b);
#endif
}

daxa_f32mat3x3 identity_mat3()
{
    return daxa_f32mat3x3(
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    );
}

daxa_f32mat4x4 identity_mat4()
{
    return daxa_f32mat4x4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

daxa_f32 aabb_get_hit(Aabb aabb, daxa_f32vec3 ray_origin, daxa_f32vec3 ray_direction, 
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  daxa_f32mat4x3
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  daxa_f32mat3x4 
#endif // DAXA_SHADERLANG
  objectToWorld4x3) 
{
  // Convert object to world matrix to 4x4
  daxa_f32mat4x4 worldToObject = transpose(Convert3x4To4x4(objectToWorld4x3));

  // Transform origin
  daxa_f32vec4 ray_origin4 = mat4_vec4_mul(worldToObject, daxa_f32vec4(ray_origin, 1.0f));

  daxa_f32vec3 origin = ray_origin4.xyz;
  
  // Transform direction
  daxa_f32vec4 ray_direction4 = mat4_vec4_mul(worldToObject, daxa_f32vec4(ray_direction, 0.0f));

  daxa_f32vec3 direction = ray_direction4.xyz;
  
  // Normalize direction
  direction = normalize(direction);

  // Create ray
  Ray r = {origin, direction};
  
  return hitAabb(aabb, r);
}


void check_boundaries(inout daxa_f32vec3 pos, inout Particle particle, daxa_f32 wall_min, daxa_f32 wall_max) {
  // Check boundaries
  if (pos.x < wall_min)
  {
      pos.x = wall_min;
      particle.v.x = -particle.v.x;
  }
  else if (pos.x > wall_max)
  {
      pos.x = wall_max;
      particle.v.x = -particle.v.x;
  }

  if (pos.y < wall_min)
  {
      pos.y = wall_min;
      particle.v.y = -particle.v.y;
  }
  else if (pos.y > wall_max)
  {
      pos.y = wall_max;
      particle.v.y = -particle.v.y;
  }

  if (pos.z < wall_min)
  {
      pos.z = wall_min;
      particle.v.z = -particle.v.z;
  }
  else if (pos.z > wall_max)
  {
      pos.z = wall_max;
      particle.v.z = -particle.v.z;
  }
}


void particle_apply_external_force(inout Particle particle, daxa_f32vec3 pos, daxa_f32 wall_min, daxa_f32 wall_max, daxa_f32vec3 mouse_target, daxa_f32 mouse_radius, daxa_u32 flags) {
  // Repulsion force
  if (((flags & MOUSE_TARGET_FLAG) == MOUSE_TARGET_FLAG) &&
  ((flags & PARTICLE_FORCE_ENABLED_FLAG) == PARTICLE_FORCE_ENABLED_FLAG))
  {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL   
      if (all(greaterThan(mouse_target, daxa_f32vec3(wall_min))) &&
          all(lessThan(mouse_target, daxa_f32vec3(wall_max))))
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
      if(all(mouse_target > daxa_f32vec3(wall_min)) &&
          all(mouse_target < daxa_f32vec3(wall_max)))
#endif // DAXA_SHADERLANG
      {          

          daxa_f32vec3 dist = pos - mouse_target;
          if (dot(dist, dist) < mouse_radius * mouse_radius)
          {
              daxa_f32vec3 force = normalize(dist) * 0.05f;
              particle.v += force;
          }
      }
  }
}



#if defined(DAXA_RIGID_BODY_FLAG)
daxa_f32mat3x3 rigid_body_get_rotation_matrix(daxa_f32vec4 rotation) {
  daxa_f32vec4 quaternion = normalize(rotation);
  daxa_f32 x = quaternion.x;
  daxa_f32 y = quaternion.y;
  daxa_f32 z = quaternion.z;
  daxa_f32 w = quaternion.w;

  daxa_f32 x2 = x + x;
  daxa_f32 y2 = y + y;
  daxa_f32 z2 = z + z;

  daxa_f32 xx = x * x2;
  daxa_f32 xy = x * y2;
  daxa_f32 xz = x * z2;

  daxa_f32 yy = y * y2;
  daxa_f32 yz = y * z2;
  daxa_f32 zz = z * z2;

  daxa_f32 wx = w * x2;
  daxa_f32 wy = w * y2;
  daxa_f32 wz = w * z2;
  
  daxa_f32vec3 col0 = daxa_f32vec3(1.0f - (yy + zz), xy + wz, xz - wy);
  daxa_f32vec3 col1 = daxa_f32vec3(xy - wz, 1.0f - (xx + zz), yz + wx);
  daxa_f32vec3 col2 = daxa_f32vec3(xz + wy, yz - wx, 1.0f - (xx + yy));

  return daxa_f32mat3x3(col0, col1, col2);
}

daxa_f32mat4x4 rigid_body_get_transform_matrix(RigidBody rigid_body) {
  daxa_f32mat3x3 rotation = rigid_body_get_rotation_matrix(rigid_body.rotation);
  daxa_f32mat4x4 transform = mat4_from_mat3(rotation);
  transform[3] = daxa_f32vec4(rigid_body.position, 1.0);

#if DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG 
    return transpose(transform);
#else
    return transform;
#endif
}

daxa_f32mat4x4 rigid_body_get_transform_matrix_from_rotation_translation(daxa_f32vec4 rotation, daxa_f32vec3 position) {
  daxa_f32mat3x3 _rotation = rigid_body_get_rotation_matrix(rotation);
  daxa_f32mat4x4 transform = mat4_from_mat3(_rotation);
  transform[3] = daxa_f32vec4(position, 1.0);
#if DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG 
    return transpose(transform);
#else
    return transform;
#endif
}
#endif // DAXA_RIGID_BODY_FLAG

void check_mouse_input(daxa_u32vec2 pixel_coord, daxa_f32vec2 mouse_pos, Ray ray, hitPayload payload, DAXA_PTR(GpuStatus) status)
{

    if (pixel_coord.x == daxa_u32(mouse_pos.x) && pixel_coord.y == daxa_u32(mouse_pos.y))
    {
        if ((DAXA_DEREF(status) flags & MOUSE_DOWN_FLAG) == MOUSE_DOWN_FLAG)
        {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
        if (all(notEqual(payload.hit_pos, daxa_f32vec3(MAX_DIST))))
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
        if (all(payload.hit_pos != daxa_f32vec3(MAX_DIST)))
#endif // DAXA_SHADERLANG
        {
          DAXA_DEREF(status)flags |= MOUSE_TARGET_FLAG;
          DAXA_DEREF(status)mouse_target = payload.hit_pos;
          DAXA_DEREF(status)hit_distance = length(payload.hit_pos - ray.origin);
          DAXA_DEREF(status)rigid_body_index = payload.rigid_body_index;
          DAXA_DEREF(status)rigid_element_index = payload.rigid_element_index;
        } 
#if defined(DAXA_RIGID_BODY_FLAG)
          if (((DAXA_DEREF(status)flags & RIGID_BODY_PICK_UP_ENABLED_FLAG) == RIGID_BODY_PICK_UP_ENABLED_FLAG) || ((DAXA_DEREF(status)flags & RIGID_BODY_PUSHING_ENABLED_FLAG) == RIGID_BODY_PUSHING_ENABLED_FLAG)
          || ((DAXA_DEREF(status)flags & RIGID_BODY_REST_POS_ENABLED_FLAG) == RIGID_BODY_REST_POS_ENABLED_FLAG)){
            DAXA_DEREF(status)flags &= ~MOUSE_DOWN_FLAG;
            if (payload.rigid_body_index != -1)
            {
              daxa_f32vec3 rigid_body_pos = rigid_body_get_position_by_index(payload.rigid_body_index);
              daxa_f32vec4 rigid_body_rot = rigid_body_get_rotation_by_index(payload.rigid_body_index);
              daxa_f32mat4x4 transform = rigid_body_get_transform_matrix_from_rotation_translation(
                rigid_body_rot, rigid_body_pos);
              DAXA_DEREF(status)local_hit_position = mat4_vec4_mul(inverse(transform), daxa_f32vec4(payload.hit_pos, 1)).xyz;
            }
          }
#endif // DAXA_RIGID_BODY_FLAG
    }
    if((DAXA_DEREF(status)flags & MOUSE_TARGET_FLAG) == MOUSE_TARGET_FLAG) {
      DAXA_DEREF(status)hit_origin = ray.origin;
      DAXA_DEREF(status)hit_direction = ray.direction;
    }
  }
}



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
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
    return outerProduct(a, b);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
    return daxa_f32mat3x3(
        a.x * b.x, a.y * b.x, a.z * b.x,
        a.x * b.y, a.y * b.y, a.z * b.y,
        a.x * b.z, a.y * b.z, a.z * b.z
    );
    // return daxa_f32mat3x3(
    //     a.x * b.x, a.x * b.y, a.x * b.z,
    //     a.y * b.x, a.y * b.y, a.y * b.z,
    //     a.z * b.x, a.z * b.y, a.z * b.z
    // );
#endif // DAXA_SHADERLANG
}

daxa_f32mat4x4 outer_product_mat4(daxa_f32vec4 a, daxa_f32vec4 b)
{
    return daxa_f32mat4x4(
        a.x * b.x, a.x * b.y, a.x * b.z, a.x * b.w,
        a.y * b.x, a.y * b.y, a.y * b.z, a.y * b.w,
        a.z * b.x, a.z * b.y, a.z * b.z, a.z * b.w,
        a.w * b.x, a.w * b.y, a.w * b.z, a.w * b.w
    );
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
# if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  daxa_f32vec3 temp = daxa_f32vec3(mat[0][col1], mat[1][col1], mat[2][col1]);
  mat[0][col1] = mat[0][col2];
  mat[1][col1] = mat[1][col2];
  mat[2][col1] = mat[2][col2];
  mat[0][col2] = temp.x;
  mat[1][col2] = temp.y;
  mat[2][col2] = temp.z;
# elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  daxa_f32vec3 temp = daxa_f32vec3(mat[col1][0], mat[col1][1], mat[col1][2]);
  mat[col1][0] = mat[col2][0];
  mat[col1][1] = mat[col2][1];
  mat[col1][2] = mat[col2][2];
  mat[col2][0] = temp.x;
  mat[col2][1] = temp.y;
  mat[col2][2] = temp.z;
# endif // DAXA_SHADERLANG
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
  U = identity_mat3();
  V = identity_mat3();
  S = A;

#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
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
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  // Perform Jacobi iterations
  for (int sweep = 0; sweep < iters; sweep++)
  {
      daxa_f32 Sch, Ssh, Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;

      // First rotation (zero out Ss21)
      Ssh = S[0][1] * 0.5f;
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
      daxa_f32 Sc = Stmp2 - Stmp1;
      daxa_f32 Ss = 2.0f * Sch * Ssh;

      Stmp1 = Ss * S[0][2];
      Stmp2 = Ss * S[0][0];
      S[0][2] = Sc * S[0][2] + Stmp2;
      S[0][0] = Sc * S[0][0] - Stmp1;

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
      Ssh = S[1][2] * 0.5f;
      Stmp5 = S[2][2] - S[0][0];
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
      Stmp2 = Ss * S[0][0];
      S[1][0] = Sc * S[1][0] + Stmp2;
      S[0][0] = Sc * S[0][0] - Stmp1;
      
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
      Ssh = S[2][0] * 0.5f;
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
      Sc = Stmp2 - Stmp1;
      Ss = 2.0f * Sch * Ssh;

      Stmp1 = Ss * S[1][1];
      Stmp2 = Ss * S[1][2];
      S[1][1] = Sc * S[1][1] + Stmp2;
      S[1][2] = Sc * S[1][2] - Stmp1;

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
#endif // DAXA_SHADERLANG

  // Construct diagonal matrix S
  S = daxa_f32mat3x3(sigma1, 0.0f, 0.0f,
                     0.0f, sigma2, 0.0f,
                     0.0f, 0.0f, sigma3);
}

daxa_f32mat3x3 element_wise_mul(daxa_f32mat3x3 a, daxa_f32mat3x3 b)
{
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
    return daxa_f32mat3x3(
        a[0][0] * b[0][0], a[0][1] * b[0][1], a[0][2] * b[0][2],
        a[1][0] * b[1][0], a[1][1] * b[1][1], a[1][2] * b[1][2],
        a[2][0] * b[2][0], a[2][1] * b[2][1], a[2][2] * b[2][2]
    );
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
    return daxa_f32mat3x3(
        a[0][0] * b[0][0], a[1][0] * b[1][0], a[2][0] * b[2][0],
        a[0][1] * b[0][1], a[1][1] * b[1][1], a[2][1] * b[2][1],
        a[0][2] * b[0][2], a[1][2] * b[1][2], a[2][2] * b[2][2]
    );
#endif // DAXA_SHADERLANG
}

daxa_f32vec3 mat3_vec3_mul(daxa_f32mat3x3 m, daxa_f32vec3 v)
{
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
    return daxa_f32vec3(
        dot(m[0], v),
        dot(m[1], v),
        dot(m[2], v)
    );
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
    return daxa_f32vec3(
        dot(v, daxa_f32vec3(m[0][0], m[1][0], m[2][0])),
        dot(v, daxa_f32vec3(m[0][1], m[1][1], m[2][1])),
        dot(v, daxa_f32vec3(m[0][2], m[1][2], m[2][2]))
    );
#endif // DAXA_SHADERLANG
}

daxa_f32mat3x3 mat3_mul(daxa_f32mat3x3 a, daxa_f32mat3x3 b) {
    daxa_f32mat3x3 result;
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
    for (int i = 0; i < 3; i++) {
        result[i] = daxa_f32vec3(
            dot(daxa_f32vec3(a[i][0], a[i][1], a[i][2]), daxa_f32vec3(b[0][0], b[1][0], b[2][0])),
            dot(daxa_f32vec3(a[i][0], a[i][1], a[i][2]), daxa_f32vec3(b[0][1], b[1][1], b[2][1])),
            dot(daxa_f32vec3(a[i][0], a[i][1], a[i][2]), daxa_f32vec3(b[0][2], b[1][2], b[2][2]))
        );
    }
#endif // DAXA_SHADERLANG
    return result;
}

daxa_f32mat3x3 calculate_stress(daxa_f32mat3x3 F, daxa_f32mat3x3 U, daxa_f32mat3x3 V, daxa_f32 mu, daxa_f32 la, daxa_f32 J) {
    daxa_f32mat3x3 V_T = transpose(V); // Transpuesta de V
    daxa_f32mat3x3 U_V_T = mat3_mul(U, V_T); // U @ V.transpose()
    daxa_f32mat3x3 F_T = transpose(F); // Transpuesta de F_dg[p]

    daxa_f32mat3x3 term1 = 2.0 * mu * mat3_mul((F - U_V_T), F_T); // 2 * mu * (F_dg[p] - U @ V.transpose()) @ F_dg[p].transpose()
    daxa_f32mat3x3 term2 = identity_mat3() * la * J * (J - 1.0); // ti.Matrix.identity(daxa_f32, 3) * la * J * (J - 1)

    return term1 + term2;
}

daxa_f32mat3x3 update_deformation_gradient(daxa_f32mat3x3 F, daxa_f32mat3x3 C, daxa_f32 dt) {
    return element_wise_mul(identity_mat3() + dt * C, F); // deformation gradient update
}

daxa_i32vec3 calculate_particle_grid_pos(Aabb aabb, daxa_f32 inv_dx) {
  
  daxa_f32vec3 particle_center = (aabb.min + aabb.max) * 0.5f * inv_dx;
  daxa_f32vec3 particle_center_dx = particle_center - daxa_f32vec3(0.5f, 0.5f, 0.5f);

  return daxa_i32vec3(particle_center_dx); // Floor
}

daxa_i32vec3 calculate_particle_grid_pos_and_center(Aabb aabb, daxa_f32 inv_dx, out daxa_f32vec3 particle_center) {
  
  particle_center = (aabb.min + aabb.max) * 0.5f * inv_dx;
  daxa_f32vec3 particle_center_dx = particle_center - daxa_f32vec3(0.5f, 0.5f, 0.5f);

  return daxa_i32vec3(particle_center_dx); // Floor
}

daxa_i32vec3 calculate_particle_status(Aabb aabb, daxa_f32 inv_dx, out daxa_f32vec3 fx, out daxa_f32vec3 w[3]) {

  daxa_f32vec3 particle_center;
  daxa_i32vec3 base_coord = calculate_particle_grid_pos_and_center(aabb, inv_dx, particle_center);

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



daxa_f32vec3 calculate_p2g(inout Particle particle, daxa_f32 dt, daxa_f32 p_vol, daxa_f32 mu_0, daxa_f32 lambda_0, daxa_f32 inv_dx, daxa_f32 p_mass, inout daxa_f32mat3x3 affine) {

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
  svd(particle.F, U, sig, V, 5); // Singular Value Decomposition

  daxa_f32 J = 1.0f;
  // Calculate J
  for (daxa_u32 i = 0; i < 3; ++i)
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
      daxa_f32mat3x3 new_F = identity_mat3();
      new_F[0][0] = J;
      particle.F = new_F;
  }
  else if (particle.type == MAT_SNOW)
  {
      particle.F = mat3_mul(U, mat3_mul(sig, transpose(V)));
  }

  // Fixed Corotated
  // APIC C (Mp = 1/4 * ∆x^2 * I for quadratic Ni(x))
  // S = ∆t * Vp * Mp-1 * @I/@F * (Fp)
  daxa_f32mat3x3 stress = calculate_stress(particle.F, U, V, mu, la, J); // Stress tensor

  stress = (-dt * p_vol * (4 * inv_dx * inv_dx)) * stress;

  affine = stress + p_mass * particle.C;

  // Transactional momentum
  return daxa_f32vec3(p_mass * particle.v);
}


daxa_f32 calculate_p2g_water(inout Particle particle, daxa_f32 p_vol, daxa_f32 w_k, daxa_f32 w_gamma, daxa_f32 inv_dx) {
  daxa_f32 stress = w_k * (1 - 1 / pow(particle.J, w_gamma));
  return -p_vol * (4 * inv_dx * inv_dx) * stress;
}


daxa_f32vec3 calculate_weighted_p2g_velocity(daxa_f32vec3 dpos, daxa_f32 weight, daxa_f32vec3 mv, daxa_f32mat3x3 affine) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  return weight * (mv + mat3_vec3_mul(affine, dpos));
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return weight * (mv + mat3_vec3_mul(affine, dpos));
#endif // DAXA_SHADERLANG
}

daxa_f32mat3x3 calculate_weighted_g2p_deformation(daxa_f32vec3 dpos, daxa_f32 weight, daxa_f32vec3 vel_value) {
  return weight * outer_product(vel_value, dpos);
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



bool inside_triangle(daxa_f32vec3 p, daxa_f32vec3 v0, daxa_f32vec3 v1, daxa_f32vec3 v2) {
  // determine the barycentric coordinates to determine if the point is inside the triangle
  // from: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
  daxa_f32vec3 ab = v1 - v0;
  daxa_f32vec3 ac = v2 - v0;
  daxa_f32vec3 ap = p - v0;

  daxa_f32 d00 = dot(ab, ab);
  daxa_f32 d01 = dot(ab, ac);
  daxa_f32 d11 = dot(ac, ac);
  daxa_f32 d20 = dot(ap, ab);
  daxa_f32 d21 = dot(ap, ac);
  daxa_f32 denom = d00 * d11 - d01 * d01;

  daxa_f32 alpha = (d11 * d20 - d01 * d21) / denom;
  daxa_f32 beta = (d00 * d21 - d01 * d20) / denom;
  daxa_f32 gamma = 1.0 - alpha - beta;

  // slight tolerance to avoid discarding valid points on the edge
  // this might not bee needed at all
  // Todo: reconsider this
  daxa_f32 min = -0.0000001;
  daxa_f32 max = 1.0000001;

  return min <= alpha && alpha <= max && min <= beta && beta <= max && min <= gamma && gamma <= max;
}

daxa_f32 vec3_abs_max(daxa_f32vec3 v)
{
  return max(max(abs(v.x), abs(v.y)), abs(v.z));
}

#elif defined(__cplusplus) // C++
#include <cmath> // std::sqrt

#if defined(DAXA_SIMULATION_WATER_MPM_MLS)
#define TIME_STEP 1e-3f
#define MAX_VELOCITY 20.0f
#else // DAXA_SIMULATION_WATER_MPM_MLS
#if defined(DAXA_SIMULATION_MANY_MATERIALS)
#define MAX_VELOCITY 4.0f
#define TIME_STEP 1e-4f
#else 
#define MAX_VELOCITY 12.0f
#define TIME_STEP 2e-4f
#endif

#if defined(DAXA_RIGID_BODY_FLAG)
#define MAX_RIGID_VELOCITY 5.0f
#endif // DAXA_RIGID_BODY_FLAG

#endif // DAXA_SIMULATION_WATER_MPM_MLS


inline daxa_f32 length(const daxa_f32vec3 &v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline daxa_f32vec3 normalize(const daxa_f32vec3 &v) {
    daxa_f32 len = length(v);
    if (len == 0) {
        // Maneja el caso de la normalización de un vector cero
        return {0.0f, 0.0f, 0.0f};
    }
    return {v.x / len, v.y / len, v.z / len};
}

#define float_to_int(f) (*reinterpret_cast<const int*>(&static_cast<const float&>(f)))

#define int_to_float(i) (*reinterpret_cast<const float*>(&static_cast<const int&>(i)))

daxa_i32
to_emulated_float(daxa_f32 f)
{
   daxa_i32 bits = float_to_int(f);
   return f < 0 ? -2147483648 - bits : bits;
}

daxa_f32
from_emulated_float(daxa_i32 bits)
{
   return int_to_float(bits < 0 ? -2147483648 - bits : bits);
}

daxa_i32
to_emulated_positive_float(daxa_f32 f)
{
   return float_to_int(f);
}

daxa_f32
from_emulated_positive_float(daxa_i32 bits)
{
   return int_to_float(bits);
}


// support for addition, subtraction, multiplication, division
inline daxa_f32vec3 operator+(daxa_f32vec3 a, daxa_f32vec3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline daxa_f32vec3 operator-(daxa_f32vec3 a, daxa_f32vec3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline daxa_f32vec3 operator*(daxa_f32vec3 a, daxa_f32vec3 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline daxa_f32vec3 operator/(daxa_f32vec3 a, daxa_f32vec3 b)
{
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

inline daxa_f32vec3 operator+(daxa_f32vec3 a, daxa_f32 b)
{
    return {a.x + b, a.y + b, a.z + b};
}

inline daxa_f32vec3 operator-(daxa_f32vec3 a, daxa_f32 b)
{
    return {a.x - b, a.y - b, a.z - b};
}

inline daxa_f32vec3 operator*(daxa_f32vec3 a, daxa_f32 b)
{
    return {a.x * b, a.y * b, a.z * b};
}

inline daxa_f32vec3 operator/(daxa_f32vec3 a, daxa_f32 b)
{
    return {a.x / b, a.y / b, a.z / b};
}

inline daxa_f32vec3 operator+=(daxa_f32vec3 &a, const daxa_f32vec3 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline daxa_f32vec3 operator-= (daxa_f32vec3 &a, const daxa_f32vec3 &b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

inline daxa_f32vec3 operator*= (daxa_f32vec3 &a, const daxa_f32vec3 &b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

inline daxa_f32vec3 operator/= (daxa_f32vec3 &a, const daxa_f32vec3 &b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

inline daxa_f32mat3x3 mat3_inverse(const daxa_f32mat3x3 &m)
{
    daxa_f32 a00 = m.y.y * m.z.z - m.y.z * m.z.y;
    daxa_f32 a01 = m.y.x * m.z.z - m.y.z * m.z.x;
    daxa_f32 a02 = m.y.x * m.z.y - m.y.y * m.z.x;
    daxa_f32 a10 = m.x.y * m.z.z - m.x.z * m.z.y;
    daxa_f32 a11 = m.x.x * m.z.z - m.x.z * m.z.x;
    daxa_f32 a12 = m.x.x * m.z.y - m.x.y * m.z.x;
    daxa_f32 a20 = m.x.y * m.y.z - m.x.z * m.y.y;
    daxa_f32 a21 = m.x.x * m.y.z - m.x.z * m.y.x;
    daxa_f32 a22 = m.x.x * m.y.y - m.x.y * m.y.x;

    daxa_f32 det = m.x.x * a00 - m.x.y * a01 + m.x.z * a02;
    daxa_f32 inv_det = 1.0f / det;

    return daxa_f32mat3x3(daxa_f32vec3(a00 * inv_det, -a10 * inv_det, a20 * inv_det),
                          daxa_f32vec3(-a01 * inv_det, a11 * inv_det, -a21 * inv_det),
                          daxa_f32vec3(a02 * inv_det, -a12 * inv_det, a22 * inv_det));
}


inline daxa_f32vec3 cross(const daxa_f32vec3 &a, const daxa_f32vec3 &b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline daxa_f32mat3x3 inverse(const daxa_f32mat3x3 &m)
{
    daxa_f32 a00 = m.y.y * m.z.z - m.y.z * m.z.y;
    daxa_f32 a01 = m.y.x * m.z.z - m.y.z * m.z.x;
    daxa_f32 a02 = m.y.x * m.z.y - m.y.y * m.z.x;
    daxa_f32 a10 = m.x.y * m.z.z - m.x.z * m.z.y;
    daxa_f32 a11 = m.x.x * m.z.z - m.x.z * m.z.x;
    daxa_f32 a12 = m.x.x * m.z.y - m.x.y * m.z.x;
    daxa_f32 a20 = m.x.y * m.y.z - m.x.z * m.y.y;
    daxa_f32 a21 = m.x.x * m.y.z - m.x.z * m.y.x;
    daxa_f32 a22 = m.x.x * m.y.y - m.x.y * m.y.x;

    daxa_f32 det = m.x.x * a00 - m.x.y * a01 + m.x.z * a02;
    daxa_f32 inv_det = 1.0f / det;

    return daxa_f32mat3x3(daxa_f32vec3(a00 * inv_det, -a10 * inv_det, a20 * inv_det),
                          daxa_f32vec3(-a01 * inv_det, a11 * inv_det, -a21 * inv_det),
                          daxa_f32vec3(a02 * inv_det, -a12 * inv_det, a22 * inv_det));
}


#if defined(DAXA_RIGID_BODY_FLAG)
daxa_f32mat3x4 rigid_body_get_transform_matrix(const RigidBody &rigid_body) {
    daxa_f32vec3 translation = rigid_body.position;
    daxa_f32vec4 rotation = rigid_body.rotation;

    // transform quaternion to matrix
    daxa_f32 x2 = rotation.x + rotation.x;
    daxa_f32 y2 = rotation.y + rotation.y;
    daxa_f32 z2 = rotation.z + rotation.z;
    daxa_f32 xx = rotation.x * x2;
    daxa_f32 xy = rotation.x * y2;
    daxa_f32 xz = rotation.x * z2;
    daxa_f32 yy = rotation.y * y2;
    daxa_f32 yz = rotation.y * z2;
    daxa_f32 zz = rotation.z * z2;
    daxa_f32 wx = rotation.w * x2;
    daxa_f32 wy = rotation.w * y2;
    daxa_f32 wz = rotation.w * z2;

    daxa_f32mat3x3 rotation_matrix = daxa_f32mat3x3(daxa_f32vec3(1.0f - (yy + zz), xy - wz, xz + wy),
                                                    daxa_f32vec3(xy + wz, 1.0f - (xx + zz), yz - wx),
                                                    daxa_f32vec3(xz - wy, yz + wx, 1.0f - (xx + yy)));

    return daxa_f32mat3x4(daxa_f32vec4(rotation_matrix.x.x, rotation_matrix.y.x, rotation_matrix.z.x, translation.x),
                        daxa_f32vec4(rotation_matrix.x.y, rotation_matrix.y.y, rotation_matrix.z.y, translation.y),
                        daxa_f32vec4(rotation_matrix.x.z, rotation_matrix.y.z, rotation_matrix.z.z, translation.z));
}


auto rigid_body_get_transform_mat4(const RigidBody &rigid_body) -> daxa_f32mat4x4 {
    daxa_f32vec3 translation = rigid_body.position;
    daxa_f32vec4 rotation = rigid_body.rotation;

    // transform quaternion to matrix
    daxa_f32 x2 = rotation.x + rotation.x;
    daxa_f32 y2 = rotation.y + rotation.y;
    daxa_f32 z2 = rotation.z + rotation.z;
    daxa_f32 xx = rotation.x * x2;
    daxa_f32 xy = rotation.x * y2;
    daxa_f32 xz = rotation.x * z2;
    daxa_f32 yy = rotation.y * y2;
    daxa_f32 yz = rotation.y * z2;
    daxa_f32 zz = rotation.z * z2;
    daxa_f32 wx = rotation.w * x2;
    daxa_f32 wy = rotation.w * y2;
    daxa_f32 wz = rotation.w * z2;

    daxa_f32mat3x3 rotation_matrix = daxa_f32mat3x3(daxa_f32vec3(1.0f - (yy + zz), xy - wz, xz + wy),
                                                    daxa_f32vec3(xy + wz, 1.0f - (xx + zz), yz - wx),
                                                    daxa_f32vec3(xz - wy, yz + wx, 1.0f - (xx + yy)));

    return daxa_f32mat4x4(daxa_f32vec4(rotation_matrix.x.x, rotation_matrix.y.x, rotation_matrix.z.x, translation.x),
                        daxa_f32vec4(rotation_matrix.x.y, rotation_matrix.y.y, rotation_matrix.z.y, translation.y),
                        daxa_f32vec4(rotation_matrix.x.z, rotation_matrix.y.z, rotation_matrix.z.z, translation.z),
                        daxa_f32vec4(0.0f, 0.0f, 0.0f, 1.0f));
}


auto dot(const daxa_f32vec3 &a, const daxa_f32vec3 &b) -> daxa_f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

auto dot(const daxa_f32vec4 &a, const daxa_f32vec4 &b) -> daxa_f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

auto mat4_vec4_mul(const daxa_f32mat4x4 &m, const daxa_f32vec4 &v) -> daxa_f32vec4 {
    return daxa_f32vec4(dot(m.x, v), dot(m.y, v), dot(m.z, v), dot(m.w, v));
}

#endif // DAXA_RIGID_BODY_FLAG

#endif // GLSL & HLSL & SLANG & C++


// RIGID BODY SHADER CODE
#if defined(DAXA_RIGID_BODY_FLAG)

daxa_f32vec3 get_normal_by_vertices(daxa_f32vec3 v0, daxa_f32vec3 v1, daxa_f32vec3 v2) {
  daxa_f32vec3 u = v1 - v0;
  daxa_f32vec3 v = v2 - v0;
#if TRIANGLE_ORIENTATION == CLOCKWISE
  return normalize(cross(v, u));
#else
  return normalize(cross(u, v));
#endif  
}

#if !defined(__cplusplus) // GLSL & HLSL & SLANG
struct InterpolatedParticleData {
  daxa_u32 color;
  daxa_f32 weighted_tags[MAX_RIGID_BODY_COUNT];
  daxa_f32mat4x4 weighted_matrix;
  daxa_f32vec4 weighted_vector;
};

void interpolated_particle_data_init(inout InterpolatedParticleData data) {
  data.color = 0;
  for(int i = 0; i < MAX_RIGID_BODY_COUNT; i++) {
    data.weighted_tags[i] = 0.0f;
  }
  data.weighted_matrix = daxa_f32mat4x4(0.0f);
  data.weighted_vector = daxa_f32vec4(0.0f);
}

void cdf_update_tag(inout daxa_u32 color, daxa_u32 rigid_body_index, daxa_f32 signed_distance) {
  daxa_u32 tag = signed_distance < 0.0f ? 0x1 : 0x0;
  daxa_u32 offset = rigid_body_index + TAG_DISPLACEMENT;
  color = color & ~(1u << offset) | (tag << offset);
}

bool cdf_get_tag(daxa_u32 color, daxa_u32 rigid_body_index) {
  return ((color >> (TAG_DISPLACEMENT + rigid_body_index)) & 0x1) != 0;
}

bool cdf_get_affinity(daxa_u32 color, daxa_u32 rigid_body_index) {
  return ((color >> (rigid_body_index)) & 0x1) != 0;
}

daxa_u32 cdf_get_affinities(daxa_u32 color) {
  return ((color << TAG_DISPLACEMENT) >> TAG_DISPLACEMENT);
}

daxa_u32 cdf_get_tags(daxa_u32 color) {
  return (color >> TAG_DISPLACEMENT);
}

bool cdf_is_compatible(daxa_u32 color1, daxa_u32 color2) {
  daxa_u32 shared_affinities = cdf_get_affinities(color1) & cdf_get_affinities(color2);
  return (shared_affinities & cdf_get_tags(color1)) == (shared_affinities & cdf_get_tags(color2));
}


daxa_f32 node_cdf_signed_distance(NodeCDF node_cdf, daxa_u32 rigid_body_index) {
  daxa_f32 sign = cdf_get_tag(node_cdf.color, rigid_body_index) ? 1.0f : -1.0f;
  return sign * to_emulated_positive_float(node_cdf.unsigned_distance);
}

void interpolate_color(inout InterpolatedParticleData data, NodeCDF node_cdf, daxa_f32 weight, daxa_u32 rigid_body_count) {
  data.color |= cdf_get_affinities(node_cdf.color);

  rigid_body_count = min(rigid_body_count, MAX_RIGID_BODY_COUNT);

  for(daxa_u32 r = 0; r < rigid_body_count; r++) {
    daxa_f32 signed_distance = node_cdf_signed_distance(node_cdf, r);
    data.weighted_tags[r] += signed_distance * weight;
  }
}


// turn the weighted tags into the proper tags of the particle
void interpolated_particle_data_compute_tags(inout InterpolatedParticleData data, daxa_u32 rigid_body_count) {
  for(daxa_u32 r = 0; r < rigid_body_count; r++) {
    daxa_f32 weighted_tag = data.weighted_tags[r];
    cdf_update_tag(data.color, r, weighted_tag);
  }
}


void interpolate_distance_and_normal(inout InterpolatedParticleData data, NodeCDF node_cdf, daxa_f32 weight, daxa_f32vec3 dpos) {
  if(cdf_get_affinities(node_cdf.color) == 0) {
    return;
  }

  if(node_cdf.rigid_id == -1) {
    return;
  }

  bool particle_tag = cdf_get_tag(data.color, node_cdf.rigid_id);
  bool node_tag = cdf_get_tag(node_cdf.color, node_cdf.rigid_id);
  
  daxa_f32 sign = (particle_tag == node_tag) ? 1.0f : -1.0f;

  daxa_f32 signed_distance = sign * to_emulated_positive_float(node_cdf.unsigned_distance);
  daxa_f32 weight_signed_distance = weight * signed_distance;
  daxa_f32mat3x3 outer_product = outer_product(dpos, dpos);

#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  data.weighted_vector += daxa_f32vec4(1.0f, dpos);
  data.weighted_matrix += daxa_f32mat4x4(1.0f, dpos.x , dpos.y, dpos.z,
                                         dpos.x, outer_product[0],
                                         dpos.y, outer_product[1],
                                         dpos.z, outer_product[2]);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  data.weighted_vector += daxa_f32vec4(dpos, 1.0f);
  data.weighted_matrix += daxa_f32mat4x4(daxa_f32vec4(outer_product[0], dpos.x),
                                          daxa_f32vec4(outer_product[1], dpos.y),
                                          daxa_f32vec4(outer_product[2], dpos.z),
                                          daxa_f32vec4(dpos, 1.0f));
                                          
#endif
}


ParticleCDF interpolated_particle_data_compute_particle_cdf(InterpolatedParticleData data, daxa_f32 dx) {
  ParticleCDF particle_cdf;
  particle_CDF_init(particle_cdf);
  if (abs(determinant(data.weighted_matrix)) > RECONSTRUCTION_GUARD)
  {
    daxa_f32vec4 result = mat4_vec4_mul(inverse(data.weighted_matrix), data.weighted_vector);

    particle_cdf.color = data.color;
    particle_cdf.distance = result.x * dx;
    particle_cdf.normal = normalize(result.yzw);
  }
  return particle_cdf;
}

ParticleCDF particle_CDF_check_and_correct_penetration(ParticleCDF particle_cdf, daxa_u32 previous_color) {
  daxa_u32 shared_affinities = cdf_get_affinities(particle_cdf.color) & cdf_get_affinities(previous_color);
  daxa_u32 difference = (shared_affinities & cdf_get_tags(particle_cdf.color)) ^ (shared_affinities & cdf_get_tags(previous_color));

  bool penetration = difference != 0;

  ParticleCDF new_particle_cdf = particle_cdf;

  if (penetration)
  {
    new_particle_cdf.color = ((cdf_get_tags(particle_cdf.color)  ^ difference) << TAG_DISPLACEMENT) | cdf_get_affinities(particle_cdf.color);
    new_particle_cdf.difference = difference;
    new_particle_cdf.distance = -new_particle_cdf.distance;
    new_particle_cdf.normal = -new_particle_cdf.normal;
  }

  return new_particle_cdf;
}

void gather_CDF_compute(daxa_u32 particle_index, Aabb aabb, DAXA_PTR(GpuInput) config) {

  float dx = DAXA_DEREF(config)dx;
  float inv_dx = DAXA_DEREF(config)inv_dx;
  daxa_u32 rigid_body_count = DAXA_DEREF(config)rigid_body_count;

  daxa_f32vec3 w[3];
  daxa_f32vec3 fx;
  daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

  daxa_u32vec3 array_grid = daxa_u32vec3(base_coord);

  InterpolatedParticleData data;
  interpolated_particle_data_init(data);

  for (daxa_u32 i = 0; i < 3; ++i)
  {
      for (daxa_u32 j = 0; j < 3; ++j)
      {
          for (daxa_u32 k = 0; k < 3; ++k)
          {
              daxa_u32vec3 coord = array_grid + daxa_u32vec3(i, j, k);
              if (coord.x >= DAXA_DEREF(config)grid_dim.x || coord.y >= DAXA_DEREF(config)grid_dim.y || coord.z >= DAXA_DEREF(config)grid_dim.z)
              {
                  continue;
              }

              daxa_u32 index = coord.x + coord.y * DAXA_DEREF(config)grid_dim.x + coord.z * DAXA_DEREF(config)grid_dim.x * DAXA_DEREF(config)grid_dim.y;

              daxa_f32 weight = w[i].x * w[j].y * w[k].z;

              NodeCDF nodeCDF = get_node_cdf_by_index(index);
              interpolate_color(data, nodeCDF, weight, rigid_body_count);
          }
      }
  }

  interpolated_particle_data_compute_tags(data, rigid_body_count);

  for (daxa_u32 i = 0; i < 3; ++i)
  {
      for (daxa_u32 j = 0; j < 3; ++j)
      {
          for (daxa_u32 k = 0; k < 3; ++k)
          {
              daxa_u32vec3 coord = array_grid + daxa_u32vec3(i, j, k);
              if (coord.x >= DAXA_DEREF(config)grid_dim.x || coord.y >= DAXA_DEREF(config)grid_dim.y || coord.z >= DAXA_DEREF(config)grid_dim.z)
              {
                  continue;
              }

              daxa_u32 index = coord.x + coord.y * DAXA_DEREF(config)grid_dim.x + coord.z * DAXA_DEREF(config)grid_dim.x * DAXA_DEREF(config)grid_dim.y;

              daxa_f32 weight = w[i].x * w[j].y * w[k].z;

              daxa_f32vec3 dpos = (daxa_f32vec3(i, j, k) - fx) * dx;

              NodeCDF nodeCDF = get_node_cdf_by_index(index);
              interpolate_distance_and_normal(data, nodeCDF, weight, dpos);
          }
      }
  }
  
  daxa_u32 particle_states = get_rigid_particle_CDF_color_by_index(particle_index);

  ParticleCDF particleCDF = interpolated_particle_data_compute_particle_cdf(data, dx);

  particleCDF = particle_CDF_check_and_correct_penetration(particleCDF, particle_states);

  set_rigid_particle_CDF_by_index(particle_index, particleCDF);
}

daxa_f32vec3 rigid_body_get_velocity_at(RigidBody r, daxa_f32vec3 position) {

  return r.velocity + cross(r.omega, position - r.position);
}

daxa_f32vec3 particle_collision(daxa_f32vec3 velocity, daxa_f32vec3 normal,RigidBody r, daxa_f32vec3 particle_position, daxa_f32 dt, daxa_f32 dx) {

  daxa_f32 friction = r.friction;
  daxa_f32 pushing_force = r.pushing_force;
  daxa_f32vec3 rigid_velocity = rigid_body_get_velocity_at(r, particle_position);

  daxa_f32vec3 relative_velocity = velocity - rigid_velocity;

  daxa_f32 normal_vel_norm = dot(normal, relative_velocity);

  daxa_f32vec3 tangential_relative_velocity = relative_velocity - normal_vel_norm * normal;

  daxa_f32 tangential_norm = length(tangential_relative_velocity);

  daxa_f32 tangential_scale = max(tangential_norm + min(normal_vel_norm, 0.0f) * friction, 0.0f) / max(1e-23f, tangential_norm);

  daxa_f32vec3 projected_velocity = tangential_scale * tangential_relative_velocity + max(0.0f, normal_vel_norm) * normal;

  projected_velocity += rigid_velocity;

  projected_velocity += dt * dx * pushing_force * normal;

  return projected_velocity;
}

daxa_f32mat3x3 rigid_body_get_transformed_inversed_inertia(RigidBody r) {
  daxa_f32mat3x3 rotation = rigid_body_get_rotation_matrix(r.rotation);
// #if DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
//   return mat3_mul(transpose(rotation), mat3_mul(r.inv_inertia, rotation));
// #elif DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  return mat3_mul(rotation, mat3_mul(r.inv_inertia, transpose(rotation)));
// #endif
}

void rigid_body_apply_delta_impulse(RigidBody r, daxa_u32 rigid_index, daxa_f32vec3 impulse, daxa_f32vec3 position) {
  daxa_f32vec3 torque = cross(position - r.position, impulse);
  
  daxa_f32vec3 lineal_velocity = impulse * r.inv_mass;
  daxa_f32vec3 angular_velocity = mat3_vec3_mul(rigid_body_get_transformed_inversed_inertia(r), torque);

  // TODO: this is too slow
  rigid_body_add_atomic_velocity_delta_by_index(rigid_index, lineal_velocity);
  rigid_body_add_atomic_omega_delta_by_index(rigid_index, angular_velocity);
}

void rigid_body_apply_impulse(inout RigidBody r, daxa_f32vec3 impulse,  daxa_f32vec3 position) {
  daxa_f32vec3 torque = cross(position - r.position, impulse);
  
  daxa_f32vec3 lineal_velocity = impulse * r.inv_mass;
  daxa_f32vec3 angular_velocity = mat3_vec3_mul(rigid_body_get_transformed_inversed_inertia(r), torque);

  r.velocity += lineal_velocity;
  r.omega += angular_velocity;
}

daxa_f32mat3x3 cross_product_matrix(daxa_f32vec3 v) {
#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  return daxa_f32mat3x3(0, -v.z, v.y,
                        v.z, 0, -v.x,
                        -v.y, v.x, 0);
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  return daxa_f32mat3x3(0, v.z, -v.y,
                          -v.z, 0, v.x,
                          v.y, -v.x, 0);
#endif
}

daxa_f32 rigid_body_get_impulse_contribution(RigidBody r, daxa_f32vec3 position, daxa_f32vec3 direction) {
  daxa_f32 ret = r.inv_mass;
  daxa_f32mat3x3 inversed_inertia = rigid_body_get_transformed_inversed_inertia(r);
  daxa_f32mat3x3 rn = cross_product_matrix(position);
// #if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL
  inversed_inertia = mat3_mul(transpose(rn), mat3_mul(inversed_inertia, rn));
// #elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
//   inversed_inertia = mat3_mul(rn, mat3_mul(inversed_inertia, transpose(rn)));
// #endif
  ret += dot(direction, mat3_vec3_mul(inversed_inertia, direction));
  return ret;
}

void rigid_body_apply_temporal_velocity(inout RigidBody r) {
  r.velocity += r.velocity_delta;
  r.omega += r.omega_delta;
}

void rigid_body_save_velocity(RigidBody r, daxa_u32 rigid_index) {
  rigid_body_set_velocity_by_index(rigid_index, r.velocity);
  rigid_body_set_omega_by_index(rigid_index, r.omega);
}

void rigid_body_save_parameters(RigidBody r, daxa_u32 rigid_index) {
  rigid_body_save_velocity(r, rigid_index);
  rigid_body_set_position_by_index(rigid_index, r.position);
  rigid_body_set_rotation_by_index(rigid_index, r.rotation);

  rigid_body_reset_velocity_delta_by_index(rigid_index);
  rigid_body_reset_omega_delta_by_index(rigid_index);
}

void rigid_body_enforce_angular_velocity_parallel_to(inout RigidBody r, daxa_f32vec3 direction) {
  direction = normalize(direction);

  r.omega = dot(r.omega, direction) * direction;
}


daxa_f32vec4 quaternion_multiply(daxa_f32vec4 q1, daxa_f32vec4 q2) {
  daxa_f32vec3 v1 = q1.xyz;
  daxa_f32vec3 v2 = q2.xyz;
  daxa_f32 w1 = q1.w;
  daxa_f32 w2 = q2.w;

  daxa_f32vec3 v = w1 * v2 + w2 * v1 + cross(v1, v2);
  daxa_f32 w = w1 * w2 - dot(v1, v2);

  return daxa_f32vec4(v, w);
}

daxa_f32vec4 rigid_body_apply_angular_velocity(daxa_f32vec4 rotation, daxa_f32vec3 omega, daxa_f32 dt) {
  daxa_f32vec3 axis = omega;
  daxa_f32 angle = length(omega);
  if(angle < 1e-23f) {
      return rotation;
  }

  axis = normalize(axis);
  daxa_f32 ot = angle * dt;
  daxa_f32 s = sin(ot * 0.5f);
  daxa_f32 c = cos(ot * 0.5f);

  daxa_f32vec4 q = daxa_f32vec4(s * axis, c);
  return quaternion_multiply(rotation, q);
}

void rigid_body_advance(inout RigidBody r, daxa_f32 dt) {
  // linear velocity
  r.velocity *= exp(-dt * r.linear_damping);
  r.position += dt * r.velocity;
  // angular velocity
  r.omega *= exp(-dt * r.angular_damping);
  r.rotation = rigid_body_apply_angular_velocity(r.rotation, r.omega, dt);
}

daxa_f32 rigid_body_get_boundary_friction(RigidBody r) {
  return BOUNDARY_FRICTION;
}



void rigid_body_raster_rigid_bound(daxa_f32 dx, daxa_f32 inv_dx, daxa_u32 particle_index, DAXA_PTR(GpuInput) config) {

  RigidParticle particle = get_rigid_particle_by_index(particle_index);

  if (particle.rigid_id > MAX_RIGID_BODY_COUNT)
  {
      return;
  }

  RigidBody r = get_rigid_body_by_index(particle.rigid_id);

  daxa_f32mat4x4 transform = rigid_body_get_transform_matrix(r);

  particle.min = mat4_vec4_mul(transform, daxa_f32vec4(particle.min, 1)).xyz;
  particle.max = mat4_vec4_mul(transform, daxa_f32vec4(particle.max, 1)).xyz;

  Aabb aabb = Aabb(particle.min, particle.max);

  daxa_f32vec3 p_pos = (aabb.min + aabb.max) * 0.5f;

#if DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL 
  if(any(lessThan(p_pos, daxa_f32vec3(0))) || any(greaterThanEqual(p_pos, daxa_f32vec3(1))))
#elif DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
  if(any(p_pos < daxa_f32vec3(0)) || any(p_pos >= daxa_f32vec3(1)))
#endif
      return;

  daxa_i32vec3 base_coord = calculate_particle_grid_pos(aabb, inv_dx);

  daxa_u32vec3 array_grid = daxa_u32vec3(base_coord);

  // get primitive position and orientation
  daxa_f32vec3 p0 = get_first_vertex_by_triangle_index(particle.triangle_id);
  daxa_f32vec3 p1 = get_second_vertex_by_triangle_index(particle.triangle_id);
  daxa_f32vec3 p2 = get_third_vertex_by_triangle_index(particle.triangle_id);

  p0 = mat4_vec4_mul(transform, daxa_f32vec4(p0, 1)).xyz;
  p1 = mat4_vec4_mul(transform, daxa_f32vec4(p1, 1)).xyz;
  p2 = mat4_vec4_mul(transform, daxa_f32vec4(p2, 1)).xyz;
  
  daxa_f32vec3 normal = get_normal_by_vertices(p0, p1, p2);

  // Scatter to grid
  for (daxa_u32 i = 0; i < 3; ++i)
  {
      for (daxa_u32 j = 0; j < 3; ++j)
      {
          for (daxa_u32 k = 0; k < 3; ++k)
          {
              daxa_u32vec3 coord = array_grid + daxa_u32vec3(i, j, k);
              if (coord.x >= DAXA_DEREF(config)grid_dim.x || coord.y >= DAXA_DEREF(config)grid_dim.y || coord.z >= DAXA_DEREF(config)grid_dim.z)
              {
                  continue;
              }

              daxa_f32vec3 grid_pos = daxa_u32vec3(coord) * dx;

              // TODO: check if this vector is correct (p_pos - grid_pos)
              daxa_f32 signed_distance = dot(grid_pos - p_pos, normal);
              daxa_f32vec3 projected_point = grid_pos - signed_distance * normal;

              if(!inside_triangle(projected_point, p0, p1, p2)) {
                  continue;
              }

              daxa_f32 unsigned_distance = abs(signed_distance);
              bool negative = signed_distance < 0;

              daxa_u32 index = (coord.x + coord.y * DAXA_DEREF(config)grid_dim.x + coord.z * DAXA_DEREF(config)grid_dim.x * DAXA_DEREF(config)grid_dim.y);

              if (set_atomic_rigid_cell_distance_by_index(index, unsigned_distance) > unsigned_distance)
              {
                  if (set_atomic_rigid_cell_distance_by_index(index, unsigned_distance) == unsigned_distance)
                  {
                      set_atomic_rigid_cell_rigid_id_by_index(index, particle.rigid_id);
                      set_atomic_rigid_cell_rigid_particle_index_by_index(index, particle_index);
                  }
              }

              set_atomic_rigid_cell_color_by_index(index, particle.rigid_id, negative);
          }
      }
  }
}



void rigid_body_g2p_check_particle_interaction(out daxa_f32vec3 vel_value, daxa_u32 index, daxa_u32 particle_color, daxa_f32vec3 particle_pos, daxa_f32vec3 particle_vel, daxa_f32vec3 particle_normal, daxa_f32 weight, daxa_f32 p_mass, daxa_f32 dt, daxa_f32 dx) {
  vel_value = get_cell_by_index(index).v;

  NodeCDF rigid_cell = get_node_cdf_by_index(index);
  daxa_u32 grid_color = rigid_cell.color;
  // the particle has collided and needs to be projected along the collider
  if(!cdf_is_compatible(grid_color, particle_color)) {
      daxa_u32 rigid_id = rigid_cell.rigid_id;

      if(rigid_id == -1) {
          return;
      }

      RigidBody r = get_rigid_body_by_index(rigid_id);

      // Particle in collision with rigid body
      daxa_f32vec3 projected_velocity = particle_collision(particle_vel, particle_normal, r, particle_pos, dt, dx);

      if(any(isnan(projected_velocity)) || any(isinf(projected_velocity))) {
          return;
      }
      
      vel_value = projected_velocity;

      // Apply impulse to rigid body
      daxa_f32vec3 impulse = weight * (particle_vel - projected_velocity) * p_mass;
      rigid_body_apply_delta_impulse(r, rigid_id, impulse, particle_pos);
  }
}



void rigid_body_advect(daxa_f32 dt, daxa_f32 gravity, daxa_u32 flags, daxa_u32 rigid_index) {
  RigidBody r = get_rigid_body_by_index(rigid_index);

  // Apply delta velocity
  rigid_body_apply_temporal_velocity(r);

  // // Apply angular velocity
  if(vec3_abs_max(r.rotation_axis) > 0.1f) {
      rigid_body_enforce_angular_velocity_parallel_to(r, r.rotation_axis);
  }

  // Advance rigid body simulation
  rigid_body_advance(r, dt);

  if ((flags & RIGID_BODY_ADD_GRAVITY_FLAG) == RIGID_BODY_ADD_GRAVITY_FLAG) {
      // Apply gravity force
      rigid_body_apply_impulse(r, daxa_f32vec3(0, gravity, 0) * r.mass * dt, r.position);
  }

  // // Apply angular velocity
  if(vec3_abs_max(r.rotation_axis) > 0.1f) {
      rigid_body_enforce_angular_velocity_parallel_to(r, r.rotation_axis);
  }

  // Save parameters
  rigid_body_save_parameters(r, rigid_index);
}


void rigid_body_check_boundaries(daxa_f32 dt, daxa_f32 dx, DAXA_PTR(GpuStatus) status, DAXA_PTR(GpuInput) config, daxa_u32 rigid_index) {
  RigidBody r = get_rigid_body_by_index(rigid_index);

  daxa_f32mat4x4 transform = rigid_body_get_transform_matrix(r);
  daxa_f32vec3 minimum = mat4_vec4_mul(transform, daxa_f32vec4(r.min, 1)).xyz;
  daxa_f32vec3 maximum = mat4_vec4_mul(transform, daxa_f32vec4(r.max, 1)).xyz;

  // Repulsion force
  if (((DAXA_DEREF(status)flags & MOUSE_TARGET_FLAG) == MOUSE_TARGET_FLAG) && 
  DAXA_DEREF(status)rigid_body_index == rigid_index)
  {
      if ((DAXA_DEREF(status)flags & RIGID_BODY_PICK_UP_ENABLED_FLAG) == RIGID_BODY_PICK_UP_ENABLED_FLAG) {
          daxa_f32vec3 hit_position = mat4_vec4_mul(transform, daxa_f32vec4(DAXA_DEREF(status)local_hit_position, 1.0)).xyz;
          daxa_f32vec3 impulse_src_pos = DAXA_DEREF(status)hit_origin + DAXA_DEREF(status)hit_direction * DAXA_DEREF(status)hit_distance;
          daxa_f32vec3 impulse_dir = normalize(impulse_src_pos - hit_position);
          daxa_f32vec3 impulse = (impulse_dir * DAXA_DEREF(config)applied_force * r.mass * dt);
          rigid_body_apply_impulse(r, impulse, r.position);

      } else if ((DAXA_DEREF(status)flags & RIGID_BODY_PUSHING_ENABLED_FLAG) == RIGID_BODY_PUSHING_ENABLED_FLAG) {
          daxa_f32vec3 impulse_position = DAXA_DEREF(status)mouse_target;
          daxa_f32vec3 impulse_dir = normalize(impulse_position - DAXA_DEREF(status)hit_origin);
          daxa_f32vec3 impulse = (impulse_dir * DAXA_DEREF(config)applied_force * r.mass * dt);
          rigid_body_apply_impulse(r, impulse, r.position);
      } else if ((DAXA_DEREF(status)flags & RIGID_BODY_REST_POS_ENABLED_FLAG) == RIGID_BODY_REST_POS_ENABLED_FLAG) {
          r.velocity = daxa_f32vec3(0.0f);
          r.omega = daxa_f32vec3(0.0f);
          r.rotation = daxa_f32vec4(0.0f, 0.0f, 0.0f, 1.0f);
      } 
  }
  
  const daxa_u32 bound = BOUNDARY;
  daxa_f32vec3 min_bound = daxa_f32vec3((dx * bound));
  daxa_f32vec3 max_bound = daxa_f32vec3(1 - dx * bound);


  // check collision of aabb against planes
  daxa_f32vec3 penetrations[4] = {daxa_f32vec3(0), daxa_f32vec3(0), daxa_f32vec3(0), daxa_f32vec3(0)};
  daxa_u32 collision_count = 0;

  // Get OBB corners
  daxa_f32vec3 corners[8] = {
      {minimum.x, minimum.y, minimum.z}, {minimum.x, minimum.y, maximum.z}, {minimum.x, maximum.y, minimum.z}, {minimum.x, maximum.y, maximum.z}, 
      {maximum.x, minimum.y, minimum.z}, {maximum.x, minimum.y, maximum.z}, {maximum.x, maximum.y, minimum.z}, {maximum.x, maximum.y, maximum.z}
  };

  // Check for collisions and calculate penetration
  for (int i = 0; i < 8; i++) {
    daxa_f32vec3 corner = corners[i];
    daxa_b32 collision = false;
    if (corner.x < min_bound.x) {
      penetrations[collision_count].x = max(penetrations[collision_count].x, min_bound.x - corner.x);
      collision = true;
    }
    if (corner.x > max_bound.x) {
      penetrations[collision_count].x = min(penetrations[collision_count].x, corner.x - max_bound.x);
      collision = true;
    }
    if (corner.y < min_bound.y) {
      penetrations[collision_count].y = max(penetrations[collision_count].y, min_bound.y - corner.y);
      collision = true;
    }
    if (corner.y > max_bound.y) {
      penetrations[collision_count].y = min(penetrations[collision_count].y, corner.y - max_bound.y);
      collision = true;
    }
    if (corner.z < min_bound.z) {
      penetrations[collision_count].z = max(penetrations[collision_count].z, min_bound.z - corner.z);
      collision = true;
    }
    if (corner.z > max_bound.z) {
      penetrations[collision_count].z = min(penetrations[collision_count].z, corner.z - max_bound.z);
      collision = true;
    }

    if(collision) {
      collision_count++;
    }
  }

  // Handle collision response
  if (collision_count > 0) {
    for(int i = 0; i < collision_count; i++) {
      daxa_f32vec3 penetration = penetrations[i];
      daxa_f32vec3 normal = normalize(penetration);
      daxa_f32vec3 collision_point = r.position + normal * length(penetration);
      
      daxa_f32 friction = rigid_body_get_boundary_friction(r);
      daxa_f32 restitution = r.restitution;
      daxa_f32vec3 v_at_point = rigid_body_get_velocity_at(r, collision_point);
      daxa_f32vec3 r0 = collision_point - r.position;
      daxa_f32 v_normal = dot(normal, v_at_point);

      daxa_f32 impulse_denom = rigid_body_get_impulse_contribution(r, r0, normal);
      daxa_f32 J = -(1 + restitution) * v_normal / impulse_denom;

      if (J > 0) {
        daxa_f32vec3 impulse = J * normal;
        rigid_body_apply_impulse(r, impulse, collision_point);

        // Friction
        daxa_f32vec3 v_after = rigid_body_get_velocity_at(r, collision_point);
        daxa_f32vec3 v_tangent = v_after - dot(v_after, normal) * normal;
        
        if (length(v_tangent) > COLLISION_GUARD) {
            daxa_f32vec3 tangent = normalize(v_tangent);
            daxa_f32 j_tangent = -dot(v_after, tangent) / rigid_body_get_impulse_contribution(r, r0, tangent);
            j_tangent = clamp(j_tangent, -friction * J, friction * J);
            daxa_f32vec3 friction_impulse = j_tangent * tangent;
            rigid_body_apply_impulse(r, friction_impulse, collision_point);
        }

        // // Move the object out of penetration
        // r.position += penetration;
      }
    }
  }

  daxa_f32 max_v = DAXA_DEREF(config)max_rigid_velocity;
  // cap velocity
  if (length(r.velocity) > max_v)
  { 
    r.velocity = normalize(r.velocity) * max_v;
  }

  // Advance rigid body simulation
  rigid_body_advance(r, dt);

  // Save parameters
  rigid_body_save_parameters(r, rigid_index);
}


#endif // GLSL & HLSL & SLANG

#endif // DAXA_RIGID_BODY_FLAG
