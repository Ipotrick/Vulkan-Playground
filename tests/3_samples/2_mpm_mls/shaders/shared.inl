#pragma once

#define DAXA_RAY_TRACING 1
#if defined(GL_core_profile) // GLSL
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_ARB_gpu_shader_int64 : enable
#endif // GL_core_profile


#include "daxa/daxa.inl"

#define DAXA_SIMULATION_WATER_MPM_MLS
#if !defined(DAXA_SIMULATION_WATER_MPM_MLS)
// #define DAXA_SIMULATION_MANY_MATERIALS
#define DAXA_RIGID_BODY_FLAG
// WARN: rigid bodies don't collide with each other
// #define DAXA_SIMULATION_MANY_RIGID_BODIES
#endif // DAXA_SIMULATION_WATER_MPM_MLS

#if defined(DAXA_RIGID_BODY_FLAG)
// #define DAXA_LEVEL_SET_FLAG
#endif // DAXA_RIGID_BODY_FLAG

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
#define RIGID_BODY_IMPULSE_ENABLED_FLAG (1u << 5)
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
// const daxa_f32 rigid_body_densities[NUM_RIGID_BOX_COUNT] = {300.0f, 1500.0f, 10000.0f};
const daxa_f32 rigid_body_frictions[NUM_RIGID_BOX_COUNT] = {0.1f, 0.3f, -0.1f};
#else 
#define NUM_RIGID_BOX_COUNT 1
const daxa_f32 rigid_body_densities[NUM_RIGID_BOX_COUNT] = {5000.0f};
const daxa_f32 rigid_body_frictions[NUM_RIGID_BOX_COUNT] = {0.1f};
#endif
#define NUM_RIGID_PARTICLES 32768

#define BOX_VOLUME 0.47684f // dim.x * dim.y * dim.z

#define BOX_VERTEX_COUNT 8
#define BOX_INDEX_COUNT 36
#define BOX_TRIANGLE_COUNT 12

// #define STATE_MASK 0xAAAAAAAAU
// #define SIGN_MASK 0x55555555U
#define TAG_DISPLACEMENT MAX_RIGID_BODY_COUNT
#define RECONSTRUCTION_GUARD 1e-10f
#define COLLISION_GUARD 1e-7f
#define EPSILON 1e-6f

#define COUNTER_CLOCKWISE 0
#define CLOCKWISE 1
#define TRIANGLE_ORIENTATION COUNTER_CLOCKWISE

#define PENALTY_FORCE 1e3f
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

#if defined(DAXA_LEVEL_SET_FLAG)
struct NodeLevelSet {
  daxa_f32 distance;
};
#endif // DAXA_LEVEL_SET_FLAG
#endif // DAXA_RIGID_BODY_FLAG

DAXA_DECL_BUFFER_PTR(GpuInput)
DAXA_DECL_BUFFER_PTR(GpuStatus)
DAXA_DECL_BUFFER_PTR(Particle)
#if defined(DAXA_RIGID_BODY_FLAG)
DAXA_DECL_BUFFER_PTR(RigidBody)
DAXA_DECL_BUFFER_PTR(RigidParticle)
DAXA_DECL_BUFFER_PTR(NodeCDF)
DAXA_DECL_BUFFER_PTR(ParticleCDF)
#if defined(DAXA_LEVEL_SET_FLAG)
DAXA_DECL_BUFFER_PTR(NodeLevelSet)
#endif // DAXA_LEVEL_SET_FLAG
#endif // DAXA_RIGID_BODY_FLAG
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
#if defined(DAXA_RIGID_BODY_FLAG)
    daxa_BufferPtr(RigidBody) rigid_bodies;
    daxa_BufferPtr(daxa_u32) indices;
    daxa_BufferPtr(daxa_f32vec3) vertices;
    daxa_RWBufferPtr(RigidParticle) rigid_particles;
    daxa_RWBufferPtr(NodeCDF) rigid_cells;
    daxa_RWBufferPtr(ParticleCDF) rigid_particle_color;
#if defined(DAXA_LEVEL_SET_FLAG)
    daxa_BufferPtr(NodeLevelSet) level_set_grid;
#endif // DAXA_LEVEL_SET_FLAG
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
const daxa_f32vec3 RIGID_BODY_PARTICLE_COLOR = daxa_f32vec3(0.6f, 0.4f, 0.2f);
const daxa_f32vec3 WATER_HIGH_SPEED_COLOR = daxa_f32vec3(0.3f, 0.5f, 1.0f);
const daxa_f32vec3 WATER_LOW_SPEED_COLOR = daxa_f32vec3(0.1f, 0.2f, 0.4f);
const daxa_f32vec3 SNOW_HIGH_SPEED_COLOR = daxa_f32vec3(0.9f, 0.9f, 1.0f);
const daxa_f32vec3 SNOW_LOW_SPEED_COLOR = daxa_f32vec3(0.5f, 0.5f, 0.6f);
const daxa_f32vec3 JELLY_HIGH_SPEED_COLOR = daxa_f32vec3(1.0f, 0.5f, 0.5f);
const daxa_f32vec3 JELLY_LOW_SPEED_COLOR = daxa_f32vec3(0.7f, 0.2f, 0.2f);



#if defined(GL_core_profile) // GLSL
#extension GL_EXT_shader_atomic_float : enable

DAXA_DECL_PUSH_CONSTANT(ComputePush, p)

layout(buffer_reference, scalar) buffer PARTICLE_BUFFER {Particle particles[]; }; // Particle buffer
layout(buffer_reference, scalar) buffer CELL_BUFFER {Cell cells[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer AABB_BUFFER {Aabb aabbs[]; }; // Particle positions
#if defined(DAXA_RIGID_BODY_FLAG)
layout(buffer_reference, scalar) buffer RIGID_BODY_BUFFER {RigidBody rigid_bodies[]; }; // Rigid body information
layout(buffer_reference, scalar) buffer INDEX_BUFFER {uint indices[]; }; // Rigid body indices info
layout(buffer_reference, scalar) buffer VERTEX_BUFFER {vec3 vertices[]; }; // Rigid body vertices info
layout(buffer_reference, scalar) buffer RIGID_PARTICLE_BUFFER {RigidParticle particles[]; }; // Rigid particle buffer
layout(buffer_reference, scalar) buffer RIGID_CELL_BUFFER {NodeCDF cells[]; }; // Rigid cell buffer
layout(buffer_reference, scalar) buffer RIGID_PARTICLE_STATUS_BUFFER {ParticleCDF particles[]; }; // Rigid  Particle color buffer
#if defined(DAXA_LEVEL_SET_FLAG)
layout(buffer_reference, scalar) buffer LEVEL_SET_NODE_BUFFER {NodeLevelSet nodes[]; }; // Rigid cell level set buffer
#endif // DAXA_LEVEL_SET_FLAG
#endif 

daxa_i32
to_emulated_float(daxa_f32 f)
{
   daxa_i32 bits = floatBitsToInt(f);
   return f < 0 ? -2147483648 - bits : bits;
}

daxa_f32
from_emulated_float(daxa_i32 bits)
{
   return intBitsToFloat(bits < 0 ? -2147483648 - bits : bits);
}

daxa_i32
to_emulated_positive_float(daxa_f32 f)
{
   return floatBitsToInt(f);
}

daxa_f32
from_emulated_positive_float(daxa_i32 bits)
{
   return intBitsToFloat(bits);
}

daxa_f32 inverse_f32(daxa_f32 f) {
  // check for divide by zero
  return f == 0.0f ? 0.0f : 1.0f / f;
}


Particle get_particle_by_index(daxa_u32 particle_index) {
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  return particle_buffer.particles[particle_index];
}

#if defined(DAXA_RIGID_BODY_FLAG)
RigidBody get_rigid_body_by_index(daxa_u32 rigid_body_index) {
  RIGID_BODY_BUFFER rigid_body_buffer =
      RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index];
}

daxa_f32vec3 get_rigid_body_color_by_index(daxa_u32 rigid_body_index) {
  RIGID_BODY_BUFFER rigid_body_buffer =
      RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].color;
}


daxa_f32vec3 rigid_body_add_atomic_velocity_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 velocity) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  daxa_f32 x = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity.x, velocity.x);
  daxa_f32 y = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity.y, velocity.y);
  daxa_f32 z = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity.z, velocity.z);
  return vec3(x, y, z);
}

daxa_f32vec3 rigid_body_add_atomic_omega_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 omega) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  daxa_f32 x = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega.x, omega.x);
  daxa_f32 y = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega.y, omega.y);
  daxa_f32 z = atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega.z, omega.z);
  return vec3(x, y, z);
}

void rigid_body_add_atomic_velocity_delta_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 velocity_delta) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity_delta.x, velocity_delta.x);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity_delta.y, velocity_delta.y);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].velocity_delta.z, velocity_delta.z);
}

void rigid_body_add_atomic_omega_delta_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 omega_delta) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega_delta.x, omega_delta.x);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega_delta.y, omega_delta.y);
  atomicAdd(rigid_body_buffer.rigid_bodies[rigid_body_index].omega_delta.z, omega_delta.z);
}

daxa_f32vec3 rigid_body_get_position_by_index(daxa_u32 rigid_body_index) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].position;
}

void rigid_body_set_position_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 position) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].position = position;
}

void rigid_body_set_rotation_by_index(daxa_u32 rigid_body_index, daxa_f32vec4 rotation) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].rotation = rotation;
}

daxa_f32vec3 rigid_body_get_velocity_by_index(daxa_u32 rigid_body_index) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].velocity;
}

void rigid_body_set_velocity_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 velocity) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].velocity = velocity;
}

daxa_f32vec3 rigid_body_get_omega_by_index(daxa_u32 rigid_body_index) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].omega;
}

void rigid_body_set_omega_by_index(daxa_u32 rigid_body_index, daxa_f32vec3 omega) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].omega = omega;
}

daxa_f32vec3 rigid_body_get_velocity_delta_by_index(daxa_u32 rigid_body_index) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].velocity_delta;
}

void rigid_body_reset_velocity_delta_by_index(daxa_u32 rigid_body_index) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].velocity_delta = vec3(0, 0, 0);
}

daxa_f32vec3 rigid_body_get_omega_delta_by_index(daxa_u32 rigid_body_index) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].omega_delta;
}

void rigid_body_reset_omega_delta_by_index(daxa_u32 rigid_body_index) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  rigid_body_buffer.rigid_bodies[rigid_body_index].omega_delta = vec3(0, 0, 0);
}

daxa_f32vec4 rigid_body_get_rotation_by_index(daxa_u32 rigid_body_index) {
  RIGID_BODY_BUFFER rigid_body_buffer = RIGID_BODY_BUFFER(p.rigid_bodies);
  return rigid_body_buffer.rigid_bodies[rigid_body_index].rotation;
}

uint get_first_index_by_triangle_index(daxa_u32 triangle_index) {
  INDEX_BUFFER index_buffer = INDEX_BUFFER(p.indices);
  return index_buffer.indices[triangle_index * 3];
}

uint get_second_index_by_triangle_index(daxa_u32 triangle_index) {
  INDEX_BUFFER index_buffer = INDEX_BUFFER(p.indices);
  return index_buffer.indices[triangle_index * 3 + 1];
}

uint get_third_index_by_triangle_index(daxa_u32 triangle_index) {
  INDEX_BUFFER index_buffer = INDEX_BUFFER(p.indices);
  return index_buffer.indices[triangle_index * 3 + 2];
}


uvec3 get_indices_by_triangle_index(daxa_u32 triangle_index) {
  INDEX_BUFFER index_buffer = INDEX_BUFFER(p.indices);
  return uvec3(get_first_index_by_triangle_index(triangle_index), get_second_index_by_triangle_index(triangle_index), get_third_index_by_triangle_index(triangle_index));
}

vec3 get_vertex_by_index(daxa_u32 vertex_index) {
  VERTEX_BUFFER vertex_buffer = VERTEX_BUFFER(p.vertices);
  return vertex_buffer.vertices[vertex_index];
}

vec3 get_first_vertex_by_triangle_index(daxa_u32 triangle_index) {
  return get_vertex_by_index(get_first_index_by_triangle_index(triangle_index));
}

vec3 get_second_vertex_by_triangle_index(daxa_u32 triangle_index) {
  return get_vertex_by_index(get_second_index_by_triangle_index(triangle_index));
}

vec3 get_third_vertex_by_triangle_index(daxa_u32 triangle_index) {
  return get_vertex_by_index(get_third_index_by_triangle_index(triangle_index));
}

mat3 get_vertices_by_triangle_index(daxa_u32 triangle_index) {
  uvec3 indices = get_indices_by_triangle_index(triangle_index);
  return mat3(get_vertex_by_index(indices.x), get_vertex_by_index(indices.y), get_vertex_by_index(indices.z));
}

RigidParticle get_rigid_particle_by_index(daxa_u32 particle_index) {
  RIGID_PARTICLE_BUFFER rigid_particle_buffer = RIGID_PARTICLE_BUFFER(p.rigid_particles);
  return rigid_particle_buffer.particles[particle_index];
}

NodeCDF get_node_cdf_by_index(daxa_u32 cell_index) {
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  return rigid_cell_buffer.cells[cell_index];
}

void node_cdf_set_by_index(daxa_u32 cell_index, NodeCDF cell) {
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  rigid_cell_buffer.cells[cell_index] = cell;
}


daxa_u32 get_node_cdf_color_by_index(daxa_u32 cell_index) {
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  return rigid_cell_buffer.cells[cell_index].color;
}

void zeroed_out_node_cdf_by_index(daxa_u32 cell_index) {
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  rigid_cell_buffer.cells[cell_index].unsigned_distance = to_emulated_positive_float(MAX_DIST);
  rigid_cell_buffer.cells[cell_index].color = 0;
  rigid_cell_buffer.cells[cell_index].rigid_id = -1;
  rigid_cell_buffer.cells[cell_index].rigid_particle_index = -1;
}

daxa_f32 set_atomic_rigid_cell_distance_by_index(daxa_u32 cell_index, daxa_f32 unsigned_distance) {
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  daxa_i32 bits = to_emulated_positive_float(unsigned_distance);
  daxa_i32 result = atomicMin(rigid_cell_buffer.cells[cell_index].unsigned_distance, bits);
  return from_emulated_positive_float(result);
}

daxa_u32 set_atomic_rigid_cell_color_by_index(daxa_u32 cell_index, daxa_u32 rigid_id, bool negative) {
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  daxa_u32 negative_flag = negative ? 1u : 0u;
  daxa_u32 flags = (negative_flag << (TAG_DISPLACEMENT + rigid_id)) | (1u << rigid_id);
  return atomicOr(rigid_cell_buffer.cells[cell_index].color, flags);
}

daxa_u32 set_atomic_rigid_cell_rigid_id_by_index(daxa_u32 cell_index, daxa_u32 rigid_id) {
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  return atomicExchange(rigid_cell_buffer.cells[cell_index].rigid_id, rigid_id);
}

daxa_u32 set_atomic_rigid_cell_rigid_particle_index_by_index(daxa_u32 cell_index, daxa_u32 rigid_particle_index) {
  RIGID_CELL_BUFFER rigid_cell_buffer = RIGID_CELL_BUFFER(p.rigid_cells);
  return atomicExchange(rigid_cell_buffer.cells[cell_index].rigid_particle_index, rigid_particle_index);
}

void particle_CDF_init(inout ParticleCDF particle_CDF) {
  particle_CDF.distance = MAX_DIST;
  particle_CDF.color = 0;
  particle_CDF.difference = 0;
  particle_CDF.normal = vec3(0, 0, 0);
}

ParticleCDF get_rigid_particle_CDF_by_index(daxa_u32 particle_index) {
  RIGID_PARTICLE_STATUS_BUFFER rigid_particle_color_buffer = RIGID_PARTICLE_STATUS_BUFFER(p.rigid_particle_color);
  return rigid_particle_color_buffer.particles[particle_index];
}

daxa_u32 get_rigid_particle_CDF_color_by_index(daxa_u32 particle_index) {
  RIGID_PARTICLE_STATUS_BUFFER rigid_particle_color_buffer = RIGID_PARTICLE_STATUS_BUFFER(p.rigid_particle_color);
  return rigid_particle_color_buffer.particles[particle_index].color;
}

void set_rigid_particle_CDF_by_index(daxa_u32 particle_index, ParticleCDF color) {
  RIGID_PARTICLE_STATUS_BUFFER rigid_particle_color_buffer = RIGID_PARTICLE_STATUS_BUFFER(p.rigid_particle_color);
  rigid_particle_color_buffer.particles[particle_index] = color;
}

#if defined(DAXA_LEVEL_SET_FLAG)
NodeLevelSet level_set_get_node_by_index(daxa_u32 node_index) {
  LEVEL_SET_NODE_BUFFER level_set_buffer = LEVEL_SET_NODE_BUFFER(p.level_set_grid);
  return level_set_buffer.nodes[node_index];
}

daxa_f32 level_set_get_distance_by_index(daxa_u32 node_index) {
  LEVEL_SET_NODE_BUFFER level_set_buffer = LEVEL_SET_NODE_BUFFER(p.level_set_grid);
  return level_set_buffer.nodes[node_index].distance;
}

void level_set_node_set_distance_by_index(daxa_u32 node_index, daxa_f32 distance) {
  LEVEL_SET_NODE_BUFFER level_set_buffer = LEVEL_SET_NODE_BUFFER(p.level_set_grid);
  level_set_buffer.nodes[node_index].distance = distance;
}
#endif // DAXA_LEVEL_SET_FLAG

#endif // DAXA_RIGID_BODY_FLAG

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

void particle_set_velocity_by_index(daxa_u32 particle_index, daxa_f32vec3 v) {
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  particle_buffer.particles[particle_index].v = v;
}

void particle_set_F_by_index(daxa_u32 particle_index, daxa_f32mat3x3 F) {
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  particle_buffer.particles[particle_index].F = F;
}

void particle_set_C_by_index(daxa_u32 particle_index, daxa_f32mat3x3 C) {
  PARTICLE_BUFFER particle_buffer =
      PARTICLE_BUFFER(p.particles);
  particle_buffer.particles[particle_index].C = C;
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

#if defined(DAXA_RIGID_BODY_FLAG)
daxa_f32mat3x3 rigid_body_get_rotation_matrix(daxa_f32vec4 rotation) {
    daxa_f32vec4 quaternion = rotation;
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
  daxa_f32mat4x4 translation = daxa_f32mat4x4(1.0);
  translation[3] = daxa_f32vec4(rigid_body.position, 1.0);
  daxa_f32mat3x3 rotation = rigid_body_get_rotation_matrix(rigid_body.rotation);
  daxa_f32mat4x4 rotation_matrix = daxa_f32mat4x4(rotation);
  return translation * rotation_matrix;
}

daxa_f32mat4x4 rigid_body_get_transform_matrix_from_rotation_translation(daxa_f32vec4 rotation, daxa_f32vec3 position) {
  daxa_f32mat4x4 translation = daxa_f32mat4x4(1.0);
  translation[3] = daxa_f32vec4(position, 1.0);
  daxa_f32mat3x3 rotation_mat3 = rigid_body_get_rotation_matrix(rotation);
  daxa_f32mat4x4 rotation_matrix = daxa_f32mat4x4(rotation_mat3);
  return translation * rotation_matrix;
}
#endif

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
#endif // DAXA_RIGID_BODY_FLAG

#endif // GLSL & HLSL



#if defined(DAXA_RIGID_BODY_FLAG)
// Credits: https://github.com/taichi-dev/taichi/blob/c5af2f92bc481e99cac2bc548dfa98e188bbcc44/include/taichi/geometry/mesh.h
// Note: assuming world origin aligns with elem.v[0]
daxa_f32mat3x3 get_world_to_object_matrix(daxa_f32vec3 v0, daxa_f32vec3 v1, daxa_f32vec3 v2) {
  daxa_f32vec3 u = v1 - v0;
  daxa_f32vec3 v = v2 - v0;
#if TRIANGLE_ORIENTATION == CLOCKWISE
  daxa_f32vec3 w = normalize(cross(v, u));
#else
  daxa_f32vec3 w = normalize(cross(u, v));
#endif
  return inverse(daxa_f32mat3x3(u, v, w));
}



daxa_f32vec3 get_normal_by_vertices(daxa_f32vec3 v0, daxa_f32vec3 v1, daxa_f32vec3 v2) {
  daxa_f32vec3 u = v1 - v0;
  daxa_f32vec3 v = v2 - v0;
#if TRIANGLE_ORIENTATION == CLOCKWISE
  return normalize(cross(v, u));
#else
  return normalize(cross(u, v));
#endif  
}


#if defined(GL_core_profile) // GLSL
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

  data.weighted_vector += daxa_f32vec4(1.0f, dpos);
  data.weighted_matrix += daxa_f32mat4x4(1.0f, dpos.x , dpos.y, dpos.z,
                                         dpos.x, outer_product[0],
                                         dpos.y, outer_product[1],
                                         dpos.z, outer_product[2]);
  
}


ParticleCDF interpolated_particle_data_compute_particle_cdf(InterpolatedParticleData data, daxa_f32 dx) {
  ParticleCDF particle_cdf;
  particle_CDF_init(particle_cdf);
  if (abs(determinant(data.weighted_matrix)) > RECONSTRUCTION_GUARD)
  {
    daxa_f32vec4 result = inverse(data.weighted_matrix) * data.weighted_vector;

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

#endif // GLSL

#endif // DAXA_RIGID_BODY_FLAG
