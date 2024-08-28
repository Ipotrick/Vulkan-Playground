#include <shared.inl>
// #include <custom file!!>



#if defined(CHECK_RIGID_BODY_FLAG)
void gather_CDF_compute(daxa_u32 particle_index) {
    daxa_BufferPtr(GpuInput) config = daxa_BufferPtr(GpuInput)(daxa_id_to_address(p.input_buffer_id));

    float dx = deref(config).dx;
    float inv_dx = deref(config).inv_dx;
    daxa_u32 rigid_body_count = deref(config).rigid_body_count;

    Aabb aabb = get_aabb_by_index(particle_index);

    daxa_f32vec3 w[3];
    daxa_f32vec3 fx;
    daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

    uvec3 array_grid = uvec3(base_coord);

    InterpolatedParticleData data;
    interpolated_particle_data_init(data);

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

                daxa_f32 weight = w[i].x * w[j].y * w[k].z;

                NodeCDF nodeCDF = get_node_cdf_by_index(index);
                interpolate_color(data, nodeCDF, weight, rigid_body_count);
            }
        }
    }

    interpolated_particle_data_compute_tags(data, rigid_body_count);

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


daxa_f32vec3 get_rigid_body_velocity_at(RigidBody r, daxa_f32vec3 position) {

    return r.velocity + cross(r.omega, position - r.position);
}


daxa_f32vec3 particle_collision(daxa_f32vec3 velocity, daxa_f32vec3 normal,RigidBody r, daxa_f32vec3 particle_position, daxa_f32 dt, daxa_f32 dx) {

    daxa_f32 friction = r.friction;
    daxa_f32 pushing_force = r.pushing_force;
    daxa_f32vec3 rigid_velocity = get_rigid_body_velocity_at(r, particle_position);

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
    daxa_f32mat3x3 rotation = rigid_body_get_rotation_matrix(r);
    return rotation * r.inv_inertia * transpose(rotation);
}

void rigid_body_apply_delta_impulse(RigidBody r, daxa_u32 rigid_index, daxa_f32vec3 impulse, daxa_f32vec3 position) {
    daxa_f32vec3 torque = cross(position - r.position, impulse);
    
    daxa_f32vec3 lineal_velocity = impulse * r.inv_mass;
    daxa_f32vec3 angular_velocity = rigid_body_get_transformed_inversed_inertia(r) * torque;

    // TODO: this is too slow
    rigid_body_add_atomic_velocity_delta_by_index(rigid_index, lineal_velocity);
    rigid_body_add_atomic_omega_delta_by_index(rigid_index, angular_velocity);
}

void rigid_body_apply_impulse(daxa_f32vec3 impulse, inout RigidBody r, daxa_f32vec3 position) {
    daxa_f32vec3 torque = cross(position - r.position, impulse);
    
    daxa_f32vec3 lineal_velocity = impulse * r.inv_mass;
    daxa_f32vec3 angular_velocity = rigid_body_get_transformed_inversed_inertia(r) * torque;

    r.velocity += lineal_velocity;
    r.omega += angular_velocity;
}
    

void rigid_body_apply_temporal_velocity(inout RigidBody r) {
    r.velocity += r.velocity_delta;
    r.omega += r.omega_delta;
}

void rigid_body_save_parameters(RigidBody r, daxa_u32 rigid_index) {
    rigid_body_set_velocity_by_index(rigid_index, r.velocity);
    rigid_body_set_omega_by_index(rigid_index, r.omega);
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

daxa_f32vec4 rigid_body_aply_angular_velocity(daxa_f32vec4 rotation, daxa_f32vec3 omega, daxa_f32 dt) {
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
    r.rotation = rigid_body_aply_angular_velocity(r.rotation, r.omega, dt);
}

#endif // CHECK_RIGID_BODY_FLAG

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
#elif LEVEL_SET_COLLISION_COMPUTE_FLAG == 1
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
    
    RigidParticle particle = get_rigid_particle_by_index(pixel_i_x);
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

    if (particle.rigid_id > MAX_RIGID_BODY_COUNT)
    {
        return;
    }

    RigidBody r = get_rigid_body_by_index(particle.rigid_id);

    daxa_f32mat4x4 transform = rigid_body_get_transform_matrix(r);

    particle.min = (transform * vec4(particle.min, 1)).xyz;
    particle.max = (transform * vec4(particle.max, 1)).xyz;

    Aabb aabb = Aabb(particle.min, particle.max);

    daxa_f32vec3 p_pos = (aabb.min + aabb.max) * 0.5f;

    daxa_i32vec3 base_coord = calculate_particle_grid_pos(aabb, inv_dx);

    uvec3 array_grid = uvec3(base_coord);

    // get primitive position and orientation
    vec3 p0 = get_first_vertex_by_triangle_index(particle.triangle_id);
    vec3 p1 = get_second_vertex_by_triangle_index(particle.triangle_id);
    vec3 p2 = get_third_vertex_by_triangle_index(particle.triangle_id);

    p0 = (transform * vec4(p0, 1)).xyz;
    p1 = (transform * vec4(p1, 1)).xyz;
    p2 = (transform * vec4(p2, 1)).xyz;
    
    daxa_f32vec3 normal = get_normal_by_vertices(p0, p1, p2);

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

                // TODO: check if this vector is correct (p_pos - grid_pos)
                daxa_f32 signed_distance = dot(grid_pos - p_pos, normal);
                daxa_f32vec3 projected_point = grid_pos - signed_distance * normal;

                if(!inside_triangle(projected_point, p0, p1, p2)) {
                    continue;
                }

                daxa_f32 unsigned_distance = abs(signed_distance);
                bool negative = signed_distance < 0;

                uint index = (coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y);

                if (set_atomic_rigid_cell_distance_by_index(index, unsigned_distance) > unsigned_distance)
                {
                    if (set_atomic_rigid_cell_distance_by_index(index, unsigned_distance) == unsigned_distance)
                    {
                        set_atomic_rigid_cell_rigid_id_by_index(index, particle.rigid_id);
                        set_atomic_rigid_cell_rigid_particle_index_by_index(index, pixel_i_x);
                    }
                }

                set_atomic_rigid_cell_color_by_index(index, particle.rigid_id, negative);
            }
        }
    }
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
    daxa_i32vec3 base_coord = calculate_particle_color(aabb, inv_dx, fx, w);

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
    float const rest_density = 0.2f;
    float const dynamic_viscosity = 0.1f;
    // equation of state
    float const eos_stiffness = 4.0f;
    float const eos_power = 4;

    Particle particle = get_particle_by_index(pixel_i_x);

    Aabb aabb = get_aabb_by_index(pixel_i_x);

    daxa_f32vec3 w[3];
    daxa_f32vec3 fx;
    daxa_i32vec3 base_coord = calculate_particle_color(aabb, inv_dx, fx, w);

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

#if defined(CHECK_RIGID_BODY_FLAG)
    ParticleCDF particle_CDF = get_rigid_particle_CDF_by_index(pixel_i_x);
#endif // CHECK_RIGID_BODY_FLAG

    Aabb aabb = get_aabb_by_index(pixel_i_x);

    daxa_f32vec3 w[3];
    daxa_f32vec3 fx;
    daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

    mat3 stress = calculate_p2g(particle, dt, p_vol, mu_0, lambda_0, inv_dx);

    mat3 affine = stress + p_mass * particle.C;

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

                uint index = (coord.x + coord.y * deref(config).grid_dim.x + coord.z * deref(config).grid_dim.x * deref(config).grid_dim.y);

#if defined(CHECK_RIGID_BODY_FLAG)
                daxa_u32 grid_color = get_node_cdf_color_by_index(index);

                daxa_u32 particle_color = get_rigid_particle_CDF_color_by_index(pixel_i_x);

                // only update compatible particles
                if(!cdf_is_compatible(grid_color, particle_color)) {
                    continue;
                }
#endif // CHECK_RIGID_BODY_FLAG

                vec3 dpos = (vec3(i, j, k) - fx) * dx;

                float weight = w[i].x * w[j].y * w[k].z;

                vec3 velocity_mass = weight * (mv + affine * dpos);
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
    uint bound = 3;

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

    Particle particle = get_particle_by_index(pixel_i_x);
    Aabb aabb = get_aabb_by_index(pixel_i_x);

    daxa_f32vec3 w[3];
    daxa_f32vec3 fx;
    daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

    particle.C = mat3(0);
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

                vec3 grid_value = get_cell_by_index(index).v;

                vec3 w_grid = vec3(grid_value * weight);

                particle.v += w_grid; // Velocity
                particle.C += 4 * weight * outer_product(vel_value, dpos);
            }
        }
    }

    aabb.min += dt * particle.v;
    aabb.max += dt * particle.v;

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

#if defined(CHECK_RIGID_BODY_FLAG)
    gather_CDF_compute(pixel_i_x);
#endif // CHECK_RIGID_BODY_FLAG

    daxa_BufferPtr(GpuStatus) status = daxa_BufferPtr(GpuStatus)(daxa_id_to_address(p.status_buffer_id));

    float dx = deref(config).dx;
    float inv_dx = deref(config).inv_dx;
    float dt = deref(config).dt;
    float p_mass = 1.0f;

    Particle particle = get_particle_by_index(pixel_i_x);
    Aabb aabb = get_aabb_by_index(pixel_i_x);

#if defined(CHECK_RIGID_BODY_FLAG)
    ParticleCDF particle_CDF = get_rigid_particle_CDF_by_index(pixel_i_x);
#endif // CHECK_RIGID_BODY_FLAG

    daxa_f32vec3 w[3];
    daxa_f32vec3 fx;
    daxa_i32vec3 base_coord = calculate_particle_status(aabb, inv_dx, fx, w);

    daxa_f32mat3x3 particle_C = mat3(0);
    daxa_f32vec3 particle_velocity = daxa_f32vec3(0);

    uvec3 array_grid = uvec3(base_coord);

    vec3 pos_x = (aabb.min + aabb.max) * 0.5f;

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

                vec3 dpos = (vec3(i, j, k) - fx) * dx;
                float weight = w[i].x * w[j].y * w[k].z;

                vec3 vel_value;

#if defined(CHECK_RIGID_BODY_FLAG)
                
                vel_value = get_cell_by_index(index).v;

                NodeCDF rigid_cell = get_node_cdf_by_index(index);
                daxa_u32 grid_color = rigid_cell.color;

                daxa_u32 particle_color = get_rigid_particle_CDF_color_by_index(pixel_i_x);

                // the particle has collided and needs to be projected along the collider
                if(!cdf_is_compatible(grid_color, particle_color)) {
                    daxa_u32 rigid_id = rigid_cell.rigid_id;

                    if(rigid_id == -1) {
                        continue;
                    }

                    RigidBody r = get_rigid_body_by_index(rigid_id);

                    daxa_u32 rigid_particle_index = rigid_cell.rigid_particle_index;

                    // Particle in collision with rigid body
                    daxa_f32vec3 projected_velocity = particle_collision(particle.v, particle_CDF.normal, r, pos_x, dt, dx);
                    
                    vel_value = projected_velocity;

                    // Apply impulse to rigid body
                    daxa_f32vec3 impulse = weight * (particle.v - projected_velocity) * p_mass;
                    rigid_body_apply_delta_impulse(r, rigid_id, impulse, pos_x);
                    
                }
#else
                vel_value = get_cell_by_index(index).v;
#endif // CHECK_RIGID_BODY_FLAG
                vec3 w_grid = vec3(vel_value * weight);

                particle_velocity += w_grid; // Velocity
                particle_C += 4 * inv_dx * inv_dx * weight * outer_product(vel_value, dpos);
            }
        }
    }

    particle.v = particle_velocity;
    particle.C = particle_C;

#if defined(CHECK_RIGID_BODY_FLAG)
    if(particle_CDF.difference != 0) {
        daxa_f32vec3 f_penalty = abs(particle_CDF.distance) * particle_CDF.normal * PENALTY_FORCE;
        particle.v += dt * f_penalty / p_mass;
    }
    // }
#endif // CHECK_RIGID_BODY_FLAG

    aabb.min += dt * particle.v;
    aabb.max += dt * particle.v;

    vec3 pos = (aabb.min + aabb.max) * 0.5f;
    const float wall_min = 3 * dx;
    float wall_max = (float(deref(config).grid_dim.x) - 3) * dx;

    // Repulsion force
    if ((deref(status).flags & MOUSE_TARGET_FLAG) == MOUSE_TARGET_FLAG)
    {
        if (all(greaterThan(deref(status).mouse_target, vec3(wall_min))) &&
            all(lessThan(deref(status).mouse_target, vec3(wall_max))))
        {
            vec3 dist = pos - deref(status).mouse_target;
            if (dot(dist, dist) < deref(config).mouse_radius * deref(config).mouse_radius)
            {
                vec3 force = normalize(dist) * 0.05f;
                particle.v += force;
            }
        }
    }

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

    RigidBody r = get_rigid_body_by_index(pixel_i_x);

    // Apply delta velocity
    rigid_body_apply_temporal_velocity(r);

    // // Apply angular velocity
    if(vec3_abs_max(r.rotation_axis) > 0.1f) {
        rigid_body_enforce_angular_velocity_parallel_to(r, r.rotation_axis);
    }

    // Advance rigid body simulation
    rigid_body_advance(r, dt);

    // // Apply gravity force
    // rigid_body_apply_impulse(daxa_f32vec3(0, deref(config).gravity, 0) * r.mass * dt, r, r.position);

    // // Apply angular velocity
    if(vec3_abs_max(r.rotation_axis) > 0.1f) {
        rigid_body_enforce_angular_velocity_parallel_to(r, r.rotation_axis);
    }

    // Save parameters
    rigid_body_save_parameters(r, pixel_i_x);
}
#else
// Main compute shader
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main()
{
}
#endif
