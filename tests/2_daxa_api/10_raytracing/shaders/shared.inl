#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>

#define MAX_PRIMITIVES 2
#define NUM_VERTICES 3
// #define ACTIVATE_ATOMIC_FLOAT // this will throws an exception if the device does not support atomic daxa_f32
// #define DAXA_SHADERLANG_COMPILE_SLANG

struct CameraView
{
    daxa_f32mat4x4 inv_view;
    daxa_f32mat4x4 inv_proj;
#if defined(ACTIVATE_ATOMIC_FLOAT)
    daxa_f32 hit_count;
#endif
};
DAXA_DECL_BUFFER_PTR(CameraView)

struct Aabb
{
    daxa_f32vec3 minimum;
    daxa_f32vec3 maximum;
};

struct Aabbs
{
    Aabb aabbs[MAX_PRIMITIVES];
};
DAXA_DECL_BUFFER_PTR(Aabbs)

struct Vertices
{
    daxa_f32vec3 vertices[NUM_VERTICES];
};  
DAXA_DECL_BUFFER_PTR(Vertices)

struct PushConstant
{
    daxa_u32 frame;
    daxa_u32vec2 size;
    daxa_TlasId tlas;
#if defined(DAXA_SHADERLANG_COMPILE_SLANG)
    daxa::RWTexture2DId<daxa_f32vec4> swapchain;
#else
    daxa_ImageViewId swapchain;
#endif
    daxa_BufferPtr(CameraView) camera_buffer;
    daxa_BufferPtr(Vertices) vertex_buffer;
    daxa_BufferPtr(Aabbs) aabb_buffer;
};

struct rayLight
{
    daxa_f32vec3 inHitPosition;
    daxa_f32 outLightDistance;
    daxa_f32vec3 outLightDir;
    daxa_f32 outIntensity;
};
struct hitPayload
{
    daxa_f32vec3 hitValue;
    daxa_u32 seed;
#if defined(ACTIVATE_ATOMIC_FLOAT)
    daxa_b32 is_hit;
#endif
};

struct Ray
{
    daxa_f32vec3 origin;
    daxa_f32vec3 direction;
};

#if !defined(__cplusplus)


// Ray-AABB intersection
daxa_f32 hitAabb(const Aabb aabb, const Ray r)
{
    daxa_f32vec3 invDir = 1.0 / r.direction;
    daxa_f32vec3 tbot = invDir * (aabb.minimum - r.origin);
    daxa_f32vec3 ttop = invDir * (aabb.maximum - r.origin);
    daxa_f32vec3 tmin = min(ttop, tbot);
    daxa_f32vec3 tmax = max(ttop, tbot);
    daxa_f32 t0 = max(tmin.x, max(tmin.y, tmin.z));
    daxa_f32 t1 = min(tmax.x, min(tmax.y, tmax.z));
    return t1 > max(t0, 0.0) ? t0 : -1.0;
}


/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

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

// Generate a random daxa_f32 in [0, 1) given the previous RNG state
daxa_f32 rnd(inout daxa_u32 prev)
{
  return (daxa_f32(lcg(prev)) / daxa_f32(0x01000000));
}


#if !defined(GL_core_profile)

daxa_f32mat4x4 inverse(daxa_f32mat4x4 m)
{
    daxa_f32mat4x4 inv;
    daxa_f32 det;
    daxa_f32 inv_det;
    daxa_f32mat4x4 adj;

    inv[0][0] = m[1][1] * m[2][2] * m[3][3] - m[1][1] * m[2][3] * m[3][2] - m[2][1] * m[1][2] * m[3][3] + m[2][1] * m[1][3] * m[3][2] + m[3][1] * m[1][2] * m[2][3] - m[3][1] * m[1][3] * m[2][2];
    inv[1][0] = -m[1][0] * m[2][2] * m[3][3] + m[1][0] * m[2][3] * m[3][2] + m[2][0] * m[1][2] * m[3][3] - m[2][0] * m[1][3] * m[3][2] - m[3][0] * m[1][2] * m[2][3] + m[3][0] * m[1][3] * m[2][2];
    inv[2][0] = m[1][0] * m[2][1] * m[3][3] - m[1][0] * m[2][3] * m[3][1] - m[2][0] * m[1][1] * m[3][3] + m[2][0] * m[1][3] * m[3][1] + m[3][0] * m[1][1] * m[2][3] - m[3][0] * m[1][3] * m[2][1];
    inv[3][0] = -m[1][0] * m[2][1] * m[3][2] + m[1][0] * m[2][2] * m[3][1] + m[2][0] * m[1][1] * m[3][2] - m[2][0] * m[1][2] * m[3][1] - m[3][0] * m[1][1] * m[2][2] + m[3][0] * m[1][2] * m[2][1];
    inv[0][1] = -m[0][1] * m[2][2] * m[3][3] + m[0][1] * m[2][3] * m[3][2] + m[2][1] * m[0][2] * m[3][3] - m[2][1] * m[0][3] * m[3][2] - m[3][1] * m[0][2] * m[2][3] + m[3][1] * m[0][3] * m[2][2];
    inv[1][1] = m[0][0] * m[2][2] * m[3][3] - m[0][0] * m[2][3] * m[3][2] - m[2][0] * m[0][2] * m[3][3] + m[2][0] * m[0][3] * m[3][2] + m[3][0] * m[0][2] * m[2][3] - m[3][0] * m[0][3] * m[2][2];
    inv[2][1] = -m[0][0] * m[2][1] * m[3][3] + m[0][0] * m[2][3] * m[3][1] + m[2][0] * m[0][1] * m[3][3] - m[2][0] * m[0][3] * m[3][1] - m[3][0] * m[0][1] * m[2][3] + m[3][0] * m[0][3] * m[2][1];
    inv[3][1] = m[0][0] * m[2][1] * m[3][2] - m[0][0] * m[2][2] * m[3][1] - m[2][0] * m[0][1] * m[3][2] + m[2][0] * m[0][2] * m[3][1] + m[3][0] * m[0][1] * m[2][2] - m[3][0] * m[0][2] * m[2][1];
    inv[0][2] = m[0][1] * m[1][2] * m[3][3] - m[0][1] * m[1][3] * m[3][2] - m[1][1] * m[0][2] * m[3][3] + m[1][1] * m[0][3] * m[3][2] + m[3][1] * m[0][2] * m[1][3] - m[3][1] * m[0][3] * m[1][2];
    inv[1][2] = -m[0][0] * m[1][2] * m[3][3] + m[0][0] * m[1][3] * m[3][2] + m[1][0] * m[0][2] * m[3][3] - m[1][0] * m[0][3] * m[3][2] - m[3][0] * m[0][2] * m[1][3] + m[3][0] * m[0][3] * m[1][2];
    inv[2][2] = m[0][0] * m[1][1] * m[3][3] - m[0][0] * m[1][3] * m[3][1] - m[1][0] * m[0][1] * m[3][3] + m[1][0] * m[0][3] * m[3][1] + m[3][0] * m[0][1] * m[1][3] - m[3][0] * m[0][3] * m[1][1];
    inv[3][2] = -m[0][0] * m[1][1] * m[3][2] + m[0][0] * m[1][2] * m[3][1] + m[1][0] * m[0][1] * m[3][2] - m[1][0] * m[0][2] * m[3][1] - m[3][0] * m[0][1] * m[1][2] + m[3][0] * m[0][2] * m[1][1];
    inv[0][3] = -m[0][1] * m[1][2] * m[2][3] + m[0][1] * m[1][3] * m[2][2] + m[1][1] * m[0][2] * m[2][3] - m[1][1] * m[0][3] * m[2][2] - m[2][1] * m[0][2] * m[1][3] + m[2][1] * m[0][3] * m[1][2]; 
    inv[1][3] = m[0][0] * m[1][2] * m[2][3] - m[0][0] * m[1][3] * m[2][2] - m[1][0] * m[0][2] * m[2][3] + m[1][0] * m[0][3] * m[2][2] + m[2][0] * m[0][2] * m[1][3] - m[2][0] * m[0][3] * m[1][2];
    inv[2][3] = -m[0][0] * m[1][1] * m[2][3] + m[0][0] * m[1][3] * m[2][1] + m[1][0] * m[0][1] * m[2][3] - m[1][0] * m[0][3] * m[2][1] - m[2][0] * m[0][1] * m[1][3] + m[2][0] * m[0][3] * m[1][1];
    inv[3][3] = m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1] - m[1][0] * m[0][1] * m[2][2] + m[1][0] * m[0][2] * m[2][1] + m[2][0] * m[0][1] * m[1][2] - m[2][0] * m[0][2] * m[1][1];

    det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];
    inv_det = 1.0f / det;

    inv[0][0] *= inv_det;
    inv[0][1] *= inv_det;
    inv[0][2] *= inv_det;
    inv[0][3] *= inv_det;
    inv[1][0] *= inv_det;
    inv[1][1] *= inv_det;
    inv[1][2] *= inv_det;
    inv[1][3] *= inv_det;
    inv[2][0] *= inv_det;
    inv[2][1] *= inv_det;
    inv[2][2] *= inv_det;
    inv[2][3] *= inv_det;
    inv[3][0] *= inv_det;
    inv[3][1] *= inv_det;
    inv[3][2] *= inv_det;
    inv[3][3] *= inv_det;

    return inv;
}

daxa_f32mat4x4 Convert3x4To4x4(daxa_f32mat3x4 objectToWorld4x3)
{
    daxa_f32mat4x4 objectToWorld4x4;

    objectToWorld4x4[0] = float4(objectToWorld4x3[0], 0.0f);
    objectToWorld4x4[1] = float4(objectToWorld4x3[1], 0.0f);
    objectToWorld4x4[2] = float4(objectToWorld4x3[2], 0.0f);
    objectToWorld4x4[3] = float4(objectToWorld4x3[3], 1.0f);

    return objectToWorld4x4;
}

daxa_f32 aabb_get_hit(Aabb aabb, daxa_f32vec3 ray_origin, daxa_f32vec3 ray_direction, daxa_f32mat3x4 worldToObject) 
{
    // Convertir la matriz de transformación del objeto al espacio del objeto
    daxa_f32mat4x4 worldToObject = transpose(Convert3x4To4x4(worldToObject));

    // Transformar el origen y la dirección del rayo al espacio del objeto
    daxa_f32vec3 origin = (mul(worldToObject, daxa_f32vec4(ray_origin, 1.0f))).xyz;
    
    // Aquí se transforma la dirección sin normalización
    daxa_f32vec3 direction = (mul((daxa_f32mat3x3)worldToObject, ray_direction)).xyz;
    
    // Normalizar la dirección después de la transformación
    direction = normalize(direction);

    // Crear el rayo transformado
    Ray r = {origin, direction};

    // Calcular la intersección con el AABB
    return hitAabb(aabb, r);
}

#endif 
#endif // !defined(__cplusplus)