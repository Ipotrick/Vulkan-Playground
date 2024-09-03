#include "shared.inl"

struct RayPayload
{
    float4 color;
};

struct MyAttributes
{
    float3 normal;
    float3 position;
};

[shader("raygeneration")]
void rayGenShader()
{
}

[shader("miss")]
void missShader(inout RayPayload payload)
{
    payload.color = float4(0.5, 0.3, 0.1, 1);
}

[shader("closesthit")]
void closestHitShader(inout RayPayload payload, 
in MyAttributes attr)
{
  payload.color = float4(0.3, 0.8, 1, 1.0);
}

[shader("intersection")]
void intersectionShader()
{
  float THit = RayTCurrent();
}