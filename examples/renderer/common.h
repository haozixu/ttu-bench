#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

struct Ray { float ox, oy, oz, tmin, dx, dy, dz, tmax; };
struct Hit { unsigned primitive; float t, u, v; unsigned front_face, instance_id; };
struct Triangle { float3 vertices[3]; };
struct Node { float lower[3], upper[3]; unsigned left, right, start, count; };
struct RenderParams {
    const Ray* rays;
    Hit* hits;
    const Triangle* triangles;
    const Node* nodes;
    const unsigned* indices;
    unsigned count, root_low, root_high, dummy_low, dummy_high;
};
static_assert(sizeof(Ray) == 32 && sizeof(Hit) == 24 && sizeof(Node) == 40, "record ABI");
static_assert(sizeof(RenderParams) == 64, "parameter ABI");

__device__ float3 subtract(float3 first, float3 second) {
    return make_float3(first.x-second.x, first.y-second.y, first.z-second.z);
}
__device__ float dot_product(float3 first, float3 second) {
    return first.x*second.x + first.y*second.y + first.z*second.z;
}
__device__ float3 cross_product(float3 first, float3 second) {
    return make_float3(first.y*second.z-first.z*second.y,
                      first.z*second.x-first.x*second.z, first.x*second.y-first.y*second.x);
}
__device__ Hit empty_hit(float tmax) { return {0xffffffffu, tmax, 0, 0, 0, 0xffffffffu}; }
