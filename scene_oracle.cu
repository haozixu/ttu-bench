#include <optix.h>

struct Ray { float ox, oy, oz, tmin, dx, dy, dz, tmax; };
struct Hit { unsigned primitive, t, u, v, front_face, instance_id; };
struct Params { OptixTraversableHandle gas; const Ray* rays; Hit* hits; unsigned ray_flags, has_instances; };
extern "C" { __constant__ Params params; }

extern "C" __global__ void __raygen__scene() {
    const unsigned index = optixGetLaunchIndex().x;
    const Ray ray = params.rays[index];
    unsigned primitive = 0xffffffffu, t = __float_as_uint(ray.tmax), u = 0, v = 0, front_face = 0;
    unsigned instance_id = 0xffffffffu;
    optixTrace(params.gas, make_float3(ray.ox, ray.oy, ray.oz),
               make_float3(ray.dx, ray.dy, ray.dz), ray.tmin, ray.tmax, 0.0f,
               OptixVisibilityMask(255), params.ray_flags,
               0, 1, 0, primitive, t, u, v, front_face, instance_id);
    params.hits[index] = {primitive, t, u, v, front_face, instance_id};
}

extern "C" __global__ void __miss__scene() {}

extern "C" __global__ void __closesthit__scene() {
    const float2 bary = optixGetTriangleBarycentrics();
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(__float_as_uint(bary.x));
    optixSetPayload_3(__float_as_uint(bary.y));
    optixSetPayload_4(optixIsFrontFaceHit());
    optixSetPayload_5(params.has_instances ? optixGetInstanceId() : 0xffffffffu);
}
