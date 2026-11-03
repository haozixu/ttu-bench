#include <optix.h>

struct Params {
    OptixTraversableHandle gas;
    unsigned int* out;
};
extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    float3 origin = make_float3(0.25f, 0.25f, -5.0f);
    float3 dir    = make_float3(0.0f, 0.0f, 1.0f);
    unsigned int p0 = 0;
    optixTrace(params.gas, origin, dir, 0.0f, 100.0f, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
               0, 1, 0, p0);
    params.out[idx.x] = p0;
}

extern "C" __global__ void __miss__ms() {
    optixSetPayload_0(0xDEAD0000u);           // miss marker
}

extern "C" __global__ void __closesthit__ch() {
    optixSetPayload_0(0x10000000u | optixGetPrimitiveIndex());
}
