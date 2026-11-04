// Only an ELF container. All instructions are replaced before loading.
extern "C" __global__ void ttu_timing(float* out) {
    float value = threadIdx.x;
#pragma unroll
    for (int index = 0; index < 8192; ++index)
        asm volatile("add.f32 %0, %0, 0f3f800000;" : "+f"(value));
    out[threadIdx.x] = value;
}
