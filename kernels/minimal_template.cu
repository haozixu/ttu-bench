extern "C" __global__ void ttu_minimal(float* out) {
    float value = threadIdx.x;
#pragma unroll
    for (int index = 0; index < 512; ++index)
        asm volatile("add.f32 %0, %0, 0f3f800000;" : "+f"(value));
    out[threadIdx.x] = value;
}
