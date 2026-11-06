extern "C" __global__ void ttu_multiwarp(unsigned* output) {
    __syncthreads();
    float value=threadIdx.x;
#pragma unroll
    for (int index=0;index<8192;++index)
        asm volatile("add.f32 %0, %0, 0f3f800000;" : "+f"(value));
    __syncthreads();
    output[threadIdx.x]=value;
}
