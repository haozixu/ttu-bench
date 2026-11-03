// Phase-3b probe: unambiguous TTULD result capture.
// acc2/acc3 chains do NOT depend on cfg. The cfg loads are kept alive by a
// never-taken comparison. Therefore for tid==0:
//   out[0]  == 0.0f  and  out2[0] == 1.0f  IFF TTULD did not write.
// Any other value in out[0]/out2[0] is the TTULD result qword.
extern "C" __global__ void ttu_probe2(float* out, unsigned long long* cfg, float* out2) {
    float a = (float)threadIdx.x;
    float pad = a + 1.0f;              // shifts register allocation parity
    float acc = a;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        asm volatile("add.f32 %0, %1, %2;" : "=f"(acc) : "f"(acc), "f"(a));
    }

    unsigned long long root = __ldg(cfg + 0);
    unsigned long long oxy  = __ldg(cfg + 1);
    unsigned long long ozz  = __ldg(cfg + 2);
    unsigned long long dxy  = __ldg(cfg + 3);
    unsigned long long dzz  = __ldg(cfg + 4);

    float acc2 = acc;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        asm volatile("add.f32 %0, %1, %2;" : "=f"(acc2) : "f"(acc2), "f"(a));
    }

    float acc3 = acc2 + 1.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        asm volatile("add.f32 %0, %1, %2;" : "=f"(acc3) : "f"(acc3), "f"(a));
    }

    // never taken; keeps all five 64-bit loads alive (both halves used)
    if ((root ^ oxy ^ ozz ^ dxy ^ dzz) == 0xDEADBEEFCAFEBABEULL) {
        out[31] = 1.0f;
    }

    out[threadIdx.x] = acc2 + pad * 0.0f;
    out2[threadIdx.x] = acc3;
}
