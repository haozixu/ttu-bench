extern "C" __global__ void native_trace(const RenderParams* params) {
    unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= params->count) return;
    Ray ray = params->rays[index];
    uint32_t state[76] = {};
    state[2] = state[14] = __float_as_uint(ray.dx);
    state[3] = state[15] = __float_as_uint(ray.dy);
    state[12] = state[36] = __float_as_uint(ray.oz);
    state[13] = state[37] = __float_as_uint(ray.tmin);
    state[16] = state[32] = __float_as_uint(ray.dz);
    state[17] = state[33] = __float_as_uint(ray.tmax);
    state[24] = state[66] = __float_as_uint(ray.ox);
    state[25] = state[67] = __float_as_uint(ray.oy);
    state[40] = 0xff030407;
    state[41] = 0x700428;
    state[42] = state[72] = 0xff0000;
    state[50] = params->dummy_low;
    state[51] = params->dummy_high;
    state[52] = 0xff030207;
    state[53] = 0x400428;
    state[54] = params->root_low;
    state[55] = params->root_high;
    uint64_t input[18], output[18];
    const unsigned pairs[18] = {42,52,12,66,16,14,72,40,36,24,32,2,34,50,56,54,60,58};
    for (unsigned pair = 0; pair < 18; ++pair) {
        unsigned start = pairs[pair];
        input[pair] = uint64_t(state[start]) | uint64_t(state[start+1]) << 32;
    }
    ttu_issue(input, output);
    unsigned result = unsigned(output[6]);
    unsigned distance = unsigned(output[6] >> 32);
    unsigned bary_u = unsigned(output[10]);
    unsigned bary_v = unsigned(output[10] >> 32);
    Hit hit = empty_hit(ray.tmax);
    if (result != 0xffffffffu || distance != 0xffffffffu) {
        unsigned tag = result >> 28;
        if (tag == 4 || tag == 6 || tag == 8 || tag == 10) {
            hit = {result & 0x0fffffffu, fabsf(__uint_as_float(distance)),
                   fabsf(__uint_as_float(bary_u)), __uint_as_float(bary_v),
                   unsigned((bary_u >> 31) == 0), 0xffffffffu};
        } else hit = {0xfffffffeu, __uint_as_float(distance), __uint_as_float(bary_u),
                      __uint_as_float(bary_v), 0, result};
    }
    params->hits[index] = hit;
}
