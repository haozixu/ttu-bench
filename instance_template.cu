#include <stdint.h>
#include <math.h>

struct DeviceParams {
    uint64_t ias_base, ias_size, dummy, child_base, child_size;
    uint32_t root_low, root_high, dummy_low, dummy_high;
    float ray[8];
};

struct DeviceResult {
    uint32_t status, requests, instance_id, parent_slot_nonzero;
    uint32_t initial[76], child[76];
    float local_ray[8];
    uint64_t event, child_root;
};

static_assert(sizeof(DeviceParams) == 88, "DeviceParams ABI mismatch");
static_assert(sizeof(DeviceResult) == 672, "DeviceResult ABI mismatch");

extern "C" __device__ __noinline__ void ttu_issue(const volatile uint64_t* input, volatile uint64_t* output);

__device__ void prepare(uint32_t* state, const float* ray, uint32_t root_low, uint32_t root_high,
                        uint32_t dummy_low, uint32_t dummy_high) {
    for (int index = 0; index < 76; ++index) state[index] = 0;
    state[2] = state[14] = __float_as_uint(ray[4]);
    state[3] = state[15] = __float_as_uint(ray[5]);
    state[12] = state[36] = __float_as_uint(ray[2]);
    state[13] = state[37] = __float_as_uint(ray[3]);
    state[16] = state[32] = __float_as_uint(ray[6]);
    state[17] = state[33] = __float_as_uint(ray[7]);
    state[24] = state[66] = __float_as_uint(ray[0]);
    state[25] = state[67] = __float_as_uint(ray[1]);
    state[40] = 0xff030407;
    state[41] = 0x700428;
    state[42] = state[72] = 0xff0000;
    state[50] = dummy_low;
    state[51] = dummy_high;
    state[52] = 0xff030207;
    state[53] = 0x400428;
    state[54] = root_low;
    state[55] = root_high;
}

__device__ void issue(uint32_t* state) {
    const unsigned pairs[18] = {42,52,12,66,16,14,72,40,36,24,32,2,34,50,56,54,60,58};
    uint64_t input[18], output[18];
    for (int index = 0; index < 18; ++index) {
        unsigned start = pairs[index];
        input[index] = uint64_t(state[start]) | uint64_t(state[start + 1]) << 32;
    }
    ttu_issue(input, output);
    for (int index = 0; index < 18; ++index) {
        unsigned start = pairs[index];
        state[start] = uint32_t(output[index]);
        state[start + 1] = uint32_t(output[index] >> 32);
    }
}

extern "C" __global__ void instance_probe(const DeviceParams* params, DeviceResult* result) {
    if (threadIdx.x || blockIdx.x) return;
    result->status = 0;
    result->requests = 0;
    result->instance_id = 0xffffffff;
    result->event = result->child_root = 0;
    uint32_t state[76];
    prepare(state, params->ray, params->root_low, params->root_high, params->dummy_low, params->dummy_high);
    issue(state);
    result->requests = 1;
    for (int index = 0; index < 76; ++index) result->initial[index] = state[index];
    // Diagnostic only: this is not proof that a parent can be resumed.
    result->parent_slot_nonzero = state[51] != 0;
    if (state[72] == 0xffffffff && state[73] == 0xffffffff) {
        result->status = 1;
        return;
    }
    if ((state[33] >> 27) != 0x16) {
        result->status = 2;
        return;
    }
    uint64_t event = uint64_t(state[72]) | uint64_t(state[73] & 0x1ffff) << 32;
    result->event = event;
    if (event < params->ias_base + 0x80 || event + 0x40 > params->ias_base + params->ias_size || (event & 63)) {
        result->status = 3;
        return;
    }
    uint64_t child_root = *reinterpret_cast<const uint64_t*>(event + 8) & ((uint64_t(1) << 49) - 1);
    result->child_root = child_root;
    if (child_root < params->child_base || child_root + 128 > params->child_base + params->child_size || (child_root & 127)) {
        result->status = 4;
        return;
    }
    uint32_t index_word = *reinterpret_cast<const uint32_t*>(event + 4);
    uint64_t table_offset = *reinterpret_cast<const uint64_t*>(params->ias_base + 0x30);
    uint32_t instance_count = *reinterpret_cast<const uint32_t*>(params->ias_base + 0x70);
    uint64_t metadata_offset = table_offset + uint64_t(index_word - 1) * 0x80;
    if (!index_word || index_word > instance_count || metadata_offset > params->ias_size ||
        params->ias_size - metadata_offset < 0x80) {
        result->status = 6;
        return;
    }
    result->instance_id = *reinterpret_cast<const uint32_t*>(params->ias_base + metadata_offset + 0x74) & 0xffffff;
    const float* inverse = reinterpret_cast<const float*>(event + 0x10);
    float local_ray[8];
    for (int axis = 0; axis < 3; ++axis) {
        local_ray[axis] = inverse[4 * axis + 3];
        local_ray[axis + 4] = 0;
        for (int component = 0; component < 3; ++component) {
            local_ray[axis] += inverse[4 * axis + component] * params->ray[component];
            local_ray[axis + 4] += inverse[4 * axis + component] * params->ray[component + 4];
        }
    }
    local_ray[3] = params->ray[3];
    local_ray[7] = params->ray[7];
    for (int index = 0; index < 8; ++index) result->local_ray[index] = local_ray[index];
    prepare(state, local_ray, uint32_t(child_root >> 7), uint32_t(child_root >> 39) | 0x02484000,
            params->dummy_low, params->dummy_high);
    issue(state);
    result->requests = 2;
    for (int index = 0; index < 76; ++index) result->child[index] = state[index];
    result->status = 5;
}
