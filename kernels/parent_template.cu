// Appended to instance_template.cu by the compiler-managed TTU harness.
// Restricted to one flat IAS with at most three static instances of one GAS.
struct PendingEntry { uint32_t low, high, entry_low, entry_high; };
struct ParentParams { DeviceParams base; uint32_t stop_after_first, reserved; };
struct ParentTrace {
    uint32_t kind, instance_id, index_word, pending_after;
    uint64_t event, child_root;
    uint32_t input[76], parent[76], child[76];
    float local_ray[8];
};
struct ParentResult {
    uint32_t status, requests, steps, pending_count, maximum_pending;
    uint32_t instance_id, primitive, front_face;
    float t, u, v;
    uint32_t reserved;
    ParentTrace trace[8];
};
static_assert(sizeof(ParentTrace) == 976, "ParentTrace ABI mismatch");
static_assert(sizeof(ParentResult) == 7856, "ParentResult ABI mismatch");
static_assert(sizeof(ParentParams) == 96, "ParentParams ABI mismatch");

extern "C" __global__ void parent_probe(const ParentParams* control, ParentResult* result) {
    if (threadIdx.x || blockIdx.x) return;
    const DeviceParams* params = &control->base;
    result->status = 0;
    result->requests = result->steps = result->pending_count = result->maximum_pending = 0;
    result->instance_id = result->primitive = 0xffffffff;
    result->t = params->ray[7];
    uint32_t count = *reinterpret_cast<const uint32_t*>(params->ias_base + 0x70);
    if (!count || count > 3) { result->status = 7; return; }
    PendingEntry pending[16];
    unsigned top = 0;
    uint32_t state[76];
    prepare(state, params->ray, params->root_low, params->root_high, params->dummy_low, params->dummy_high);
    for (unsigned step = 0; step < 8; ++step) {
        ParentTrace& trace = result->trace[step];
        for (int index = 0; index < 76; ++index) trace.input[index] = state[index];
        issue(state);
        ++result->requests;
        ++result->steps;
        for (int index = 0; index < 76; ++index) trace.parent[index] = state[index];
        if (state[72] == 0xffffffff && state[73] == 0xffffffff) {
            trace.kind = 1;
        } else {
            if ((state[33] >> 27) != 0x16) { result->status = 3; return; }
            trace.kind = 2;
            uint64_t event = uint64_t(state[72]) | uint64_t(state[73] & 0x1ffff) << 32;
            trace.event = event;
            if (event < params->ias_base + 0x80 || event + 0x40 > params->ias_base + params->ias_size || (event & 63)) {
                result->status = 4; return;
            }
            uint32_t n = (state[55] >> 25) & 7;
            if (n > 3 || top + n > 16) { result->status = 7; return; }
            for (int entry = int(n) - 1; entry >= 0; --entry) {
                uint32_t high = state[57 + 2 * entry];
                // Only the observed relative-reference entry type is implemented.
                if (((high >> 17) & 0x1ff) != 0x21) { result->status = 7; return; }
                pending[top++] = {state[54], (state[55] & ~(7u << 25)) | (1u << 25),
                                  state[56 + 2 * entry], high};
            }
            if (top > result->maximum_pending) result->maximum_pending = top;
            trace.pending_after = top;
            uint32_t index_word = *reinterpret_cast<const uint32_t*>(event + 4);
            uint64_t table = *reinterpret_cast<const uint64_t*>(params->ias_base + 0x30);
            if (!index_word || index_word > count || table > params->ias_size ||
                uint64_t(index_word) * 0x80 > params->ias_size - table) {
                result->status = 6; return;
            }
            uint64_t metadata = params->ias_base + table + uint64_t(index_word - 1) * 0x80;
            trace.instance_id = *reinterpret_cast<const uint32_t*>(metadata + 0x74) & 0xffffff;
            trace.index_word = index_word;
            uint64_t root = *reinterpret_cast<const uint64_t*>(event + 8) & ((uint64_t(1) << 49) - 1);
            trace.child_root = root;
            if (root < params->child_base || root + 128 > params->child_base + params->child_size || (root & 127)) {
                result->status = 5; return;
            }
            const float* matrix = reinterpret_cast<const float*>(event + 0x10);
            for (int axis = 0; axis < 3; ++axis) {
                float origin = matrix[4 * axis + 3], direction = 0;
                for (int component = 0; component < 3; ++component) {
                    origin += matrix[4 * axis + component] * params->ray[component];
                    direction += matrix[4 * axis + component] * params->ray[4 + component];
                }
                trace.local_ray[axis] = origin;
                trace.local_ray[4 + axis] = direction;
            }
            trace.local_ray[3] = params->ray[3];
            trace.local_ray[7] = result->t;
            prepare(state, trace.local_ray, uint32_t(root >> 7), uint32_t(root >> 39) | 0x02484000,
                    params->dummy_low, params->dummy_high);
            issue(state);
            ++result->requests;
            for (int index = 0; index < 76; ++index) trace.child[index] = state[index];
            if (!(state[72] == 0xffffffff && state[73] == 0xffffffff)) {
                float t = fabsf(__uint_as_float(state[73]));
                float u = fabsf(__uint_as_float(state[32])), v = __uint_as_float(state[33]);
                unsigned tag = state[72] >> 28;
                if ((tag != 8 && tag != 10) || !isfinite(t) || !isfinite(u) || !isfinite(v) ||
                    u < 0 || u > 1 || v < 0 || v > 1) { result->status = 8; return; }
                if (t <= result->t) {
                    result->t = t;
                    result->u = u;
                    result->v = v;
                    result->instance_id = trace.instance_id;
                    result->primitive = state[72] & 0x0fffffff;
                    result->front_face = !(state[32] >> 31);
                }
            }
        }
        result->pending_count = top;
        if (control->stop_after_first) { result->status = 9; return; }
        if (!top) { result->status = 1; return; }
        PendingEntry next = pending[--top];
        float ray[8];
        for (int index = 0; index < 8; ++index) ray[index] = params->ray[index];
        ray[7] = result->t;
        prepare(state, ray, params->root_low, params->root_high, params->dummy_low, params->dummy_high);
        state[50] = next.low;
        state[51] = next.high;
        state[34] = next.entry_low;
        state[35] = next.entry_high;
        for (int index = 54; index < 62; ++index) state[index] = 0;
    }
    result->status = 2;
}
