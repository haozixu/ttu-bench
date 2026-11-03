"""Raw 19-instruction TTU request/return harness for parent-state experiments.

No guessed continuation policy is implied by this transport. Host can submit
all observed state slots, separated from scene/oracle and resume policy.
"""
import ctypes
from pathlib import Path

from device_instance import DeviceProbe, Params

State = ctypes.c_uint32 * 76
RAW_SOURCE = '''
extern "C" __global__ void state_probe(const uint32_t* input, uint32_t* output) {
    if (threadIdx.x || blockIdx.x) return;
    uint32_t state[76];
    for (int i = 0; i < 76; ++i) state[i] = input[i];
    issue(state);
    for (int i = 0; i < 76; ++i) output[i] = state[i];
}
'''


class RawStateProbe(DeviceProbe):
    def __init__(self):
        super().__init__(extra_source=RAW_SOURCE, kernel_name='state_probe', params_type=State, result_type=State)

    def query(self, state):
        if len(state) < 76:
            raise ValueError('expected at least 76 state words')
        return tuple(self.launch(State(*state[:76])))


class Trace(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ('kind', 'instance_id', 'index_word', 'pending_after')] + [
        ('event', ctypes.c_uint64), ('child_root', ctypes.c_uint64),
        ('input', State), ('parent', State), ('child', State), ('local_ray', ctypes.c_float * 8)]


class ParentResult(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ('status', 'requests', 'steps', 'pending_count', 'maximum_pending',
                                                   'instance_id', 'primitive', 'front_face')] + [
        (name, ctypes.c_float) for name in ('t', 'u', 'v')] + [
        ('reserved', ctypes.c_uint32), ('trace', Trace * 8)]


class ParentProbe(DeviceProbe):
    def __init__(self):
        if ctypes.sizeof(Trace) != 976 or ctypes.sizeof(ParentResult) != 7856:
            raise ValueError('unexpected parent-result ABI')
        source = Path(__file__).with_name('parent_template.cu').read_text()
        super().__init__(extra_source=source, kernel_name='parent_probe', params_type=ParentParams, result_type=ParentResult)


class ParentParams(ctypes.Structure):
    _fields_ = [('base', Params), ('stop_after_first', ctypes.c_uint32), ('reserved', ctypes.c_uint32)]
