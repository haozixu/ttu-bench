"""CUDA Driver API bindings and ELF section parsing used by the TTU probe."""
import ctypes
import struct


cuda = ctypes.CDLL('libcuda.so.1')
for name, arguments in [
    ('cuInit', [ctypes.c_uint]),
    ('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
    ('cuCtxCreate_v2', [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]),
    ('cuCtxDestroy_v2', [ctypes.c_void_p]),
    ('cuModuleLoadData', [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]),
    ('cuModuleUnload', [ctypes.c_void_p]),
    ('cuModuleGetFunction', [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]),
    ('cuMemAlloc_v2', [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]),
    ('cuMemFree_v2', [ctypes.c_uint64]),
    ('cuMemcpyHtoD_v2', [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]),
    ('cuMemcpyDtoH_v2', [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]),
    ('cuLaunchKernel', [ctypes.c_void_p] + [ctypes.c_uint] * 7 +
     [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]),
    ('cuCtxSynchronize', []),
    ('cuGetErrorString', [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]),
]:
    function = getattr(cuda, name)
    function.argtypes = arguments
    function.restype = ctypes.c_int


def check(result, operation):
    if result:
        description = ctypes.c_char_p()
        cuda.cuGetErrorString(result, ctypes.byref(description))
        message = (description.value or b'unknown CUDA error').decode()
        raise RuntimeError(f'{operation}: CUDA error {result} ({message})')


def sections(image):
    section_table = struct.unpack_from('<Q', image, 40)[0]
    entry_size, count, names_index = struct.unpack_from('<HHH', image, 58)
    result = []
    for index in range(count):
        header = section_table + index * entry_size
        name, kind, flags, address, offset, size, link, info, alignment, stride = (
            struct.unpack_from('<IIQQQQIIQQ', image, header)
        )
        result.append(dict(idx=index, nameoff=name, typ=kind, off=offset,
                           size=size, shdr=header))
    names_offset = result[names_index]['off']
    for section in result:
        start = names_offset + section['nameoff']
        end = image.index(b'\0', start)
        section['name'] = image[start:end].decode()
    return result
