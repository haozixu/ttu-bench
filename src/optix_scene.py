"""OptiX GAS builder/oracle that keeps allocations in the caller's context."""
from project_paths import ROOT as HERE, KERNELS, BUILD
import ctypes
from pathlib import Path
import shutil
import subprocess

SDK = HERE.parent / 'OptiX-8.0.0'


class SceneInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64)
                for name in ('base', 'size', 'handle', 'gas_base', 'gas_size', 'gas_handle')]


class Ray(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in ('ox', 'oy', 'oz', 'tmin', 'dx', 'dy', 'dz', 'tmax')]


class Hit(ctypes.Structure):
    _fields_ = [('primitive', ctypes.c_uint32), ('t', ctypes.c_float),
                ('u', ctypes.c_float), ('v', ctypes.c_float), ('front_face', ctypes.c_uint32),
                ('instance_id', ctypes.c_uint32)]


def build_bridge():
    BUILD.mkdir(parents=True, exist_ok=True)
    toolkit = Path(shutil.which('nvcc')).resolve().parent.parent
    source = KERNELS / 'scene_bridge.cpp'
    library = BUILD / 'scene_bridge.so'
    oracle = KERNELS / 'scene_oracle.cu'
    ptx = BUILD / 'scene_oracle.ptx'
    if not ptx.exists() or ptx.stat().st_mtime < oracle.stat().st_mtime:
        subprocess.run(['nvcc', '-ptx', '-arch=sm_86', '-I', str(SDK / 'include'),
                        str(oracle), '-o', str(ptx)], check=True)
    if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-O2', str(source),
                        '-I', str(SDK / 'include'), '-I', str(toolkit / 'include'),
                        '-L', str(toolkit / 'lib64'), '-lcuda', '-ldl', '-o', str(library)], check=True)
    return library, ptx


class Scene:
    def __init__(self, triangles, *, geometry_flags=0, instances=()):
        library, ptx = build_bridge()
        self.library = ctypes.CDLL(str(library))
        self.library.scene_create.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint,
            ctypes.c_char_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint), ctypes.c_uint,
            ctypes.POINTER(SceneInfo), ctypes.POINTER(ctypes.c_void_p)]
        self.library.scene_create.restype = ctypes.c_int
        self.library.scene_trace.argtypes = [ctypes.c_void_p, ctypes.POINTER(Ray), ctypes.c_uint, ctypes.c_uint,
                                              ctypes.POINTER(Hit)]
        self.library.scene_trace.restype = ctypes.c_int
        self.library.scene_launch_device.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint]
        self.library.scene_launch_device.restype = ctypes.c_int
        self.library.scene_build_seconds.argtypes = [ctypes.c_void_p]
        self.library.scene_build_seconds.restype = ctypes.c_double
        self.library.scene_destroy.argtypes = [ctypes.c_void_p]
        self.library.scene_destroy.restype = None
        self.library.scene_error.argtypes = []
        self.library.scene_error.restype = ctypes.c_char_p
        self.info = SceneInfo()
        self.object = ctypes.c_void_p()
        coordinates = [coordinate for triangle in triangles for vertex in triangle for coordinate in vertex]
        vertices = (ctypes.c_float * len(coordinates))(*coordinates)
        transforms = (ctypes.c_float * (12 * len(instances)))(
            *(value for instance in instances for value in instance['transform']))
        instance_ids = (ctypes.c_uint * len(instances))(*(instance['id'] for instance in instances))
        self.check(self.library.scene_create(vertices, len(triangles), ptx.read_bytes(), geometry_flags,
                                              transforms, instance_ids, len(instances),
                                              ctypes.byref(self.info), ctypes.byref(self.object)))

    def check(self, result):
        if result:
            raise RuntimeError(self.library.scene_error().decode())

    def trace(self, rays, *, ray_flags=1):
        inputs = (Ray * len(rays))(*(Ray(*ray) for ray in rays))
        outputs = (Hit * len(rays))()
        self.check(self.library.scene_trace(self.object, inputs, len(rays), ray_flags, outputs))
        return [dict(primitive=None if hit.primitive == 0xffffffff else hit.primitive,
                     t=None if hit.primitive == 0xffffffff else hit.t,
                     front_face=None if hit.primitive == 0xffffffff else bool(hit.front_face),
                     instance_id=None if hit.instance_id == 0xffffffff else hit.instance_id,
                     barycentrics=None if hit.primitive == 0xffffffff else [hit.u, hit.v]) for hit in outputs]

    def close(self):
        if self.object:
            self.library.scene_destroy(self.object)
            self.object = ctypes.c_void_p()

    def launch_device(self, device_params, count):
        """Enqueue using persistent caller-owned Params/rays/hits on stream 0."""
        self.check(self.library.scene_launch_device(self.object, device_params, count))

    def build_seconds(self):
        """Host time for AS sizing/allocation/build/sync, excluding pipeline JIT."""
        return self.library.scene_build_seconds(self.object)
