#!/usr/bin/env python3
"""Small single-GAS renderer: CUDA BVH, OptiX, experimental native TTU."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import subprocess
import tempfile
import time
import zlib

from cuda_driver import check, cuda, sections
from device_instance import issue_source, patch_compiled
from optix_scene import Ray, Scene
from project_paths import ROOT, BUILD, hash_sources
from run_scene import reference
from mesh import load_ply
from prepare_bunny import bunny_path
from scene import build_bvh, geometry, pack_triangles, validation_rays


HERE = Path(__file__).resolve().parent
DESTINATION = BUILD / 'renderer'
MISSING = 0xffffffff


class Params(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('rays', 'hits', 'triangles', 'nodes', 'indices')] + [
        (name, ctypes.c_uint) for name in ('count', 'root_low', 'root_high', 'dummy_low', 'dummy_high')]


class OptixParams(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('gas', 'rays', 'hits')] + [
        ('flags', ctypes.c_uint), ('instances', ctypes.c_uint)]


class Launch(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ('camera', 'trace', 'shade', 'optix', 'scene')] + [
        (name, ctypes.c_uint64) for name in ('params', 'rays', 'hits', 'triangles', 'colors')] + [
        (name, ctypes.c_uint) for name in ('count', 'width', 'height', 'mode', 'block')]


class Reflection(ctypes.Structure):
    _fields_ = [('primary',Launch),('secondary',Launch),('generate',ctypes.c_void_p),
                ('compose',ctypes.c_void_p),('floor_start',ctypes.c_uint),('floor_y',ctypes.c_float)]


def run_compile(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'Compilation failed: {command}\n{result.stdout}\n{result.stderr}')
    return result.stderr


def compile_sources(destination, native, native_build='monolithic', native_opt=None):
    if native_opt is None:
        native_opt = 0 if native_build == 'monolithic' else 3
    compiler = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True).stdout
    options = ['nvcc', '-arch=sm_86', '-cubin', '-O3', '--fmad=false', '-Xptxas=-v']
    software = destination / 'software.cubin'
    software_log = run_compile([*options, str(HERE/'software.cu'), '-o', str(software)])
    logs = dict(software=software_log, options=options, compiler=compiler, directory=str(destination))
    images = dict(software=software.read_bytes())
    if native:
        cubin = destination / 'native-original.cubin'
        logs['native_build'] = native_build
        if native_build == 'monolithic':
            generated = destination / 'native-generated.cu'
            generated.write_text('#include "common.h"\n' + issue_source() + '\n' + (HERE/'native.cu').read_text())
            native_options = ['nvcc', '-arch=sm_86', '-cubin', f'-O{native_opt}', '--fmad=false',
                              f'-Xptxas=-O{native_opt},-v']
            commands = [[*native_options, '-I', str(HERE), str(generated), '-o', str(cubin)]]
            symbol = 'native_trace'
        else:
            generated = destination / 'native-caller.cu'
            generated.write_text('#include "common.h"\nextern "C" __device__ void ttu_issue('
                                 'const volatile uint64_t*, volatile uint64_t*);\n' + (HERE/'native.cu').read_text())
            issue = destination / 'native-issue.cu'
            issue.write_text('#include <stdint.h>\n' + issue_source())
            commands = [
                ['nvcc', '-arch=sm_86', '-dc', f'-O{native_opt}', '--fmad=false', f'-Xptxas=-O{native_opt},-v',
                 '-I', str(HERE), str(generated), '-o', str(destination/'caller.o')],
                ['nvcc', '-arch=sm_86', '-dc', '-O0', '--fmad=false', '-Xptxas=-O0,-v',
                 str(issue), '-o', str(destination/'issue.o')],
                ['nvcc', '-arch=sm_86', '-dlink', '-cubin', str(destination/'caller.o'), str(destination/'issue.o'),
                 '-o', str(cubin)]]
            symbol = 'ttu_issue'
        logs['native_commands'] = commands
        logs['native'] = '\n'.join(run_compile(command) for command in commands)
        images['native'], logs['patch'] = patch_compiled(cubin.read_bytes(), symbol, separate_maps=native_build == 'linked')
        (destination/'native-patched.cubin').write_bytes(images['native'])
        logs['native_linked_resources'] = subprocess.run(['cuobjdump', '--dump-resource-usage', str(cubin)],
            capture_output=True, text=True, check=True).stdout
    toolkit = Path(subprocess.check_output(['which', 'nvcc'], text=True).strip()).resolve().parent.parent
    timing = destination / 'timing.so'
    run_compile(['g++', '-std=c++17', '-O3', '-ffp-contract=off', '-fPIC', '-shared', str(HERE/'timing.cpp'),
                 str(HERE/'cpu_reference.cpp'),
                 '-I', str(toolkit/'include'), '-L', str(toolkit/'lib64'), '-lcuda', '-o', str(timing)])
    logs['images'] = {name: hashlib.sha256(image).hexdigest() for name, image in images.items()}
    logs['timing_library_sha256'] = hashlib.sha256(timing.read_bytes()).hexdigest()
    return images, timing, logs


def inventory():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,clocks.sm,power.draw,temperature.gpu,utilization.gpu',
                                    '--format=csv'], text=True)


class Renderer:
    def __init__(self, triangles, capacity, backends, build_directory, block=128, native_build='monolithic', native_opt=None):
        if ctypes.sizeof(Params) != 64 or ctypes.sizeof(OptixParams) != 32:
            raise ValueError('host parameter ABI differs')
        self.backends, self.capacity, self.block = backends, capacity, block
        self.allocations, self.modules = [], []
        self.scene = None
        self.context = ctypes.c_void_p()
        self.closed = False
        self.build_seconds = {}
        started = time.perf_counter()
        images, timing, self.build = compile_sources(build_directory, 'native' in backends, native_build, native_opt)
        self.build_seconds['compile_patch_seconds_host'] = time.perf_counter()-started
        self.timer = ctypes.CDLL(str(timing))
        self.timer.renderer_launch.argtypes = [ctypes.POINTER(Launch), ctypes.c_uint]
        self.timer.renderer_time.argtypes = [ctypes.POINTER(Launch), ctypes.c_uint, ctypes.c_uint,
                                            ctypes.POINTER(ctypes.c_float)]
        self.timer.renderer_launch.restype = self.timer.renderer_time.restype = ctypes.c_int
        self.timer.renderer_reflect_launch.argtypes = [ctypes.POINTER(Reflection),ctypes.c_uint]
        self.timer.renderer_reflect_time.argtypes = [ctypes.POINTER(Reflection),ctypes.c_uint,ctypes.c_uint,
                                                   ctypes.POINTER(ctypes.c_float)]
        self.timer.renderer_reflect_launch.restype = self.timer.renderer_reflect_time.restype = ctypes.c_int
        self.timer.cpu_reference.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                                             ctypes.c_uint, ctypes.c_void_p]
        self.timer.cpu_reference.restype = None
        self.cpu_triangles = pack_triangles(triangles)
        self.triangle_count = len(triangles)
        check(cuda.cuInit(0), 'init')
        device = ctypes.c_int()
        check(cuda.cuDeviceGet(ctypes.byref(device), 0), 'device')
        check(cuda.cuCtxCreate_v2(ctypes.byref(self.context), 0, device), 'context')
        try:
            self.functions = {}
            for name, image in images.items():
                module = ctypes.c_void_p()
                check(cuda.cuModuleLoadData(ctypes.byref(module), image), 'module load')
                self.modules.append(module)
                for symbol in (('software_trace', 'camera', 'shade','reflect_floor','shade_mirror') if name == 'software' else ('native_trace',)):
                    function = ctypes.c_void_p()
                    check(cuda.cuModuleGetFunction(ctypes.byref(function), module, symbol.encode()), symbol)
                    self.functions[symbol] = function
            started = time.perf_counter()
            nodes, indices, self.bvh = build_bvh(triangles)
            self.build_seconds['software_bvh_seconds_host'] = time.perf_counter()-started
            self.triangles = self.allocate(pack_triangles(triangles))
            self.nodes, self.indices = self.allocate(nodes), self.allocate(indices)
            self.rays = self.allocate(capacity*32)
            self.colors = self.allocate(capacity*3)
            self.hits = {backend: self.allocate((capacity+1)*24) for backend in backends}
            self.params = {backend: self.allocate(64) for backend in backends}
            self.root, self.dummy = (0, 0), (0, 0)
            if 'native' in backends or 'optix' in backends:
                started = time.perf_counter()
                self.scene = Scene(triangles, geometry_flags=1)
                self.build_seconds['optix_scene_build_pipeline_seconds_host'] = time.perf_counter()-started
                self.build_seconds['optix_as_size_allocate_build_sync_seconds_host'] = self.scene.build_seconds()
                from sm86_fixture import FIXTURES
                image = (FIXTURES/'jit_16.elf').read_bytes()
                static = next(section for section in sections(image) if section['name'] == '.nv.global.init')
                dummy = self.allocate(4096)
                self.upload(dummy, image[static['off']:static['off']+128])
                as_data = self.download(self.scene.info.base, self.scene.info.size)
                offset = struct.unpack_from('<Q', as_data, 0x20)[0]
                if offset%128 or not 0 <= offset <= self.scene.info.size-128:
                    raise ValueError('unsupported GAS root offset')
                root = self.scene.info.base+offset
                self.root = ((root>>7)&MISSING, (root>>39)|0x02484000)
                self.dummy = ((dummy>>6)&MISSING, (dummy>>38)|0x02402800)
                self.build['gas'] = dict(bytes=self.scene.info.size, root_offset=offset,
                                         sha256=hashlib.sha256(as_data).hexdigest())
        except BaseException:
            self.close()
            raise

    def allocate(self, value):
        data = value if isinstance(value, bytes) else None
        size = len(data) if data is not None else value
        pointer = ctypes.c_uint64()
        check(cuda.cuMemAlloc_v2(ctypes.byref(pointer), size), 'allocate')
        self.allocations.append(pointer.value)
        if data is not None:
            self.upload(pointer.value, data)
        return pointer.value

    @staticmethod
    def upload(pointer, data):
        check(cuda.cuMemcpyHtoD_v2(pointer, data, len(data)), 'upload')

    @staticmethod
    def download(pointer, size):
        buffer = ctypes.create_string_buffer(size)
        check(cuda.cuMemcpyDtoH_v2(buffer, pointer, size), 'download')
        return buffer.raw

    def job(self, backend, count, width=0, height=0, mode=1):
        if not 0 < count <= self.capacity:
            raise ValueError('ray count outside allocation')
        params = (OptixParams(self.scene.info.handle, self.rays, self.hits[backend], 1, 0)
                  if backend == 'optix' else Params(self.rays, self.hits[backend], self.triangles,
                    self.nodes, self.indices, count, *self.root, *self.dummy))
        self.upload(self.params[backend], bytes(params))
        return Launch(self.functions['camera'], self.functions.get(backend+'_trace'), self.functions['shade'],
                      ctypes.cast(self.scene.library.scene_launch_device, ctypes.c_void_p) if backend == 'optix' else None,
                      self.scene.object if backend == 'optix' else None,
                      self.params[backend], self.rays, self.hits[backend], self.triangles, self.colors,
                      count, width, height, mode, self.block)

    def launch(self, job, stage):
        check(self.timer.renderer_launch(ctypes.byref(job), stage), 'renderer launch')

    def sync(self):
        check(cuda.cuCtxSynchronize(), 'renderer synchronize')

    def timed(self, job, stage, iterations):
        elapsed = ctypes.c_float()
        check(self.timer.renderer_time(ctypes.byref(job), stage, iterations, ctypes.byref(elapsed)), 'event timing')
        return elapsed.value/iterations

    def reference(self, rays):
        data = b''.join(bytes(Ray(*ray)) for ray in rays)
        hits = ctypes.create_string_buffer(len(rays)*24)
        self.timer.cpu_reference(self.cpu_triangles, self.triangle_count, data, len(rays), hits)
        return unpack_hits(hits.raw)

    def reflected_job(self, backend, count, width, height, floor_start, floor_y):
        primary = self.job(backend,count,width,height)
        secondary_rays = self.allocate((count+1)*32)
        secondary_hits = self.allocate((count+1)*24)
        secondary_params = self.allocate(64)
        params = (OptixParams(self.scene.info.handle,secondary_rays,secondary_hits,1,0) if backend == 'optix' else
            Params(secondary_rays,secondary_hits,self.triangles,self.nodes,self.indices,count,*self.root,*self.dummy))
        self.upload(secondary_params,bytes(params))
        secondary = Launch.from_buffer_copy(bytes(primary))
        secondary.rays, secondary.hits, secondary.params = secondary_rays,secondary_hits,secondary_params
        return Reflection(primary,secondary,self.functions['reflect_floor'],self.functions['shade_mirror'],floor_start,floor_y)

    def reflect_launch(self, job, stage):
        check(self.timer.renderer_reflect_launch(ctypes.byref(job),stage),'reflection launch')

    def reflect_timed(self, job, stage, iterations):
        elapsed = ctypes.c_float()
        check(self.timer.renderer_reflect_time(ctypes.byref(job),stage,iterations,ctypes.byref(elapsed)),
              'reflection timing')
        return elapsed.value/iterations

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.scene:
            self.scene.close()
        for pointer in reversed(self.allocations):
            cuda.cuMemFree_v2(pointer)
        for module in reversed(self.modules):
            cuda.cuModuleUnload(module)
        if self.context:
            cuda.cuCtxDestroy_v2(self.context)


def unpack_hits(data):
    hits = []
    for index, (primitive, distance, bary_u, bary_v, face, instance) in enumerate(struct.iter_unpack('<IfffII', data)):
        if primitive == 0xfffffffe:
            hits.append(dict(primitive=primitive, t=None, barycentrics=[], front_face=None,
                             raw_words=[f'{word:08x}' for word in struct.unpack_from('<6I',data,index*24)]))
        else:
            hits.append(dict(primitive=None if primitive == MISSING else primitive, t=distance,
                             barycentrics=[bary_u,bary_v], front_face=bool(face)))
    return hits


def geometric_agreement(first, second, triangle, ray):
    coordinates = [value for vertex in triangle for value in vertex]
    scale = max(1, *map(abs,coordinates), *map(abs,ray[:3]),
                *(abs(ray[4+axis]*hit['t']) for hit in (first,second) for axis in range(3)))
    budget = 16*2**-23*scale
    points, residuals = [], []
    for hit in (first,second):
        bary_u, bary_v = hit['barycentrics']
        if not (0 <= bary_u <= 1 and 0 <= bary_v <= 1 and bary_u+bary_v <= 1+2**-20):
            return False, dict(reason='barycentrics outside triangle')
        point = [triangle[0][axis]+bary_u*(triangle[1][axis]-triangle[0][axis])+
                 bary_v*(triangle[2][axis]-triangle[0][axis]) for axis in range(3)]
        residuals.append(max(abs(point[axis]-ray[axis]-hit['t']*ray[4+axis]) for axis in range(3)))
        points.append(point)
    difference = max(abs(first_value-second_value) for first_value,second_value in zip(*points))
    return max(residuals) <= budget and difference <= 2*budget, dict(
        world_difference=difference, ray_residual=max(residuals), budget=budget,
        bary_difference=max(abs(first_value-second_value) for first_value,second_value in
                            zip(first['barycentrics'],second['barycentrics'])))


def compare(actual, expected, *, geometry=None, diagnostics=None):
    failures = []
    for index, (first, second) in enumerate(zip(actual, expected)):
        okay = first['primitive'] == second['primitive'] and first['primitive'] != 0xfffffffe
        if okay and first['primitive'] is not None:
            finite = all(math.isfinite(value) for hit in (first,second)
                         for value in (hit['t'],*hit['barycentrics']))
            identity = first['front_face'] == second['front_face']
            okay = finite and identity and all(
                math.isfinite(value) and math.isclose(value, target, rel_tol=3e-5, abs_tol=2e-5)
                for value, target in zip([first['t'], *first['barycentrics']], [second['t'], *second['barycentrics']]))
            if (not okay and geometry is not None and finite and identity and
                    math.isclose(first['t'],second['t'],rel_tol=3e-5,abs_tol=2e-5)):
                triangles, rays = geometry
                okay, details = geometric_agreement(first,second,triangles[first['primitive']],rays[index])
                if okay and diagnostics is not None:
                    diagnostics.append(dict(ray=index,**details))
        if not okay:
            failures.append(dict(ray=index, actual=first, expected=second))
    if len(actual) != len(expected):
        failures.append(dict(error='hit buffer length differs'))
    return failures


def check_hits(engine, actual, expected, triangles, rays, policy, report, stage, backend):
    accepted = []
    geometry = (triangles,rays) if policy == 'geometric' else None
    failures = compare(actual,expected,geometry=geometry,diagnostics=accepted)
    if accepted:
        indices = [entry['ray'] for entry in accepted]
        selected_rays = [rays[index] for index in indices]
        cpu = engine.reference(selected_rays)
        for values in (actual,expected):
            failures += compare([values[index] for index in indices],cpu,
                                geometry=(triangles,selected_rays))
    report['validation'].append(dict(stage=stage,backend=backend,rays=len(rays),failures=len(failures),
        strict_barycentric_differences=len(accepted),cpu_adjudicated=len(accepted),examples=accepted[:16],
        maximum_bary_difference=max((entry['bary_difference'] for entry in accepted),default=0),
        maximum_world_difference=max((entry['world_difference'] for entry in accepted),default=0),
        maximum_ray_residual=max((entry['ray_residual'] for entry in accepted),default=0)))
    return failures


def png(path, width, height, rgb):
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data)&MISSING)
    scanlines = b''.join(b'\0'+rgb[row*width*3:(row+1)*width*3] for row in range(height))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>2I5B', width, height, 8, 2, 0, 0, 0))
                     +chunk(b'IDAT', zlib.compress(scanlines))+chunk(b'IEND', b''))


def stats(values):
    return dict(min_ms=min(values), median_ms=statistics.median(values), max_ms=max(values), samples_ms=values)


def reflection_demo(engine, triangles, rays, buffers, args, report):
    count = len(rays)
    floor_start = len(triangles)-2
    floor_y = triangles[-1][0][1]
    jobs = {backend:engine.reflected_job(backend,count,args.width,args.height,floor_start,floor_y) for backend in args.backends}
    hits, secondary = {}, {}
    for backend, job in jobs.items():
        primary = buffers[backend]
        for repeat in range(3):
            engine.upload(job.secondary.rays,b'\xa5'*((count+1)*32))
            engine.upload(job.secondary.hits,b'\xa5'*((count+1)*24))
            engine.reflect_launch(job,0)
            engine.reflect_launch(job,1)
            engine.sync()
            if (engine.download(job.secondary.rays+count*32,32) != b'\xa5'*32 or
                    engine.download(job.secondary.hits+count*24,24) != b'\xa5'*24):
                raise ValueError('reflection tail guard overwritten')
        ray_data = engine.download(job.secondary.rays,count*32)
        hit_data = engine.download(job.secondary.hits,count*24)
        if args.artifacts:
            (args.output/(backend+'-secondary-rays.bin')).write_bytes(ray_data)
            (args.output/(backend+'-secondary-hits.bin')).write_bytes(hit_data)
        secondary[backend] = list(struct.iter_unpack('<8f',ray_data))
        hits[backend] = unpack_hits(hit_data)
        indices = [index for index,hit in enumerate(primary) if hit['primitive'] is not None and
                   floor_start <= hit['primitive'] < floor_start+2]
        mirror_indices = set(indices)
        selected = indices[::max(1,len(indices)//128)][:128]
        for index,ray in enumerate(secondary[backend]):
            source = rays[index]
            hit = primary[index]
            if index in mirror_indices:
                distance = (floor_y-source[1])/source[5]
                expected = (source[0]+distance*source[4],floor_y+0.001,
                    source[2]+distance*source[6],0.001,source[4],-source[5],source[6],source[7])
            else:
                expected = (20,21,22,0.001,0,0,1,100)
            if not all(math.isclose(value,target,rel_tol=2e-6,abs_tol=2e-6) for value,target in zip(ray,expected)):
                raise ValueError('GPU reflected ray differs from host formula')
        expected = engine.reference([secondary[backend][index] for index in selected])
        failures = check_hits(engine,[hits[backend][index] for index in selected],expected,triangles,
            [secondary[backend][index] for index in selected],args.validation,report,'reflection-cpu',backend)
        if failures:
            report['failures'].append(dict(stage='reflection-cpu',backend=backend,examples=failures[:16]))
        engine.reflect_launch(job,2)
        engine.sync()
        if args.images or args.artifacts:
            png(args.output/(backend+'-mirror.png'),args.width,args.height,engine.download(engine.colors,count*3))
    reference_backend = 'optix' if 'optix' in hits else args.backends[0]
    for backend in args.backends:
        if secondary[backend] != secondary[reference_backend]:
            raise ValueError('reflection backends generated different rays')
        failures = check_hits(engine,hits[backend],hits[reference_backend],triangles,secondary[backend],
            args.validation,report,'reflection-cross',backend)
        if failures:
            report['failures'].append(dict(stage='reflection-cross',backend=backend,examples=failures[:16]))
    if report['failures']:
        raise ValueError('reflection validation failed')
    mirror_count = sum(hit['primitive'] is not None and floor_start <= hit['primitive'] < floor_start+2
                       for hit in buffers[reference_backend])
    report['reflection'] = dict(bounces=1,mirror='last two horizontal ground triangles',mirror_rays=mirror_count,
        dispatched_secondary_rays=count,dummy_miss_rays=count-mirror_count,origin_offset_y=0.001,tmin=0.001,
        hit_point='analytic horizontal plane intersection after closest-hit classification; identical secondary rays across backends',
        direction='(dx,-dy,dz), equivalent to reflection about floor normal (0,1,0)',
        secondary_hits={backend:sum(hit['primitive'] is not None for hit in values) for backend,values in hits.items()},
        tail_guards=True,host_ray_formula_checked=count*len(jobs),timings={})
    if not args.check_only:
        for job in jobs.values():
            started = time.perf_counter()
            while time.perf_counter()-started < 0.2:
                engine.reflect_timed(job,3,64)
        for stage,name in ((0,'generate'),(1,'secondary_trace'),(2,'compose'),(3,'camera_primary_generate_secondary_compose')):
            timings = {backend:[] for backend in args.backends}
            for sample in range(args.samples):
                ordered = args.backends[sample%len(args.backends):]+args.backends[:sample%len(args.backends)]
                for backend in ordered:
                    timings[backend].append(engine.reflect_timed(jobs[backend],stage,args.iterations))
            report['reflection']['timings'][name] = {backend:stats(values) for backend,values in timings.items()}
        for backend,job in jobs.items():
            actual = unpack_hits(engine.download(job.secondary.hits,count*24))
            failures = check_hits(engine,actual,hits[backend],triangles,secondary[backend],args.validation,
                                  report,'reflection-pipeline-tail',backend)
            if failures:
                report['failures'].append(dict(stage='reflection-pipeline-tail',backend=backend,examples=failures[:16]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--grid', type=int, default=4)
    model = parser.add_mutually_exclusive_group()
    model.add_argument('--mesh', type=Path, help='triangular ASCII PLY; adds two ground triangles')
    model.add_argument('--bunny', choices=('res4','res3','res2','full'), help='prepared Stanford Bunny level')
    parser.add_argument('--mesh-yaw', type=float, default=25)
    parser.add_argument('--mesh-deduplicate', action='store_true', help='explicitly remove exact coincident faces, keeping first winding')
    parser.add_argument('--validation', choices=('strict','geometric'), default='strict',
                        help='geometric keeps exact ID/face and t checks; bary differences need ray/triangle residual and CPU adjudication')
    parser.add_argument('--backends', nargs='+', choices=('software', 'optix', 'native'),
                        default=['software', 'optix', 'native'])
    parser.add_argument('--block', type=int, choices=(32, 64, 128, 256), default=128)
    parser.add_argument('--native-build', choices=('monolithic', 'linked'), default='monolithic')
    parser.add_argument('--native-opt', type=int, choices=(0, 1, 2, 3),
                        help='monolithic default O0; linked caller default O3, issue function fixed O0')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--output', type=Path, default=ROOT/'results/local/renderer')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--mirror-floor',action='store_true',help='one reflected ray per floor hit; dummy misses elsewhere')
    parser.add_argument('--images', action='store_true', help='save demo PNGs in addition to the report')
    parser.add_argument('--artifacts', action='store_true', help='opt in to raw buffers, PNGs and retained build files')
    args = parser.parse_args()
    if args.native_opt is None:
        args.native_opt = 0 if args.native_build == 'monolithic' else 3
    if min(args.width, args.height, args.grid, args.samples, args.iterations) < 1 or args.grid > 16:
        parser.error('positive dimensions/samples/iterations required; grid <= 16')
    if args.width*args.height > 4*1024*1024:
        parser.error('first demo limits images to four million rays')
    args.backends = list(dict.fromkeys(args.backends))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
                  contract='Single opaque triangle GAS; software BVH differs from OptiX AS. Native TTU is experimental compiler-marker patch, not a minimal timing kernel. Event times include kernel/interface work, exclude build/JIT/transfers; no pure RT speedup claim.',
                  inventory_before=inventory(), failures=[], validation=[], timings={})
    engine = None
    temporary_build = None
    try:
        DESTINATION.mkdir(parents=True, exist_ok=True)
        if args.artifacts:
            build_directory = Path(tempfile.mkdtemp(prefix='run-', dir=DESTINATION))
        else:
            temporary_build = tempfile.TemporaryDirectory(prefix='run-', dir=DESTINATION)
            build_directory = Path(temporary_build.name)
        report['build_directory'] = str(build_directory)
        report['build_retained'] = args.artifacts
        if not math.isfinite(args.mesh_yaw):
            raise ValueError('mesh yaw must be finite')
        provenance = None
        if args.bunny:
            args.mesh, provenance = bunny_path(args.bunny)
        triangles, scene_info = load_ply(args.mesh, args.mesh_yaw,
            provenance['normalization_bounds'] if provenance else None,args.mesh_deduplicate) if args.mesh else (
            geometry(args.grid), dict(kind='boxes', grid=args.grid))
        if provenance:
            scene_info['provenance'] = provenance
        if 'face_source_ids' in scene_info:
            source_ids = scene_info.pop('face_source_ids')+[MISSING,MISSING]
            if args.artifacts:
                (args.output/'mesh-face-ids.bin').write_bytes(struct.pack('<'+'I'*len(source_ids),*source_ids))
            scene_info['face_mapping'] = dict(file='mesh-face-ids.bin',format='little-endian uint32 original face index; ffffffff for floor')
        report['scene'] = scene_info
        report['validation_policy'] = dict(name=args.validation, t_rel=3e-5, t_abs=2e-5,
            strict_bary_rel=3e-5, strict_bary_abs=2e-5,
            geometric_budget='16 * float32 epsilon * max(1, abs(vertices), abs(origin), abs(t*direction))',
            contract='Exact primitive and face; geometric policy changes only barycentric comparison and independently checks disputed hits using double CPU brute force')
        report['native_decoder'] = dict(empirical_triangle_tags=[4,6,8,10],primitive_mask='0x0fffffff',
            contract='Opaque triangle GAS only; additional tags checked against CPU/OptiX, not a general event decoder')
        report['scene_sha256'] = hashlib.sha256(pack_triangles(triangles)).hexdigest()
        engine = Renderer(triangles, max(257, args.width*args.height), args.backends, build_directory, args.block,
                          args.native_build, args.native_opt)
        report.update(build=engine.build, build_seconds=engine.build_seconds, software_bvh=engine.bvh)
        controlled = list(struct.iter_unpack('<8f', b''.join(bytes(Ray(*ray)) for ray in validation_rays(triangles,257))))
        controlled_expected = engine.reference(controlled)
        python_indices = list(range(min(8,len(controlled))))
        python_expected = [reference(triangles,controlled[index],ray_flags=1) for index in python_indices]
        oracle_failures = compare([controlled_expected[index] for index in python_indices],python_expected)
        report['cpu_oracle'] = dict(algorithm='double-precision brute force, float32 hit ABI',
                                   python_cross_checks=len(python_indices), failures=oracle_failures)
        if oracle_failures:
            raise ValueError('compiled CPU oracle differs from Python reference')
        for count in (1, 31, 32, 33, 127, 129, 257):
            rays = controlled[:count]
            packed = b''.join(bytes(Ray(*ray)) for ray in rays)
            rays = list(struct.iter_unpack('<8f', packed))
            expected = controlled_expected[:count]
            engine.upload(engine.rays, packed)
            for backend in args.backends:
                job = engine.job(backend, count)
                for repeat in range(3):
                    engine.upload(engine.hits[backend], b'\xa5'*((count+1)*24))
                    engine.launch(job, 0)
                    engine.sync()
                    if engine.download(engine.hits[backend]+count*24, 24) != b'\xa5'*24:
                        raise ValueError(f'{backend}: tail guard overwritten at ray count {count}')
                    failures = check_hits(engine,unpack_hits(engine.download(engine.hits[backend], count*24)),
                        expected,triangles,rays,args.validation,report,f'controlled-{count}-{repeat}',backend)
                    if failures:
                        if args.artifacts:
                            failure_name = f'failed-{backend}-{count}-{repeat}'
                            (args.output/(failure_name+'-rays.bin')).write_bytes(packed)
                            (args.output/(failure_name+'-hits.bin')).write_bytes(
                                engine.download(engine.hits[backend],count*24))
                        report['failures'].append(dict(stage='validation', backend=backend, count=count,
                                                        repeat=repeat, examples=failures[:16]))
                        raise ValueError('functional validation failed')
        count = args.width*args.height
        camera_job = engine.job(args.backends[0], count, args.width, args.height)
        engine.launch(camera_job, 1)
        engine.sync()
        ray_data = engine.download(engine.rays, count*32)
        if args.artifacts:
            (args.output/'rays.bin').write_bytes(ray_data)
        rays = list(struct.iter_unpack('<8f', ray_data))
        cpu_indices = sorted({int(index*(count-1)/min(256, count-1)) for index in range(min(257,count))}) if count > 1 else [0]
        cpu = engine.reference([rays[index] for index in cpu_indices])
        buffers = {}
        for backend in args.backends:
            job = engine.job(backend, count, args.width, args.height)
            engine.launch(job, 0)
            engine.sync()
            data = engine.download(engine.hits[backend], count*24)
            if args.artifacts:
                (args.output/(backend+'-hits.bin')).write_bytes(data)
            buffers[backend] = unpack_hits(data)
            failures = check_hits(engine,[buffers[backend][index] for index in cpu_indices],cpu,
                triangles,[rays[index] for index in cpu_indices],args.validation,report,'image-cpu',backend)
            if failures:
                report['failures'].append(dict(stage='image-cpu', backend=backend, examples=failures[:16]))
        reference_backend = 'optix' if 'optix' in buffers else args.backends[0]
        for backend in args.backends:
            failures = check_hits(engine,buffers[backend],buffers[reference_backend],triangles,rays,
                args.validation,report,'image-cross',backend)
            if failures:
                report['failures'].append(dict(stage='image-cross', backend=backend, examples=failures[:16]))
            for mode, name in enumerate(('depth', 'normal', 'primitive')):
                job = engine.job(backend, count, args.width, args.height, mode)
                engine.launch(job, 2)
                engine.sync()
                if args.images or args.artifacts:
                    png(args.output/(backend+'-'+name+'.png'), args.width, args.height, engine.download(engine.colors, count*3))
        if report['failures']:
            raise ValueError('image validation failed')
        if not args.check_only:
            jobs = {backend: engine.job(backend, count, args.width, args.height) for backend in args.backends}
            for job in jobs.values():
                for warmup in range(3):
                    engine.launch(job, 3)
            engine.sync()
            warmup = {}
            for backend in args.backends:
                started = time.perf_counter()
                launches = 0
                while time.perf_counter()-started < 0.2:
                    engine.timed(jobs[backend],0,128)
                    launches += 128
                warmup[backend] = dict(launches=launches,host_seconds=time.perf_counter()-started)
            report['timing_protocol'] = dict(sustained_warmup=warmup,
                order='alternate coherent/shuffled first each sample; rotate backend order; upload outside events',
                clock_lock=False,shuffle_seed=20261104)
            indices = list(range(count))
            random.Random(20261104).shuffle(indices)
            ray_orders = dict(coherent=ray_data,shuffled=b''.join(ray_data[index*32:(index+1)*32] for index in indices))
            timings = {order:{backend:[] for backend in args.backends} for order in ray_orders}
            for sample in range(args.samples):
                orders = ('coherent','shuffled') if sample%2==0 else ('shuffled','coherent')
                for order in orders:
                    engine.upload(engine.rays,ray_orders[order])
                    ordered = args.backends[sample%len(args.backends):]+args.backends[:sample%len(args.backends)]
                    for backend in ordered:
                        timings[order][backend].append(engine.timed(jobs[backend], 0, args.iterations))
            for order, backend_times in timings.items():
                report['timings'][order+'_trace'] = {backend: stats(values)|dict(mrays_per_second=count/(statistics.median(values)*1000))
                                                     for backend,values in backend_times.items()}
            engine.upload(engine.rays,ray_orders['shuffled'])
            for backend in args.backends:
                engine.launch(jobs[backend],0)
            engine.sync()
            for backend in args.backends:
                actual = unpack_hits(engine.download(engine.hits[backend], count*24))
                failures = check_hits(engine,actual,[buffers[backend][index] for index in indices],
                    triangles,[rays[index] for index in indices],args.validation,report,'shuffled',backend)
                if failures:
                    report['failures'].append(dict(stage='shuffled', backend=backend, examples=failures[:16]))
                    raise ValueError('shuffled ray validation failed')
            engine.upload(engine.rays, ray_data)
            for job in jobs.values():
                engine.launch(job, 0)
            engine.sync()
            for stage, name in ((1, 'camera'), (2, 'shade'), (3, 'camera_trace_shade')):
                report['timings'][name] = {}
                for backend in args.backends:
                    values = [engine.timed(jobs[backend], stage, args.iterations) for sample in range(args.samples)]
                    report['timings'][name][backend] = stats(values)
            for backend in args.backends:
                failures = check_hits(engine,unpack_hits(engine.download(engine.hits[backend], count*24)),
                    buffers[backend],triangles,rays,args.validation,report,'pipeline-tail',backend)
                if failures:
                    report['failures'].append(dict(stage='pipeline-tail', backend=backend, examples=failures[:16]))
        if args.mirror_floor:
            reflection_demo(engine,triangles,rays,buffers,args,report)
        report['hit_counts'] = {backend: sum(hit['primitive'] is not None for hit in hits) for backend,hits in buffers.items()}
        report['validation_ray_counts'] = dict(tail_guard_checked=True,
            distinct_rays=len(set(validation_rays(triangles, 257))), repeats_per_case=3)
        report['passed'] = not report['failures']
    except Exception as error:
        report['fatal_error'] = str(error)
        report['passed'] = False
    finally:
        if engine:
            engine.close()
        if temporary_build:
            temporary_build.cleanup()
        report['inventory_after'] = inventory()
        sources = [str(path.relative_to(ROOT)) for path in HERE.iterdir() if path.suffix in ('.py','.cu','.cpp','.h')]
        sources += ['src/device_instance.py','src/optix_scene.py','kernels/scene_bridge.cpp',
                    'kernels/scene_oracle.cu','src/cuda_driver.py','src/sm86_fixture.py','src/ttu_isa.py',
                    'tools/run_scene.py']
        sources += ['examples/renderer/assets/stanford-bunny.json']
        report['source_sha256'] = hash_sources(sources)
        report['artifacts'] = {path.name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                               for path in args.output.iterdir() if path.suffix in ('.bin','.png')}
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'], fatal_error=report.get('fatal_error'),
                          failures=report['failures'], timings=report['timings'], output=str(args.output))))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
