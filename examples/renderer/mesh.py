"""Load triangular ASCII PLY meshes without changing face order or winding."""
import hashlib
import math
from pathlib import Path

from scene import quantize


def load_ply(path, yaw_degrees=25, normalization_bounds=None, deduplicate=False):
    path = Path(path)
    data = path.read_bytes()
    lines = iter(data.decode('ascii').splitlines())
    if next(lines, '') != 'ply' or next(lines, '') != 'format ascii 1.0':
        raise ValueError('mesh input requires ASCII PLY 1.0')
    elements, current = [], None
    for line in lines:
        fields = line.split()
        if not fields or fields[0] in ('comment', 'obj_info'):
            continue
        if fields[0] == 'end_header':
            break
        if fields[0] == 'element' and len(fields) == 3:
            current = dict(name=fields[1], count=int(fields[2]), properties=[])
            elements.append(current)
        elif fields[0] == 'property' and current is not None:
            current['properties'].append(fields[1:])
        else:
            raise ValueError(f'unsupported PLY header: {line}')
    else:
        raise ValueError('PLY has no end_header')
    if [element['name'] for element in elements] != ['vertex', 'face']:
        raise ValueError('PLY must contain vertex then face elements')
    vertex_info, face_info = elements
    if not 0 < vertex_info['count'] <= 2000000 or not 0 < face_info['count'] <= 4000000:
        raise ValueError('mesh vertex/face count outside demo bounds')
    properties = vertex_info['properties']
    if any(len(prop) != 2 or prop[0] == 'list' for prop in properties):
        raise ValueError('PLY vertex properties must be scalars')
    names = [prop[1] for prop in properties]
    if len(names) != len(set(names)) or not all(name in names for name in ('x','y','z')):
        raise ValueError('PLY must have unique x/y/z properties')
    coordinates = [names.index(name) for name in ('x','y','z')]
    vertices = []
    for index in range(vertex_info['count']):
        fields = next(lines, '').split()
        if len(fields) != len(properties):
            raise ValueError(f'bad PLY vertex {index}')
        vertex = tuple(float(fields[axis]) for axis in coordinates)
        if not all(math.isfinite(value) for value in vertex):
            raise ValueError('nonfinite PLY vertex')
        vertices.append(vertex)
    if face_info['properties'] not in ([['list','uchar','int','vertex_indices']],
                                       [['list','uchar','int','vertex_index']],
                                       [['list','uint8','int32','vertex_indices']]):
        raise ValueError('PLY face must have one vertex_indices list')
    faces = []
    for index in range(face_info['count']):
        fields = next(lines, '').split()
        if len(fields) != 4 or fields[0] != '3':
            raise ValueError(f'PLY face {index} must be triangular')
        face = tuple(int(value) for value in fields[1:])
        if len(set(face)) != 3 or not all(0 <= vertex < len(vertices) for vertex in face):
            raise ValueError(f'invalid indices in PLY face {index}')
        faces.append(face)
    if any(line.strip() for line in lines):
        raise ValueError('unexpected trailing PLY records')
    lower = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    upper = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    reference_lower, reference_upper = normalization_bounds or (lower,upper)
    extent = max(high-low for low,high in zip(reference_lower,reference_upper))
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError('invalid mesh extent')
    scale = 4/extent
    center = [(low+high)/2 for low,high in zip(reference_lower,reference_upper)]
    cosine, sine = math.cos(math.radians(yaw_degrees)), math.sin(math.radians(yaw_degrees))
    matrix = [[scale*cosine, 0, scale*sine, -scale*(cosine*center[0]+sine*center[2])],
              [0, scale, 0, -scale*reference_lower[1]-1.8],
              [-scale*sine, 0, scale*cosine, scale*(sine*center[0]-cosine*center[2])],
              [0,0,0,1]]
    transformed = [tuple(sum(row[axis]*vertex[axis] for axis in range(3))+row[3]
                         for row in matrix[:3]) for vertex in vertices]
    triangles = quantize([[transformed[index] for index in face] for face in faces])
    for index, triangle in enumerate(triangles):
        first = [triangle[1][axis]-triangle[0][axis] for axis in range(3)]
        second = [triangle[2][axis]-triangle[0][axis] for axis in range(3)]
        cross = [first[1]*second[2]-first[2]*second[1], first[2]*second[0]-first[0]*second[2],
                 first[0]*second[1]-first[1]*second[0]]
        if sum(value*value for value in cross) == 0:
            raise ValueError(f'mesh face {index} degenerates after float32 normalization')
    groups = {}
    for index, triangle in enumerate(triangles):
        key = tuple(sorted(tuple(vertex) for vertex in triangle))
        groups.setdefault(key,[]).append(index)
    duplicates = [group for group in groups.values() if len(group)>1]
    source_ids = list(range(len(triangles)))
    if deduplicate:
        source_ids = sorted(group[0] for group in groups.values())
        triangles = [triangles[index] for index in source_ids]
    model_count = len(triangles)
    triangles += quantize([[(-6,-1.95,-4),(6,-1.95,8),(6,-1.95,-4)],
                           [(-6,-1.95,-4),(-6,-1.95,8),(6,-1.95,8)]])
    return triangles, dict(kind='ply', path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
        vertices=len(vertices), source_triangles=len(faces), model_triangles=model_count, floor_triangles=2,
        deduplication=dict(enabled=deduplicate, duplicate_groups=len(duplicates),
            removed_faces=len(faces)-model_count, examples=duplicates[:16],
            policy='exact float32 triangle vertex sets, either winding; retain first source face'),
        face_source_ids=source_ids,
        primitive_order='retained source face order; mapping file; two ground triangles appended', original_bounds=[lower,upper],
        normalization=dict(matrix_row_major=matrix, uniform_scale=scale, yaw_degrees=yaw_degrees,
                           reference_bounds=[reference_lower,reference_upper],
                           reference_maximum_extent_before_yaw=4, reference_minimum_y=-1.8, floor_y=-1.95))
