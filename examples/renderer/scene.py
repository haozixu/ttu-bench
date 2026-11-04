"""Deterministic geometry and a median-split software BVH with original IDs."""
import math
import struct


def geometry(grid):
    triangles = []
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    for row in range(grid):
        for column in range(grid):
            center_x = (column-(grid-1)/2)*4.6/grid
            center_y = (row-(grid-1)/2)*3.5/grid
            center_z = 0.3+0.23*((row*7+column*3)%5)
            half = 1.55/grid
            vertices = [(center_x+half*(1 if corner & 1 else -1),
                         center_y+half*(1 if corner & 2 else -1),
                         center_z+half*(1 if corner & 4 else -1)) for corner in range(8)]
            for first, second, third, fourth in faces:
                triangles += [[vertices[first], vertices[second], vertices[third]],
                              [vertices[first], vertices[third], vertices[fourth]]]
    triangles += [[(-6, -2.2, -1), (6, -2.2, 8), (6, -2.2, -1)],
                  [(-6, -2.2, -1), (-6, -2.2, 8), (6, -2.2, 8)]]
    return quantize(triangles)


def flatten(triangles):
    return [value for triangle in triangles for vertex in triangle for value in vertex]


def pack_triangles(triangles):
    values = flatten(triangles)
    return struct.pack('<'+'f'*len(values), *values)


def quantize(triangles):
    values = struct.unpack('<'+'f'*len(flatten(triangles)), pack_triangles(triangles))
    return [[values[index+vertex:index+vertex+3] for vertex in (0, 3, 6)]
            for index in range(0, len(values), 9)]


def build_bvh(triangles, leaf_size=4):
    nodes, indices = [], []
    maximum_depth = 0
    def partition(primitives, depth):
        nonlocal maximum_depth
        maximum_depth = max(maximum_depth, depth)
        lower = [min(triangles[primitive][vertex][axis] for primitive in primitives for vertex in range(3))
                 for axis in range(3)]
        upper = [max(triangles[primitive][vertex][axis] for primitive in primitives for vertex in range(3))
                 for axis in range(3)]
        position = len(nodes)
        nodes.append(None)
        if len(primitives) <= leaf_size:
            nodes[position] = (*lower, *upper, 0, 0, len(indices), len(primitives))
            indices.extend(primitives)
        else:
            centroids = {primitive: [sum(vertex[axis] for vertex in triangles[primitive])/3
                                     for axis in range(3)] for primitive in primitives}
            axis = max(range(3), key=lambda component: max(value[component] for value in centroids.values())
                       - min(value[component] for value in centroids.values()))
            ordered = sorted(primitives, key=lambda primitive: (centroids[primitive][axis], primitive))
            middle = len(ordered)//2
            left = partition(ordered[:middle], depth+1)
            right = partition(ordered[middle:], depth+1)
            nodes[position] = (*lower, *upper, left, right, 0, 0)
        return position
    partition(list(range(len(triangles))), 0)
    if maximum_depth >= 64:
        raise ValueError('BVH exceeds the checked GPU stack capacity')
    return (b''.join(struct.pack('<6f4I', *node) for node in nodes),
            struct.pack('<'+'I'*len(indices), *indices),
            dict(builder='median centroid split, longest centroid axis, original primitive IDs',
                 nodes=len(nodes), triangles=len(triangles), leaf_size=leaf_size,
                 maximum_depth=maximum_depth, stack_capacity=64))


def validation_rays(triangles, count):
    rays = []
    for index in range(count):
        triangle = triangles[(index*17)%len(triangles)]
        point = [0.23*triangle[0][axis]+0.31*triangle[1][axis]+0.46*triangle[2][axis]
                 for axis in range(3)]
        edge1 = [triangle[1][axis]-triangle[0][axis] for axis in range(3)]
        edge2 = [triangle[2][axis]-triangle[0][axis] for axis in range(3)]
        normal = [edge1[1]*edge2[2]-edge1[2]*edge2[1],
                  edge1[2]*edge2[0]-edge1[0]*edge2[2], edge1[0]*edge2[1]-edge1[1]*edge2[0]]
        scale = math.sqrt(sum(value*value for value in normal))
        direction = [value/scale*(-1 if index%2 else 1) for value in normal]
        origin = [value-2*delta for value, delta in zip(point, direction)]
        if index%7 == 0:
            origin = [20, 21, 22]
        rays.append((*origin, 0.01 if index%5 else 0.2, *direction, 1 if index%11 == 0 else 100))
    return rays
