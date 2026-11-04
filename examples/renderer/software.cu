#include "common.h"

__device__ bool intersects_box(const Node& node, const Ray& ray, float best, float& entry) {
    const float origin[3] = {ray.ox, ray.oy, ray.oz};
    const float direction[3] = {ray.dx, ray.dy, ray.dz};
    float near_value = ray.tmin, far_value = best;
    for (unsigned axis = 0; axis < 3; ++axis) {
        if (direction[axis] == 0) {
            if (origin[axis] < node.lower[axis] || origin[axis] > node.upper[axis]) return false;
        } else {
            float first = (node.lower[axis]-origin[axis])/direction[axis];
            float second = (node.upper[axis]-origin[axis])/direction[axis];
            near_value = fmaxf(near_value, fminf(first, second));
            far_value = fminf(far_value, fmaxf(first, second));
            if (near_value > far_value) return false;
        }
    }
    entry = near_value;
    return true;
}

__device__ void intersect_triangle(const Triangle& triangle, unsigned primitive, const Ray& ray, Hit& hit) {
    float3 direction = make_float3(ray.dx, ray.dy, ray.dz);
    float3 edge1 = subtract(triangle.vertices[1], triangle.vertices[0]);
    float3 edge2 = subtract(triangle.vertices[2], triangle.vertices[0]);
    float3 product = cross_product(direction, edge2);
    float determinant = dot_product(edge1, product);
    if (determinant == 0) return;
    float3 offset = subtract(make_float3(ray.ox, ray.oy, ray.oz), triangle.vertices[0]);
    float bary_u = dot_product(offset, product)/determinant;
    float3 second = cross_product(offset, edge1);
    float bary_v = dot_product(direction, second)/determinant;
    float distance = dot_product(edge2, second)/determinant;
    if (bary_u >= 0 && bary_v >= 0 && bary_u+bary_v <= 1 &&
        distance >= ray.tmin && distance <= hit.t) {
        hit = {primitive, distance, bary_u, bary_v, unsigned(determinant > 0), 0xffffffffu};
    }
}

extern "C" __global__ void software_trace(const RenderParams* params) {
    unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= params->count) return;
    Ray ray = params->rays[index];
    Hit hit = empty_hit(ray.tmax);
    unsigned stack[64], size = 0, current = 0;
    for (;;) {
        const Node node = params->nodes[current];
        float entry;
        if (intersects_box(node, ray, hit.t, entry)) {
            if (node.count) {
                for (unsigned offset = 0; offset < node.count; ++offset) {
                    unsigned primitive = params->indices[node.start+offset];
                    intersect_triangle(params->triangles[primitive], primitive, ray, hit);
                }
            } else {
                float first_entry, second_entry;
                bool first = intersects_box(params->nodes[node.left], ray, hit.t, first_entry);
                bool second = intersects_box(params->nodes[node.right], ray, hit.t, second_entry);
                if (first || second) {
                    unsigned near_child = first ? node.left : node.right;
                    if (first && second) {
                        bool left_first = first_entry <= second_entry;
                        near_child = left_first ? node.left : node.right;
                        if (size == 64) { hit.primitive = 0xfffffffeu; break; }
                        stack[size++] = left_first ? node.right : node.left;
                    }
                    current = near_child;
                    continue;
                }
            }
        }
        if (!size) break;
        current = stack[--size];
    }
    params->hits[index] = hit;
}

extern "C" __global__ void camera(Ray* rays, unsigned width, unsigned height) {
    unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= width*height) return;
    float screen_x = (2*(float(index%width)+0.37f)/width-1)*float(width)/height;
    float screen_y = 1-2*(float(index/width)+0.61f)/height;
    float3 direction = make_float3(screen_x*0.58f, screen_y*0.58f, 1);
    float inverse = rsqrtf(dot_product(direction, direction));
    rays[index] = {0, 0.7f, -7, 0.001f, direction.x*inverse,
                   direction.y*inverse, direction.z*inverse, 100};
}

extern "C" __global__ void shade(const Hit* hits, const Triangle* triangles, unsigned char* colors,
                                  unsigned count, unsigned mode) {
    unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= count) return;
    Hit hit = hits[index];
    float3 color = make_float3(0.025f, 0.04f, 0.07f);
    if (hit.primitive < 0xfffffffeu) {
        if (mode == 0) {
            float value = fminf(1, 3/fmaxf(hit.t, 0.001f));
            color = make_float3(value, value, value);
        } else if (mode == 1) {
            Triangle triangle = triangles[hit.primitive];
            float3 normal = cross_product(subtract(triangle.vertices[1], triangle.vertices[0]),
                                           subtract(triangle.vertices[2], triangle.vertices[0]));
            float inverse = rsqrtf(dot_product(normal, normal));
            color = make_float3(0.5f+0.5f*normal.x*inverse, 0.5f+0.5f*normal.y*inverse,
                                0.5f+0.5f*normal.z*inverse);
        } else {
            unsigned code = (hit.primitive+1)*2654435761u;
            color = make_float3(float(code&255)/255, float((code>>8)&255)/255, float((code>>16)&255)/255);
        }
    }
    colors[3*index] = (unsigned char)(255*fminf(1, fmaxf(0, color.x)));
    colors[3*index+1] = (unsigned char)(255*fminf(1, fmaxf(0, color.y)));
    colors[3*index+2] = (unsigned char)(255*fminf(1, fmaxf(0, color.z)));
}

extern "C" __global__ void reflect_floor(const Ray* primary_rays, const Hit* primary_hits,
                                         Ray* secondary_rays, unsigned count, unsigned floor_start, float floor_y) {
    unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= count) return;
    Ray ray = primary_rays[index];
    Hit hit = primary_hits[index];
    Ray reflected = {20,21,22,0.001f,0,0,1,100};
    if (hit.primitive >= floor_start && hit.primitive < floor_start+2) {
        float distance = (floor_y-ray.oy)/ray.dy;
        reflected = {ray.ox+distance*ray.dx, floor_y+0.001f,
                     ray.oz+distance*ray.dz, 0.001f, ray.dx, -ray.dy, ray.dz, ray.tmax};
    }
    secondary_rays[index] = reflected;
}

extern "C" __global__ void shade_mirror(const Hit* primary_hits, const Hit* secondary_hits,
                                        const Triangle* triangles, unsigned char* colors,
                                        unsigned count, unsigned floor_start) {
    unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= count) return;
    Hit primary = primary_hits[index];
    bool mirror = primary.primitive >= floor_start && primary.primitive < floor_start+2;
    Hit hit = mirror ? secondary_hits[index] : primary;
    float3 color = make_float3(0.025f,0.04f,0.07f);
    if (hit.primitive < 0xfffffffeu) {
        Triangle triangle = triangles[hit.primitive];
        float3 normal = cross_product(subtract(triangle.vertices[1],triangle.vertices[0]),
                                      subtract(triangle.vertices[2],triangle.vertices[0]));
        float inverse = rsqrtf(dot_product(normal,normal));
        color = make_float3(0.5f+0.5f*normal.x*inverse,0.5f+0.5f*normal.y*inverse,
                            0.5f+0.5f*normal.z*inverse);
    }
    if (mirror) color = make_float3(0.08f+0.85f*color.x,0.08f+0.85f*color.y,0.08f+0.85f*color.z);
    colors[3*index] = (unsigned char)(255*fminf(1,fmaxf(0,color.x)));
    colors[3*index+1] = (unsigned char)(255*fminf(1,fmaxf(0,color.y)));
    colors[3*index+2] = (unsigned char)(255*fminf(1,fmaxf(0,color.z)));
}
