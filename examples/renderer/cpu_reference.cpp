#include <cmath>
#include <cstdint>

struct Point { float x, y, z; };
struct Triangle { Point vertices[3]; };
struct Ray { float ox, oy, oz, tmin, dx, dy, dz, tmax; };
struct Hit { uint32_t primitive; float t, u, v; uint32_t front_face, instance_id; };
struct Vector { double x, y, z; };
static_assert(sizeof(Triangle) == 36 && sizeof(Ray) == 32 && sizeof(Hit) == 24, "CPU oracle ABI");

static Vector subtract(Point first, Point second) {
    return {double(first.x)-second.x, double(first.y)-second.y, double(first.z)-second.z};
}
static Vector cross(Vector first, Vector second) {
    return {first.y*second.z-first.z*second.y, first.z*second.x-first.x*second.z,
            first.x*second.y-first.y*second.x};
}
static double dot(Vector first, Vector second) {
    return first.x*second.x+first.y*second.y+first.z*second.z;
}

extern "C" void cpu_reference(const Triangle* triangles, uint32_t triangle_count,
                              const Ray* rays, uint32_t ray_count, Hit* hits) {
    for (uint32_t index = 0; index < ray_count; ++index) {
        const Ray ray = rays[index];
        const Point origin = {ray.ox, ray.oy, ray.oz};
        const Vector direction = {ray.dx, ray.dy, ray.dz};
        Hit hit = {0xffffffffu, ray.tmax, 0, 0, 0, 0xffffffffu};
        double best = ray.tmax;
        for (uint32_t primitive = 0; primitive < triangle_count; ++primitive) {
            const Triangle& triangle = triangles[primitive];
            Vector edge1 = subtract(triangle.vertices[1], triangle.vertices[0]);
            Vector edge2 = subtract(triangle.vertices[2], triangle.vertices[0]);
            Vector product = cross(direction, edge2);
            double determinant = dot(edge1, product);
            if (std::abs(determinant) < 1e-12) continue;
            Vector offset = subtract(origin, triangle.vertices[0]);
            double bary_u = dot(offset, product)/determinant;
            Vector second = cross(offset, edge1);
            double bary_v = dot(direction, second)/determinant;
            double distance = dot(edge2, second)/determinant;
            if (bary_u >= 0 && bary_v >= 0 && bary_u+bary_v <= 1 &&
                    distance >= ray.tmin && distance <= best) {
                best = distance;
                hit = {primitive, float(distance), float(bary_u), float(bary_v),
                       uint32_t(determinant > 0), 0xffffffffu};
            }
        }
        hits[index] = hit;
    }
}
