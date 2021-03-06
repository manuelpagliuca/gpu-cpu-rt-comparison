/*
    ///////////////////////////////////////////////////////////////////////////////////////
    ///  Project for the GPU COMPUTING Course @UNIMI, Manuel Pagliuca, A.Y. 2020/2021.  ///
    ///////////////////////////////////////////////////////////////////////////////////////

    This Raytracer is a makeover of the notorious Peter Shirley 'Raytracing in one weekend' book
    This is the link to the book : https://raytracing.github.io/books/RayTracingInOneWeekend.html

    I also used several chapters from a Roger Allen (principal architect from NVDIA) blog page, which
    implemented the Peter Shirley CPU-Only Raytracer for GPU execution with CUDA.
    This is the link to the blog page : https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

    The comparisons will be written in README.md and in a LaTex file.
*/

#pragma once

#include <curand_kernel.h>
#include "ray.h"

#define M_PI 3.14159265358979323846

__device__ vec3 random_in_unit_disk(curandState *local_rand_state)
{
    vec3 p;
    do
    {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0.0f) - vec3(1.0f, 1.0f, 0.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class camera
{
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, const float vfov, const float aspect, const float aperture, const float focus_dist)
    { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        const float theta = vfov * (static_cast<const float>(M_PI)) / 180.0f;
        const float half_height = tan(theta / 2.0f);
        const float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }

    __device__ ray get_ray(const float s, const float t, curandState *local_rand_state)
    {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};