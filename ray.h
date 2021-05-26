#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const Vec3 &a, const Vec3 &b)
    {
        A = a;
        B = b;
    }
    __device__ Vec3 origin() const { return A; }
    __device__ Vec3 direction() const { return B; }
    __device__ Vec3 point_at_parameter(float t) const { return A + t * B; }

    Vec3 A;
    Vec3 B;
};