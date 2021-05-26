#pragma once

#include "ray.h"

struct hit_record
{
    float t;
    Vec3 p;
    Vec3 normal;
};

class hitable
{
public:
    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const = 0;
};