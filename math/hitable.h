#pragma once

#include "ray.h"

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hitable
{
public:
    __device__ virtual bool hit(const ray &r, const float t_min, const float t_max, hit_record &rec) const = 0;
};