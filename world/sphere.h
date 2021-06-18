#pragma once

#include "../math/hitable.h"

class sphere : public hitable
{
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m){};
    __device__ virtual bool hit(const ray &r, const float tmin, const float tmax, hit_record &rec) const;
    vec3 center;
    float radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray &r, const float t_min, const float t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center;
    const float a = dot(r.direction(), r.direction());
    const float b = dot(oc, r.direction());
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = b * b - a * c;

    if (discriminant > 0)
    {
        float temp = (-b - sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }

        temp = (-b + sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}