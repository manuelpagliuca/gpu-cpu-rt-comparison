#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <iostream>
#include <vector>
#include "ray.h"
#include "vec3.h"

// STB_IMAGE
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool hit_sphere(const Vec3 &center, float radius, const ray &r)
{
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    return (discriminant > 0.0f);
}

__device__ Vec3 color(const ray &r)
{
    if (hit_sphere(Vec3(0, 0, -1), 0.5, r))
        return Vec3(1, 0, 0);
    Vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
}

__global__ void render(Vec3 *fb, int max_x, int max_y,
                       Vec3 lower_left_corner, Vec3 horizontal,
                       Vec3 vertical, Vec3 origin)
{
    int const i = threadIdx.x + blockIdx.x * blockDim.x;
    int const j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    int const pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    fb[pixel_index] = color(r);
}

int main(void)
{
    int constexpr width = 1200;
    int constexpr height = 600;
    int constexpr tx = 8;
    int constexpr ty = 8;

    std::vector<uint8_t> image;

    std::cerr << "Rendering a " << width << "x" << height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = width * height;
    size_t fb_size = num_pixels * sizeof(Vec3);

    // allocate FB
    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Render dell'immagine + timing
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    //   clock_t start, stop;
    // start = clock();

    render<<<blocks, threads>>>(fb, width, height,
                                Vec3(-2.0, -1.0, -1.0),
                                Vec3(4.0, 0.0, 0.0),
                                Vec3(0.0, 2.0, 0.0),
                                Vec3(0.0, 0.0, 0.0));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    //std::cerr << "took " << timer_seconds << " seconds.\n";

    // Salvo l'immagine (host code)
    std::cout << "P3\n"
              << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--)
    {
        for (int i = 0; i < width; i++)
        {
            size_t pixel_index = j * width + i;
            int const ir = int(255.99f * fb[pixel_index].r());
            int const ig = int(255.99f * fb[pixel_index].g());
            int const ib = int(255.99f * fb[pixel_index].b());

            image.push_back(ir);
            image.push_back(ig);
            image.push_back(ib);
        }
    }
    stbi_write_jpg("output.jpg", width, height, 3, image.data(), 100);

    checkCudaErrors(cudaFree(fb));

    return 0;
}