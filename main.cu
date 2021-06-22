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

#include <iostream>
#include <chrono>
#include <cfloat>
#include <vector>
#include <string>

#include <curand_kernel.h>

#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "math/vec3.h"
#include "math/ray.h"
#include "world/sphere.h"
#include "math/hitable_list.h"
#include "math/camera.h"
#include "world/material.h"
#include "utils/utils.h"

// This function in the sequential code version used heavilty recursion
// in this case since we are running this kernel on a thread we don't want
// to used it too much, and we need to impose a static limit (of 50).
// In the CPU-only raytracer the function is called 'pixelColorFunction'
__device__ vec3 color(const ray &r, hitable **world, curandState *local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0.0f, 0.0f, 0.0f);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            const float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
    }

    // Case of exceeded recursion
    return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void rand_init(curandState *rand_state)
{
    // first thread initialize the seed with no offsets
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(2048, 0, 0, rand_state);
    }
}

__global__ void render_init(const int max_x, const int max_y, curandState *rand_state)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    const int pixel_index = j * max_x + i;

    // Each thread gets a different seed and got a unique initialized curand state
    curand_init(2048 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, const int max_x, const int max_y, const int ns, camera **cam, hitable **world, curandState *rand_state)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    const int pixel_index = j * max_x + i;

    curandState local_rand_state = rand_state[pixel_index];

    vec3 col(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < ns; s++)
    {
        const float u = static_cast<const float>(i + curand_uniform(&local_rand_state)) / static_cast<const float>(max_x);
        const float v = static_cast<const float>(j + curand_uniform(&local_rand_state)) / static_cast<const float>(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;

    col /= static_cast<const float>(ns);
    col[0] = sqrtf(col[0]);
    col[1] = sqrtf(col[1]);
    col[2] = sqrtf(col[2]);

    fb[pixel_index] = col;
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, const int nx, const int ny)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, new dielectric(1.5f));
        d_list[1] = new sphere(vec3(1.5f, 0.0f, -2.0f), 0.5f, new metal(vec3(0.8f, 0.3f, 0.2f), 0.0f));
        d_list[2] = new sphere(vec3(-1.5f, 0.0f, -2.0f), 0.5f, new metal(vec3(0.2f, 0.5f, 0.9f), 0.0f));
        d_list[3] = new sphere(vec3(0.0f, -50.5f, -1.0f), 50.0f, new lambertian(vec3(0.1f, 0.4f, 0.1f)));
        d_list[4] = new sphere(vec3(-2.0f, 0.8f, 0.0f), 1.4f, new lambertian(vec3(0.3f, 0.3f, 0.8f)));
        d_list[5] = new sphere(vec3(-0.8f, 1.8f, 4.0f), 2.6f, new dielectric(1.5f)); /// 2.417f: diamond
        d_list[6] = new sphere(vec3(3.0f, 0.9f, -3.0f), 1.5f, new metal(vec3(0.4f, 0.7f, 0.1f), 0.0f));
        d_list[7] = new sphere(vec3(-3.0f, 2.5f, -6.0f), 3.6f, new metal(vec3(0.1f, 0.1f, 0.1f), 0.0f));

        *d_world = new hitable_list(d_list, 8);

        vec3 lookfrom(15.0f, 3.0f, 15.0f);
        vec3 lookat(0.0f, 2.0f, 0.0f);
        const float dist_to_focus = 20.0f;
        const float aperture = 0.1f;
        *d_camera = new camera(lookfrom,
                               lookat,
                               vec3(0.0f, 1.0f, 0.0f),
                               30.0f,
                               static_cast<float>(nx) / static_cast<float>(ny),
                               aperture,
                               dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera)
{
    for (int i = 0; i < 8; i++)
    {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main()
{
    int nx = 800;
    int ny = 600;
    int ns = 10;
    constexpr int tx = 8;
    constexpr int ty = 8;

    std::vector<uint8_t> image;

    std::cout << "Insert resolutions : "
                 "Respect the aspect ratio of 16:9\n"
                 "Width : ";
    std::cin >> nx;
    std::cout << "Height : ";
    std::cin >> ny;
    std::cout << "Number of samples : ";
    std::cin >> ns;
    std::cout << std::endl;

    std::cerr << "Framebuffer : " << nx << "x" << ny << "\n";
    std::cerr << "Grid dim(" << nx / tx + 1 << ", " << ny / ty + 1 << ")\n";
    std::cerr << "Thread-block dim(" << tx << ", " << ty << ")" << std::endl;

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Allocazione del framebuffer
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Allocazione dei RandomState per ogni pixel (MSAA)
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    // Allocazione della hitables_list, world e della camera
    hitable **d_list;
    constexpr int num_hitables = 8;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Initialisation
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render
    auto start = std::chrono::system_clock::now();

    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "It took " << elapsed_seconds.count() << " seconds \n\n\n";

    // Salvataggio dell'immagine su disco
    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            const size_t pixel_index = j * nx + i;
            const int ir = static_cast<const int>(255.99f * fb[pixel_index].r());
            const int ig = static_cast<const int>(255.99f * fb[pixel_index].g());
            const int ib = static_cast<const int>(255.99f * fb[pixel_index].b());

            image.push_back(ir);
            image.push_back(ig);
            image.push_back(ib);
        }
    }

    std::string filename = std::to_string(nx) + "x" + std::to_string(ny) + "_" + std::to_string(ns) + "samples.png";
    stbi_write_png(filename.c_str(), nx, ny, 3, image.data(), 0);

    // Clean up
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    d_list = NULL;
    d_camera = NULL;
    d_world = NULL;
    fb = NULL;
    d_rand_state = NULL;

    cudaDeviceReset();
}
