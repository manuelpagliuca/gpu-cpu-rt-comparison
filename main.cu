#include <iostream>
#include <ctime>
#include <cfloat>
#include <vector>

#include <curand_kernel.h>

#include "math/vec3.h"
#include "math/ray.h"
#include "world/sphere.h"
#include "math/hitable_list.h"
#include "math/camera.h"
#include "world/material.h"

#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Prima di uscire chiama la cudaDeviceReset()
        cudaDeviceReset();
        exit(99);
    }
}

// This function in the sequential code version used heavilty recursion
// in this case since we are running this kernel on a thread we don't want
// to used it too much, and we need to impose a static limit (of 50).
// In my hostcode-only raytracer the function is called 'pixelColorFunction'

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
            float t = 0.5f * (unit_direction.y() + 1.0f);
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
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y))
        return;

    const int pixel_index = j * max_x + i;

    // Each thread gets a different seed and got a unique initialized curand state
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
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
        const float u = static_cast<float>(i + curand_uniform(&local_rand_state)) / static_cast<float>(max_x);
        const float v = static_cast<float>(j + curand_uniform(&local_rand_state)) / static_cast<float>(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;

    col /= static_cast<float>(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);

    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;

        d_list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, new dielectric(1.5f));
        d_list[1] = new sphere(vec3(1.5f, 0.0f, -2.0f), 0.5f, new metal(vec3(0.8f, 0.3f, 0.2f), 0.0f));
        d_list[2] = new sphere(vec3(-1.5f, 0.0f, -2.0f), 0.5f, new metal(vec3(0.2f, 0.5f, 0.9f), 0.0f));
        d_list[3] = new sphere(vec3(0.0f, -50.5f, -1.0f), 50.0f, new lambertian(vec3(0.1f, 0.4f, 0.1f)));
        d_list[4] = new sphere(vec3(-2.0f, 0.8f, 0.0f), 1.4f, new lambertian(vec3(0.3f, 0.3f, 0.8f)));
        d_list[5] = new sphere(vec3(-0.8f, 1.8f, 4.0f), 2.6f, new dielectric(1.5f)); /// 2.417f: diamond
        d_list[6] = new sphere(vec3(3.0f, 0.9f, -3.0f), 1.5f, new metal(vec3(0.4f, 0.7f, 0.1f), 0.0f));
        d_list[7] = new sphere(vec3(-3.0f, 2.5f, -6.0f), 3.6f, new metal(vec3(0.1f, 0.1f, 0.1f), 0.0f));

        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 8);

        vec3 lookfrom(15.0f, 3.0f, 15.0f);
        vec3 lookat(0.0f, 2.0f, 0.0f);
        float dist_to_focus = 20.0f;
        (lookfrom - lookat).length();
        float aperture = 0.1f;
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
    const int tx = 8;
    const int ty = 8;

    std::cout << "Inserire la risoluzione, si consiglia caldamente di rispettare l'aspect ratio di 16:9\n";
    std::cout << "Inserire la larghezza : ";
    std::cin >> nx;
    std::cout << "Inserire l'altezza : ";
    std::cin >> ny;
    std::cout << "Inserire il numero di samples da utilizzare : ";
    std::cin >> ns;
    std::cout << std::endl;

    std::vector<uint8_t> image;

    std::cerr << "Dimensione del framebuffer, " << nx << "x" << ny << "\n";
    std::cerr << "Dimensione della griglia (" << nx / tx + 1 << ", " << ny / ty + 1 << ")\n"
              << std::endl;
    std::cerr << "Dimensione dei blocchi (" << tx << ", " << ty << ")\n";
    std::cout << std::endl;

    const int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Allocazione del framebuffer
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Allocazione dei CUDA RandomState
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    // Inizializzazione del secondo random state
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocazione della hitables_list, world e della camera
    hitable **d_list;
    const int num_hitables = 8;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Rendering
    clock_t start, stop;
    start = clock();
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    const double timer_seconds = static_cast<const double>(stop - start) / static_cast<const double>(CLOCKS_PER_SEC);

    std::cerr << "Il rendering ha impiegato " << timer_seconds << " secondi.\n";

    // Salvataggio dell'immagine su disco
    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            size_t pixel_index = j * nx + i;
            const int ir = static_cast<const int>(255.99f * fb[pixel_index].r());
            const int ig = static_cast<const int>(255.99f * fb[pixel_index].g());
            const int ib = static_cast<const int>(255.99f * fb[pixel_index].b());

            image.push_back(ir);
            image.push_back(ig);
            image.push_back(ib);
        }
    }

    stbi_write_png("render.png", nx, ny, 3, image.data(), 0);

    // Clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    d_list = NULL;
    d_camera = NULL;
    d_world = NULL;
    fb = NULL;
    d_rand_state = NULL;
    d_rand_state2 = NULL;

    cudaDeviceReset();
}
