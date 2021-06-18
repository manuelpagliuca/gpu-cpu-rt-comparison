#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <ctime>
#include <iostream>
#include <vector>
#include <cfloat>

// STB_IMAGE
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Classe vec3
#include "vec3.h"

// Classe per il raggio
class ray
{
public:
    __device__ ray() {}
    __device__ ray(const vec3 &a, const vec3 &b)
    {
        A = a;
        B = b;
    }
    __device__ vec3 origin() const { return A; }
    __device__ vec3 direction() const { return B; }
    __device__ vec3 point_at_parameter(float t) const { return A + t * B; }

    vec3 A;
    vec3 B;
};

// Record per gli oggetti
struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
};

// Classe astratta di oggetti che possono essere colpiti
class hitable
{
public:
    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const = 0;
};

// SFERA (hitable)
class sphere : public hitable
{
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r) : center(cen), radius(r){};
    __device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
    vec3 center;
    float radius;
};

__device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center;

    float const a = dot(r.direction(), r.direction());
    float const b = dot(oc, r.direction());
    float const c = dot(oc, oc) - radius * radius;
    float const discriminant = b * b - a * c;

    if (discriminant > 0)
    {
        float temp = (-b - sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            return true;
        }

        temp = (-b + sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            return true;
        }
    }
    return false;
};

// Lista di oggetti colpibili da Ray
class hitable_list : public hitable
{
public:
    __device__ hitable_list() {}

    __device__ hitable_list(hitable **l, int n)
    {
        list = l;
        list_size = n;
    }
    __device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;

    hitable **list;
    int list_size;
};

// Si occupa di lanciare tutte le hit function presenti nella lista di hitable
__device__ bool hitable_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < list_size; i++)
    {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

class camera
{
public:
    __device__ camera()
    {
        lower_left_corner = vec3(-2.0, -1.0, -1.0);
        horizontal = vec3(4.0, 0.0, 0.0);
        vertical = vec3(0.0, 2.0, 0.0);
        origin = vec3(0.0, 0.0, 0.0);
    }

    __device__ ray get_ray(float u, float v)
    {
        // Costruzione del raggio (parte dal TLC e va fino al LRC)
        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
};

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

__device__ bool hit_sphere(const vec3 &center, float radius, const ray &r)
{
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    return (discriminant > 0.0f);
}

__device__ vec3 color(const ray &r, hitable **world)
{
    hit_record rec;

    if ((*world)->hit(r, 0.0f, FLT_MAX, rec))
    {
        // Faccio rientrare nel range dei color [0, 1] (normal map)
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    }
    else
    {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        // Interpolazione che da un effetto "cielo"
        return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    // Indici della griglia bidimensionale
    int const i = threadIdx.x + blockIdx.x * blockDim.x;
    int const j = threadIdx.y + blockIdx.y * blockDim.y;

    // Check bounds
    if ((i >= max_x) || (j >= max_y))
        return;

    // Indice del pixel su memoria contigua
    int const pixel_index = j * max_x + i;

    // Stesso seme per ogni thread, che genererà una differente sequenza numerica
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns,
                       camera **cam, hitable **world, curandState *rand_state)
{
    // Indici della griglia bidimensionale
    int const i = threadIdx.x + blockIdx.x * blockDim.x;
    int const j = threadIdx.y + blockIdx.y * blockDim.y;

    // Check bounds
    if ((i >= max_x) || (j >= max_y))
        return;

    // Indice del pixel su memoria contigua
    int const pixel_index = j * max_x + i;

    // Preleva un numero randomico per il thread apposito
    curandState local_rand_state = rand_state[pixel_index];

    vec3 col(0, 0, 0);

    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, world);
    }

    // Traccia il raggio
    fb[pixel_index] = col / float(ns);
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera)
{
    // Ci assicuriamo che il popolamento di entrambe le liste avvenga soltanto una volta
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *(d_list) = new sphere(vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *d_world = new hitable_list(d_list, 2);
        *d_camera = new camera();
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera)
{
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
    delete *d_camera;
}

int main(void)
{
    int constexpr width = 1200;
    int constexpr height = 600;
    int constexpr ns = 1000;
    int constexpr tx = 8;
    int constexpr ty = 8;

    std::vector<uint8_t> image;

    std::cerr << "Dimensione del framebuffer, " << width << "x" << height << "\n";
    std::cerr << "Dimensione della griglia (" << width / tx + 1 << ", " << height / ty + 1 << ")\n"
              << std::endl;
    std::cerr << "Dimensione dei blocchi (" << tx << ", " << ty << ")\n";

    int constexpr num_pixels = width * height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Allocazione del FrameBuffer che conterrà l'immagine (memoria unificata)
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Allocazione di un Random State
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    // Allocazione sulla GPU della lista di hitables
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(hitable *)));

    // Allocazione del mondo che conterrà gli oggetti
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));

    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    // Kernel che si occupa del popolamento del mondo
    create_world<<<1, 1>>>(d_list, d_world, d_camera);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render dell'immagine + timing
    dim3 blocks(width / tx + 1, height / ty + 1);
    dim3 threads(tx, ty);

    render_init<<<blocks, threads>>>(width, height, d_rand_state);

    clock_t start, stop;
    start = clock();

    render<<<blocks, threads>>>(fb, width, height, ns,
                                d_camera,
                                d_world, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = static_cast<double>((stop - start)) / static_cast<double>(CLOCKS_PER_SEC);
    std::cerr << "Il rendering ha impiegato " << timer_seconds << " secondi.\n";

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

    stbi_write_png("output.png", width, height, 3, image.data(), 0);

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
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

    return 0;
}
