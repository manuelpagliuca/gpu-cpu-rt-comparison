#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include "vec3.h"

// STB_IMAGE
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void render(Vec3* fb, int max_x, int max_y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	fb[pixel_index] = Vec3(float(i) / max_x, float(j) / max_y, 0.2f);
}

int main(void) {
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
	Vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	clock_t start, stop;
	start = clock();

	// Render our buffer
	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);
	render << <blocks, threads >> > (fb, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	// Output FB as Image
	std::cout << "P3\n" << width << " " << height << "\n255\n";
	for (int j = height - 1; j >= 0; j--) {
		for (int i = 0; i < width; i++) {
			size_t pixel_index = j * width + i;
			int const ir = int(255.99 * fb[pixel_index].r());
			int const ig = int(255.99 * fb[pixel_index].g());
			int const ib = int(255.99 * fb[pixel_index].b());

			image.push_back(ir);
			image.push_back(ig);
			image.push_back(ib);
		}
	}
	stbi_write_jpg("output.jpg", width, height, 3, image.data(), 100);

	checkCudaErrors(cudaFree(fb));

	return 0;
}