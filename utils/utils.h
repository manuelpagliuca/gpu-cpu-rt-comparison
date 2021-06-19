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

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, const int line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Prima di uscire chiama la cudaDeviceReset()
        cudaDeviceReset();
        exit(99);
    }
}