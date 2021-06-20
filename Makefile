CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG      = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_60,code=sm_60

SRCS = main.cu
INCS = math/vec3.h math/ray.h math/hitable.h math/hitable_list.h math/camera.h world/sphere.h  world/material.h utils/utils.h

rt: rt.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o rt rt.o

rt.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o rt.o -c main.cu

# use nvprof --query-metrics
profile_metrics: rt
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer,dram_read_throughput,dram_write_throughput,ipc,ipc_instance,sm_efficiency,warp_execution_efficiency ./rt

clean:
	rm -f rt rt.o render.png