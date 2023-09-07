# CUDA Ray Tracer: implementation, comparison, and profiling
This is a project for the **GPU COMPUTING** course at the University of Milan, M.Sc. in Computer Science, A.Y. 2020/2021.
The project I made consists of a comparison between a CPU implementation of the Ray Tracer and a GPU implementation
made with CUDA. To read the report just click this [link](https://github.com/manuelpagliuca/Ray-Tracer-CUDA/blob/main/GPU_COMPUTING___RAY_TRACER%20.pdf).
## Compiling
`nvcc main.cu -o rt`

## Profiling
`nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer,dram_read_throughput,dram_write_throughput,ipc,ipc_instance,sm_efficiency,warp_execution_efficiency ./rt`

For other information about the comparison with the CPU version of the same scene, check the PDF paper.
