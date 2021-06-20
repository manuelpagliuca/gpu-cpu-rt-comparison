# Ray Tracing in One Weekend with CUDA

This is a project for the GPU COMPUTING course during the A.Y. 2020/2021 @UNIMI, Milan.

Compiling : `nvcc main.cu -o rt`

Profiling : `nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer,dram_read_throughput,dram_write_throughput,ipc,ipc_instance,sm_efficiency,warp_execution_efficiency ./rt`

For other informations about the comparison with the CPU versione of the same scene, check the PDF paper.
