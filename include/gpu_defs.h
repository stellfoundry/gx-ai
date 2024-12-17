#if defined(__HIPCC__)
#include "gpu_defs_hip.h"
//#pragma message("COMPILING USING HIP DEFS")
#elif defined(__CUDACC__)
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cufft.h>
#include <cufftXt.h>
#include <nccl.h>
#include <cutensor.h>
#include <cub/cub.cuh>
#define  GPU_SYMBOL(X) X
//#pragma message("COMPILING USING CUDA DEFS")
#else
#error message("UNKNOWN GPU ARCHITECTURE")
#endif
