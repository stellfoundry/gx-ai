#pragma once
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
//inline
//cudaError_t checkCuda(cudaError_t result)
//{
//  cudaDeviceSynchronize();
//  if (result != cudaSuccess) {
//    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
//    assert(result == cudaSuccess);
//  } 
//  return result;
//}

#define checkCuda(val)           __checkCudaErrors__ ( (val), #val, __FILE__, __LINE__ )
 
template <typename T>
inline T __checkCudaErrors__(T code, const char *func, const char *file, int line) 
{
  if (code) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, (unsigned int)code, func);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
  return code;
}

inline cudaError_t print_cudims(dim3 dimGrid, dim3 dimBlock) {
  printf("dimBlock: %d %d %d dimGrid: %d %d %d\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);
  return checkCuda(cudaGetLastError());
}


//int getError();
//int getError(char* message);
//int getError(char* message,int i);
//int getError(info_struct * info, char* message);
//int getError(info_struct * info, char* message, int i);
