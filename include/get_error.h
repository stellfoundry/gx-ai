#pragma once
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  } 
  return result;
}

//int getError();
//int getError(char* message);
//int getError(char* message,int i);
//int getError(info_struct * info, char* message);
//int getError(info_struct * info, char* message, int i);
