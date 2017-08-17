#include "laguerre_transform.h"

Laguerre::Laguerre(Grids* grids) 
{
  int L = grids->Nlaguerre - 1;
  int J = (3*L-1)/2;
  float *toGrid_h, *toSpectral_h;
  cudaMallocHost((void**) &toGrid_h, sizeof(float)*(L+1)*(J+1));
  cudaMallocHost((void**) &toSpectral_h, sizeof(float)*(L+1)*(J+1));

  cudaMalloc((void**) &toGrid, sizeof(float)*(L+1)*(J+1));
  cudaMalloc((void**) &toSpectral, sizeof(float)*(L+1)*(J+1));

  initTransforms(toGrid_h, toSpectral_h);
}

Laguerre::~Laguerre()
{
  cudaFreeHost(toGrid_h);
  cudaFreeHost(toSpectral_h);
  cudaFree(toGrid);
  cudaFree(toSpectral);
}

int initTransforms(float* toGrid, float* toSpectral)
{
  return 0;
}

int transformToGrid(Moments* m)
{
  return 0;
}

int transformToSpectral(Moments* m)
{
  return 0;
}
