#include "laguerre_transform.h"

Laguerre::Laguerre(Grids* grids) 
{
  L = grids->Nlaguerre - 1;
  J = (3*L-1)/2;
  float *toGrid_h, *toSpectral_h;
  cudaMallocHost((void**) &toGrid_h, sizeof(float)*(L+1)*(J+1));
  cudaMallocHost((void**) &toSpectral_h, sizeof(float)*(L+1)*(J+1));

  cudaMalloc((void**) &toGrid, sizeof(float)*(L+1)*(J+1));
  cudaMalloc((void**) &toSpectral, sizeof(float)*(L+1)*(J+1));

  initTransforms(toGrid_h, toSpectral_h);
  cudaMemcpy(toGrid, toGrid_h, sizeof(float)*(L+1)*(J+1), cudaMemcpyHostToDevice);
  cudaMemcpy(toSpectral, toSpectral_h, sizeof(float)*(L+1)*(J+1), cudaMemcpyHostToDevice);

  cublasCreate(&handle);
  cudaFreeHost(toGrid_h);
  cudaFreeHost(toSpectral_h);
}

Laguerre::~Laguerre()
{
  cudaFree(toGrid);
  cudaFree(toSpectral);
}

int initTransforms(float* toGrid, float* toSpectral)
{
  return 0;
}

int transformToGrid(Moments* m)
{
  int stride, stride2 = 0;
  cublasCgemv(handle, CUBLAS_OP_N, J+1, L+1,
              &make_cuComplex(1.,0.), toGrid, J+1, 
              m->ghl, stride, &make_cuComplex(0.,0.),
              m->ghl, stride2);
  return 0;
}

int transformToSpectral(Moments* m)
{
  return 0;
}
