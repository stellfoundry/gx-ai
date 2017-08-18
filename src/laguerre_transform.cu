#include "laguerre_transform.h"

Laguerre::Laguerre(Grids* grids) :
  grids_(grids)
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
  // toGrid = toGrid[l + (L+1)*j] = Psi^l(x_j)
  // toSpectral = toSpectral[j + (J+1)*l] = w_j Psi_l(x_j)
  return 0;
}

int transformToGrid(RealMoments* m)
{
  return cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
     grids_->NxNyNz, J+1, L+1, 1.,
     m->ghl, grids_->NxNyNz, grids_->NxNyNz*(L+1),
     toGrid, J+1, 0,
     0., m->ghl, grids_->NxNyNz, grids_->NxNyNz*(L+1), 
     grids_->Nhermite);
}

int transformToSpectral(Moments* m)
{
  return 0;
}
