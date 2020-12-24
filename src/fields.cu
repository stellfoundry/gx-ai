#include "fields.h"
#include "get_error.h"

Fields::Fields(Grids* grids) : size_(sizeof(cuComplex)*grids->NxNycNz),
			       N(grids->NxNycNz), grids_(grids)
{
  checkCuda(cudaMalloc((void**) &phi, size_));
  cudaMemset(phi, 0., size_);

  cudaMallocHost((void**) &phih, size_);
}

Fields::~Fields() {
  cudaFree(phi);
  cudaFreeHost(phih);
}

void Fields::print_phi(void)
{
  CP_TO_CPU(phih, phi, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("phi(%d) = (%e, %e) \n",j, phih[j].x, phih[j].y);
  printf("\n");
}

