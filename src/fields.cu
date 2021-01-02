#include "fields.h"
#include "get_error.h"

Fields::Fields(Parameters* pars, Grids* grids) : size_(sizeof(cuComplex)*grids->NxNycNz),
						 N(grids->NxNycNz), grids_(grids), pars_(pars)
{
  checkCuda(cudaMalloc((void**) &phi, size_));
  cudaMemset(phi, 0., size_);

  cudaMallocHost((void**) &phi_h, size_);

  if (pars_->beta > 0.) {
    checkCuda(cudaMalloc((void**) &apar, size_));
    cudaMemset(apar, 0., size_);

    cudaMallocHost((void**) &apar_h, size_);
  }
}

Fields::~Fields() {
  if (phi)     cudaFree(phi);
  if (phi_h)   cudaFreeHost(phi_h);
  if (apar)    cudaFree(apar);
  if (apar_h)  cudaFreeHost(apar_h);
}

void Fields::print_phi(void)
{
  CP_TO_CPU(phi_h, phi, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("phi(%d) = (%e, %e) \n",j, phi_h[j].x, phi_h[j].y);
  printf("\n");
}

void Fields::print_apar(void)
{
  CP_TO_CPU(apar_h, apar, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("apar(%d) = (%e, %e) \n",j, apar_h[j].x, apar_h[j].y);
  printf("\n");
}

