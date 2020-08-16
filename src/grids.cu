#include "grids.h"
#include "cuda_constants.h"
#include "get_error.h"
#include "device_funcs.h"

Grids::Grids(Parameters* pars) :
  // copy from input parameters
  Nx(pars->nx_in),
  Ny(pars->ny_in),
  Nz(pars->nz_in),
  Nspecies(pars->nspec_in),
  Nm(pars->nm_in),
  Nl(pars->nl_in),
  Nj(3*pars->nl_in/2-1),
  // some additional derived grid sizes
  Nyc(Ny/2+1),
  Naky((Ny-1)/3+1),
  Nakx(1 + 2*((Nx-1)/3)),
  NxNyc(Nx * Nyc),
  NxNy(Nx * Ny),
  NxNycNz(Nx * Nyc * Nz),
  NxNyNz(Nx * Ny * Nz),
  NxNz(Nx * Nz),
  NycNz(Nyc * Nz),
  Nmoms(Nm*Nl),
  pars_(pars)
{
  cudaDeviceSynchronize();
  cudaMallocHost((void**) &theta0_h, sizeof(float)*Nakx);

  cudaMallocHost((void**) &kx_outh, sizeof(float)*Nx); 
  cudaMallocHost((void**) &kx_h, sizeof(float)*Nakx); 
  cudaMallocHost((void**) &ky_h, sizeof(float)*Nyc);
  cudaMallocHost((void**) &kz_h, sizeof(float)*Nz);

  cudaMalloc((void**) &kx, sizeof(float)*Nx);
  cudaMalloc((void**) &ky, sizeof(float)*Nyc);
  cudaMalloc((void**) &kz, sizeof(float)*Nz);

  //  printf("In grids constructor. Nyc = %i \n",Nyc);
  
  // copy some parameters to device constant memory 
  cudaMemcpyToSymbol(nx,  &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ny,  &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nyc, &Nyc, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nz,  &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nspecies, &Nspecies, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nm,  &Nm, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nl,  &Nl, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nj,  &Nj, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(zp,  &pars_->Zp, sizeof(float),0,cudaMemcpyHostToDevice);  
  cudaMemcpyToSymbol(ikx_fixed, &pars_->ikx_fixed, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(iky_fixed, &pars_->iky_fixed, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // initialize k arrays
  int Nmax = max(max(Nx, Nyc),Nz);
  kInit<<<1, Nmax>>>(kx, ky, kz, pars_->x0, pars_->y0, pars_->Zp);

  CP_TO_CPU(kx_outh, kx, sizeof(float)*Nx);
  CP_TO_CPU(ky_h, ky, sizeof(float)*Nyc);
  CP_TO_CPU(kz_h, kz, sizeof(float)*Nz);

  for (int i=0; i<((Nx-1)/3+1); i++) kx_h[i]=kx_outh[i];
  for (int i=2*Nx/3+1; i<Nx; i++) kx_h[i-2*Nx/3+(Nx-1)/3] = kx_outh[i];
}

Grids::~Grids() {
  cudaFree(kx);
  cudaFree(ky);
  cudaFree(kz);
  
  cudaFreeHost(kx_h);
  cudaFreeHost(ky_h);
}


