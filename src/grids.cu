#include "grids.h"
//#include "grids_kernel.cu"

Grids::Grids(Parameters* pars) :
  // copy from input parameters
  Nx(pars->nx_in),
  Ny(pars->ny_in),
  Nz(pars->nz_in),
  Nspecies(pars->nspec_in),
  Nhermite(pars->nhermite_in),
  Nlaguerre(pars->nlaguerre_in),
  // some additional derived grid sizes
  Nyc(Ny/2+1),
  Naky((Ny-1)/3+1),
  Nakx(Nx - (2*Nx/3+1 - ((Nx-1)/3+1))),
  NxNyc(Nx * Nyc),
  NxNy(Nx * Ny),
  NxNycNz(Nx * Nyc * Nz),
  NxNz(Nx * Nz),
  NycNz(Nyc * Nz),
  Nmoms(Nhermite*Nlaguerre)
{
  // copy some of these to device constant memory 
  // cudaMemcpyToSymbol(nx, &Nx, sizeof(int),0,cudaMemcpyHostToDevice);

  cudaMallocManaged((void**) &kx, sizeof(float)*Nx);
  cudaMallocManaged((void**) &ky, sizeof(float)*Nyc);
  cudaMallocManaged((void**) &kz, sizeof(float)*Nz);
}

Grids::~Grids() {
  cudaFree(kx);
  cudaFree(ky);
  cudaFree(kz);
}

void Grids::initialize() {
  //kInit<<<1,1>>>(kx, ky, kz);
}

