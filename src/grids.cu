#include "grids.h"
#include "cuda_constants.h"


__global__ void kInit(float* kx, float* ky, float* kz, 
                      float X0, float Y0, int Zp, 
                      float qsf, float shat, bool no_zderiv) 
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if(id<nyc) {
    ky[id] = (float) id/Y0;
  }
  if(id<nx/2+1) {
    kx[id] = (float) id/X0;
  } else if (id<nx) {
    kx[id] = (float) (id - nx)/X0;
  }
  if(id<nz) {
    if(id<(nz/2+1)) {
      kz[id] = (float) id/Zp;
    }
    else if(id<nz) {
      kz[id] = (float) (id - nz)/Zp;
    }
    //if(qsf<0.) kz[id] = shat; // local limit
    //if(no_zderiv) kz[id] = 0; 
  }	
}

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
  Nmoms(Nhermite*Nlaguerre),
  pars_(pars)
{
  cudaMallocManaged((void**) &kx, sizeof(float)*Nx);
  cudaMallocManaged((void**) &ky, sizeof(float)*Nyc);
  cudaMallocManaged((void**) &kz, sizeof(float)*Nz);

  // copy some parameters to device constant memory 
  cudaMemcpyToSymbol(nx, &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ny, &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nyc, &Nyc, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nz, &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nspecies, &Nspecies, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nhermite, &Nhermite, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nlaguerre, &Nlaguerre, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();


  // initialize k arrays
  int Nmax = max(max(Nx, Nyc),Nz);
  kInit<<<1, Nmax>>>(kx, ky, kz,
                     pars_->x0, pars_->y0, pars_->Zp,
                     pars_->qsf, pars_->shat, pars_->no_zderiv);
}

Grids::~Grids() {
  cudaFree(kx);
  cudaFree(ky);
  cudaFree(kz);
}


