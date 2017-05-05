#include "solver.h"
#include "qneut_kernel.cu"
#include "get_error.h"

Solver::Solver(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{
  cudaMalloc((void**) &nbar, sizeof(cuComplex)*grids_->NxNycNz);
  cudaMalloc((void**) &tmp, sizeof(cuComplex)*grids_->NxNycNz);

  // set up phiavgdenom, which is stored for quasineutrality calculation
  cudaMalloc((void**) &phiavgdenom, sizeof(float)*grids_->Nx);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  maxThreadsPerBlock_ = prop.maxThreadsPerBlock;
  float* tmpXZ;
  cudaMalloc((void**) &tmpXZ, sizeof(float)*grids_->Nx*grids_->Nz);
  dim3 dimBlock = dim3(1, maxThreadsPerBlock_/grids_->Nz, grids_->Nz);
  dim3 dimGrid = dim3(1, grids_->Nx/dimBlock.y+1, 1);
  calc_phiavgdenom<<<dimGrid,dimBlock>>>(phiavgdenom, tmpXZ, geo_->jacobian, pars_->ti_ov_te, geo_->kperp2_t); 
  cudaFree(tmpXZ);
}

Solver::~Solver() 
{
  cudaFree(nbar);
  cudaFree(tmp);
  cudaFree(phiavgdenom);
}

int Solver::fieldSolve(Moments* moms, Fields* fields)
{
  if(pars_->adiabatic_electrons) {
    cudaMemset(nbar, 0., sizeof(cuComplex)*grids_->NxNycNz);
    real_space_density<<<grids_->NxNycNz/dimBlock.x+1, maxThreadsPerBlock_>>>(nbar, moms->ghl, geo_->kperp2_t);
    if(pars_->iphi00==2) {
      dimBlock = dim3(maxThreadsPerBlock_/grids_->Nz/2, 1, grids_->Nz);
      dimGrid = dim3(grids_->Nyc/dimBlock.x+1, grids_->Nx, 1);
      qneutAdiab_part1<<<dimGrid, dimBlock>>>(tmp, nbar, geo_->jacobian, pars_->ti_ov_te, geo_->kperp2_t);
      qneutAdiab_part2<<<dimGrid, dimBlock>>>(fields->phi, tmp, nbar, phiavgdenom, geo_->jacobian, pars_->ti_ov_te, geo_->kperp2_t);
    }
 }
  return 0;
}

