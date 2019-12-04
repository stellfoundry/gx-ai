#include "solver.h"
#include "qneut_kernel.cu"
#include "get_error.h"
#include "cuda_constants.h"

Solver::Solver(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{
  cudaMalloc((void**) &nbar, sizeof(cuComplex)*grids_->NxNycNz);
  cudaMalloc((void**) &tmp, sizeof(cuComplex)*grids_->NxNycNz);

  // set up phiavgdenom, which is stored for quasineutrality calculation
  cudaMalloc((void**) &phiavgdenom, sizeof(float)*grids_->Nx);
  cudaMemset(phiavgdenom, 0., sizeof(float)*grids_->Nx);
  
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  maxThreadsPerBlock_ = prop.maxThreadsPerBlock;

  float* tmpXZ;
  cudaMalloc((void**) &tmpXZ, sizeof(float)*grids_->NxNz);
  cudaMemset(tmpXZ, 0., grids_->NxNz);

  dim3 dimBlock = dim3(maxThreadsPerBlock_/grids_->Nz, grids_->Nz, 1);
  dim3 dimGrid  = dim3(grids_->Nx/dimBlock.x+1, 1, 1);
  
  //  printf("calc_phiavgdenom:\n");
  //  print_cudims(dimGrid, dimBlock);

  calc_phiavgdenom <<<dimGrid,dimBlock>>> (phiavgdenom, tmpXZ, geo_->kperp2, geo_->jacobian,
     pars_->species, pars_->ti_ov_te); 
  
  //  printf("tmpXZ: \n");
  //  this->svar(tmpXZ, grids_->Nx*grids_->Nz);
  cudaFree(tmpXZ);

  //  printf("phiavgdenom: \n");
  //  this->svar(phiavgdenom, grids_->Nx);
  
  // cuda dims for qneut calculation
  dimBlock_qneut = dim3(32, 4, 4);
  dimGrid_qneut = dim3(grids_->Nyc/dimBlock.x+1, grids_->Nx/dimBlock.y+1, grids_->Nz/dimBlock.z+1);
  //  printf("_qneut:\n");
  //  print_cudims(dimGrid_qneut, dimBlock_qneut);
}

Solver::~Solver() 
{
  cudaFree(nbar);
  cudaFree(tmp);
  cudaFree(phiavgdenom);
}

void Solver::svar (cuComplex* f, int N)
{
  cuComplex* f_h;
  cudaMallocHost((void**) &f_h, sizeof(cuComplex)*N);
  CP_TO_CPU (f_h, f, N);

  for (int i=0; i<N; i++) printf("solver: var(%d) = (%f, %f) \n", i, f_h[i].x, f_h[i].y);
  printf("\n");

  cudaFreeHost (f_h);
}

void Solver::svar (float* f, int N)
{
  float* f_h;
  cudaMallocHost((void**) &f_h, sizeof(float)*N);

  CP_TO_CPU (f_h, f, N*sizeof(float));

  for (int i=0; i<N; i++) printf("solver: var(%d) = %f \n", i, f_h[i]);
  printf("\n");
  
  cudaFreeHost (f_h);
}

int Solver::fieldSolve(MomentsG* G, Fields* fields)
{
  if(pars_->adiabatic_electrons) {

    real_space_density <<<grids_->NxNycNz/maxThreadsPerBlock_+1, maxThreadsPerBlock_>>>
      (nbar, G->G(), geo_->kperp2, pars_->species);
    
    if(pars_->iphi00==2) {
      qneutAdiab_part1 <<<dimGrid_qneut, dimBlock_qneut>>>
	(tmp, nbar, geo_->kperp2, geo_->jacobian, pars_->species, pars_->ti_ov_te);

      qneutAdiab_part2 <<<dimGrid_qneut, dimBlock_qneut>>>

	(fields->phi, tmp, nbar, phiavgdenom, geo_->kperp2,
	 geo_->jacobian, pars_->species, pars_->ti_ov_te);
    } else {
      
    }
  }
  if(pars_->source_option==PHIEXT) {
    add_source <<<dimGrid_qneut, dimBlock_qneut>>> (fields->phi, pars_->phi_ext);
  }
  return 0;
}

