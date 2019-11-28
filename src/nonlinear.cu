#include "nonlinear.h"
#include "cuda_constants.h"
#include "device_funcs.h"
#include "get_error.h"
#include "species.h"
#include "cudaReduc_kernel.cu"
#include "maxReduc.cu"

__global__ void J0phiToGrid(cuComplex* J0phi, cuComplex* phi, float* b,
			    float* muB, float rho2_s);

__global__ void bracket(float* g_res, float* dg_dx, float* dJ0phi_dy,
			float* dg_dy, float* dJ0Phi_dx, float kxfac);

Nonlinear::Nonlinear(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{

  if (grids_->Nl==1) {
    printf("\n");
    printf("Cannot do a nonlinear run with nlaguerre < 2\n");
    printf("\n");
    exit(1);
  }

  laguerre =        new LaguerreTransform(grids_, 1);
  grad_perp_G =     new GradPerp(grids_, grids_->Nz*grids_->Nl);
  grad_perp_J0phi = new GradPerp(grids_, grids_->Nz*laguerre->J);

  checkCuda(cudaMalloc((void**) &dG,    sizeof(float)*grids_->NxNyNz*grids_->Nl));
  checkCuda(cudaMalloc((void**) &dg_dx, sizeof(float)*grids_->NxNyNz*laguerre->J));
  checkCuda(cudaMalloc((void**) &dg_dy, sizeof(float)*grids_->NxNyNz*laguerre->J));

  checkCuda(cudaMalloc((void**) &J0phi, sizeof(cuComplex)*grids_->NxNycNz*laguerre->J));
  checkCuda(cudaMalloc((void**) &dJ0phi_dx,  sizeof(float)*grids_->NxNyNz*laguerre->J));
  checkCuda(cudaMalloc((void**) &dJ0phi_dy,  sizeof(float)*grids_->NxNyNz*laguerre->J));

  checkCuda(cudaMalloc((void**) &g_res, sizeof(float)*grids_->NxNyNz*laguerre->J));

  dimBlock = dim3(32, 4, 1);
  dimGrid = dim3(grids_->NxNyNz / dimBlock.x + 1, 1, 1); 

  dt_cfl = 0.;

  cudaMallocHost((void**) &vmax_x, sizeof(float));
  cudaMallocHost((void**) &vmax_y, sizeof(float));
}

Nonlinear::~Nonlinear() 
{
  delete laguerre;
  delete grad_perp_G;
  delete grad_perp_J0phi;

  cudaFree(dG);
  cudaFree(dg_dx);
  cudaFree(dg_dy);
  cudaFree(J0phi);
  cudaFree(dJ0phi_dx);
  cudaFree(dJ0phi_dy);
  cudaFree(g_res);
}

void Nonlinear::nlps5d(MomentsG* G, Fields* f, MomentsG* G_res)
{
  for(int s=0; s<grids_->Nspecies; s++) {

    // BD  J0phiToGrid does not use a Laguerre transform. Implications?
    // BD  If we use alternate forms for <J0> then that would need to be reflected here
    J0phiToGrid <<<dimGrid,dimBlock>>>
      (J0phi, f->phi, geo_->kperp2, laguerre->get_roots(), pars_->species_h[s].rho2);

    grad_perp_J0phi->dxC2R(J0phi, dJ0phi_dx);
    grad_perp_J0phi->dyC2R(J0phi, dJ0phi_dy);

    // loop over m to save memory. also makes it easier to parallelize later...
    // no extra computation though, just no batching in m in FFTs and matrix multiplies
    for(int m=0; m<grids_->Nm; m++) {

      grad_perp_G->dxC2R(G->Gm(m,s), dG);
      laguerre->transformToGrid(dG, dg_dx);
    
      grad_perp_G->dyC2R(G->Gm(m,s), dG);
      laguerre->transformToGrid(dG, dg_dy);
    
      bracket <<<dimGrid,dimBlock>>> (g_res, dg_dx, dJ0phi_dy, dg_dy, dJ0phi_dx, pars_->kxfac);
    
      laguerre->transformToSpectral(g_res, dG);
      grad_perp_G->R2C(dG, G_res->Gm(m,s));
    }
  }
}

/*
__global__ void max_abs(float *f, float *res)
{
  float vmax = 0.;
  for(int idxyz=blockIdx.x*blockDim.x+threadIdx.x; idxyz<nx*ny*nz; idxyz+=blockDim.x*gridDim.x) {
    vmax = max(abs(f[idxyz]), vmax); 
  }
  atomicMaxFloat(res, vmax);
}
*/

// Note: should only be called after nlps5d, as it assumes 
// dJ0phi_dx, dJ0phi_dy have been calculated.
double Nonlinear::cfl(double dt_max)
{
  float vmax = 1.e-10;
  vmax_x[0] = vmax;
  vmax_y[0] = vmax;
  //  int threads=256;
  //  int blocks=min((grids_->NxNyNz+threads-1)/threads,2048);

  // BD dJ0phi_dx is defined on the pseudo-spectral v_perp**2 grid
  // BD so why is the reduction only over Nl? Should be J
  //  vmax_x[0] = maxReduc(dJ0phi_dx, grids_->NxNyNz*grids_->Nl, dJ0phi_dx, dJ0phi_dx);
  //  vmax_y[0] = maxReduc(dJ0phi_dy, grids_->NxNyNz*grids_->Nl, dJ0phi_dy, dJ0phi_dy);
  vmax_x[0] = maxReduc(dJ0phi_dx, grids_->NxNyNz*laguerre->J, dJ0phi_dx, dJ0phi_dx);
  vmax_y[0] = maxReduc(dJ0phi_dy, grids_->NxNyNz*laguerre->J, dJ0phi_dy, dJ0phi_dy);
  vmax = max(vmax_x[0], vmax_y[0]);

  dt_cfl = (pars_->cfl/vmax < dt_max) ? pars_->cfl/vmax : dt_max;
  //  printf("dt_cfl = %f \n", dt_cfl);
  return dt_cfl;
}

__global__ void J0phiToGrid(cuComplex* J0phi, cuComplex* phi, float* kperp2,
			    float* muB, float rho2_s)
{
  unsigned int idxyz = get_id1();
  unsigned int J = (3*nl/2-1);

  if(idxyz<nx*nyc*nz) {
    for (int j = threadIdx.y; j < J; j += blockDim.y) {
      J0phi[idxyz + nx*nyc*nz*j] = j0f(sqrtf(2. * muB[j] * kperp2[idxyz]*rho2_s)) * phi[idxyz];
    }
  }
}

__global__ void bracket(float* g_res, float* dg_dx, float* dJ0phi_dy,
			float* dg_dy, float* dJ0phi_dx, float kxfac)
{
  unsigned int idxyz = get_id1();
  unsigned int J = (3*nl/2-1);

  if(idxyz<nx*ny*nz) {
    for (int j = threadIdx.y; j < J; j += blockDim.y) {
      unsigned int ig = idxyz + nx*ny*nz*j;

      g_res[ig] = ( dg_dx[ig] * dJ0phi_dy[ig] - dg_dy[ig] * dJ0phi_dx[ig] ) * kxfac;

    }
  }
}
