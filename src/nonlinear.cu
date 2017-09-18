#include "nonlinear.h"
#include "cuda_constants.h"
#include "device_funcs.h"
#include "get_error.h"

__global__ void J0phiToGrid(cuComplex* phi, float* b, float* muB, cuComplex* J0phi);
__global__ void bracket(float* dg_dx, float* dJ0phi_dy, float* dg_dy, float* dJ0Phi_dx, float* g_res);

Nonlinear::Nonlinear(Grids* grids, Geometry* geo) :
  grids_(grids), geo_(geo)
{
  laguerre = new LaguerreTransform(grids_, 1);
  grad_perp_G = new GradPerp(grids_, grids_->Nz*grids_->Nl);
  grad_perp_J0phi = new GradPerp(grids_, grids_->Nz*(laguerre->J+1));

  checkCuda(cudaMalloc((void**) &dG, sizeof(float)*grids_->NxNyNz*grids_->Nl));
  checkCuda(cudaMalloc((void**) &dg_dx, sizeof(float)*grids_->NxNyNz*(laguerre->J+1)));
  checkCuda(cudaMalloc((void**) &dg_dy, sizeof(float)*grids_->NxNyNz*(laguerre->J+1)));

  checkCuda(cudaMalloc((void**) &J0phi, sizeof(cuComplex)*grids_->NxNycNz*(laguerre->J+1)));
  checkCuda(cudaMalloc((void**) &dJ0phi_dx, sizeof(float)*grids_->NxNyNz*(laguerre->J+1)));
  checkCuda(cudaMalloc((void**) &dJ0phi_dy, sizeof(float)*grids_->NxNyNz*(laguerre->J+1)));

  checkCuda(cudaMalloc((void**) &g_res, sizeof(float)*grids_->NxNyNz*(laguerre->J+1)));

  dimBlock = dim3(32, 4, 1);
  dimGrid = dim3(grids_->NxNyNz/dimBlock.x+1, 1, 1);

  dt_cfl = 0.;
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
  J0phiToGrid<<<dimGrid,dimBlock>>>(f->phi, geo_->kperp2, laguerre->get_roots(), J0phi);
  grad_perp_J0phi->dxC2R(J0phi, dJ0phi_dx);
  grad_perp_J0phi->dyC2R(J0phi, dJ0phi_dy);

  // loop over m to save memory. also makes it easier to parallelize later...
  // no extra computation though, just no batching in m in FFTs and matrix multiplies
  for(int m=0; m<grids_->Nm; m++) {
    grad_perp_G->dxC2R(G->Gm(m), dG);
    laguerre->transformToGrid(dG, dg_dx);
  
    grad_perp_G->dyC2R(G->Gm(m), dG);
    laguerre->transformToGrid(dG, dg_dy);
  
    bracket<<<dimGrid,dimBlock>>>(dg_dx, dJ0phi_dy, dg_dy, dJ0phi_dx, g_res);
  
    laguerre->transformToSpectral(g_res, dG);
    grad_perp_G->R2C(dG, G_res->Gm(m));
  }
}

__global__ void max_abs(float *f, float *res)
{
  float vmax = 0.;
  for(int idxyz=blockIdx.x*blockDim.x+threadIdx.x;idxyz<nx*nyc*nz;idxyz+=blockDim.x*gridDim.x) {
    vmax = fmaxf(abs(f[idxyz]), vmax);
  }
  atomicMaxFloat(res, vmax);
}

// Note: should only be called after nlps5d, as it assumes 
// dJ0phi_dx, dJ0phi_dy have been calculated.
double Nonlinear::cfl(double dt_max)
{
  float vmax = 1.e-10;
  float vmax_x = 1.e-10, vmax_y = 1.e-10;
  int threads=256;
  int blocks=min((grids_->NxNyNz+threads-1)/threads,2048);
  max_abs<<<blocks,threads>>>(dJ0phi_dx, &vmax_x);
  max_abs<<<blocks,threads>>>(dJ0phi_dy, &vmax_y);
  vmax = max(vmax_x, vmax_y);
  dt_cfl = 1./vmax < dt_max ? 1./vmax : dt_max;
  
  return dt_cfl;
}

__global__ void J0phiToGrid(cuComplex* phi, float* b, float* muB, cuComplex* J0phi)
{
  unsigned int idxyz = get_id1();
  unsigned int J = (3*(nl-1)-1)/2;

  if(idxyz<nx*nyc*nz) {
    for (int j = threadIdx.y; j < J+1; j += blockDim.y) {
      J0phi[idxyz + nx*nyc*nz*j] = j0f(sqrtf(2.*muB[j]*b[idxyz]))*phi[idxyz];
    }
  }
}

__global__ void bracket(float* dg_dx, float* dJ0phi_dy, float* dg_dy, float* dJ0phi_dx, float* g_res)
{
  unsigned int idxyz = get_id1();
  unsigned int J = (3*(nl-1)-1)/2;

  if(idxyz<nx*ny*nz) {
    for (int j = threadIdx.y; j < J+1; j += blockDim.y) {
      unsigned int globalIdx = idxyz + nx*ny*nz*j;
      g_res[globalIdx] = dg_dx[globalIdx]*dJ0phi_dy[globalIdx] - dg_dy[globalIdx]*dJ0phi_dx[globalIdx];
    }
  }
}
