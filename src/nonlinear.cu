#include "nonlinear.h"
#include "cuda_constants.h"
#include "device_funcs.h"
#include "get_error.h"
#include "species.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

Nonlinear::Nonlinear(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{

  if (grids_ -> Nl < 2) {
    printf("\n");
    printf("Cannot do a nonlinear run with nlaguerre < 2\n");
    printf("\n");
    exit(1);
  }

  laguerre =        new LaguerreTransform(grids_, 1);
  red =             new Red(grids_->NxNyNz);

  nBatch = grids_->Nz*grids_->Nl; 
  grad_perp_G =     new GradPerp(grids_, nBatch, grids_->NxNycNz*grids_->Nl); 
  
  nBatch = grids_->Nz*grids_->Nj; 
  grad_perp_J0phi = new GradPerp(grids_, nBatch, grids_->NxNycNz*grids_->Nj); 

  nBatch = grids_->Nz;
  grad_perp_phi =   new GradPerp(grids_, nBatch, grids_->NxNycNz);
  
  checkCuda(cudaMalloc(&dG,    sizeof(float)*grids_->NxNyNz*grids_->Nl));
  checkCuda(cudaMalloc(&dg_dx, sizeof(float)*grids_->NxNyNz*grids_->Nj));
  checkCuda(cudaMalloc(&dg_dy, sizeof(float)*grids_->NxNyNz*grids_->Nj));

  checkCuda(cudaMalloc(&J0phi,      sizeof(cuComplex)*grids_->NxNycNz*grids_->Nj));
  checkCuda(cudaMalloc(&dJ0phi_dx,  sizeof(float)*grids_->NxNyNz*grids_->Nj));
  checkCuda(cudaMalloc(&dJ0phi_dy,  sizeof(float)*grids_->NxNyNz*grids_->Nj));

  checkCuda(cudaMalloc(&dphi,  sizeof(float)*grids_->NxNyNz));
  
  checkCuda(cudaMalloc(&g_res, sizeof(float)*grids_->NxNyNz*grids_->Nj));

  checkCuda(cudaMalloc(&val1,  sizeof(float)));
  cudaMemset(val1, 0., sizeof(float));

  int nxyz = grids_->NxNyNz;
  int nlag = grids_->Nj;

  int nbx = 32;
  int ngx = (nxyz-1)/nbx + 1;
 
  int nby = 8;
  int ngy = (nlag-1)/nby + 1;

  dBx = dim3(nbx, nby, 1);
  dGx = dim3(ngx, ngy, 1);

  nxyz = grids_->NxNycNz;
  nbx = 32;
  ngx = (nxyz-1)/nbx + 1;

  dBk = dim3(nbx, nby, 1);
  dGk = dim3(ngx, ngy, 1);

  cfl_x_inv = (float) grids_->Nx / (pars_->cfl * 2 * M_PI * pars_->x0);
  cfl_y_inv = (float) grids_->Ny / (pars_->cfl * 2 * M_PI * pars_->y0); 
  
  dt_cfl = 0.;

  cudaMallocHost((void**) &vmax_x, sizeof(float));
  cudaMallocHost((void**) &vmax_y, sizeof(float));
}

Nonlinear::~Nonlinear() 
{
  delete red;
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

void Nonlinear::qvar (cuComplex* G, int N)
{
  cuComplex* G_h;
  int Nk = grids_->Nyc*grids_->Nx;
  G_h = (cuComplex*) malloc (sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {G_h[i].x = 0.; G_h[i].y = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(cuComplex));

  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d) = (%e, %e) \n", i%Nk, i/Nk, G_h[i].x, G_h[i].y);
  printf("\n");

  free (G_h);
}

void Nonlinear::qvar (float* G, int N)
{
  float* G_h;
  int N_ = grids_->Ny*grids_->Nx*grids_->Nz;
  G_h = (float*) malloc (sizeof(float)*N);
  for (int i=0; i<N; i++) {G_h[i] = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(float));

  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d) = %e \n", i%N_, i/N_, G_h[i]);
  printf("\n");

  free (G_h);
}

void Nonlinear::nlps5d(MomentsG* G, Fields* f, MomentsG* G_res)
{
  for(int s=0; s<grids_->Nspecies; s++) {

    // BD  J0phiToGrid does not use a Laguerre transform. Implications?
    // BD  If we use alternate forms for <J0> then that would need to be reflected here

    //    print_cudims(dimGrid, dimBlock);

    //    printf("\n");
    //    printf("Phi:\n");
    //    qvar(f->phi, grids_->NxNycNz);

    
    J0phiToGrid <<< dGk, dBk >>> (J0phi, f->phi, geo_->kperp2, laguerre->get_roots(), pars_->species_h[s].rho2);

    grad_perp_J0phi -> dxC2R(J0phi, dJ0phi_dx);
    grad_perp_J0phi -> dyC2R(J0phi, dJ0phi_dy);

    // loop over m to save memory. also makes it easier to parallelize later...
    // no extra computation: just no batching in m in FFTs and in the matrix multiplies
    for(int m=0; m<grids_->Nm; m++) {

      grad_perp_G -> dxC2R(G->Gm(m,s), dG);
      laguerre    -> transformToGrid(dG, dg_dx);
    
      grad_perp_G -> dyC2R(G->Gm(m,s), dG);      
      laguerre    -> transformToGrid(dG, dg_dy);
         
      bracket <<< dGx, dBx >>> (g_res, dg_dx, dJ0phi_dy, dg_dy, dJ0phi_dx, pars_->kxfac);
      laguerre->transformToSpectral(g_res, dG);
      grad_perp_G->R2C(dG, G_res->Gm(m,s));

    }

  }
}
double Nonlinear::cfl(Fields *f, double dt_max)
{
  grad_perp_phi -> dxC2R(f->phi, dphi);  red->Max(dphi, val1); CP_TO_CPU(vmax_y, val1, sizeof(float));
  grad_perp_phi -> dyC2R(f->phi, dphi);  red->Max(dphi, val1); CP_TO_CPU(vmax_x, val1, sizeof(float));
  float vmax = max(vmax_x[0]*cfl_x_inv, vmax_y[0]*cfl_y_inv);
  dt_cfl = min(dt_max, 1./vmax);
  return dt_cfl;
}
