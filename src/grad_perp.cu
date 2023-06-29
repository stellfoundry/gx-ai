#include "grad_perp.h"
#include "device_funcs.h"
#include "get_error.h"
#define GBK <<< dGk, dBk >>>
#define GBX <<< dGx, dBx >>>
#define GBX_single <<< dGx_single, dBx_single >>>


GradPerp::GradPerp(Grids* grids, int batch_size, int mem_size)
  : grids_(grids), batch_size_(batch_size), mem_size_(mem_size), tmp(nullptr)
{
  cufftCreate(&gradperp_plan_R2C);
  cufftCreate(&gradperp_plan_C2R);
  cufftCreate(&gradperp_plan_dxC2R);
  cufftCreate(&gradperp_plan_dyC2R);
  cufftCreate(&gradperp_plan_R2Cntft);
  cufftCreate(&gradperp_plan_C2Rntft);

  // Use MakePlanMany to enable callbacks
  // Order of Nx, Ny is correct here

  cudaMalloc (&tmp, sizeof(cuComplex)*mem_size_);

  int maxthreads = 1024; 
  int nthreads = min(maxthreads, mem_size_);
  int nblocks = 1 + (mem_size_-1)/nthreads;

  dB = dim3(nthreads, 1, 1);
  dG = dim3(nblocks,  1, 1);
  
  int NLPSfftdims[2] = {grids->Nx, grids->Ny};
  int NLPSfftdimy = grids->Ny;
  size_t workSize;
  cufftMakePlanMany(gradperp_plan_C2R,    2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_R2C,    2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dxC2R,  2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dyC2R,  2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_R2Cntft,   1, &NLPSfftdimy, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, batch_size_*grids_->Nx, &workSize);
  cufftMakePlanMany(gradperp_plan_C2Rntft,   1, &NLPSfftdimy, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_*grids_->Nx, &workSize);

  cudaDeviceSynchronize();
  cufftCallbackLoadC i_kxs_callbackPtr_h;
  cufftCallbackLoadC i_kx_callbackPtr_h; 
  cufftCallbackLoadC i_ky_callbackPtr_h; 
  cufftCallbackLoadC phasefac_callbackPtr_h;
  cufftCallbackStoreC mask_and_scale_callbackPtr_h;
  cufftCallbackStoreC scale_ky_callbackPtr_h;


  checkCuda(cudaMemcpyFromSymbol(&i_kxs_callbackPtr_h, 
                     i_kxs_callbackPtr, 
                     sizeof(i_kxs_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&i_kx_callbackPtr_h, 
                     i_kx_callbackPtr, 
                     sizeof(i_kx_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&i_ky_callbackPtr_h, 
                     i_ky_callbackPtr, 
                     sizeof(i_ky_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&mask_and_scale_callbackPtr_h, 
                     mask_and_scale_callbackPtr, 
                     sizeof(mask_and_scale_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&scale_ky_callbackPtr_h, 
                     scale_ky_callbackPtr, 
                     sizeof(scale_ky_callbackPtr_h)));

  checkCuda(cufftXtSetCallback(gradperp_plan_dxC2R, (void**) &i_kx_callbackPtr_h, 
                     CUFFT_CB_LD_COMPLEX, 
                     (void**)&grids_->kx));

  checkCuda(cufftXtSetCallback(gradperp_plan_dyC2R, (void**) &i_ky_callbackPtr_h, 
                     CUFFT_CB_LD_COMPLEX, 
                     (void**)&grids_->ky));

  checkCuda(cufftXtSetCallback(gradperp_plan_R2C,   (void**) &mask_and_scale_callbackPtr_h, 
                     CUFFT_CB_ST_COMPLEX, 
                     NULL));
  
  checkCuda(cufftXtSetCallback(gradperp_plan_R2Cntft,   (void**) &scale_ky_callbackPtr_h, 
                     CUFFT_CB_ST_COMPLEX, 
                     NULL)); // is this scale_ky callback necessary? //JMH
  
  cudaDeviceSynchronize();
  
  int nxyz = grids_->NxNyNz;
  int nlag = grids_->Nj;
  int nher = grids_->Nm;

  int nbx = min(32, nxyz);  int ngx = 1 + (nxyz-1)/nbx;
  int nby = min(4, nlag);  int ngy = 1 + (nlag-1)/nby;
  int nbz = min(4, nher);  int ngz = 1 + (nher-1)/nbz;

  dBx = dim3(nbx, nby, nbz);
  dGx = dim3(ngx, ngy, ngz);

  dBx_single = dim3(nbx, nby, 1);
  dGx_single = dim3(ngx, ngy, 1);

  int nxkyz = grids_->NxNycNz;

  nbx = min(32, nxkyz);      ngx = 1 + (nxkyz-1)/nbx;
  nby = min(16, nlag);       ngy = 1 + (nlag-1)/nby;

  dBk = dim3(nbx, nby, 1);
  dGk = dim3(ngx, ngy, 1);

}

GradPerp::~GradPerp()
{
  if (tmp)      cudaFree (tmp);
  cufftDestroy ( gradperp_plan_R2C    );
  cufftDestroy ( gradperp_plan_C2R    );
  cufftDestroy ( gradperp_plan_dxC2R  );
  cufftDestroy ( gradperp_plan_dyC2R  );
  cufftDestroy ( gradperp_plan_R2Cntft   );
  cufftDestroy ( gradperp_plan_C2Rntft  );
}

// Out-of-place 2D transforms in cufft now overwrite the input data. 

void GradPerp::dxC2R(cuComplex* G, float* dxG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  checkCuda(cufftExecC2R(gradperp_plan_dxC2R, tmp, dxG));
}

void GradPerp::phase_mult_ntft(float* G, bool positive_phase)
{
  cufftExecR2C(gradperp_plan_R2Cntft, G, tmp); //1D FFT in y

  // multiplying by exp(i*deltaKx*x) or exp(-i*deltaKx*x)
  if (positive_phase) {
    if (batch_size_ == grids_->Nz*grids_->Nl*grids_->Nm) { // if multiplying G
      iKxgtoGrid GBX (tmp, grids_->phasefac_ntft);
    } else if (batch_size_ == grids_->Nz*grids_->Nj) { // if multiplying J0phi or J0apar
      iKxJ0ftoGrid GBK (tmp, grids_->phasefac_ntft); 
    } // add in an else for single moment here?
  } else {
    if (batch_size_ == grids_->Nz*grids_->Nl*grids_->Nm) { // if multiplying G
      iKxgtoGrid GBX (tmp, grids_->phasefacminus_ntft);
    } else if (batch_size_ == grids_->Nz*grids_->Nj) { // if multiplying J0phi or J0apar
      iKxJ0ftoGrid GBK (tmp, grids_->phasefacminus_ntft); 
    } // add in an else statement for single moment here?
  }
  
  cufftExecC2R(gradperp_plan_C2Rntft, tmp, G); //1D FFT in y

}

void GradPerp::qvar (cuComplex* G, int N)
{
  cuComplex* G_h;
  int Nk = grids_->Nyc*grids_->Nx;
  G_h = (cuComplex*) malloc (sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {G_h[i].x = 0.; G_h[i].y = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(cuComplex));

  printf("\n");
  for (int i=0; i<N; i++) printf("grad_perp: var(%d,%d,%d) = (%e, %e)  \n", i%grids_->Nyc, i/grids_->Nyc%grids_->Nx, i/Nk, G_h[i].x, G_h[i].y);
  printf("\n");

  free (G_h);
}

void GradPerp::qvar (float* G, int N)
{
  float* G_h;
  int Nx = grids_->Ny*grids_->Nx;
  G_h = (float*) malloc (sizeof(float)*N);
  for (int i=0; i<N; i++) G_h[i] = 0.;
  CP_TO_CPU (G_h, G, N*sizeof(float));

  printf("\n");
  for (int i=0; i<N; i++) printf("grad_perp: var(%d,%d,%d) = %e \n", i%grids_->Ny, i/grids_->Ny%grids_->Nx, i/Nx, G_h[i]);
  printf("\n");

  free (G_h);
}

void GradPerp::dyC2R(cuComplex* G, float* dyG)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  checkCuda(cufftExecC2R(gradperp_plan_dyC2R, tmp, dyG));
}

void GradPerp::C2R(cuComplex* G, float* Gy)
{
  CP_ON_GPU (tmp, G, sizeof(cuComplex)*mem_size_);
  checkCuda(cufftExecC2R(gradperp_plan_C2R, tmp, Gy));
}

// An R2C that accumulates -- will be very useful
void GradPerp::R2C(float* G, cuComplex* res, bool accumulate)
{
  if (accumulate) {
    checkCuda(cufftExecR2C(gradperp_plan_R2C, G, tmp));
    add_section <<< dG, dB >>> (res, tmp, mem_size_);
  } else {
    checkCuda(cufftExecR2C(gradperp_plan_R2C, G, res));
  }
}


