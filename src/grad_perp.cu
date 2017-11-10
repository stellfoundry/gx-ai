#include "cufftXt.h"
#include "cufft.h"
#include "device_funcs.h"
#include "grad_perp.h"
#include "cuda_constants.h"

__device__ cuComplex i_kx(void *dataIn, size_t offset, void *kxData, void *sharedPtr)
{
  float *kx = (float*) kxData;
  unsigned int idx = offset % (nx*nyc) / nyc; 
  cuComplex Ikx = make_cuComplex(0., kx[idx]);
  return Ikx*((cuComplex*)dataIn)[offset];
}

__device__ cuComplex i_ky(void *dataIn, size_t offset, void *kyData, void *sharedPtr)
{
  float *ky = (float*) kyData;
  unsigned int idy = offset % (nx*nyc) % nyc; 
  cuComplex Iky = make_cuComplex(0., ky[idy]);
  return Iky*((cuComplex*)dataIn)[offset];
}

__device__ void mask_and_scale(void *dataOut, size_t offset, cufftComplex element, void *data, void * sharedPtr)
{
  unsigned int idx = offset % (nx*nyc) / nyc; 
  unsigned int idy = offset % (nx*nyc) % nyc; 
  int ikx = get_ikx(idx);
  if( idy>(ny-1)/3 || ikx>(nx-1)/3 || ikx<-(nx-1)/3 || (ikx==0 && idy==0)) {
    // mask
    ((cuComplex*)dataOut)[offset].x = 0.;
    ((cuComplex*)dataOut)[offset].y = 0.;
  } else {
    // scale
    ((cuComplex*)dataOut)[offset] = element/(nx*ny);
  }
}

__managed__ cufftCallbackLoadC i_kx_callbackPtr = i_kx;
__managed__ cufftCallbackLoadC i_ky_callbackPtr = i_ky;
__managed__ cufftCallbackStoreC mask_and_scale_callbackPtr = mask_and_scale;

GradPerp::GradPerp(Grids* grids, int batch_size)
 : grids_(grids), batch_size_(batch_size)
{
  cufftCreate(&gradperp_plan_R2C);
  cufftCreate(&gradperp_plan_dxC2R);
  cufftCreate(&gradperp_plan_dyC2R);

  int NLPSfftdims[2] = {grids->Nx, grids->Ny};
  size_t workSize;
  cufftMakePlanMany(gradperp_plan_R2C, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, batch_size_, &workSize);
  // need separate plans for dx and dy in order to use callbacks... what is the memory cost?
  cufftMakePlanMany(gradperp_plan_dxC2R, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);
  cufftMakePlanMany(gradperp_plan_dyC2R, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, batch_size_, &workSize);

  cudaDeviceSynchronize();
  cufftXtSetCallback(gradperp_plan_dxC2R, 
                     (void**) &i_kx_callbackPtr, 
                     CUFFT_CB_LD_COMPLEX, 
                     (void**)&grids_->kx);

  cufftXtSetCallback(gradperp_plan_dyC2R, 
                     (void**) &i_ky_callbackPtr, 
                     CUFFT_CB_LD_COMPLEX, 
                     (void**)&grids_->ky);

  cufftXtSetCallback(gradperp_plan_R2C, 
                     (void**) &mask_and_scale_callbackPtr, 
                     CUFFT_CB_ST_COMPLEX, 
                     NULL);
  cudaDeviceSynchronize();
}

GradPerp::~GradPerp()
{
  cufftDestroy(gradperp_plan_R2C);
  cufftDestroy(gradperp_plan_dxC2R);
  cufftDestroy(gradperp_plan_dyC2R);
}

void GradPerp::dxC2R(cuComplex* G, float* dxG)
{
  cufftExecC2R(gradperp_plan_dxC2R, G, dxG);
}

void GradPerp::dyC2R(cuComplex* G, float* dyG)
{
  cufftExecC2R(gradperp_plan_dyC2R, G, dyG);
}

void GradPerp::R2C(float* G, cuComplex* res)
{
  cufftExecR2C(gradperp_plan_R2C, G, res);
}
