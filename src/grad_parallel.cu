#include "grad_parallel.h"
#include "cuda_constants.h"
#include "cufftXt.h"
#include "cufft.h"
#include "device_funcs.h"

__device__ void i_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  unsigned int idz = offset / (nx*nyc);
  cuComplex Ikz = make_cuComplex(0., kz[idz]);
  ((cuComplex*)dataOut)[offset] = Ikz*element/nz;
}

__device__ void abs_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  unsigned int idz = offset / (nx*nyc);
  ((cuComplex*)dataOut)[offset] = abs(kz[idz])*element/nz;
}

__global__ void zderiv(cuComplex* res, cuComplex* f, float* kz)
{
  unsigned int idx = get_id1();

  if (idx < nz) {
    cuComplex tmp;
    tmp.x = -kz[idx]*f[idx].y;
    tmp.y = kz[idx]*f[idx].x;
    res[idx] = tmp;
  }
}

__managed__ cufftCallbackStoreC i_kz_callbackPtr = i_kz;
__managed__ cufftCallbackStoreC abs_kz_callbackPtr = abs_kz;

GradParallel::GradParallel(Grids* grids, bool abs, bool single) :
  grids_(grids)
{

  // (ky, kx, theta) <-> (ky, kx, kpar)
  cufftCreate(&gradpar_plan_forward);
  cufftCreate(&gradpar_plan_inverse);

  // MFM: Plan for 1d FFT
  if(single) {
    cufftPlan1d(&gradpar_plan_forward, grids_->Nz, CUFFT_R2C, 1);
    cufftPlan1d(&gradpar_plan_inverse, grids_->Nz, CUFFT_C2R, 1);
  }  
  else {
    int n = grids_->Nz;
    int inembed = grids_->NxNycNz;
    int onembed = grids_->NxNycNz;
    size_t workSize;
    cufftMakePlanMany(gradpar_plan_forward, 1,   &n, &inembed, grids_->NxNyc, 1,
		      //  dim,  n,  isize,   istride,       idist,
		      &onembed, grids_->NxNyc, 1,     CUFFT_C2C, grids_->NxNyc, &workSize);
    // osize,   ostride,       odist, type,      batchsize
    cufftMakePlanMany(gradpar_plan_inverse, 1,   &n, &inembed, grids_->NxNyc, 1,
		      //  dim,  n,  isize,   istride,       idist,
		      &onembed, grids_->NxNyc, 1,     CUFFT_C2C, grids_->NxNyc, &workSize);
    // osize,   ostride,       odist, type,      batchsize
    // isize = size of input data
    // istride = distance between two elements in a batch = distance between (ky,kx,z=1) and (ky,kx,z=2) = Nx*(Ny/2+1)
    // idist = distance between first element of consecutive batches = distance between (ky=1,kx=1,z=1) and (ky=2,kx=1,z=1) = 1
  

  // set up callback functions
    if(abs) {
      cufftXtSetCallback(gradpar_plan_forward, (void**) &abs_kz_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz);
    } else {
      cufftXtSetCallback(gradpar_plan_forward, (void**) &i_kz_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz);
    }
  }
}

GradParallel::~GradParallel() {
  cufftDestroy(gradpar_plan_forward);
  cufftDestroy(gradpar_plan_inverse);
}

// FFT and derivative for all moments
void GradParallel::eval(Moments* m)
{
  // FFT and derivative on parallel term
  // i*kz*ghl calculated via callback, defined as part of gradpar_plan_forward
  // for now, loop over all l and m because cannot batch 
  // eventually will optimize by first transposing so that z is fastest index
  for(int i = 0; i < grids_->Nmoms*grids_->Nspecies; i++) {
    // forward FFT (z -> kz) & multiply by i kz (via callback)
    cufftExecC2C(gradpar_plan_forward, &m->ghl[grids_->NxNycNz*i], &m->ghl[grids_->NxNycNz*i], CUFFT_FORWARD);

    // backward FFT (kz -> z)
    cufftExecC2C(gradpar_plan_inverse, &m->ghl[grids_->NxNycNz*i], &m->ghl[grids_->NxNycNz*i], CUFFT_INVERSE);
  }
}

// FFT and derivative for a single moment
void GradParallel::eval(cuComplex* m, cuComplex* res)
{
  cufftExecC2C(gradpar_plan_forward, m, res, CUFFT_FORWARD);
  cufftExecC2C(gradpar_plan_inverse, res, res, CUFFT_INVERSE);
}

void GradParallel::eval_1d(float* bmag_t, cuComplex* bmag_complex_t)
{
  cufftExecR2C(gradpar_plan_forward, bmag_t, bmag_complex_t);
  zderiv<<<1,grids_->Nz>>>(bmag_complex_t,bmag_complex_t, grids_->kz);
  cufftExecC2R(gradpar_plan_inverse, bmag_complex_t, bmag_t);
}