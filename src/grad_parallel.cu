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

__device__ void i_kz_1d(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  unsigned int idz = offset;
  cuComplex Ikz = make_cuComplex(0., kz[idz]);
  ((cuComplex*)dataOut)[offset] = Ikz*element/nz;
}

__managed__ cufftCallbackStoreC i_kz_callbackPtr = i_kz;
__managed__ cufftCallbackStoreC i_kz_1d_callbackPtr = i_kz_1d;
__managed__ cufftCallbackStoreC abs_kz_callbackPtr = abs_kz;

GradParallelPeriodic::GradParallelPeriodic(Grids* grids, bool abs) :
  grids_(grids)
{
  // (ky, kx, theta) <-> (ky, kx, kpar)
  cufftCreate(&gradpar_plan_forward);
  cufftCreate(&gradpar_plan_inverse);

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

GradParallelPeriodic::~GradParallelPeriodic() {
  cufftDestroy(gradpar_plan_forward);
  cufftDestroy(gradpar_plan_inverse);
}

// FFT and derivative for all moments
void GradParallelPeriodic::eval(MomentsG* G)
{
  // FFT and derivative on parallel term
  // i*kz*G calculated via callback, defined as part of gradpar_plan_forward
  // for now, loop over all l and m because cannot batch 
  // eventually will optimize by first transposing so that z is fastest index
  G->reality();
  for(int i = 0; i < grids_->Nmoms*grids_->Nspecies; i++) {
    // forward FFT (z -> kz) & multiply by i kz (via callback)
    cufftExecC2C(gradpar_plan_forward, G->G(i), G->G(i), CUFFT_FORWARD);

    // backward FFT (kz -> z)
    cufftExecC2C(gradpar_plan_inverse, G->G(i), G->G(i), CUFFT_INVERSE);
  }
  G->reality();
}

// FFT and derivative for a single moment
void GradParallelPeriodic::eval(cuComplex* mom, cuComplex* res)
{
  reality_kernel<<<dim3(32,32,1),dim3(grids_->Nx/32+1, grids_->Nz/32+1,1)>>>(res);
  cufftExecC2C(gradpar_plan_forward, mom, res, CUFFT_FORWARD);
  cufftExecC2C(gradpar_plan_inverse, res, res, CUFFT_INVERSE);
  reality_kernel<<<dim3(32,32,1),dim3(grids_->Nx/32+1, grids_->Nz/32+1,1)>>>(res);
}

// FFT only for a single moment
void GradParallelPeriodic::fft_only(cuComplex* mom, cuComplex* res, int dir)
{
  // use gradpar_plan_inverse since it does not multiply by i kz via callback 
  cufftExecC2C(gradpar_plan_inverse, mom, res, dir);
}

GradParallelLocal::GradParallelLocal(Grids* grids, bool abs) :
  grids_(grids), abs_(abs)
{
  dimBlock = 512;
  dimGrid = grids_->NxNycNz/dimBlock.x+1;
}

void GradParallelLocal::eval(MomentsG *G)
{
  if(!abs_) {
    G->scale(make_cuComplex(0.,1.));
  }
}

// single moment
void GradParallelLocal::eval(cuComplex* mom, cuComplex* res) 
{
  if(!abs_) {
    scale_singlemom_kernel<<<dimGrid,dimBlock>>>(res, mom, make_cuComplex(0.,1.));
  } else {
    scale_singlemom_kernel<<<dimGrid,dimBlock>>>(res, mom, make_cuComplex(1.,0.));
  }
}

GradParallel1D::GradParallel1D(Grids* grids)
{
  // (theta) <-> (kpar)
  cufftCreate(&gradpar_plan_forward);
  cufftCreate(&gradpar_plan_inverse);

  // MFM: Plan for 1d FFT
  cufftPlan1d(&gradpar_plan_forward, grids_->Nz, CUFFT_R2C, 1);
  cufftPlan1d(&gradpar_plan_inverse, grids_->Nz, CUFFT_C2R, 1);

  cufftXtSetCallback(gradpar_plan_forward, (void**) &i_kz_1d_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz);

  cudaMalloc((void**) &b_complex, sizeof(cuComplex)*(grids_->Nz/2+1));
}

GradParallel1D::~GradParallel1D() {
  cufftDestroy(gradpar_plan_forward);
  cufftDestroy(gradpar_plan_inverse);
  cudaFree(b_complex);
}

void GradParallel1D::eval1D(float* b)
{
  cufftExecR2C(gradpar_plan_forward, b, b_complex);
  cufftExecC2R(gradpar_plan_inverse, b_complex, b);
}

