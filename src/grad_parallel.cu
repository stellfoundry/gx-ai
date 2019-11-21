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

GradParallelPeriodic::GradParallelPeriodic(Grids* grids) :
  grids_(grids)
{
  // (ky, kx, theta) <-> (ky, kx, kpar)
  cufftCreate(&gradpar_plan_forward);
  cufftCreate(&abs_gradpar_plan_forward);
  cufftCreate(&gradpar_plan_inverse);

  int n = grids_->Nz; 			// size of FFT
  int isize = grids_->NxNycNz;		// size of input data
  int osize = grids_->NxNycNz;		// size of output data
  int dim = 1;				// 1 dimensional
  int istride = grids_->NxNyc;		// distance between two elements in a batch 
					// = distance between (ky,kx,z=1) and (ky,kx,z=2) = Nx*(Ny/2+1)
  int idist = 1;			// idist = distance between first element of consecutive batches 
					// = distance between (ky=1,kx=1,z=1) and (ky=2,kx=1,z=1) = 1
  int ostride = grids_->NxNyc;
  int odist = 1;
  int batchsize = grids_->NxNyc;	// number of consecutive transforms
  size_t workSize;

  cufftMakePlanMany(gradpar_plan_forward, dim, &n, &isize, istride, idist,
      	      &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany(abs_gradpar_plan_forward, dim, &n, &isize, istride, idist,
      	      &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany(gradpar_plan_inverse, dim, &n, &isize, istride, idist,
      	      &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);

  //cufftCallbackStoreC i_kz_callbackPtr_host;
  //cufftCallbackStoreC abs_kz_callbackPtr_host;

  //cudaMemcpyFromSymbol(&i_kz_callbackPtr_host, i_kz_callbackPtr, sizeof(i_kz_callbackPtr_host));
  //cudaMemcpyFromSymbol(&abs_kz_callbackPtr_host, abs_kz_callbackPtr, sizeof(abs_kz_callbackPtr_host));

  // set up callback functions
  cudaDeviceSynchronize();
  cufftXtSetCallback(gradpar_plan_forward, (void**) &i_kz_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz);
  cufftXtSetCallback(abs_gradpar_plan_forward, (void**) &abs_kz_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz);
  cudaDeviceSynchronize();
}

GradParallelPeriodic::~GradParallelPeriodic() {
  cufftDestroy(gradpar_plan_forward);
  cufftDestroy(abs_gradpar_plan_forward);
  cufftDestroy(gradpar_plan_inverse);
}

// FFT and derivative for all moments
void GradParallelPeriodic::dz(MomentsG* G)
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
void GradParallelPeriodic::dz(cuComplex* mom, cuComplex* res)
{
  reality_singlemom_kernel<<<dim3(32,32,1),dim3(grids_->Nx/32+1, grids_->Nz/32+1,1)>>>(mom);
  cufftExecC2C(gradpar_plan_forward, mom, res, CUFFT_FORWARD);
  cufftExecC2C(gradpar_plan_inverse, res, res, CUFFT_INVERSE);
  reality_singlemom_kernel<<<dim3(32,32,1),dim3(grids_->Nx/32+1, grids_->Nz/32+1,1)>>>(res);
}

// FFT and |kz| operator for a single moment
void GradParallelPeriodic::abs_dz(cuComplex* mom, cuComplex* res)
{
  reality_singlemom_kernel<<<dim3(32,32,1),dim3(grids_->Nx/32+1, grids_->Nz/32+1,1)>>>(mom);
  cufftExecC2C(abs_gradpar_plan_forward, mom, res, CUFFT_FORWARD);
  cufftExecC2C(gradpar_plan_inverse, res, res, CUFFT_INVERSE);
  reality_singlemom_kernel<<<dim3(32,32,1),dim3(grids_->Nx/32+1, grids_->Nz/32+1,1)>>>(res);
}

// FFT only for a single moment
void GradParallelPeriodic::fft_only(cuComplex* mom, cuComplex* res, int dir)
{
  // use gradpar_plan_inverse since it does not multiply by i kz via callback 
  cufftExecC2C(gradpar_plan_inverse, mom, res, dir);
}

GradParallelLocal::GradParallelLocal(Grids* grids) :
  grids_(grids)
{
  dimBlock = 512;
  dimGrid = grids_->NxNycNz/dimBlock.x+1;
}

void GradParallelLocal::dz(MomentsG *G)
{
  G->scale(make_cuComplex(0.,1.));
}

// single moment
void GradParallelLocal::dz(cuComplex* mom, cuComplex* res) 
{
  scale_singlemom_kernel<<<dimGrid,dimBlock>>>(res, mom, make_cuComplex(0.,1.));
}

// single moment
void GradParallelLocal::abs_dz(cuComplex* mom, cuComplex* res) 
{
  scale_singlemom_kernel<<<dimGrid,dimBlock>>>(res, mom, make_cuComplex(1.,0.));
}

GradParallel1D::GradParallel1D(Grids* grids)
{
  // (theta) <-> (kpar)
  cufftCreate(&gradpar_plan_forward);
  cufftCreate(&gradpar_plan_inverse);

  // MFM: Plan for 1d FFT
  cufftPlan1d(&gradpar_plan_forward, grids_->Nz, CUFFT_R2C, 1);
  cufftPlan1d(&gradpar_plan_inverse, grids_->Nz, CUFFT_C2R, 1);

  cudaDeviceSynchronize();
  cufftXtSetCallback(gradpar_plan_forward, (void**) &i_kz_1d_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz);
  cudaDeviceSynchronize();

  cudaMalloc((void**) &b_complex, sizeof(cuComplex)*(grids_->Nz/2+1));
}

GradParallel1D::~GradParallel1D() {
  cufftDestroy(gradpar_plan_forward);
  cufftDestroy(gradpar_plan_inverse);
  cudaFree(b_complex);
}

void GradParallel1D::dz1D(float* b)
{
  cufftExecR2C(gradpar_plan_forward, b, b_complex);
  cufftExecC2R(gradpar_plan_inverse, b_complex, b);
}

