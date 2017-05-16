#include "linear.h"
#include "device_funcs.h"
#include "cufft.h"
#include "cufftXt.h"
#include "get_error.h"
#include "species.h"
#include "cuda_constants.h"

__global__ void rhs_linear(cuComplex *g, cuComplex* phi, float* b, float* iomegad, float* bgrad, float* ky, specie* s,
                           cuComplex* rhs_par, cuComplex* rhs);

__device__ void i_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  unsigned int idz = offset / (nx*nyc);
  cuComplex Ikz = make_cuComplex(0., kz[idz]);
  ((cuComplex*)dataOut)[offset] = Ikz*element;
}

__managed__ cufftCallbackStoreC i_kz_callbackPtr = i_kz;

Linear::Linear(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{
  mRhs_par = new Moments(grids);

  // set up z fft
  int n = grids_->Nz;
  int inembed = grids_->NxNycNz;
  int onembed = grids_->NxNycNz;
  size_t workSize;
  // (ky, kx, z) <-> (ky, kx, kz)
  cufftCreate(&ZDerivplanHL_forward);
  cufftCreate(&ZDerivplanHL_inverse);
  cufftMakePlanMany(ZDerivplanHL_forward, 1,   &n, &inembed, grids_->NxNyc, 1,
                              //  dim,  n,  isize,   istride,       idist,
                                &onembed, grids_->NxNyc, 1,     CUFFT_C2C, grids_->NxNyc, &workSize);
                              // osize,   ostride,       odist, type,      batchsize
  cufftMakePlanMany(ZDerivplanHL_inverse, 1,   &n, &inembed, grids_->NxNyc, 1,
                              //  dim,  n,  isize,   istride,       idist,
                                &onembed, grids_->NxNyc, 1,     CUFFT_C2C, grids_->NxNyc, &workSize);
                              // osize,   ostride,       odist, type,      batchsize
  // isize = size of input data
  // istride = distance between two elements in a batch = distance between (ky,kx,z=1) and (ky,kx,z=2) = Nx*(Ny/2+1)
  // idist = distance between first element of consecutive batches = distance between (ky=1,kx=1,z=1) and (ky=2,kx=1,z=1) = 1
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  // set up callback functions
  cufftXtSetCallback(ZDerivplanHL_forward, (void**) &i_kz_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  dimBlock = dim3(32, min(4, grids_->Nlaguerre), min(4, grids_->Nhermite));
  dimGrid = dim3(grids_->NxNycNz/dimBlock.x, 1, 1);
  sharedSize = 32*(grids_->Nlaguerre+2)*(grids_->Nhermite+4)*sizeof(cuComplex);
  printf("For linear RHS: sharedSize = %f KB\n", sharedSize/1024.);
}

Linear::~Linear()
{
  cufftDestroy(ZDerivplanHL);

  delete mRhs_par;
}

int Linear::zderiv(Moments* m)
{
  // FFT and derivative on parallel term
  // i*kz*ghl calculated via callback, defined as part of ZDerivplanHL_forward
  for(int i = 0; i < grids_->Nmoms*grids_->Nspecies; i++) {
    cufftExecC2C(ZDerivplanHL_forward, &m->ghl[grids_->NxNycNz*i], &m->ghl[grids_->NxNycNz*i], CUFFT_FORWARD);
    cufftExecC2C(ZDerivplanHL_inverse, &m->ghl[grids_->NxNycNz*i], &m->ghl[grids_->NxNycNz*i], CUFFT_INVERSE);
  }

  return 0;
}

int Linear::rhs(Moments* m, Fields* f, Moments* mRhs) {
  // calculate RHS
  rhs_linear<<<dimGrid, dimBlock, sharedSize>>>(m->ghl, f->phi, geo_->kperp2, geo_->omegad, geo_->bgrad, grids_->ky, pars_->species,
                                                mRhs_par->ghl, mRhs->ghl);

  // FFT and derivative on parallel term
  // i*kz*ghl calculated via callback, defined as part of ZDerivplanHL_forward
  // for now, loop over all l and m because cannot batch 
  // eventually will optimize by first transposing so that z is fastest index
  for(int i = 0; i < grids_->Nmoms*grids_->Nspecies; i++) {
    cufftExecC2C(ZDerivplanHL_forward, &mRhs_par->ghl[grids_->NxNycNz*i], &mRhs_par->ghl[grids_->NxNycNz*i], CUFFT_FORWARD);
    cufftExecC2C(ZDerivplanHL_inverse, &mRhs_par->ghl[grids_->NxNycNz*i], &mRhs_par->ghl[grids_->NxNycNz*i], CUFFT_INVERSE);
  }

  // combine
  mRhs->add_scaled(1., mRhs, (float) geo_->gradpar/grids_->Nz, mRhs_par);

  // closures... TO DO!

  return 0;
}

// main kernel function for calculating RHS
# define S_G(L, M) s_g[sidxyz + sDimx*(L) + sDimx*sDimy*(M)]
__global__ void rhs_linear(cuComplex *g, cuComplex* phi, float* b, float* omegad, float* bgrad, float* ky, specie* species,
                           cuComplex* rhs_par, cuComplex* rhs)
{
  extern __shared__ cuComplex s_g[];

  unsigned int idxyz = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int sidxyz = threadIdx.x;
  // these modulo operations are expensive... better way to get these indices?
  unsigned int idy = idxyz % (nx*nyc) % nyc; 
  unsigned int idz = idxyz / (nx*nyc);

  int sDimx = 32;
  int sDimy = nlaguerre+2;
  //int sDimz = nhermite+4;
  //int sDims_g = sDimx*sDimy*sDimz;

  // read these values into (hopefully) register memory. 
  // local to each thread (i.e. each idxyz).
  // since idxyz is linear, these accesses are coalesced.
  cuComplex phi_ = phi[idxyz];
  float b_ = b[idxyz];
  cuComplex iomegad_ = make_cuComplex(0., omegad[idxyz]);

  // all threads in a block will likely have same value of idz, so they will be reading same value of bgrad[idz].
  // if bgrad was in shared memory, would have bank conflicts.
  // no bank conflicts for reading from global memory though. 
  float bgrad_ = bgrad[idz];  

  // this is coalesced?
  cuComplex iomegastar_ = make_cuComplex(0., ky[idy]); 

 //#pragma unroll
 for(int is=0; is<nspecies; is++) { // might be a better way to handle species loop here...
  specie s = species[is];

  // species-specific constants
  float nu_ = s.nu_ss; 
  float tprim_ = s.tprim;
  float fprim_ = s.fprim;
  //const float rho_ = s.rho;

  // read tile of g into shared mem
  // each thread in the block reads in multiple values of l and m
  for (int l = threadIdx.z; l < nhermite; l += blockDim.z) {
   for (int m = threadIdx.y; m < nlaguerre; m += blockDim.y) {
    int globalIdx = idxyz + nx*nyc*nz*m + nx*nyc*nz*nlaguerre*l + nx*nyc*nz*nlaguerre*nhermite*is; 
    int sl = l + 2;
    int sm = m + 1;
    S_G(sl, sm) = g[globalIdx];
   }
  }

  __syncthreads();

  // set up ghost cells in l (for all m's)
  for (int m = threadIdx.y; m < nlaguerre; m += blockDim.y) {
    int sm = m + 1;
    int sl = threadIdx.z + 2;
    if(sl < 4) {
      // set ghost to zero at low l
      S_G(sl-2, sm) = make_cuComplex(0., 0.);

      // set ghost with closures at high l
      S_G(sl+nhermite, sm) = make_cuComplex(0., 0.);
    }
  }

  // set up ghost cells in m (for all l's)
  for (int l = threadIdx.z; l < nhermite+2; l += blockDim.z) {
    int sl = l + 1; // this takes care of corners...
    int sm = threadIdx.y + 1;
    if(sm < 2) {
      // set ghost to zero at low m
      S_G(sl, sm-1) = make_cuComplex(0., 0.);

      // set ghost with closures at high m
      S_G(sl, sm+nlaguerre) = make_cuComplex(0., 0.);
    }
  }

  __syncthreads();

  // stencil (on non-ghost cells)
  for (int l = threadIdx.z; l < nhermite; l += blockDim.z) {
   for (int m = threadIdx.y; m < nlaguerre; m += blockDim.y) {
    int globalIdx = idxyz + nx*nyc*nz*m + nx*nyc*nz*nlaguerre*l + nx*nyc*nz*nlaguerre*nhermite*is; 
    int sl = l + 2;
    int sm = m + 1;

    // need to calculate parallel terms separately because need to take derivative via fft 
    rhs_par[globalIdx] = -( sqrtf(l+1)*S_G(sl+1,sm) + sqrtf(l)*S_G(sl-1,sm) );

    // remaining terms
    rhs[globalIdx] = 
     - bgrad_ * ( -sqrtf(l+1)*(m+1)*S_G(sl+1,sm) - sqrtf(l+1)*m*S_G(sl+1,sm-1)

                  + sqrtf(l)*m*S_G(sl-1,sm) + sqrtf(l)*(m+1)*S_G(sl-1,sm+1) )

        - iomegad_ * ( sqrtf((l+1)*(l+2))*S_G(sl+2,sm) + sqrtf(l*(l-1))*S_G(sl-2,sm)

                      + (m+1)*S_G(sl,sm+1) + m*S_G(sl,sm-1) + 2.*(l+m+1)*S_G(sl,sm) );

         - nu_ * ( b_ + l + 2*m ) * ( S_G(sl,sm) + Jflr(m, b_)*phi_ );

    // add drive and conservation terms
    if(l==0) {
      rhs[globalIdx] = rhs[globalIdx] + phi_*(
                Jflr(m-1,b_)*( -m*iomegad_ + m*tprim_*iomegastar_ ) 
              + Jflr(m,b_) * ( -2*(m+1)*iomegad_ + (fprim_ + tprim_*2*m)*iomegastar_ )
              + Jflr(m+1,b_)*( -(m+1)*iomegad_ + (m+1)*tprim_*iomegastar_ ) );  
    }
    if(l==1) {
      rhs_par[globalIdx] = rhs_par[globalIdx] - Jflr(m,b_)*phi_;
      rhs[globalIdx] = rhs[globalIdx] - phi_ * ( m*Jflr(m,b_) + (m+1)*Jflr(m+1,b_) ) * bgrad_;
    }
    if(l==2) {
      rhs[globalIdx] = rhs[globalIdx] + phi_ * Jflr(m,b_) * (-2*iomegad_ + tprim_*iomegastar_);
    }
   }

  }

 } // species loop
}
