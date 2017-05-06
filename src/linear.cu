#include "linear.h"
#include "device_funcs.h"
#include "cufft.h"

__global__ void rhs_linear(cuComplex *g, cuComplex* phi, cuComplex* rhs_par, cuComplex* rhs);

Linear::Linear(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{
  mRhs_par = new Moments(grids);

  dimBlock = dim3(32, min(4, grids_->Nlaguerre), min(4, grids_->Nhermite));
  dimGrid = dim3(grids_->NxNycNz/dimBlock.x, 1, 1);
  sharedSize = 32*(grids_->Nlaguerre+2)*(grids_->Nhermite+4)*sizeof(cuComplex);
  printf("sharedSize = %f KB\n", sharedSize/1024.);
}

Linear::~Linear()
{
  delete mRhs_par;
}

int Linear::rhs(Moments* m, Fields* f, Moments* mRhs) {
  // calculate RHS
  rhs_linear<<<dimGrid, dimBlock, sharedSize>>>(m->ghl, f->phi, mRhs_par->ghl, mRhs->ghl);

  // FFT and derivative on parallel term

  // combine
  mRhs->add_scaled(1., mRhs, 1., mRhs_par);
  return 0;
}

// main kernel function for calculating RHS
# define S_G(L, M) s_g[sidxyz + sDimx*L + sDimx*sDimy*M]
__global__ void rhs_linear(cuComplex *g, cuComplex* phi, 
                           cuComplex* rhs_par, cuComplex* rhs)
{
  extern __shared__ cuComplex s_g[];

  unsigned int idxyz = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int sidxyz = threadIdx.x;
  // these modulo operations are expensive... better way to get these indices?
  unsigned int idy = idxyz % (nx*nyc) % nx; 
  unsigned int idz = idxyz / (nx*nyc);

  int sDimx = 32;
  int sDimy = nlaguerre+2;
  //int sDimz = nhermite+4;
  //int sDims_g = sDimx*sDimy*sDimz;

  // read these values into (hopefully) register memory. 
  // local to each thread (i.e. each idxyz).
  // since idxyz is linear, these accesses are coalesced.
  const cuComplex phi_ = phi[idxyz];
  const float b_ = 0.;//b[idxyz];
  const cuComplex iomegad_ = make_cuFloatComplex(0., 0.);//omegad[idxyz]};

  // all threads in a block will likely have same value of idz, so they will be reading same value of bgrad[idz].
  // if bgrad was in shared memory, would have bank conflicts.
  // no bank conflicts for reading from global memory though. 
  const float bgrad_ = 0.;// bgrad[idz];  

  // this is coalesced?
  const cuComplex iomegastar_ = make_cuFloatComplex(0., 0.);// ky[idy]}; 

 #pragma unroll
 for(int is=0; is<nspecies; is++) { // might be a better way to handle species loop here...

  // species-specific constants
  const float nu_ = 0.; //species[is].nu;
  const float tprim_ = 0.;// species[is].tprim;
  const float fprim_ = 0.;//species[is].fprim;

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
      S_G(sl-2, sm) = make_cuFloatComplex(0., 0.);

      // set ghost with closures at high l
      S_G(sl+nhermite, sm) = make_cuFloatComplex(0., 0.);
    }
  }

  // set up ghost cells in m (for all l's)
  for (int l = threadIdx.z; l < nhermite; l += blockDim.z) {
    int sl = l + 2;
    int sm = threadIdx.y + 1;
    if(sm < 2) {
      // set ghost to zero at low m
      S_G(sl, sm-1) = make_cuFloatComplex(0., 0.);

      // set ghost with closures at high m
      S_G(sl, sm+nlaguerre) = make_cuFloatComplex(0., 0.);
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

                      + (m+1)*S_G(sl,sm+1) + m*S_G(sl,sm-1) + 2.*(l+m+1)*S_G(sl,sm) )

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
