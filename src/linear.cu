#include "linear.h"
#include "device_funcs.h"
#include "cufft.h"
#include "get_error.h"
#include "species.h"
#include "cuda_constants.h"

__global__ void rhs_linear(cuComplex *g, cuComplex* phi, cuComplex* upar_bar,
			   cuComplex* uperp_bar, cuComplex* t_bar,
			   float* b, float* cv_d, float* gb_d, float* bgrad,
			   float* ky, specie* s, cuComplex* rhs_par, cuComplex* rhs);

__global__ void conservation_terms(cuComplex* upar_bar, cuComplex* uperp_bar,
				   cuComplex* t_bar, cuComplex* G, cuComplex* phi,
				   float *b, specie* species);

__global__ void hypercollisions(cuComplex* g, float nu_hyper_l, float nu_hyper_m,
				int p_hyper_l, int p_hyper_m, cuComplex* rhs);


Linear::Linear(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{
  GRhs_par = new MomentsG(pars_, grids_);

  // set up parallel ffts
  if(pars_->local_limit) {
    DEBUGPRINT("Using local limit for grad parallel.\n");
    grad_par = new GradParallelLocal(grids_);
  }
  else if(pars_->boundary_option_periodic) {
    DEBUGPRINT("Using periodic for grad parallel.\n");
    grad_par = new GradParallelPeriodic(grids_);
  }
  else {
    DEBUGPRINT("Using twist-and-shift for grad parallel.\n");
    grad_par = new GradParallelLinked(grids_, pars_->jtwist);
  }
 
  if(pars_->closure_model_opt==BEER42) {
    DEBUGPRINT("Initializing Beer 4+2 closures\n");
    closures = new Beer42(grids_, geo_, grad_par);
  } else if (pars_->closure_model_opt==SMITHPERP) {
    DEBUGPRINT("Initializing Smith perpendicular toroidal closures\n");
    closures = new SmithPerp(grids_, geo_, pars_->smith_perp_q, pars_->smith_perp_w0);
  } else if (pars_->closure_model_opt == SMITHPAR) {
    DEBUGPRINT("Initializing Smith parallel closures\n");
    closures = new SmithPar(grids_, geo_, grad_par, pars_->smith_par_q);
  }


  // allocate conservation terms for collision operator
  int size = sizeof(cuComplex)*grids_->NxNycNz*grids_->Nspecies;
  cudaMalloc((void**) &upar_bar, size);
  cudaMalloc((void**) &uperp_bar, size);
  cudaMalloc((void**) &t_bar, size);
  cudaMemset(upar_bar, 0., size);
  cudaMemset(uperp_bar, 0., size);
  cudaMemset(t_bar, 0., size);

  // set up CUDA grids for main linear kernel
  // NOTE: dimBlock.x = sharedSize.x = 32 gives best performance, but using 8 is only 5% worse.
  // this allows use of 4x more LH resolution without changing shared memory layouts.
  // dimBlock = dim3(8, min(4, grids_->Nl), min(4, grids_->Nm));
  dimBlock = dim3(8, min(4, grids_->Nl), min(4, grids_->Nm));
  dimGrid = dim3(grids_->NxNycNz/dimBlock.x+1, 1, 1);
  sharedSize = dimBlock.x*(grids_->Nl+2)*(grids_->Nm+4)*sizeof(cuComplex);
  DEBUGPRINT("For linear RHS: size of shared memory block = %f KB\n", sharedSize/1024.);
  if(sharedSize/1024.>48.) {
    printf("Error: currently cannot support this velocity resolution due to shared memory constraints.\n");
    printf("size of shared memory block must be less than 48 KB, so make sure (nm+4)*(nlaaguerre+2)<%d.\n", 48*1024/8/dimBlock.x);
    exit(1);
  }
}

Linear::~Linear()
{
  if(pars_->closure_model_opt>0) delete closures;
  delete grad_par;
  delete GRhs_par;
}

int Linear::rhs(MomentsG* G, Fields* f, MomentsG* GRhs) {

  // calculate conservation terms for collision operator
  conservation_terms<<<grids_->NxNycNz/256+1, 256>>>
	(upar_bar, uperp_bar, t_bar, G->G(), f->phi, geo_->kperp2, pars_->species);

  // calculate RHS
  rhs_linear<<<dimGrid, dimBlock, sharedSize>>>
      	(G->G(), f->phi, upar_bar, uperp_bar, t_bar,
        geo_->kperp2, geo_->cv_d, geo_->gb_d, geo_->bgrad, 
       	grids_->ky, pars_->species, GRhs_par->G(), GRhs->G());

  // hypercollisions
  if(pars_->hypercollisions) {
    hypercollisions<<<dimGrid,dimBlock>>>(G->G(), pars_->nu_hyper_l, pars_->nu_hyper_m,
					  pars_->p_hyper_l, pars_->p_hyper_m, GRhs->G());
  }

  // parallel gradient term
  grad_par->dz(GRhs_par);

  // combine
  GRhs->add_scaled(1., GRhs, (float) geo_->gradpar, GRhs_par);

  // closures
  if(pars_->closure_model_opt>0) {
    closures->apply_closures(G, GRhs);
  }

  return 0;
}

// main kernel function for calculating RHS
# define S_G(L, M) s_g[sidxyz + (sDimx)*(L) + (sDimx)*(sDimy)*(M)]
__global__ void rhs_linear(cuComplex *g, cuComplex* phi, 
	cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar,
	float* kperp2, float* cv_d, float*gb_d, float* bgrad, float* ky, specie* species,
	cuComplex* rhs_par, cuComplex* rhs)
{
  extern __shared__ cuComplex s_g[]; // aliased below by macro S_G, defined above

  unsigned int idxyz = threadIdx.x + blockIdx.x*blockDim.x;
  if(idxyz<nx*nyc*nz) {
    const unsigned int sidxyz = threadIdx.x;
    // these modulo operations are expensive... better way to get these indices?
    const unsigned int idy = idxyz % (nx*nyc) % nyc; 
    const unsigned int idz = idxyz / (nx*nyc);
  
    // shared memory blocks of size blockDim.x * (nl+2) * (nm+4)
    const int sDimx = blockDim.x;
    const int sDimy = nl+2;
  
    // read these values into (hopefully) register memory. 
    // local to each thread (i.e. each idxyz).
    // since idxyz is linear, these accesses are coalesced.
    const cuComplex phi_ = phi[idxyz];
  
    // all threads in a block will likely have same value of idz, so they will be reading same value of bgrad[idz].
    // if bgrad were in shared memory, would have bank conflicts.
    // no bank conflicts for reading from global memory though. 
    const float bgrad_ = bgrad[idz];  
  
    // this is coalesced?
    const cuComplex iky_ = make_cuComplex(0., ky[idy]); 
  
   //#pragma unroll
   for(int is=0; is<nspecies; is++) { // might be a better way to handle species loop here...
    specie s = species[is];

    // species-specific constants
    const float vt_ = s.vt;
    const float zt_ = s.zt;
    const float nu_ = s.nu_ss; 
    const float tprim_ = s.tprim;
    const float fprim_ = s.fprim;
    const float b_s = kperp2[idxyz] * s.rho2;
    const cuComplex icv_d_s = 2. * s.tz * make_cuComplex(0., cv_d[idxyz]);
    const cuComplex igb_d_s = 2. * s.tz * make_cuComplex(0., gb_d[idxyz]);

    // conservation terms (species-specific)
    cuComplex upar_bar_  =  upar_bar[idxyz + is*nx*nyc*nz];
    cuComplex uperp_bar_ = uperp_bar[idxyz + is*nx*nyc*nz];
    cuComplex t_bar_     =     t_bar[idxyz + is*nx*nyc*nz];
  
    // read tile of g into shared mem
    // each thread in the block reads in multiple values of l and m
    for (int m = threadIdx.z; m < nm; m += blockDim.z) {
     for (int l = threadIdx.y; l < nl; l += blockDim.y) {
      int globalIdx = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*is; 
      int sl = l + 1;
      int sm = m + 2;
      S_G(sl, sm) = g[globalIdx];
     }
    }
  
    // this syncthreads is not necessary unless ghosts require information from interior cells
    //__syncthreads();
  
    // set up ghost cells in m (for all l's)
    for (int l = threadIdx.y; l < nl; l += blockDim.y) {
      int sl = l + 1;
      int sm = threadIdx.z + 2;
      if(sm < 4) {
        // set ghost to zero at low m
        S_G(sl, sm-2) = make_cuComplex(0., 0.);
  
        // set ghost with closures at high m
        S_G(sl, sm+nm) = make_cuComplex(0., 0.);
      }
    }
  
    // set up ghost cells in l (for all m's)
    for (int m = threadIdx.z; m < nm+2; m += blockDim.z) {
      int sm = m + 1; // this takes care of corners...
      int sl = threadIdx.y + 1;
      if(sl < 2) {
        // set ghost to zero at low l
        S_G(sl-1, sm) = make_cuComplex(0., 0.);
  
        // set ghost with closures at high l
        S_G(sl+nl, sm) = make_cuComplex(0., 0.);
      }
    }
  
    __syncthreads();
  
    // stencil (on non-ghost cells)
    for (int m = threadIdx.z; m < nm; m += blockDim.z) {
     for (int l = threadIdx.y; l < nl; l += blockDim.y) {
      int globalIdx = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*is; 
      int sl = l + 1; // offset to get past ghosts
      int sm = m + 2; // offset to get past ghosts
  
      // need to calculate parallel terms separately because need to take derivative via fft 
      rhs_par[globalIdx] = -vt_*( sqrtf(m+1)*S_G(sl,sm+1) + sqrtf(m)*S_G(sl,sm-1) );
  
      // remaining terms
      rhs[globalIdx] = 
       - vt_ * bgrad_ * ( - sqrtf(m+1)*(l+1)*S_G(sl,sm+1) - sqrtf(m+1)* l   *S_G(sl-1,sm+1)  
                          + sqrtf(m  )* l   *S_G(sl,sm-1) + sqrtf(m  )*(l+1)*S_G(sl+1,sm-1) )
  
	- icv_d_s * ( sqrtf((m+1)*(m+2))*S_G(sl,sm+2) + 2.*m*S_G(sl,sm) + sqrtf(m*(m-1))*S_G(sl,sm-2) )
	- igb_d_s * (              (l+1)*S_G(sl+1,sm) + 2.*(l+1)*S_G(sl,sm)          + l*S_G(sl-1,sm) )
  
           - nu_ * ( b_s + 2*l + m ) * ( S_G(sl,sm) );
  
      // add potential, drive, and conservation terms in low hermite moments
      if(m==0) {
        rhs[globalIdx] = rhs[globalIdx] + phi_ * (
            Jflr(l-1,b_s)*(      -l *igb_d_s * zt_ +           tprim_  *l  * iky_ )
	  + Jflr(l,  b_s)*( -2*(l+1)*igb_d_s * zt_ + (fprim_ + tprim_*2*l) * iky_ )
	  + Jflr(l+1,b_s)*(   -(l+1)*igb_d_s * zt_ )
	  + Jflr(l+1,b_s,false)*                              tprim_*(l+1) * iky_ )
	  + nu_ * sqrtf(b_s) * ( Jflr(l, b_s) + Jflr(l-1, b_s) ) * uperp_bar_
	  + nu_ * 2. * ( l*Jflr(l-1,b_s) + 2.*l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) ) * t_bar_ ;
      }
 //  - nu_ * ( b_s + 2*l ) * Jflr(l, b_s) * phi_ * zt_ ; // BD What was this doing here?!

      if(m==1) {
        rhs_par[globalIdx] = rhs_par[globalIdx] - Jflr(l,b_s)*phi_ * zt_ * vt_;

        rhs[globalIdx] = rhs[globalIdx] - phi_ * (
	          l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) ) * bgrad_ * vt_ * zt_
      		+ nu_ * Jflr(l,b_s) * upar_bar_;
      }
      if(m==2) {
        rhs[globalIdx] = rhs[globalIdx] + phi_ *
	          Jflr(l,b_s) * (-2*icv_d_s * zt_ + tprim_ * iky_)/sqrtf(2) 
		+ nu_ * sqrtf(2) * Jflr(l,b_s) * t_bar_;
      }  
     } // l loop
    } // m loop
  
   } // species loop
  } // idxyz < NxNycNz
}

# define H_(XYZ, L, M, S) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M) + nx*nyc*nz*nl*nm*(S)] + Jflr(L,b_s)*phi_*zt_
# define G_(XYZ, L, M, S) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M) + nx*nyc*nz*nl*nm*(S)] // H = G, except for m = 0
// C = C(H) but H and G are the same function for all m!=0. Our main array defines g so the correction to produce
// H is only appropriate for m=0. In other words, the usage here is basically handling the delta_{m0} terms
// in a clumsy way
__global__ void conservation_terms(cuComplex* upar_bar, cuComplex* uperp_bar, cuComplex* t_bar, cuComplex* g, cuComplex* phi, float *kperp2, specie* species)
{
  unsigned int idxyz = get_id1();

  if(idxyz<nx*nyc*nz) {
    cuComplex phi_ = phi[idxyz];
    for(int is=0; is<nspecies; is++) {
      const float zt_ = species[is].zt;
      int index = idxyz + nx*nyc*nz*is;
      upar_bar[index] = make_cuComplex(0., 0.);
      uperp_bar[index] = make_cuComplex(0., 0.);
      t_bar[index] = make_cuComplex(0., 0.);
      float b_s = kperp2[idxyz]*species[is].rho2;
      // sum over l
      for(int l=0; l<nl; l++) {
        upar_bar[index] = upar_bar[index] + Jflr(l,b_s)*G_(idxyz, l, 1, is);
        // H_(...) is defined by macro above. Only use H here for m=0. Confusing!
        uperp_bar[index] = uperp_bar[index] + (Jflr(l,b_s) + Jflr(l-1,b_s))*H_(idxyz, l, 0, is);

        // energy conservation correction for nlaguerre = 1
        if (nl == 1) {
            t_bar[index] = t_bar[index] + sqrtf(2.)*Jflr(l,b_s)*G_(idxyz, l, 2, is);
        } else {
            t_bar[index] = t_bar[index] + sqrtf(2.)/3.*Jflr(l,b_s)*G_(idxyz, l, 2, is)
		    + 2./3.*( l*Jflr(l-1,b_s) + 2.*l*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) )*H_(idxyz, l, 0, is);
        }
      }
      uperp_bar[index] = uperp_bar[index]*sqrtf(b_s);
    }
  }
}

__global__ void hypercollisions(cuComplex* g, float nu_hyper_l, float nu_hyper_m, int p_hyper_l, int p_hyper_m, cuComplex* rhs) {
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
   for(int is=0; is<nspecies; is++) { 
    for (int m = threadIdx.z; m < nm; m += blockDim.z) {
     for (int l = threadIdx.y; l < nl; l += blockDim.y) {
      int globalIdx = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*is; 
      if(m>2) {
        rhs[globalIdx] = rhs[globalIdx] -
	  (nu_hyper_l*pow((float) l/nl, (float) p_hyper_l)+nu_hyper_m*pow((float)m/nm, p_hyper_m))*g[globalIdx];
      }
     }
    }
   }
  }
}
