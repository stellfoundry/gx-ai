#include "moments.h"
#include "cuda_constants.h"
#include "device_funcs.cu"

Moments::Moments(Grids* grids) : 
  grids_(grids), 
  HLsize_(sizeof(cuComplex)*grids_->NxNycNz*grids_->Nmoms), 
  Momsize_(sizeof(cuComplex)*grids->NxNycNz)
{
  int Nxyz = grids_->NxNycNz;
  int Nhermite = grids_->Nhermite;
  int Nlaguerre = grids_->Nlaguerre;
  ghl = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  dens_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  upar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  tpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  tprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  qpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  qprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);

  for(int s=0; s<grids->Nspecies; s++) {
    // allocate ghl array on device only
    cudaMalloc((void**) &ghl[s], HLsize_);
    cudaMemset(ghl[s], 0., HLsize_);

    // set up pointers for named moments that point to parts of ghl
    int l,m;
    l = 0, m = 0; // density
    if(l<Nhermite && m<Nlaguerre) dens_ptr[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];
    
    l = 1, m = 0; // u_parallel
    if(l<Nhermite && m<Nlaguerre) upar_ptr[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];
    
    l = 2, m = 0; // T_parallel / sqrt(2)
    if(l<Nhermite && m<Nlaguerre) tpar_ptr[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];
    
    l = 3, m = 0; // q_parallel / sqrt(6)
    if(l<Nhermite && m<Nlaguerre) qpar_ptr[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];

    l = 0, m = 1; // T_perp 
    if(l<Nhermite && m<Nlaguerre) tprp_ptr[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];
    
    l = 1, m = 1; // q_perp
    if(l<Nhermite && m<Nlaguerre) qprp_ptr[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];
  }

  cudaMalloc((void**) &nbar, Momsize_);
}

Moments::~Moments() {
  for(int s=0; s<grids_->Nspecies; s++) {
    cudaFree(ghl[s]);
  }
  cudaFree(nbar);
  free(ghl);
  free(dens_ptr);
  free(upar_ptr);
  free(tpar_ptr);
  free(tprp_ptr);
  free(qpar_ptr);
  free(qprp_ptr);
}


int Moments::initialConditions(Fields* fields, Parameters* pars, Geometry* geo) {
 
  cudaDeviceSynchronize(); // to make sure its safe to operate on host memory
  cuComplex* init_h = (cuComplex*) malloc(Momsize_);
  if(pars->init_single) {
    //initialize single mode
    int iky = pars->iky_single;
    int ikx = pars->ikx_single;
    float fac;
    if(pars->nlpm_test && iky==0) fac = .5;
    else fac = 1.;
    for(int iz=0; iz<grids_->Nz; iz++) {
      int index = iky + grids_->Nyc*ikx + grids_->NxNyc*iz;
      init_h[index].x = pars->init_amp*fac;
      init_h[index].y = 0.; //init_amp;
    }
  }
  else {
    printf("init_amp = %e\n", pars->init_amp);
    srand(22);
    float samp;
    for(int j=0; j<grids_->Nx; j++) {
      for(int i=0; i<grids_->Nyc; i++) {
    	samp = pars->init_amp;
  
          float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
          float rb = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
  
          //loop over z here to get rid of randomness in z in initial condition
          for(int k=0; k<grids_->Nz; k++) {
              int index = i + grids_->Nyc*j + grids_->NxNyc*k;
    	      init_h[index].x = samp*cos(pars->kpar_init*geo->z[k]/pars->Zp);
              init_h[index].y = samp*cos(pars->kpar_init*geo->z[k]/pars->Zp);
          }
      }
    }
  }

  // copy initial condition into device memory
  if(pars->init == DENS) {
    cudaMemcpy(dens_ptr[0], init_h, Momsize_, cudaMemcpyHostToDevice);

    // reality condition
    //operations_->reality(ghl);

    // mask
    //operations_->mask(ghl);
  }

  free(init_h);

  return cudaGetLastError();
}

__global__ void real_space_density(cuComplex* nbar, const cuComplex** __restrict__ ghl, Geometry::kperp2_struct* __restrict__ kp2_t) 
{
  unsigned int idxyz = get_id1();
  unsigned int idx = idxyz % (nx*nyc) % nx;
  unsigned int idy = idxyz % (nx*nyc) / nx;
  unsigned int idz = idxyz / (nx*nyc);

  if(idxyz<nx*nyc*nz) {
    #pragma unroll
    for(int is=0; is<nspecies; is++) {
      double b = kperp2(kp2_t, idx, idy, idz, is);
      #pragma unroll
      for(int m=0; m<nlaguerre; m++) {
        nbar[idxyz] = nbar[idxyz] + Jflr(m,b)*ghl[is][idxyz + m*nx*nyc*nz];
      }
    }
  }
}

//__global__ void qneutAdiab(cuComplex* phi, const __restrict__ cuComplex* nbar, const __restrict__ Geometry::kperp2_struct* kp2_t)
//{
//   
//}

int Moments::fieldSolve(Fields* fields, Parameters* pars, Geometry::kperp2_struct* kp2_t)
{
  if(pars->adiabatic_electrons) {
    cudaMemset(nbar, 0., Momsize_);
    real_space_density<<<1, grids_->NxNycNz>>>(nbar, (const cuComplex**) ghl, kp2_t);
    if(pars->iphi00==2) {
      dim3 dimBlock = dim3(32, 1, grids_->Nz);
      dim3 dimGrid = dim3(grids_->Nyc/dimBlock.x, grids_->Nx, 1);
      //qneutAdiab<<<>>>(fields->phi, nbar, kp2_t, geo->jacobian);
    }
  }
  return 0;
}

