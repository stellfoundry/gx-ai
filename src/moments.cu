#include "moments.h"
#include "device_funcs.h"
#include "get_error.h"

Moments::Moments(Grids* grids) : 
  grids_(grids), 
  HLsize_(sizeof(cuComplex)*grids_->NxNycNz*grids_->Nmoms*grids_->Nspecies), 
  Momsize_(sizeof(cuComplex)*grids->NxNycNz)
{
  int Nxyz = grids_->NxNycNz;
  int Nhermite = grids_->Nhermite;
  int Nlaguerre = grids_->Nlaguerre;
  int Nmoms = grids_->Nmoms;
  checkCuda(cudaMalloc((void**) &ghl, HLsize_));
  dens_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  upar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  tpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  tprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  qpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  qprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);

  cudaMemset(ghl, 0., HLsize_);

  printf("Allocated a ghl array of size %.2f MB\n", HLsize_/1024./1024.);

  for(int s=0; s<grids->Nspecies; s++) {
    // set up pointers for named moments that point to parts of ghl
    int l,m;
    l = 0, m = 0; // density
    if(l<Nhermite && m<Nlaguerre) dens_ptr[s] = &ghl[Nxyz*m + Nxyz*Nlaguerre*l + Nxyz*Nmoms*s];
    
    l = 1, m = 0; // u_parallel
    if(l<Nhermite && m<Nlaguerre) upar_ptr[s] = &ghl[Nxyz*m + Nxyz*Nlaguerre*l + Nxyz*Nmoms*s];
    
    l = 2, m = 0; // T_parallel / sqrt(2)
    if(l<Nhermite && m<Nlaguerre) tpar_ptr[s] = &ghl[Nxyz*m + Nxyz*Nlaguerre*l + Nxyz*Nmoms*s];
    
    l = 3, m = 0; // q_parallel / sqrt(6)
    if(l<Nhermite && m<Nlaguerre) qpar_ptr[s] = &ghl[Nxyz*m + Nxyz*Nlaguerre*l + Nxyz*Nmoms*s];

    l = 0, m = 1; // T_perp 
    if(l<Nhermite && m<Nlaguerre) tprp_ptr[s] = &ghl[Nxyz*m + Nxyz*Nlaguerre*l + Nxyz*Nmoms*s];
    
    l = 1, m = 1; // q_perp
    if(l<Nhermite && m<Nlaguerre) qprp_ptr[s] = &ghl[Nxyz*m + Nxyz*Nlaguerre*l + Nxyz*Nmoms*s];
  }

  dimBlock = dim3(32, min(4, Nlaguerre), min(4, Nhermite));
  dimGrid = dim3(grids_->NxNycNz/dimBlock.x, 1, 1);
}

Moments::~Moments() {
  cudaFree(ghl);
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

  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  return cudaGetLastError();
}

__global__ void add_scaled_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int l = threadIdx.z; l < nhermite; l += blockDim.z) {
        for (int m = threadIdx.y; m < nlaguerre; m += blockDim.y) {
          int globalIdx = idxyz + nx*nyc*nz*m + nx*nyc*nz*nlaguerre*l + nx*nyc*nz*nlaguerre*nhermite*s; 
          res[globalIdx] = c1*m1[globalIdx] + c2*m2[globalIdx];
        }
      }
    }
  }
}

int Moments::add_scaled(double c1, Moments* m1, double c2, Moments* m2) {
  add_scaled_kernel<<<dimGrid,dimBlock>>>(ghl, c1, m1->ghl, c2, m2->ghl);
  return 0;
}
