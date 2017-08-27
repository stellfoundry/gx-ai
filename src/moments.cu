#include "moments.h"
#include "device_funcs.h"
#include "get_error.h"
#include "cuda_constants.h"

MomentsG::MomentsG(Grids* grids) : 
  grids_(grids), 
  LHsize_(sizeof(cuComplex)*grids_->NxNycNz*grids_->Nmoms*grids_->Nspecies), 
  Momsize_(sizeof(cuComplex)*grids->NxNycNz)
{
  int Nm = grids_->Nm;
  int Nl = grids_->Nl;
  checkCuda(cudaMalloc((void**) &G_lm, LHsize_));
  dens_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  upar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  tpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  tprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  qpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  qprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);

  cudaMemset(G_lm, 0., LHsize_);

  printf("Allocated a G_lm array of size %.2f MB\n", LHsize_/1024./1024.);

  for(int s=0; s<grids->Nspecies; s++) {
    // set up pointers for named moments that point to parts of G_lm
    int l,m;
    l = 0, m = 0; // density
    if(l<Nl && m<Nm) dens_ptr[s] = G(l,m,s);
    
    l = 0, m = 1; // u_parallel
    if(l<Nl && m<Nm) upar_ptr[s] = G(l,m,s);
    
    l = 0, m = 2; // T_parallel / sqrt(2)
    if(l<Nl && m<Nm) tpar_ptr[s] = G(l,m,s);
    
    l = 0, m = 3; // q_parallel / sqrt(6)
    if(l<Nl && m<Nm) qpar_ptr[s] = G(l,m,s);

    l = 1, m = 0; // T_perp 
    if(l<Nl && m<Nm) tprp_ptr[s] = G(l,m,s);
    
    l = 1, m = 1; // q_perp
    if(l<Nl && m<Nm) qprp_ptr[s] = G(l,m,s);
  }

  dimBlock = dim3(32, min(4, Nl), min(4, Nm));
  dimGrid = dim3(grids_->NxNycNz/dimBlock.x+1, 1, 1);
}

MomentsG::~MomentsG() {
  free(dens_ptr);
  free(upar_ptr);
  free(tpar_ptr);
  free(tprp_ptr);
  free(qpar_ptr);
  free(qprp_ptr);
  cudaFree(G_lm);
}

int MomentsG::initialConditions(Parameters* pars, Geometry* geo) {
 
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
    for(int i=0; i<grids_->Nyc; i++) {
      for(int j=0; j<grids_->Nx; j++) {
    	samp = pars->init_amp;
  
          float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
          //float rb = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
  
          //loop over z here to get rid of randomness in z in initial condition
          for(int k=0; k<grids_->Nz; k++) {
              int index = i + grids_->Nyc*j + grids_->NxNyc*k;
    	      init_h[index].x = ra*cos(pars->kpar_init*geo->z_h[k]/pars->Zp);
              init_h[index].y = ra*cos(pars->kpar_init*geo->z_h[k]/pars->Zp);
          }
      }
    }
  }
 
  // reality condition
//  for(int j=0; j<grids_->Nx/2+1; j++) {
//    for(int k=0; k<grids_->Nz; k++) {
//        int index = 0 + (grids_->Ny/2+1)*j + grids_->Nx*(grids_->Ny/2+1)*k;
//	int index2 = 0 + (grids_->Ny/2+1)*(grids_->Nx-j) + grids_->Nx*(grids_->Ny/2+1)*k;
//	init_h[index2].x = init_h[index].x;
//	init_h[index2].y = -init_h[index].y;
//    }
//  }

  // copy initial condition into device memory
  if(pars->init == DENS) {
    cudaMemcpy(dens_ptr[0], init_h, Momsize_, cudaMemcpyHostToDevice);
  }
  if(pars->init == UPAR) {
    cudaMemcpy(upar_ptr[0], init_h, Momsize_, cudaMemcpyHostToDevice);
  }
    // reality condition
  this->reality();
  
    // mask
    //operations_->mask(G_lm);

  free(init_h);

  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  return cudaGetLastError();
}

int MomentsG::zero() {
  cudaMemset(G_lm, 0., LHsize_);
  return 0;
}

int MomentsG::zero(int l, int m, int s) {
  cudaMemset(G(l,m,s), 0., Momsize_);
  return 0;
}

int MomentsG::scale(double scalar) {
  scale_kernel<<<dimGrid,dimBlock>>>(G_lm, G_lm, scalar);
  return 0;
}

int MomentsG::scale(cuComplex scalar) {
  scale_kernel<<<dimGrid,dimBlock>>>(G_lm, G_lm, scalar);
  return 0;
}

int MomentsG::add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2) {
  add_scaled_kernel<<<dimGrid,dimBlock>>>(G_lm, c1, G1->G_lm, c2, G2->G_lm);
  return 0;
}

int MomentsG::add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, 
                 double c3, MomentsG* G3, double c4, MomentsG* G4,
                 double c5, MomentsG* G5)
{
  add_scaled_kernel<<<dimGrid,dimBlock>>>(G_lm, c1, G1->G_lm, c2, G2->G_lm,
                                     c3, G3->G_lm, c4, G4->G_lm, c5, G5->G_lm);
  return 0;
  
}

int MomentsG::reality() 
{
  reality_kernel<<<dimGrid,dimBlock>>>(G_lm);
  return 0;
}
