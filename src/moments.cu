#include "moments.h"

Moments::Moments(Grids* grids) : grids_(grids), size_(grids_->NxNycNz*grids_->Nmoms)
{
  int Nxyz = grids_->NxNycNz;
  int Nhermite = grids_->Nhermite;
  int Nlaguerre = grids_->Nlaguerre;
  ghl = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  dens = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  upar = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  tpar = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  tprp = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  qpar = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  qprp = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);

  for(int s=0; s<grids->Nspecies; s++) {
    // allocate ghl array on device only
    cudaMalloc((void**) &ghl[s], size_);
    cudaMemset(ghl[s], 0., size_);

    // set up pointers for named moments that point to parts of ghl
    int l,m;
    l = 0, m = 0; // density
    if(l<Nhermite && m<Nlaguerre) dens[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];
    
    l = 1, m = 0; // u_parallel
    if(l<Nhermite && m<Nlaguerre) upar[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];
    
    l = 2, m = 0; // T_parallel / sqrt(2)
    if(l<Nhermite && m<Nlaguerre) tpar[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];
    
    l = 3, m = 0; // q_parallel / sqrt(6)
    if(l<Nhermite && m<Nlaguerre) qpar[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];

    l = 0, m = 1; // T_perp 
    if(l<Nhermite && m<Nlaguerre) tprp[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];
    
    l = 1, m = 1; // q_perp
    if(l<Nhermite && m<Nlaguerre) qprp[s] = &ghl[s][Nxyz*m + Nxyz*Nlaguerre*l];
  }
}

Moments::~Moments() {

}


