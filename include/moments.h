#pragma once

#include "grids.h"
#include "fields.h"
#include "parameters.h"
#include "geometry.h"

class MomentsG {
 public:
  MomentsG(Grids* grids);
  ~MomentsG();

  // accessor function to get pointer to specific l,m,s of G array
  // calling with no arguments gives pointer to beginning of G_lm
  cuComplex* G(int l=0, int m=0, int s=0) {
    return &G_lm[grids_->NxNycNz*l + grids_->NxNycNz*grids_->Nl*m + grids_->NxNycNz*grids_->Nmoms*s];
    // glm[ky, kx, z]
  }

  int initialConditions(Parameters *pars, Geometry* geo);

  int add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2);
  int add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, 
                 double c3, MomentsG* G3, double c4, MomentsG* G4,
                 double c5, MomentsG* G5);

  int zero();
  int zero(int l, int m, int s=0);

  int scale(double scalar);
  int scale(cuComplex scalar);

  int reality();
  
  inline void copyFrom(MomentsG* source) {
    cudaMemcpy(this->G(), source->G(), LHsize_, cudaMemcpyDeviceToDevice);
  }
 
  cuComplex** dens_ptr;
  cuComplex** upar_ptr;
  cuComplex** tpar_ptr;
  cuComplex** tprp_ptr;
  cuComplex** qpar_ptr;
  cuComplex** qprp_ptr;
 
  dim3 dimGrid, dimBlock;

 private:
  cuComplex* G_lm;
  const Grids* grids_;
  const size_t LHsize_;
  const size_t Momsize_;
};
