#pragma once

#include "grids.h"
#include "fields.h"
#include "parameters.h"
#include "geometry.h"

class Moments {
 public:
  Moments(Grids* grids);
  ~Moments();

  // accessor function to get pointer to specific l,m,s of ghl array
  cuComplex* gHL(int l, int m, int s=0) {
    return &ghl[grids_->NxNycNz*m + grids_->NxNycNz*grids_->Nlaguerre*l + grids_->NxNycNz*grids_->Nmoms*s];
    // glm[ky, kx, z]
  }
  // ghl[ky, kx, z, m, l, s]
  // ghl[ky + nyc*kx + nx*nyc*z + nx*nyc*nz*m + nx*nyc*nz*nlaguerre*l + nx*nyc*nz*nlaguerre*nhermite*s]

  int initialConditions(Parameters *pars, Geometry* geo);

  int add_scaled(double c1, Moments* m1, double c2, Moments* m2);
  int add_scaled(double c1, Moments* m1, double c2, Moments* m2, 
                 double c3, Moments* m3, double c4, Moments* m4,
                 double c5, Moments* m5);

  int zero();
  int zero(int l, int m, int s=0);

  int scale(double scalar);
  int scale(cuComplex scalar);

  int reality();
  
  inline void copyFrom(Moments* source) {
    cudaMemcpy(ghl, source->ghl, HLsize_, cudaMemcpyDeviceToDevice);
  }
 
  cuComplex* ghl;
  cuComplex** dens_ptr;
  cuComplex** upar_ptr;
  cuComplex** tpar_ptr;
  cuComplex** tprp_ptr;
  cuComplex** qpar_ptr;
  cuComplex** qprp_ptr;
 
  dim3 dimGrid, dimBlock;

 private:
  const Grids* grids_;
  const size_t HLsize_;
  const size_t Momsize_;
};
