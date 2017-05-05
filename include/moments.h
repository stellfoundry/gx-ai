#pragma once

#include "grids.h"
#include "fields.h"
#include "parameters.h"
#include "geometry.h"

class Moments {
 public:
  Moments(Grids* grids);
  ~Moments();

  int initialConditions(Fields* fields, Parameters *pars, Geometry* geo);

  int add_scaled(double c1, Moments* m1, double c2, Moments* m2);
  
  inline void copyFrom(Moments* source) {
    cudaMemcpyAsync(ghl, source->ghl, HLsize_, cudaMemcpyDeviceToDevice);
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
