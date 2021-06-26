#pragma once
#include "parameters.h"
#include "grids.h"
#include "cufft.h"

class Fields {
  public:
  Fields(Parameters *pars, Grids *grids);
  ~Fields();
  
  cuComplex * phi    ;
  cuComplex * phi_h  ;
  cuComplex * apar   ;
  cuComplex * apar_h ;
  
  void print_phi(void);
  void print_apar(void);
  void rescale(float * phi_max);
  
  inline void copyPhiFrom(Fields* source) {
    cudaMemcpyAsync(phi, source->phi, size_, cudaMemcpyDeviceToDevice);
  }
  inline void copyAparFrom(Fields* source) {
    cudaMemcpyAsync(apar, source->apar, size_, cudaMemcpyDeviceToDevice);
  }
  
private:
  const size_t size_;
  int N;
  Parameters * pars_  ;
  Grids * grids_ ;
};
