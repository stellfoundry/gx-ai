#pragma once

#include "cufft.h"
#include "grids.h"

class Fields {
  public:
  Fields(Parameters *pars, Grids* grids);
  ~Fields();
  
  cuComplex *phi, *phi_h;
  cuComplex *apar, *apar_h;
  
  void print_phi(void);
  void print_apar(void);
  
  inline void copyPhiFrom(Fields* source) {
    cudaMemcpyAsync(phi, source->phi, size_, cudaMemcpyDeviceToDevice);
  }
  inline void copyAparFrom(Fields* source) {
    cudaMemcpyAsync(apar, source->apar, size_, cudaMemcpyDeviceToDevice);
  }
  
private:
  const size_t size_;
  int N;
  Grids* grids_;
  Parameters* pars_;
};
