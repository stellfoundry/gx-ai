#pragma once

#include "cufft.h"
#include "grids.h"
#include "grad_perp.h"

class Fields {
  public:
    Fields(Grids* grids);
    ~Fields();

    cuComplex* phih;
    cuComplex* phi;
    cuComplex* apar; // for electromagnetic only
    void print_phi(void);
    void chk_fft(void);
    
    inline void copyFrom(Fields* source) {
      cudaMemcpyAsync(phi, source->phi, size_, cudaMemcpyDeviceToDevice);
    }

  private:
    const size_t size_;
    int N;
    GradPerp* fft;
    Grids* grids_;
};
