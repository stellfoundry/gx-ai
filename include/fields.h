#pragma once

#include "cufft.h"
#include "grids.h"

class Fields {
  public:
    Fields(Grids* grids);
    ~Fields();

    cuComplex* phi;
    cuComplex* apar; // for electromagnetic only

    inline void copyFrom(Fields* source) {
      cudaMemcpyAsync(phi, source->phi, size_, cudaMemcpyDeviceToDevice);
    }

  private:
    const size_t size_;
	
};
