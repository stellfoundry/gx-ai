#pragma once

#include "cufft.h"
#include "grids.h"

class Fields {
  public:
    Fields(Grids* grids);
    ~Fields();

    cuComplex* phi;
    cuComplex* apar; // for electromagnetic only

  private:
    const size_t size_;
	
};
