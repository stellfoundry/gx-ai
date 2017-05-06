#pragma once
#include "fields.h"
#include "moments.h"

class Linear {
 public:
  Linear(Parameters* pars, Grids* grids, Geometry* geo); 
  ~Linear();

  int rhs(Moments* m, Fields* f, Moments* mRhs);

  dim3 dimGrid, dimBlock;
  int sharedSize;

 private:
  Parameters* pars_;
  Grids* grids_;  
  Geometry* geo_;

  Moments* mRhs_par;
};
