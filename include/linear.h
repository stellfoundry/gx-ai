#pragma once
#include "fields.h"
#include "moments.h"
#include "grad_parallel.h"

class Linear {
 public:
  Linear(Parameters* pars, Grids* grids, Geometry* geo); 
  ~Linear();

  int rhs(Moments* m, Fields* f, Moments* mRhs);

  int zderiv(Moments *m);

  dim3 dimGrid, dimBlock;
  int sharedSize;

 private:
  Parameters* pars_;
  Grids* grids_;  
  Geometry* geo_;
  GradParallel* grad_par;

  Moments* mRhs_par;
};
