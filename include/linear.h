#pragma once
#include "fields.h"
#include "moments.h"
#include "grad_parallel.h"
#include "closures.h"

class Linear {
 public:
  Linear(Parameters* pars, Grids* grids, Geometry* geo); 
  ~Linear();

  int rhs(Moments* m, Fields* f, Moments* mRhs);

  int zderiv(Moments *m);

  dim3 dimGrid, dimBlock;
  int sharedSize;

 private:
  const Parameters* pars_;
  Grids* grids_;  
  const Geometry* geo_;
  GradParallel* grad_par;
  Closures* closures;

  Moments* mRhs_par;

  // conservation terms
  cuComplex* upar_bar;
  cuComplex* uperp_bar;
  cuComplex* t_bar;
};
