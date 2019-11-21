#pragma once
#include "fields.h"
#include "moments.h"
#include "grad_parallel.h"
#include "closures.h"

class Linear {
 public:
  Linear(Parameters* pars, Grids* grids, Geometry* geo); 
  ~Linear();

  int rhs(MomentsG* G, Fields* f, MomentsG* GRhs);

  int zderiv(MomentsG *G);

  dim3 dimGrid, dimBlock;
  int sharedSize;

 private:
  Parameters* pars_;
  Grids* grids_;  
  const Geometry* geo_;
  GradParallel* grad_par;
  Closures* closures;

  MomentsG* GRhs_par;

  //  const Parameters* pars_;

  // conservation terms
  cuComplex* upar_bar;
  cuComplex* uperp_bar;
  cuComplex* t_bar;
};
