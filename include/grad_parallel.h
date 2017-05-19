#pragma once
#include "grids.h"
#include "moments.h"

class GradParallel {
 public:
  
  GradParallel(Grids* grids, bool abs=false);
  ~GradParallel();

  void eval(Moments* m);
  void eval(cuComplex* m, cuComplex* res);
  
 
 private:
  Grids* grids_;
  
  cufftHandle gradpar_plan_forward;
  cufftHandle gradpar_plan_inverse;
};
