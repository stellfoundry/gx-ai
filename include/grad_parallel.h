#pragma once
#include "grids.h"
#include "moments.h"

class GradParallel {
 public:
  
  GradParallel(Grids* grids);
  ~GradParallel();

  void ikpar(Moments* m);
 
 private:
  Grids* grids_;
  
  cufftHandle gradpar_plan_forward;
  cufftHandle gradpar_plan_inverse;
};
