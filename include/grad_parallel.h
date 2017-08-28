#pragma once
#include "grids.h"
#include "moments.h"

class Moments; // Forward Declaration

class GradParallel {
 public:
  
  GradParallel(Grids* grids, bool abs=false, bool single=false); // MFM: added single for 1-D FFT
  ~GradParallel();

  void eval(Moments* m);
  void eval(cuComplex* m, cuComplex* res);
  void eval_1d(float* bmag_t, cuComplex* bmag_complex_t);
 
 private:
  Grids* grids_;
  
  cufftHandle gradpar_plan_forward;
  cufftHandle gradpar_plan_inverse;
};
