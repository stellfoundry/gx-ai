#pragma once
#include "grids.h"
#include "moments.h"

class MomentsG; // Forward Declaration

class GradParallel {
 public:
  GradParallel() {};
  GradParallel(Grids* grids, bool abs=false, bool single=false);
  ~GradParallel();

  virtual void eval(MomentsG* G);
  virtual void eval(cuComplex* m, cuComplex* res);
  virtual void fft_only(cuComplex* m, cuComplex* res, int dir);
  void eval_1d(float* bmag_t, cuComplex* bmag_complex_t); //MFM
  
 private:
  Grids* grids_;
  
  cufftHandle gradpar_plan_forward;
  cufftHandle gradpar_plan_inverse;
};

class GradParallelLocal : public GradParallel {
 public:
  GradParallelLocal(Grids* grids, bool abs=false);
  ~GradParallelLocal() {};

  void eval(MomentsG* G);
  void eval(cuComplex* m, cuComplex* res);
 private:
  Grids* grids_;
  const bool abs_;

  dim3 dimGrid, dimBlock;
};
