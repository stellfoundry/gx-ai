#pragma once
#include "grids.h"
#include "moments.h"

class MomentsG; // Forward Declaration

class GradParallel {
 public:
  GradParallel() {};
  virtual ~GradParallel() {};

  virtual void eval(MomentsG* G)=0;
  virtual void eval(cuComplex* m, cuComplex* res)=0;
  virtual void fft_only(cuComplex* m, cuComplex* res, int dir) {};
 private:
};

class GradParallelPeriodic : public GradParallel {
 public:
  GradParallelPeriodic(Grids* grids, bool abs=false);
  ~GradParallelPeriodic();

  void eval(MomentsG* G);
  void eval(cuComplex* m, cuComplex* res);
  void fft_only(cuComplex* m, cuComplex* res, int dir);
  
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

class GradParallelLinked : public GradParallel {
 public:
  GradParallelLinked(Grids* grids, bool abs=false);
  ~GradParallelLinked();

 private:
  
};

class GradParallel1D {
 public:
  GradParallel1D(Grids* grids);
  ~GradParallel1D();
  void eval1D(float* b); 

 private:
  Grids* grids_;
  
  cufftHandle gradpar_plan_forward;
  cufftHandle gradpar_plan_inverse;

  cuComplex *b_complex;
};
