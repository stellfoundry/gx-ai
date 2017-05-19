#pragma once

#include "grids.h"
#include "moments.h"
#include "grad_parallel.h"

class Closures {
 public:
  virtual ~Closures() {};
  virtual int apply_closures(Moments* m, Moments* mRhs) = 0;
};

class Beer42 : public Closures {
 public:
  Beer42(Grids* grids, float* omegad);
  ~Beer42();
  int apply_closures(Moments* m, Moments* mRhs);

 private:
  Grids* grids_;
  GradParallel* grad_par;
  GradParallel* abs_grad_par;

  float* omegad_;

  cuComplex* tmp;

  // closure coefficients
  float Beta_par;
  float D_par;
  float D_perp;
  cuComplex* nu;


  dim3 dimGrid, dimBlock;
};
