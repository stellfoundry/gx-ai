#pragma once

#include "parameters.h"
#include "grids.h"
#include "geometry.h"
#include "fields.h"
#include "moments.h"

class Solver {
 public:
  Solver(Parameters* pars, Grids* grids, Geometry* geo);
  ~Solver();
  
  int fieldSolve(Moments* moms, Fields* fields);
  
  cuComplex* nbar;
 private:
  float* phiavgdenom; 
  cuComplex* tmp;

  // local private copies
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;

  int maxThreadsPerBlock_;
  dim3 dimGrid_qneut, dimBlock_qneut;
};
