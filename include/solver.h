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
  
  int fieldSolve(MomentsG* G, Fields* fields);
  void svar(cuComplex* f, int N);
  void svar(float* f, int N);
  
  cuComplex * nbar = NULL;

private:

  int maxThreadsPerBlock_;
  dim3 dG, dB;

  float * phiavgdenom = NULL; 
  cuComplex * tmp = NULL;

  // local private copies
  Parameters * pars_  = NULL;
  Grids      * grids_ = NULL;
  Geometry   * geo_   = NULL;
};
