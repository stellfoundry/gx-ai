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
  
  cuComplex* nbar;
 private:
  float* phiavgdenom; 
  cuComplex* tmp;

  // local private copies
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;

  int maxThreadsPerBlock_;
  dim3 dG, dB;
};
