#pragma once

#include "parameters.h"
#include "grids.h"
#include "geometry.h"
#include "fields.h"
#include "moments.h"
#include "device_funcs.h"
#include "get_error.h"

class Solver {
 public:
  Solver(Parameters* pars, Grids* grids, Geometry* geo);
  ~Solver();
  
  void fieldSolve(MomentsG* G, Fields* fields);
  void svar(cuComplex* f, int N);
  void svar(float* f, int N);
  
  cuComplex * nbar ;

private:

  dim3 dG, dB, dg, db;

  float * phiavgdenom ;
  cuComplex * tmp ;

  // local private copies
  Parameters * pars_  ;
  Grids      * grids_ ;
  Geometry   * geo_   ;
};
