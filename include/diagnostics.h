#pragma once

#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "fields.h"

class Diagnostics {
 public:
  Diagnostics(Parameters *pars, Grids *grids, Geometry *geo);
  ~Diagnostics();

  void loop_diagnostics(Moments* moms, Fields* fields, float dt, int counter, float time) ;

  void printMomOrField(cuComplex* m, const char* filename);
  

 private:
  Fields *fields_old;

  cuComplex *growth_rates, *growth_rates_h;

  cuComplex *m_h;

  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;

  int maxThreadsPerBlock_;
  dim3 dimGrid_xy, dimBlock_xy;

  void print_growth_rates_to_screen();
};
