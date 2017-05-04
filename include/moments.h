//#pragma once

#include "grids.h"
#include "fields.h"
#include "parameters.h"
#include "geometry.h"

class Moments {
 public:
  Moments(Grids* grids);
  ~Moments();

  int initialConditions(Fields* fields, Parameters *pars, Geometry* geo);

  int advance_linear(Moments* momsStar, double dt, Moments* moms, Fields* fields, Parameters* pars);
  int fieldSolve(Fields* fields, Parameters* pars, Geometry::kperp2_struct* kp2);
 
  cuComplex** ghl;
  cuComplex** dens_ptr;
  cuComplex** upar_ptr;
  cuComplex** tpar_ptr;
  cuComplex** tprp_ptr;
  cuComplex** qpar_ptr;
  cuComplex** qprp_ptr;

  cuComplex* nbar;

 private:
  const Grids* grids_;
  const size_t HLsize_;
  const size_t Momsize_;
};
