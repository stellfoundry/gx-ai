//#pragma once

#include "grids.h"
#include "fields.h"
#include "parameters.h"

class Moments {
 public:
  Moments(Grids* grids);
  ~Moments();

  int advance_linear(Moments* momsStar, double dt, Moments* moms, Fields* fields, Parameters* parameters);
  int solve_quasineutrality(Fields* fields, Parameters* parameters);
 
  cuComplex** ghl;
  cuComplex** dens;
  cuComplex** upar;
  cuComplex** tpar;
  cuComplex** tprp;
  cuComplex** qpar;
  cuComplex** qprp;

 private:
  const Grids* grids_;
  const size_t size_;
};
