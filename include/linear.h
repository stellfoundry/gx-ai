#pragma once
#include "fields.h"
#include "moments.h"
#include "grad_parallel.h"
#include "closures.h"

class Linear {
public:
  Linear(Parameters* pars, Grids* grids, Geometry* geo); 
  Linear(Parameters* pars, Grids* grids); 
  ~Linear();

  int rhs(cuComplex *G, cuComplex *GRhs);
  int rhs(MomentsG* G, Fields* f, MomentsG* GRhs);

  int zderiv(MomentsG *G);

  dim3 dimGrid, dimBlock, dG, dB;
  int sharedSize;
  
 private:
  bool ks = false;

  Geometry       * geo_     = NULL;
  Parameters     * pars_    = NULL;
  Grids          * grids_   = NULL;  
  GradParallel   * grad_par = NULL;
  Closures       * closures = NULL;
  MomentsG       * GRhs_par = NULL;

  // conservation terms
  cuComplex * upar_bar      = NULL;
  cuComplex * uperp_bar     = NULL;
  cuComplex * t_bar         = NULL;
};
