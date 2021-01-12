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
  bool ks;

  Geometry       * geo_     ;
  Parameters     * pars_    ;
  Grids          * grids_   ;  

  GradParallel   * grad_par ;
  Closures       * closures ;
  MomentsG       * GRhs_par ;

  // conservation terms
  cuComplex * upar_bar      ;
  cuComplex * uperp_bar     ;
  cuComplex * t_bar         ;
};
