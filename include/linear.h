#pragma once
#include "cufft.h"
#include "device_funcs.h"
#include "fields.h"
#include "moments.h"
#include "grad_parallel.h"
#include "closures.h"
#include "get_error.h"

class Linear {
public:
  Linear(Parameters* pars, Grids* grids, Geometry* geo); 
  Linear(Parameters* pars, Grids* grids); 
  ~Linear();

  //  void rhs(cuComplex *G, cuComplex *GRhs);
  void rhs(MomentsG* G, Fields* f, MomentsG* GRhs);

  //  int zderiv(MomentsG *G);

  dim3 dimGrid, dimBlock, dG, dB, dGs, dBs, dimGridh, dimBlockh, dB_all, dG_all;
  int sharedSize;
  
 private:
  bool ks;
  bool vp;

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

  // Hammett-Belli hyper
  cuComplex * df            ;
  cuComplex * favg          ;
  float     * s01           ;
  float     * s10           ;
  float     * s11           ;
  float     * vol_fac       ; 

  float volDenom;
  
};
