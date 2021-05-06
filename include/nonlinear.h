#pragma once
#include "laguerre_transform.h"
#include "grids.h"
#include "geometry.h"
#include "grad_perp.h"
#include "moments.h"
#include "fields.h"
#include "reductions.h"
#include "device_funcs.h"
#include "species.h"

class Nonlinear {
 public:
  Nonlinear(Parameters* pars, Grids* grids, Geometry* geo);
  Nonlinear(Parameters* pars, Grids* grids);
  ~Nonlinear();

  void nlps(MomentsG* G, Fields* f, MomentsG* G_res);
  double cfl(Fields *f, double dt_max);
  void qvar(cuComplex* G, int N);
  void qvar(float* G, int N);
  
 private:

  int nBatch;
  size_t Size; 
  bool ks, vp;
  dim3 dGk, dBk, dGx, dBx;
  float cfl_x_inv, cfl_y_inv;
  double dt_cfl;

  Parameters        * pars_           ;
  Grids             * grids_          ;  
  Geometry          * geo_            ;
  
  Red               * red             ; 
  LaguerreTransform * laguerre        ;
  GradPerp          * grad_perp_G     ;
  GradPerp          * grad_perp_J0phi ;
  GradPerp          * grad_perp_phi   ;

  float * dG          ;
  float * dg_dx       ;
  float * dg_dy       ;
  float * val1        ;
  float * Gy          ;
  float * dJ0phi_dx   ;
  float * dJ0phi_dy   ;
  float * dphi_dy     ;
  float * dJ0_Apar_dx ;
  float * dJ0_Apar_dy ;
  float * dphi        ;
  float * g_res       ;
  float * vmax_x      ;
  float * vmax_y      ;
  cuComplex * J0phi   ;
  cuComplex * J0_Apar ;
};
