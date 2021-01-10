#pragma once

#include "laguerre_transform.h"
#include "grids.h"
#include "geometry.h"
#include "grad_perp.h"
#include "moments.h"
#include "fields.h"
#include "reductions.h"

class Nonlinear {
 public:
  Nonlinear(Parameters* pars, Grids* grids, Geometry* geo);
  Nonlinear(Parameters* pars, Grids* grids);
  ~Nonlinear();

  void nlps(MomentsG* G, Fields* f, MomentsG* G_res);
  void nlps(cuComplex *G, cuComplex *res);
  double cfl(Fields *f, double dt_max);
  void qvar(cuComplex* G, int N);
  void qvar(float* G, int N);
  
 private:

  int nBatch;

  size_t Size; 

  bool ks = false;
  
  dim3 dGk, dBk, dGx, dBx;
  
  float cfl_x_inv, cfl_y_inv;

  double dt_cfl;

  Parameters        * pars_           = NULL;
  Grids             * grids_          = NULL;  
  Geometry          * geo_            = NULL;
  
  Red               * red             = NULL; 
  LaguerreTransform * laguerre        = NULL;
  GradPerp          * grad_perp_G     = NULL;
  GradPerp          * grad_perp_J0phi = NULL;
  GradPerp          * grad_perp_phi   = NULL;

  float * dG          = NULL;
  float * dg_dx       = NULL;
  float * dg_dy       = NULL;
  float * val1        = NULL;
  float * Gy          = NULL;
  float * dJ0phi_dx   = NULL;
  float * dJ0phi_dy   = NULL;
  float * dJ0_Apar_dx = NULL;
  float * dJ0_Apar_dy = NULL;
  float * dphi        = NULL;
  float * g_res       = NULL;
  float * vmax_x      = NULL;
  float * vmax_y      = NULL;
  cuComplex * J0phi   = NULL;
  cuComplex * J0_Apar = NULL;
};
