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
  ~Nonlinear();

  void nlps5d(MomentsG* G, Fields* f, MomentsG* G_res);
  double cfl(double dt_max);
  void qvar(cuComplex* G, int N);
  
 private:

  LaguerreTransform* laguerre;
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  GradPerp* grad_perp_G;
  GradPerp* grad_perp_J0phi;
  Red* red; 
  dim3 dimGrid, dimBlock;

  float *dG, *dg_dx, *dg_dy, *val1;
  cuComplex *J0phi;
  float *dJ0phi_dx, *dJ0phi_dy;
  float *g_res;

  float *vmax_x, *vmax_y;
  float cfl_x_inv, cfl_y_inv;
  
  double dt_cfl;
};
