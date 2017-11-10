#pragma once

#include "laguerre_transform.h"
#include "grids.h"
#include "geometry.h"
#include "grad_perp.h"
#include "moments.h"
#include "fields.h"

class Nonlinear {
 public:
  Nonlinear(Parameters* pars, Grids* grids, Geometry* geo);
  ~Nonlinear();

  void nlps5d(MomentsG* G, Fields* f, MomentsG* G_res);
  double cfl(double dt_max);

 private:

  LaguerreTransform* laguerre;
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  GradPerp* grad_perp_G;
  GradPerp* grad_perp_J0phi;

  dim3 dimGrid, dimBlock;

  float *dG, *dg_dx, *dg_dy;
  cuComplex *J0phi;
  float *dJ0phi_dx, *dJ0phi_dy;
  float *g_res;

  float *vmax_x, *vmax_y;

  double dt_cfl;
};
