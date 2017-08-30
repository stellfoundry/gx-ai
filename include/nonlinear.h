#pragma once

#include "grids.h"
#include "geometry.h"
#include "laguerre_transform.h"

class Nonlinear {
 public:
  Nonlinear(Grids* grids, Geometry* geo);
  ~Nonlinear();

  nlps5d();

 private:

  LaguerreTransform* laguerre;
  Grids* grids_;
  Geometry* geo_;
  GradPerp* grad_perp;

};
