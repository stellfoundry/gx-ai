#pragma once

#include "grids.h"

class GradPerp {
 public:
  GradPerp(Grids* grids, int batch, int mem);
  ~GradPerp();

  void dxC2R(cuComplex* G, float* dxG);
  void dyC2R(cuComplex* G, float* g);
  void R2C  (float* G, cuComplex* res);

 private:
  Grids* grids_;
  const int batch_size_;
  const int mem_size_;
  cuComplex *tmp;
  cufftHandle gradperp_plan_R2C;
  cufftHandle gradperp_plan_dxC2R;
  cufftHandle gradperp_plan_dyC2R;

};
