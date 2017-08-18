#pragma once
#include "moments.h"
#include "grids.h"
#include <cublas_v2.h>

class LaguerreTransform {
 public:

  LaguerreTransform(Grids* grids);
  ~LaguerreTransform();
  
  int transformToGrid(Moments* m);
  int transformToSpectral(Moments* m);

 private:
  Grids* grids_; 

  float* toGrid;
  float* toSpectral;

  int initTransforms(float* toGrid, float* toSpectral);

  int L, J;
  cublasHandle_t handle;
  
};


