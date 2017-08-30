#pragma once
#include "moments.h"
#include "grids.h"
#include <cublas_v2.h>

class LaguerreTransform {
 public:

  LaguerreTransform(Grids* grids);
  ~LaguerreTransform();
  
  int transformToGrid(float* G_in, float* g_res);
  int transformToSpectral(float* g_in, float* G_res);

  float* get_toGrid() {return toGrid;}
  float* get_toSpectral() {return toSpectral;}
  float* get_roots() {return roots;}
  const int L;
  const int J;

 private:
  Grids* grids_; 

  float* toGrid;
  float* toSpectral;
  float* roots;

  int initTransforms(float* toGrid, float* toSpectral, float* roots);

  cublasHandle_t handle;
  
};


