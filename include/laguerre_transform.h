#pragma once
#include "moments.h"
#include "grids.h"
#include <cublas_v2.h>

class LaguerreTransform {
 public:

  LaguerreTransform(Grids* grids);
  ~LaguerreTransform();
  
  int transformToGrid(MomentsG* G);
  int transformToSpectral(MomentsG* G);

  float* get_toGrid() {return toGrid;}
  float* get_toSpectral() {return toSpectral;}
  float* get_roots() {return roots;}

 private:
  Grids* grids_; 

  float* toGrid;
  float* toSpectral;
  float* roots;

  int initTransforms(float* toGrid, float* toSpectral, float* roots);

  int L, J;
  cublasHandle_t handle;
  
};


