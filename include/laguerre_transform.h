#pragma once
#include "moments.h"

class LaguerreTransform {
 public:

  LaguerreTransform(Grids* grids);
  ~LaguerreTransform();
  
  float* toGrid;
  float* toSpectral;

  int initTransforms(float* toGrid, float* toSpectral);

  int transformToGrid(Moments* m);
  int transformToSpectral(Moments* m);

};


