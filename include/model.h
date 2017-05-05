#pragma once
#include "fields.h"
#include "moments.h"

class Model {
 public:
  Model(Parameters* pars, Grids* grids, Geometry* geo); 
  ~Model();

  int rhs(Moments* m, Fields* f, Moments* mRhs);
};
