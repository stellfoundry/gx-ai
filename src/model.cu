#include "model.h"

Model::Model(Parameters* pars, Grids* grids, Geometry* geo) 
{

}

Model::~Model()
{

}

int Model::rhs(Moments* m, Fields* f, Moments* mRhs) {
  return 0;
}
