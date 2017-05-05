#include "fields.h"
#include "get_error.h"

Fields::Fields(Grids* grids) : size_(sizeof(cuComplex)*grids->NxNycNz) {

  checkCuda(cudaMalloc((void**) &phi, size_));
  cudaMemset(phi, 0., size_);

}

Fields::~Fields() {
  cudaFree(phi);
}

