#include "fields.h"

Fields::Fields(Grids* grids) : size_(sizeof(cuComplex)*grids->NxNycNz) {

  cudaMalloc((void**) &phi, size_);
  cudaMemset(phi, 0., size_);

}

Fields::~Fields() {
  cudaFree(phi);
}
