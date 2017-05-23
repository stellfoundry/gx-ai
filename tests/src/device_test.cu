#include "gtest/gtest.h"

__device__ float val_d;

__global__ void get_val(float* val) {
  val_d = *val;
}

void EXPECT_FLOAT_EQ_D(float *check_d, float correct, float err) {
  float val;
  get_val<<<1,1>>>(check_d);
  cudaMemcpyFromSymbol(&val, val_d, sizeof(val), 0, cudaMemcpyDeviceToHost);
  if(err==0.) EXPECT_FLOAT_EQ(val, correct);
  else EXPECT_NEAR(val, correct, err);
}
