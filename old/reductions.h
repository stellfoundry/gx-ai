#pragma once
#define CUB_STDERR

#include <stdio.h>
#include <limits>
#include <typeinfo>

#include "device_funcs.h"
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
//    #include "cub.cuh"
//    #include "util_allocator.cuh"
// #include <cub/device/device_reduce.cuh>

struct CustomMax
{
  template <typename T>
  __device__ __forceinline__
  T operator()(const T &a, const T &b) const {
    return (abs(b) > abs(a)) ? abs(b) : abs(a);
  }
};

class Red {
 public:
  Red(int N);
  ~Red();
  void Sum(float* rmom, float* val, bool zero);
  void Sum(float* rmom, float* val);
  void MaxAbs(float* rmom, float* val);

 private:
  float zero = 0.;
  float* value; 
  float* dum; 
  void* work_sum, *work_max; 
  size_t nwork_sum, nwork_max;
  int N_;
  CustomMax max_op;
};  
