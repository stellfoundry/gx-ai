#pragma once
#include <cuComplex.h>

class nca {
 public:
  nca(int N, int tmp=0);
  ~nca();
  void increment_ts(void);

  int    idx, time, ikx, iky, ns;
  bool   write, write_v_time, xydata;
  size_t start[5];
  size_t count[5];
  size_t time_start[6];
  size_t time_count[6];
  int    dims[5];
  int    time_dims[6];
  float * data;
  float * cpu;
  float * tmp;
  cuComplex * z_tmp;
  cuComplex * z_cpu;
  int N_;
  int Nwrite_;
};
