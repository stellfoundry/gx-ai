#include "reductions.h"

Red::Red(int N) : N_(N)
{
  cudaMalloc(&value, sizeof(float));   cudaMemset(value, 0., sizeof(float));
  cudaMalloc(&dum, N_*sizeof(float));  cudaMemset(dum, 0., N_*sizeof(float));

  work_sum = NULL;  nwork_sum = 0;

  using namespace cub;

  // Set up work space for a summation
  CubDebugExit(DeviceReduce::Sum(work_sum, nwork_sum, dum, value, N_));

  CachingDeviceAllocator  g_allocator(true);
  g_allocator.DeviceAllocate(&work_sum, nwork_sum);
  cudaMalloc(&work_sum, nwork_sum);
  
  work_max = NULL;  nwork_max = 0;

  // Set up work space for finding max(abs(float array))
  CubDebugExit(DeviceReduce::Reduce(work_max, nwork_max, dum, value, N_, max_op, zero));

  g_allocator.DeviceAllocate(&work_max, nwork_max);
  cudaMalloc(&work_max, nwork_max);

  cudaFree(dum);
}

Red::~Red() {
  cudaFree(work_sum);  cudaFree(work_max);  cudaFree(value);
}

// val = max(abs(rmom))
void Red::MaxAbs(float* rmom, float* val)
{
  using namespace cub;
  CubDebugExit(DeviceReduce::Reduce(work_max, nwork_max, rmom, val, N_, max_op, zero));
}

// val = Sum
void Red::Sum(float* rmom, float* val, bool boo)
{ using namespace cub;
  CubDebugExit(DeviceReduce::Sum(work_sum, nwork_sum, rmom, val, N_));
}

// val += Sum
void Red::Sum(float* rmom, float* val)
{ using namespace cub;
  cudaMemcpy(value, val, sizeof(float), cudaMemcpyDeviceToDevice);
  CubDebugExit(DeviceReduce::Sum(work_sum, nwork_sum, rmom, val, N_));
  acc<<<1,1>>> (val, value);
}
