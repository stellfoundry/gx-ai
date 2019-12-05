#pragma once

//#include "geometry.h"
//#include "cuda_constants.h"
#include "cufft.h"

__device__ unsigned int get_id1(void);
__device__ unsigned int get_id2(void);
__device__ unsigned int get_id3(void);

__host__ __device__ float factorial(int m);
__device__ float Jflr(int l, float b, bool enforce_JL_0=true);

__device__ float g0(float b);
__device__ float g1(float b);
__device__ float sgam0 (float b);

__host__ __device__ bool operator>(cuComplex f, cuComplex g);

__host__ __device__ bool operator<(cuComplex f, cuComplex g);


__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g); 

__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g);

__host__ __device__ cuComplex operator-(cuComplex f);
__host__ __device__ cuComplex operator*(float scaler, cuComplex f) ;

__host__ __device__ cuComplex operator*(cuComplex f, float scaler); 

__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g);
__host__ __device__ cuDoubleComplex operator*(cuDoubleComplex f, cuDoubleComplex g);

__host__ __device__ cuComplex operator/(cuComplex f, float scaler);
__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) ;
__host__ __device__ cuDoubleComplex operator/(cuDoubleComplex f, cuDoubleComplex g) ;
__device__ int get_ikx(int idx);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2,
				  double c3, cuComplex* m3);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4,
				  double c5, cuComplex* m5);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2,
				  double c3, cuComplex* m3, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4,
				  double c5, cuComplex* m5, bool bdum);

__global__ void scale_kernel(cuComplex* res, cuComplex* m, double s);
__global__ void scale_kernel(cuComplex* res, cuComplex* m, cuComplex s);
__global__ void scale_singlemom_kernel(cuComplex* res, cuComplex* m, cuComplex s);

__global__ void add_scaled_singlemom_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2);

__global__ void add_scaled_singlemom_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2, double c3, cuComplex* m3);

__global__ void add_scaled_singlemom_kernel(cuComplex* res, cuComplex c1, cuComplex* m1, cuComplex c2, cuComplex* m2);

__global__ void reality_kernel(cuComplex* g);
__global__ void reality_singlemom_kernel(cuComplex* mom);
__global__ void set_mask(cuComplex* g);

__device__ float atomicMaxFloat(float* address, float val);
