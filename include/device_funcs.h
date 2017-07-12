#pragma once

//#include "geometry.h"
//#include "cuda_constants.h"
#include "cufft.h"

__device__ unsigned int get_id1(void);
__device__ unsigned int get_id2(void);
__device__ unsigned int get_id3(void);

__host__ __device__ float factorial(int m);
__device__ float Jflr(int m, float b);

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

__host__ __device__ cuComplex operator/(cuComplex f, float scaler);
__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) ;
__device__ int get_ikx(int idx);

__global__ void add_scaled_kernel(cuComplex* res, 
                 double c1, cuComplex* m1, double c2, cuComplex* m2, 
                 double c3, cuComplex* m3, double c4, cuComplex* m4,
                 double c5, cuComplex* m5);

__global__ void add_scaled_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2);

__global__ void add_scaled_singlemom_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2);


__global__ void add_scaled_singlemom_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2, double c3, cuComplex* m3);

