#pragma once

#include "geometry.h"
#include "cuda_constants.h"
#include "cufft.h"

__device__ unsigned int get_id1(void);
__device__ unsigned int get_id2(void);
__device__ unsigned int get_id3(void);

__device__ float kperp2(Geometry::kperp2_struct* kp2, int ix, int iy, int iz, int is);

__device__ float factorial(int m);
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

