#pragma once

#include "geometry.h"
#include "cuda_constants.h"
#include "cufft.h"

__device__ unsigned int get_id1(void);
__device__ unsigned int get_id2(void);
__device__ unsigned int get_id3(void);

__device__ double kperp2(Geometry::kperp2_struct* kp2, int ix, int iy, int iz, int is);

__device__ unsigned long long factorial(int m);
__device__ double Jflr(int m, double b);

__device__ double g0(double b);
__device__ double g1(double b);
__device__ double sgam0 (double b);

__host__ __device__ bool operator>(cuComplex f, cuComplex g);

__host__ __device__ bool operator<(cuComplex f, cuComplex g);


__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g); 

__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g);

__host__ __device__ cuComplex operator-(cuComplex f);
__host__ __device__ cuComplex operator*(double scaler, cuComplex f) ;

__host__ __device__ cuComplex operator*(cuComplex f, double scaler); 

__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g);

__host__ __device__ cuComplex operator/(cuComplex f, float scaler);
__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) ;
__device__ int get_ikx(int idx);

