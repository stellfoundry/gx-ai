#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "sys/stat.h"
#include "cufft.h"
#include "cuda_profiler_api.h"

int Nx, Ny, Nz, zThreads, totalThreads;
float X0, Y0, Zp;
__constant__ __device__ int nx,ny,nz,zthreads,totalthreads;
__constant__ __device__ float X0_d,Y0_d,Z0_d;

dim3 dimBlock, dimGrid;

#include "../device_funcs.cu"
#include "../getfcn.cu"
#include "../reduc_kernel.cu"
#include "../cudaReduc_kernel.cu"
#include "../maxReduc.cu"
#include "../sumReduc.cu"

int main(int argc, char* argv[]) {
  
  int dev;
  struct cudaDeviceProp prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop,dev);
  printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("Max Size of Block Dimension (threads): %d * %d * %d\n", prop.maxThreadsDim[0], 
                        prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Max Size of Grid Dimension (blocks): %d * %d * %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  
  float* a;
  float* a_sum_z_cpu;
  float* a_sum_z_gpu;
  
  float* a_d;
  float* a_sum_z_d;
  
  float sum_cpu = 0.;
  float sum_gpu = 0.;
  
  Nx=256;
  Nz=64;
  
  a = (float*) malloc(sizeof(float)*Nx*Nz);
  a_sum_z_cpu = (float*) malloc(sizeof(float)*Nz);
  a_sum_z_gpu = (float*) malloc(sizeof(float)*Nz);
  
  cudaMalloc((void**) &a_d, sizeof(float)*Nx*Nz);
  cudaMalloc((void**) &a_sum_z_d, sizeof(float)*Nz);
   
  getError("after host and device allocs"); 
    
  sum_cpu = 0.;  
  for(int i=0; i<Nx*Nz; i++) {
    a[i] = (float) i;   
  }
  
  sum_cpu = (double) (Nx*Nz)*(Nx*Nz-1)/2.;
  
  cudaMemcpy(a_d, a, sizeof(float)*Nx*Nz, cudaMemcpyHostToDevice);
  
  
  
  for(int idz=0; idz<Nz; idz++) {
    a_sum_z_cpu[idz] = 0.;
    for(int idx=0; idx<Nx; idx++) {
      a_sum_z_cpu[idz] = a_sum_z_cpu[idz] + a[idx + Nx*idz];
    }    
  }
  
  
  
  sumReduc_Partial(a_sum_z_d, a_d, Nx*Nz, Nz, false);
  
  sum_gpu = sumReduc(a_d, Nx*Nz, false);
  
  cudaMemcpy(a_sum_z_gpu, a_sum_z_d, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  
  
  bool CORRECT = true;
  
  for(int idz=0; idz<Nz; idz++) {
    if( a_sum_z_gpu[idz] != a_sum_z_cpu[idz]) {
      printf("MISMATCH... gpu_sum[%d]=%f\tcpu_sum[%d]=%f\n", idz,
      			a_sum_z_gpu[idz], idz, a_sum_z_cpu[idz]);
      CORRECT = false;
    }
    
  }
  if(CORRECT) {
    printf("Partial sums correct!\n\n");
  }	
  
  printf("gpu total sum = %f\tanswer = %f\tdiff = %f\n", (float) sum_gpu,
  						(float)sum_cpu, sum_gpu - sum_cpu);		
  
}
  
  
  
