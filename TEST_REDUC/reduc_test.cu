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
  double* a_sum_z_cpu;
  float* a_sum_z_gpu;
  
  float* a_d;
  float* a_sum_z_d;
  float* tmpXZ;
  float* tmpXZ2;
  
  double sum_cpu = 0.;
  float sum_gpu = 0.;
  
  Nx=256*(128/2+1);
  Nz=4;
  double init_amp = 1.e-20;
  double tol = 1.e-7;
  
  a = (float*) malloc(sizeof(float)*Nx*Nz);
  a_sum_z_cpu = (double*) malloc(sizeof(double)*Nz);
  a_sum_z_gpu = (float*) malloc(sizeof(float)*Nz);
  
  cudaMalloc((void**) &a_d, sizeof(float)*Nx*Nz);
  cudaMalloc((void**) &a_sum_z_d, sizeof(float)*Nz);
  cudaMalloc((void**) &tmpXZ, sizeof(float)*Nx*Nz);
  cudaMalloc((void**) &tmpXZ2, sizeof(float)*Nx*Nz);

  getError("after host and device allocs"); 
    
  printf("\nInitializing a[j] = %e*j\nNx = %d\tNz = %d\n\n", init_amp, Nx, Nz);

  sum_cpu = 0.;  
  for(int i=0; i<Nx*Nz; i++) {
    a[i] = (double) i*init_amp;   
  }
  
  sum_cpu = (double) (Nx*Nz)*(Nx*Nz-1)/2.;
  
  cudaMemcpy(a_d, a, sizeof(float)*Nx*Nz, cudaMemcpyHostToDevice);
  
  
  sum_cpu = 0.;
  for(int idz=0; idz<Nz; idz++) {
    a_sum_z_cpu[idz] = 0.;
    for(int idx=0; idx<Nx; idx++) {
      a_sum_z_cpu[idz] = a_sum_z_cpu[idz] + a[idx + Nx*idz];
      sum_cpu = sum_cpu + a[idx + Nx*idz]; 
    }    
  }
  
  
  
  sumReduc_Partial(a_sum_z_d, a_d, Nx*Nz, Nz, tmpXZ, tmpXZ);
  //if overwrite flag is true, a_d is changed
  
  sum_gpu = sumReduc(a_d, Nx*Nz, tmpXZ, tmpXZ);
  
  cudaMemcpy(a_sum_z_gpu, a_sum_z_d, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  
  
  bool CORRECT = true;
  
  for(int idz=0; idz<Nz; idz++) {
      printf("gpu_sum[%d]=%e\t\tcpu_sum[%d]=%e\t\tdiff[%d]=%e \n", idz,
      			a_sum_z_gpu[idz], idz, a_sum_z_cpu[idz], idz, a_sum_z_gpu[idz] - a_sum_z_cpu[idz]);
    if( abs(a_sum_z_gpu[idz] - a_sum_z_cpu[idz]) > tol*a_sum_z_gpu[idz]) {
      CORRECT = false;
    }
    
  }
  if(CORRECT) {
    printf("Partial GPU sums correct to tol of %f%!\n\n", 100*tol);
  }	
  
  printf("gpu total sum = %e\tanswer = %e\tdiff = %e\n", (float) sum_gpu,
  						(float)sum_cpu, sum_gpu - sum_cpu);		
  if(abs(sum_gpu - sum_cpu) > tol*sum_gpu) printf("GPU sum NOT correct!\n");
  else printf("Total GPU sum is correct to tol of %f%!\n", 100*tol);
  
}
  
  
  
