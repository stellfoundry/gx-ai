//  This subroutine takes as inputs two fields in (kx,ky,z) space and calculates
//  the nlps bracket, and returns it in (kx,ky,z) space

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

#include <nlps_kernel.cu>


void kInit(float *k, float *k, float *k, float *k, int N, int N, int N);

//kernels declared and defined in nlps_kernel.cu
/* __global__ void multiply(cufftComplex *c);

__global__ void deriv(cufftComplex *c, cufftComplex *d, cufftComplex *d, 
                      cufftComplex *c, cufftComplex *d, cufftComplex *d, 
                      float *k, float *k, int N, int N); 

__global__ void scalemult(cufftReal *f, cufftReal *f, cufftReal *f, cufftReal *f, cufftReal *f,
                      float a, int N, int N); */

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
cufftComplex* NLPS(cufftComplex *f_complex_d, cufftComplex *g_complex_d, int Nx, int Ny,
int Nz) 
{
    //host variables
    float *kx, *ky;
    kx = (float*) malloc(sizeof(float)*(Nx/2+1)*Ny*Nz);
    ky = (float*) malloc(sizeof(float)*(Nx/2+1)*Ny*Nz);
    
    //device variables
    float *kx_d, *ky_d;
    cufftReal *f_d, *fdxR_d, *fdyR_d;
    cufftReal *g_d, *gdxR_d, *gdyR_d;
    cufftComplex *fdx_d, *fdy_d;
    cufftComplex *gdx_d, *gdy_d;
    float scaler;
    cufftReal *multR_d;
    cufftComplex *mult_d;
    cudaMalloc((void**) &kx_d, sizeof(float)*(Nx/2+1)*Ny*Nz);
    cudaMalloc((void**) &ky_d, sizeof(float)*(Nx/2+1)*Ny*Nz);    
    cudaMalloc((void**) &f_d, sizeof(cufftReal)*Nx*Ny*Nz);            
    cudaMalloc((void**) &fdx_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz);    
    cudaMalloc((void**) &fdxR_d, sizeof(cufftReal)*(Nx)*Ny*Nz);
    cudaMalloc((void**) &fdy_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz); 
    cudaMalloc((void**) &fdyR_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &g_d, sizeof(cufftReal)*Nx*Ny*Nz);            
    cudaMalloc((void**) &gdx_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz);    
    cudaMalloc((void**) &gdxR_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &gdy_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz); 
    cudaMalloc((void**) &gdyR_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &scaler, sizeof(float));
    cudaMalloc((void**) &multR_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &mult_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz);
    
    
    
    int block_size_x=4; int block_size_y=2; 

    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid(Nx/dimBlock.x, Ny/dimBlock.y);  
    
    cufftHandle plan;
    cufftHandle plan2;
    int n[2] = {Ny, Nx};
    
    
    cufftPlanMany(&plan, 2,n,NULL,1,0,NULL,1,0,CUFFT_R2C,Nz);
    cufftPlanMany(&plan2,2,n,NULL,1,0,NULL,1,0,CUFFT_C2R,Nz);

    
    kInit(kx, ky, kx_d, ky_d, Nx, Ny, Nz);
    
    
    
    deriv<<<dimGrid, dimBlock>>> (f_complex_d, fdx_d, fdy_d, g_complex_d, gdx_d, gdy_d,
                                                      kx_d, ky_d, Nx, Ny, Nz);
    
          
    cufftExecC2R(plan2, fdx_d, fdxR_d);
    cufftExecC2R(plan2, fdy_d, fdyR_d);
    cufftExecC2R(plan2, gdx_d, gdxR_d);
    cufftExecC2R(plan2, gdy_d, gdyR_d);
     
  
    
    scaler = (float)1/(Nx*Nx*Ny*Ny);
    
    bracket<<<dimGrid,dimBlock>>> (multR_d, fdxR_d, fdyR_d, gdxR_d, gdyR_d, scaler, Nx, Ny, Nz);
    
    cufftExecR2C(plan, multR_d, mult_d);  
    
    
    cufftDestroy(plan);
    cufftDestroy(plan2);
    
    
    cudaFree(kx_d), cudaFree(ky_d);
    cudaFree(fdxR_d), cudaFree(fdyR_d);
    cudaFree(gdxR_d), cudaFree(gdyR_d);
    cudaFree(fdx_d), cudaFree(fdy_d);
    cudaFree(gdx_d), cudaFree(gdy_d);
    //cudaFree(scaler);
       
    return mult_d;
}

void kInit(float* kx, float* ky, float* kx_d, float* ky_d, int Nx, int Ny, int Nz) {
    
  for(int k=0; k<Nz; k++) {
   for(int j=0; j<Ny/2+1; j++) {
    for(int i=0; i<Nx/2+1; i++) {
      int index = i + (Nx/2+1)*j +(Ny)*(Nx/2+1)*k;
      kx[index] = i;
      ky[index] = j;
      //printf("kx[%d]=%f\n", index, kx[index]);
    }
   }
   for(int j=-Ny/2+1; j<0; j++) {
    for(int i=0; i<Nx/2+1; i++) {
      int index = i + (Nx/2+1)*(Ny+j) + (Nx/2+1)*(Ny)*k;
      kx[index] = i;
      ky[index] = j;
      //printf("kx[%d]=%f\n", index, kx[index]);
    }
   }
  }  
  cudaMemcpy(kx_d, kx, sizeof(float)*(Nx/2+1)*Ny*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(ky_d, ky, sizeof(float)*(Nx/2+1)*Ny*Nz, cudaMemcpyHostToDevice);
    
}          

