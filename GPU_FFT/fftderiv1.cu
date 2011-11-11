/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>
#include <cufft.h>


// includes, kernels
//#include <fft_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

void getfcn(float* a, float* a, float* a, float* a, int N, int N);
void getfcnC(cufftComplex *c, cufftComplex *c, int N, int N);
void kInit(float *k, float *k, float *k, float *k, int N, int N);



__global__ void deriv(cufftComplex *c, cufftComplex *d, cufftComplex *d, float *k, float *k, int N, int N); 

__global__ void scale(cufftReal *f, cufftReal *f, float a, int N, int N);



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
cufftReal* fftderiv(int k_x, int k_y, int Nx, int Ny, int sincoef, int coscoef, int derivop) 
{
    
    //host variables
    cufftReal *fcn, *fcndxR, *fcndyR;
    cufftComplex *fcn_complex;
    cufftComplex *fcndx;
    cufftComplex *fcndy;
    float *x;
    float *y;
    float *kx; 
    float *ky;
    //int Nx = 4, Ny = 8;
    fcn = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny);
    fcndxR = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny);
    fcndyR = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny);
    x = (float*) malloc(sizeof(float)*Nx);
    y = (float*) malloc(sizeof(float)*Ny);
    kx = (float*) malloc(sizeof(float)*(Nx/2+1)*Ny);
    ky = (float*) malloc(sizeof(float)*(Nx/2+1)*Ny);    
    fcn_complex = (cufftComplex*) malloc(sizeof(cufftComplex)*((Nx/2+1))*(Ny));
    fcndx = (cufftComplex*) malloc(sizeof(cufftComplex)*((Nx/2+1))*(Ny));    
    fcndy = (cufftComplex*) malloc(sizeof(cufftComplex)*((Nx/2+1))*(Ny));
    
    //device variables
    float *kx_d; 
    float *ky_d;
    cufftReal *fcn_d;
    cufftComplex *fcn_complex_d;
    cufftComplex *fcndx_d;
    cufftReal *fcndxR_d;
    cufftComplex *fcndy_d;
    cufftReal *fcndyR_d;
    float scaler;
    cudaMalloc((void**) &kx_d, sizeof(float)*(Nx/2+1)*Ny);
    cudaMalloc((void**) &ky_d, sizeof(float)*(Nx/2+1)*Ny);    
    cudaMalloc((void**) &fcn_d, sizeof(cufftReal)*Nx*Ny);    
    cudaMalloc((void**) &fcn_complex_d, sizeof(cufftComplex)*((Nx/2+1))*(Ny));    
    cudaMalloc((void**) &fcndx_d, sizeof(cufftComplex)*(Nx/2+1)*Ny);    
    cudaMalloc((void**) &fcndxR_d, sizeof(cufftReal)*Nx*Ny);
    cudaMalloc((void**) &fcndy_d, sizeof(cufftComplex)*(Nx/2+1)*Ny); 
    cudaMalloc((void**) &fcndyR_d, sizeof(cufftReal)*Nx*Ny);
    cudaMalloc((void**) &scaler, sizeof(float));
    
    //set fcn on host
    
    /* 
    0 < k_x < Nx/2
    -Ny/2+1 < k_y < Ny/2
    */
    //k_x= 2;
    //k_y= 1;
    //int coscoef=1;
    //int sincoef=0;
    
    //printf("\nfcn real, f(x,y): %d*cos(%dx + %dy) + %d*sin(%dx + %dy)\n", coscoef, k_x, k_y, sincoef, k_x, k_y);
    for(int j=0; j<Ny; j++) {
      for(int i=0; i<Nx; i++) {
      
      
      x[i] = 2*M_PI*(float)(i-Nx/2)/Nx;
      y[j] = 2*M_PI*(float)(j-Ny/2)/Ny;
      int index = i + Nx*j;
      
           
      fcn[index]= coscoef*cos( k_x*x[i] + k_y*y[j]) + 
                  sincoef*sin( k_x*x[i] + k_y*y[j]);		  
      
      
      //printf("f(%.2fPI,%.2fPI)=%.3f: %d ", x[i]/M_PI, y[j]/M_PI, fcn[index], index);
      
      }
      //printf("\n");
    }
    //printf("\n");
    
    //copy the function to the device
    cudaMemcpy(fcn_d, fcn, sizeof(cufftReal)*Nx*Ny, cudaMemcpyHostToDevice);
    
    
    
    int block_size_x=Nx/2; int block_size_y=Ny/4; 

    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid(Nx/dimBlock.x, Ny/dimBlock.y);  
    
    cufftHandle plan;
    cufftHandle plan2;
    
    cufftPlan2d(&plan, Ny, Nx, CUFFT_R2C);
    cufftPlan2d(&plan2, Ny, Nx, CUFFT_C2R);
    
    //transform the function R2C
    cufftExecR2C(plan, fcn_d, fcn_complex_d);
    
    //printf("fcn complex, F(kx,ky):\n");
    //getfcnC(fcn_complex, fcn_complex_d, Nx, Ny);
    //printf("\n");
    
    //getfcnC(fcndx, fcndx_d, Nx, Ny);
    
    //initialize k's
    kInit(kx, ky, kx_d, ky_d, Nx, Ny);
    
    //take derivatives in complex space
    deriv<<<dimGrid, dimBlock>>> (fcn_complex_d, fcndx_d, fcndy_d, kx_d, ky_d, Nx, Ny);
    
    //printf("deriv_x complex, F'_x(kx,ky):\n");
    //getfcnC(fcndx, fcndx_d, Nx, Ny);
    
    //printf("\nderiv_y complex, F'_y(kx,ky):\n");
    //getfcnC(fcndy, fcndy_d, Nx, Ny);
    
    //transform derivatives back to real space
    cufftExecC2R(plan2, fcndx_d, fcndxR_d);
    cufftExecC2R(plan2, fcndy_d, fcndyR_d);
    
    scaler = (float)1/(Nx*Ny);
    
    //scale the derivatives, necessary after inverse FT
    scale<<<dimGrid,dimBlock>>> (fcndxR_d, fcndyR_d, scaler, Nx, Ny);
    
    //printf("\nderiv_x real, f'_x(x,y): \n");
    //getfcn(fcndxR, fcndxR_d, x, y, Nx, Ny);
    
    //printf("\nderiv_y real, f'_y(x,y): \n");
    //getfcn(fcndyR, fcndyR_d, x, y, Nx, Ny);
    
    //printf("\n");
    cufftDestroy(plan);
    cufftDestroy(plan2);
    
    cudaMemcpy(fcndxR, fcndxR_d, sizeof(cufftReal)*Nx*Ny, cudaMemcpyDeviceToHost);
    cudaMemcpy(fcndyR, fcndyR_d, sizeof(cufftReal)*Nx*Ny, cudaMemcpyDeviceToHost);
    
    cudaFree(fcn_d); cudaFree(fcn_complex_d); cudaFree(fcndx_d); cudaFree(fcndy_d);
    cudaFree(fcndxR_d); cudaFree(fcndyR_d);
    
    /* for(int i=0; i<Nx; i++) {
      for(int j=0; j<Ny; j++) {
        int index = i + Nx*j;
	fcndxR[index] = 0;
      }
    }	 */
    
    if(derivop == 1) return fcndxR;
    else if(derivop == 2) return fcndyR;
    else return 0;
}

//prints a real function
void getfcn(cufftReal* fcn, cufftReal* fcn_d, float* x, float* y, int Nx, int Ny) {
  cudaMemcpy(fcn, fcn_d, sizeof(cufftReal)*Nx*Ny, cudaMemcpyDeviceToHost);
    
  for(int j=0; j<Ny; j++) {
    for(int i=0; i<Nx; i++) {
      int index = i + Nx*j;
      printf("f(%.2fPI,%.2fPI)=%.3f: %d ", x[i]/M_PI, y[j]/M_PI, fcn[index], index);     
      }
      printf("\n");
    }
  }  

//prints a complex function    
void getfcnC(cufftComplex* fcn, cufftComplex* fcn_d, int Nx, int Ny) {
  cudaMemcpy(fcn, fcn_d, sizeof(cufftComplex)*(Nx)*(Ny), cudaMemcpyDeviceToHost);
    
    
    for(int j=0; j<Ny/2+1; j++) { 
      for(int i=0; i<Nx/2+1; i++) {  
	int index = i + (Nx/2+1)*(j);
	
	printf("F(%d,%d)=%.3f+i*%.3f: %d ", i, j, -fcn[index].x*2/(Nx*Ny), -fcn[index].y*2/(Nx*Ny), index);
      }
      printf("\n");
    }  
    for(int j=-Ny/2+1; j<0; j++) {
      for(int i=0; i<Nx/2+1; i++) {
        int index = (i) + (Nx/2+1)*(j+Ny);
	
	printf("F(%d,%d)=%.3f+i*%.3f: %d ", i, j, -fcn[index].x*2/(Nx*Ny), -fcn[index].y*2/(Nx*Ny), index);
      }
        
      printf("\n");
    }
} 

//initializes k's in order required for FFT
void kInit(float* kx, float* ky, float* kx_d, float* ky_d, int Nx, int Ny) {
  /* for(int i=0; i<Nx/2+1; i++) {
      for(int j=0; j<Ny/2+1; j++) { 
        int index = i + (Nx/2+1)*j;
	
	kx[index] = i;
        ky[index] = j;
      }
      for(int j=-Ny/2+1; j<0; j++) {
        int index = i + (Nx/2+1)*(Ny+j);
	
	kx[index] = i;
        ky[index] = j;
      }
    } */
  
  for(int j=0; j<Ny/2+1; j++) {
    for(int i=0; i<Nx/2+1; i++) {
      int index = i + (Nx/2+1)*j;
      kx[index] = i;
      ky[index] = j;
      printf("ky[%d]=%f\n", index, ky[index]);
    }
  }
  for(int j=-Ny/2+1; j<0; j++) {
    for(int i=0; i<Nx/2+1; i++) {
      int index = i + (Nx/2+1)*(Ny+j);
      kx[index] = i;
      ky[index] = j;
      printf("ky[%d]=%f\n", index, ky[index]);
    }
  } 
  cudaMemcpy(kx_d, kx, sizeof(float)*(Nx/2+1)*Ny, cudaMemcpyHostToDevice);
  cudaMemcpy(ky_d, ky, sizeof(float)*(Nx/2+1)*Ny, cudaMemcpyHostToDevice);
}          

//kernel for taking derivatives
__global__ void deriv(cufftComplex* fcn, cufftComplex* fcndx, cufftComplex* fcndy, float* kx, float* ky, int Nx, int Ny) 
{
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  
  
  if(idx<(Nx/2+1) && idy<Ny) {
    int index = idx + (Nx/2+1)*idy;
    
    //da/dx
    fcndx[index].x = -kx[index]*fcn[index].y;
    fcndx[index].y =  kx[index]*fcn[index].x;
    
    //da/dy
    fcndy[index].x = -ky[index]*fcn[index].y;
    fcndy[index].y =  ky[index]*fcn[index].x;
    
  }
}  

//kernel for scaling derivatives
__global__ void scale(cufftReal* fcnx, cufftReal* fcny, float scaler, int Nx, int Ny)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  
  if(idx<Nx && idy<Ny) {
    int index = idx + Nx*idy;
    
    fcnx[index] = fcnx[index]*scaler;  
    fcny[index] = fcny[index]*scaler;
  }
}  
     					      
							 
        
