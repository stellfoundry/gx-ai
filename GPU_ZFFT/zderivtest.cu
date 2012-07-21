#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

#include <zfft_kernel.cu>
#include <zderiv.cu>

void getfcn(cufftComplex* fcn_d, int Nx, int Ny, int Nz)
{
  cufftComplex *fcnC;
  fcnC = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
  cudaMemcpy(fcnC, fcn_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz, cudaMemcpyDeviceToHost);
  for(int k=0; k<(Nz); k++) { 
    for(int j=0; j<Nx/2+1; j++) { 
      for(int i=0; i<Ny/2+1; i++) {  
	int index = i + (Ny/2+1)*(j)+Nx*(Ny/2+1)*k;
	
	printf("F(%d,%d,%.2f)=%.3f+i*%.3f: %d ", i, j, 2*M_PI*(float)(k-Nz/2)/Nz, fcnC[index].x, fcnC[index].y, index);
      }
      printf("\n");
    }  
    for(int j=-Nx/2+1; j<0; j++) {
      for(int i=0; i<Ny/2+1; i++) {
        int index = (i) + (Ny/2+1)*(j+Nx)+Nx*(Ny/2+1)*k;
	
	printf("F(%d,%d,%.2f)=%.3f+i*%.3f: %d ", i, j, 2*M_PI*(float)(k-Nz/2)/Nz, fcnC[index].x, fcnC[index].y, index);
      }
        
      printf("\n");
    }
   }  
   free(fcnC);
} 

void getfcn(cufftReal* fcn_d, int Nx, int Ny, int Nz) 
{
  cufftReal *fcn;
  fcn = (cufftReal*) malloc(sizeof(cufftReal)*Ny*Nx*Nz);
  cudaMemcpy(fcn, fcn_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
  
  for(int k=0; k<Nz; k++) {  
   for(int j=0; j<Nx; j++) {
    for(int i=0; i<Ny; i++) {
      int index = i + Ny*j + Nx*Ny*k;
      printf("f(%.2fPI,%.2fPI)=%.3f ", 2*(float)(i-Ny/2)/Ny, 2*(float)(j-Nx/2)/Nx, fcn[index]);     
      }
      printf("\n");
    } printf("\n");
   } 
  free(fcn); 
}  


cufftReal* ZDERIVtest(int akx, int aky, int akz, int asin, int acos, int Ny, int Nx, int Nz) 
{
    //host variables
    cufftReal *a, *b;
    
    
    float *y, *x, *z;
    a = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    b = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    y = (float*) malloc(sizeof(float)*Ny);                                 //
    x = (float*) malloc(sizeof(float)*Nx);				   //
    z = (float*) malloc(sizeof(float)*Nz);

    
    //device variables
    cufftReal *a_d, *b_d;
    cufftComplex *a_complex_d, *b_complex_d;
    float scaler;
    float *kz_d;
    cudaMalloc((void**) &kz_d, sizeof(float)*Nz);
    cudaMalloc((void**) &a_d, sizeof(cufftReal)*Nx*Ny*Nz); 
    cudaMalloc((void**) &b_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &a_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);
    cudaMalloc((void**) &b_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);
    cudaMalloc((void**) &scaler, sizeof(float));

     
    int xy = 512/Nz;
    int blockxy = sqrt(xy);
    //dimBlock = threadsPerBlock, dimGrid = numBlocks
    dim3 dimBlock(blockxy,blockxy,Nz);
    dim3 dimGrid(Nx/dimBlock.x+1,Ny/dimBlock.y+1,1);
    
    for(int k=0; k<Nz; k++) {
     for(int j=0; j<Nx; j++) {
      for(int i=0; i<Ny; i++) {
      
      
      y[i] = 2*M_PI*(float)(i-Ny/2)/Ny;                             //  
      x[j] = 2*M_PI*(float)(j-Nx/2)/Nx;				    //
      z[k] = 2*M_PI*(float)(k-Nz/2)/Nz;				    //
      int index = i + Ny*j + Ny*Nx*k;
      
      //we start with a field a of the form a(y,x,z)    
      a[index]= acos*cos(aky*y[i] + akx*x[j] + akz*z[k]) + 		//
                asin*sin(aky*y[i] + akx*x[j] + akz*z[k]);	        //
      
      
      
      //printf("f(%.2fPI,%.2fPI,%.2fPI)=%.3f: %d ", y[i]/M_PI, x[j]/M_PI, z[k]/M_PI, a[index], index);
      
      }
      //printf("\n");
     }
    }
    
    cudaMemcpy(a_d, a, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
    
    
    cufftHandle plan;
    cufftHandle plan2;
    int n[2] = {Nx, Ny};
    
    cufftPlanMany(&plan, 2,n,NULL,1,0,NULL,1,0,CUFFT_R2C,Nz);
    cufftPlanMany(&plan2,2,n,NULL,1,0,NULL,1,0,CUFFT_C2R,Nz);
    
    
    
    cufftExecR2C(plan, a_d, a_complex_d);
    
    
    
    //now we have a field of the form a(ky,kx,z)
    
   // kInit<<<dimGrid, dimBlock>>> (kz_d, Nz);
    
    
    /**************************************************************/
    // the ZDERIV sub-routine takes an input field a(ky,kx,z) and //
    // outputs a field b(ky,kx,z) = da/dz                         // 
    /**************************************************************/ 
    b_complex_d = ZDERIV(a_complex_d, b_complex_d, kz_d, Ny, Nx, Nz);
    
    
    
    cufftExecC2R(plan2, b_complex_d, b_d);
    
    
    scaler = (float)1/(Ny*Nx);
    
    scaleReal<<<dimGrid,dimBlock>>>(b_d, scaler, Ny, Nx, Nz);
    
    cudaMemcpy(b, b_d, sizeof(cufftReal)*Ny*Nx*Nz, cudaMemcpyDeviceToHost);
    
    cudaFree(a_d); cudaFree(b_d); cudaFree(a_complex_d); cudaFree(b_complex_d); 
    cudaFree(kz_d); 
    
    //we return a real field b(y,x,z)
    return b;
    
}    

