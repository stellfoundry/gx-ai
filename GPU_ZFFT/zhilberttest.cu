#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

#include <zhilbert.cu>



cufftReal* ZHILBERTtest(int akx, int aky, int akz, int asin, int acos, int Ny, int Nx, int Nz) 
{
    //host variables
    cufftReal *a, *c;
    
    float *y, *x, *z;
    a = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    y = (float*) malloc(sizeof(float)*Ny);                                 //
    x = (float*) malloc(sizeof(float)*Nx);				   //
    z = (float*) malloc(sizeof(float)*Nz);
    c = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    
    
    
    
    //device variables
    cufftReal *a_d, *c_d;
    cufftComplex *a_complex_d, *c_complex_d;
    float scaler;
    float *kz_d;
    cudaMalloc((void**) &kz_d, sizeof(float)*(Nz));
    cudaMalloc((void**) &a_d, sizeof(cufftReal)*Nx*Ny*Nz); 
    cudaMalloc((void**) &a_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);
    cudaMalloc((void**) &c_complex_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
    cudaMalloc((void**) &c_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &scaler, sizeof(float));
    
    // int threadsPerBlock = 512;
    int block_size_x=2; int block_size_y=2; 
    int dimGridx, dimGridy;

    dim3 dimBlock(block_size_x, block_size_y);
    if(Nx/dimBlock.x == 0) {dimGridx = 1;}
    else dimGridx = Nx/dimBlock.x;
    if(Ny/dimBlock.y == 0) {dimGridy = 1;}
    else dimGridy = Ny/dimBlock.y;
    dim3 dimGrid(dimGridx, dimGridy);  
    
    //dim3 dimGrid(100,50);
    //dim3 dimBlock(8,8,8);   
    
    for(int k=0; k<Nz; k++) {
     for(int j=0; j<Nx; j++) {
      for(int i=0; i<Ny; i++) {
      
      
      y[i] = 2*M_PI*(float)(i-Ny/2)/Ny;                             //  Ny, Nx not changed!!
      x[j] = 2*M_PI*(float)(j-Nx/2)/Nx;				    //
      z[k] = 2*M_PI*(float)(k-Nz/2)/Nz;
      int index = i + Ny*j + Ny*Nx*k;
      
      //we start with a field a of the form a(y,x,z)    
      a[index]= acos*cos( aky*y[i] + akx*x[j] + akz*z[k]) + 		//
                asin*sin(aky*y[i] + akx*x[j] + akz*z[k]);	        //
      
      
      
      //printf("f(%.2fPI,%.2fPI)=%.3f: %d ", y[i]/M_PI, x[j]/M_PI, f[index], index);
      
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
    
    kInit<<<dimGrid, dimBlock>>> (kz_d, Nz);
    
    /**************************************************************/
    // the ZHILBERT sub-routine takes an input field a(ky,kx,z) and  //
    // outputs a field c(ky,kx,z) = HC[a]                                 // 
    /**************************************************************/    
    c_complex_d = ZHILBERT(a_complex_d, c_complex_d, kz_d, Ny, Nx, Nz);
        
    
    cufftExecC2R(plan2, c_complex_d, c_d);
    
    scaler = (float)1/(Ny*Nx);
    
    scaleReal<<<dimGrid,dimBlock>>>(c_d, scaler, Ny, Nx, Nz);
    
    cudaMemcpy(c, c_d, sizeof(cufftReal)*Ny*Nx*Nz, cudaMemcpyDeviceToHost);
    
    cudaFree(kz_d); cudaFree(a_d); cudaFree(c_d); 
    cudaFree(a_complex_d); cudaFree(c_complex_d);
    
    //we return a real field c(y,x,z)
    return c;
    
}    

