/***** LINES CHANGED FOR X <-> Y MARKED BY '//' *******/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

#include <nlps.cu>


__global__ void scale(cufftReal *f, float a, int N, int N, int N);

cufftReal* NLPStest(int fkx, int fky, int fsin, int fcos, int gkx, int gky, int gsin, int gcos, int Ny, int Nx, int Nz) 
{
    //host variables
    cufftReal *f, *g, *nlps;
    
    float *y, *x;
    f = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    g = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    y = (float*) malloc(sizeof(float)*Ny);                                 //
    x = (float*) malloc(sizeof(float)*Nx);				   //
    nlps = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    
    
    
    
    //device variables
    cufftReal *f_d, *g_d, *nlps_d;
    cufftComplex *f_complex_d, *g_complex_d, *nlps_complex_d;
    float scaler;
    cudaMalloc((void**) &f_d, sizeof(cufftReal)*Nx*Ny*Nz); 
    cudaMalloc((void**) &g_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &f_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);
    cudaMalloc((void**) &g_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);
    cudaMalloc((void**) &nlps_complex_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
    cudaMalloc((void**) &nlps_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &scaler, sizeof(float));
    
    int block_size_x=2; int block_size_y=2; 
    int dimGridx, dimGridy;

    dim3 dimBlock(block_size_x, block_size_y);
    if(Ny/dimBlock.x == 0) {dimGridx = 1;}
    else dimGridx = Nx/dimBlock.x;
    if(Nx/dimBlock.y == 0) {dimGridy = 1;}
    else dimGridy = Ny/dimBlock.y;
    dim3 dimGrid(dimGridx, dimGridy);   
    
    for(int k=0; k<Nz; k++) {
     for(int j=0; j<Nx; j++) {
      for(int i=0; i<Ny; i++) {
      
      
      y[i] = 2*M_PI*(float)(i-Ny/2)/Ny;                             //  Ny, Nx not changed!!
      x[j] = 2*M_PI*(float)(j-Nx/2)/Nx;				    //
      int index = i + Ny*j + Ny*Nx*k;
      
      //fkx,fky,gkx,gky not changed!!     
      f[index]= fcos*cos( fky*y[i] + fkx*x[j]) + fsin*sin(fky*y[i] + fkx*x[j]);	        //
      g[index]= gcos*cos( gky*y[i] + gkx*x[j]) + gsin*sin(gky*y[i] + gkx*x[j]);		//
      
      
      //printf("f(%.2fPI,%.2fPI)=%.3f: %d ", y[i]/M_PI, x[j]/M_PI, f[index], index);
      
      }
      //printf("\n");
     }
    }
    
    cudaMemcpy(f_d, f, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d, g, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftHandle plan2;
    int n[2] = {Nx, Ny};
    
    cufftPlanMany(&plan, 2,n,NULL,1,0,NULL,1,0,CUFFT_R2C,Nz);
    cufftPlanMany(&plan2,2,n,NULL,1,0,NULL,1,0,CUFFT_C2R,Nz);
    
    cufftExecR2C(plan, f_d, f_complex_d);
    cufftExecR2C(plan, g_d, g_complex_d);
    
    
    
    //if nlps should receive fields on host, use these copies
    /* cudaMemcpy(f_complex, f_complex_d, sizeof(cufftComplex)*(Nz/2+1)*Ny*Nz,
                                                    cudaMemcpyDeviceToHost);
    cudaMemcpy(g_complex, g_complex_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz,
                                                    cudaMemcpyDeviceToHost); */
        
    nlps_complex_d = NLPS(f_complex_d, g_complex_d, Ny, Nx, Nz);
    
    
    
    cufftExecC2R(plan2, nlps_complex_d, nlps_d);
    
    scaler = (float)1/(Nx*Ny);
    
    scale<<<dimGrid,dimBlock>>>(nlps_d, scaler, Ny, Nx, Nz);
    
    
    
    cudaMemcpy(nlps, nlps_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    
    return nlps;
}

   
    
__global__ void scale(cufftReal* nlps, float scaler, int Ny, int Nx, int Nz)
{
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  for(int k=0; k<Nz; k++) {
   if(idy<Ny && idx<Nx) {
    int index = idy + Ny*idx + Ny*Nx*k;
    
    nlps[index] = nlps[index]*scaler;  
    
   }
  } 
}      
    
    
    
    
      
