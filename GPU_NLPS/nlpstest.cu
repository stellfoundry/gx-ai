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

cufftReal* NLPStest(int fkx, int fky, int fsin, int fcos, int gkx, int gky, int gsin, int gcos, int Nx, int Ny, int Nz) 
{
    //host variables
    cufftReal *f, *g, *nlps;
    
    float *y, *x;
    f = (cufftReal*) malloc(sizeof(cufftReal)*Ny*Nx*Nz);
    g = (cufftReal*) malloc(sizeof(cufftReal)*Ny*Nx*Nz);
    x = (float*) malloc(sizeof(float)*Nx);
    y = (float*) malloc(sizeof(float)*Ny);
    nlps = (cufftReal*) malloc(sizeof(cufftReal)*Ny*Nx*Nz);
    
    
    
    
    //device variables
    cufftReal *f_d, *g_d, *nlps_d;
    cufftComplex *f_complex_d, *g_complex_d, *nlps_complex_d;
    float scaler;
    cudaMalloc((void**) &f_d, sizeof(cufftReal)*Ny*Nx*Nz); 
    cudaMalloc((void**) &g_d, sizeof(cufftReal)*Ny*Nx*Nz);
    cudaMalloc((void**) &f_complex_d, sizeof(cufftComplex)*((Nx/2+1))*(Ny)*Nz);
    cudaMalloc((void**) &g_complex_d, sizeof(cufftComplex)*((Nx/2+1))*(Ny)*Nz);
    cudaMalloc((void**) &nlps_complex_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz);
    cudaMalloc((void**) &nlps_d, sizeof(cufftReal)*Ny*Nx*Nz);
    cudaMalloc((void**) &scaler, sizeof(float));
    
    int block_size_x=2; int block_size_y=2; 
    int dimGridx, dimGridy;

    dim3 dimBlock(block_size_x, block_size_y);
    if(Nx/dimBlock.x == 0) {dimGridx = 1;}
    else dimGridx = Nx/dimBlock.x;
    if(Ny/dimBlock.y == 0) {dimGridy = 1;}
    else dimGridy = Ny/dimBlock.y;
    dim3 dimGrid(dimGridx, dimGridy);   
    
    for(int k=0; k<Nz; k++) {
     for(int j=0; j<Ny; j++) {
      for(int i=0; i<Nx; i++) {
      
      
      x[i] = 2*M_PI*(float)(i-Nx/2)/Nx;
      y[j] = 2*M_PI*(float)(j-Ny/2)/Ny;
      int index = i + Nx*j + Nx*Ny*k;
      
           
      f[index]= fcos*cos( fkx*x[i] + fky*y[j]) + fsin*sin(fkx*x[i] + fky*y[j]);
      g[index]= gcos*cos( gkx*x[i] + gky*y[j]) + gsin*sin(gkx*x[i] + gky*y[j]);
      
      
      //printf("f(%.2fPI,%.2fPI)=%.3f: %d ", y[i]/M_PI, x[j]/M_PI, f[index], index);
      
      }
      //printf("\n");
     }
    }
    
    cudaMemcpy(f_d, f, sizeof(cufftReal)*Ny*Nx*Nz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d, g, sizeof(cufftReal)*Ny*Nx*Nz, cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftHandle plan2;
    int n[2] = {Ny, Nx};
    
    cufftPlanMany(&plan, 2,n,NULL,1,0,NULL,1,0,CUFFT_R2C,Nz);
    cufftPlanMany(&plan2,2,n,NULL,1,0,NULL,1,0,CUFFT_C2R,Nz);
    
    cufftExecR2C(plan, f_d, f_complex_d);
    cufftExecR2C(plan, g_d, g_complex_d);
    
    
    
    //if nlps should receive fields on host, use these copies
    /* cudaMemcpy(f_complex, f_complex_d, sizeof(cufftComplex)*(Nz/2+1)*Nx*Nz,
                                                    cudaMemcpyDeviceToHost);
    cudaMemcpy(g_complex, g_complex_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz,
                                                    cudaMemcpyDeviceToHost); */
        
    nlps_complex_d = NLPS(f_complex_d, g_complex_d, Nx, Ny, Nz);
    
    
    
    cufftExecC2R(plan2, nlps_complex_d, nlps_d);
    
    scaler = (float)1/(Ny*Nx);
    
    scale<<<dimGrid,dimBlock>>>(nlps_d, scaler, Nx, Ny, Nz);
    
    
    
    cudaMemcpy(nlps, nlps_d, sizeof(cufftReal)*Ny*Nx*Nz, cudaMemcpyDeviceToHost);
    
    return nlps;
}

   
    
__global__ void scale(cufftReal* nlps, float scaler, int Nx, int Ny, int Nz)
{
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  for(int k=0; k<Nz; k++) {
   if(idx<Nx && idy<Ny) {
    int index = idx + Nx*idy + Nx*Ny*k;
    
    nlps[index] = nlps[index]*scaler;  
    
   }
  } 
}      
    
    
    
    
      
