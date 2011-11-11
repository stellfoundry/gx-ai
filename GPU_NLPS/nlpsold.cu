#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

#include <nlps_kernel.cu>

void getf(float* a, float* a, float* a, float* a, int N, int N, int N);
void getfC(cufftComplex *c, cufftComplex *c, int N, int N, int N);
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
cufftReal* NLPS(int fkx, int fky, int fsin, int fcos, int gkx, int gky, int gsin, int gcos, int Nx, int Ny, int Nz) 
{
    //host variables
    cufftReal *f, *fdxR, *fdyR; 
    cufftReal *g, *gdxR, *gdyR;
    cufftComplex *f_complex, *fdx, *fdy;
    cufftComplex *g_complex, *gdx, *gdy;
    float *x, *y, *kx, *ky;
    //int Nx = 4, Ny = 8, Nz=4;
    cufftReal *multR;
    f = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    fdxR = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    fdyR = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    g = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    gdxR = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    gdyR = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    x = (float*) malloc(sizeof(float)*Nx);
    y = (float*) malloc(sizeof(float)*Ny);
    kx = (float*) malloc(sizeof(float)*(Nx/2+1)*Ny*Nz);
    ky = (float*) malloc(sizeof(float)*(Nx/2+1)*Ny*Nz);    
    f_complex = (cufftComplex*) malloc(sizeof(cufftComplex)*((Nx/2+1))*(Ny)*Nz);
    fdx = (cufftComplex*) malloc(sizeof(cufftComplex)*((Nx/2+1))*(Ny)*Nz);    
    fdy = (cufftComplex*) malloc(sizeof(cufftComplex)*((Nx/2+1))*(Ny)*Nz);
    g_complex = (cufftComplex*) malloc(sizeof(cufftComplex)*((Nx/2+1))*(Ny)*Nz);
    gdx = (cufftComplex*) malloc(sizeof(cufftComplex)*((Nx/2+1))*(Ny)*Nz);    
    gdy = (cufftComplex*) malloc(sizeof(cufftComplex)*((Nx/2+1))*(Ny)*Nz);
    multR = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    
    //device variables
    float *kx_d, *ky_d;
    cufftReal *f_d, *fdxR_d, *fdyR_d;
    cufftReal *g_d, *gdxR_d, *gdyR_d;
    cufftComplex *f_complex_d, *fdx_d, *fdy_d;
    cufftComplex *g_complex_d, *gdx_d, *gdy_d;
    float scaler;
    cufftReal *multR_d;
    cudaMalloc((void**) &kx_d, sizeof(float)*(Nx/2+1)*Ny*Nz);
    cudaMalloc((void**) &ky_d, sizeof(float)*(Nx/2+1)*Ny*Nz);    
    cudaMalloc((void**) &f_d, sizeof(cufftReal)*Nx*Ny*Nz);    
    cudaMalloc((void**) &f_complex_d, sizeof(cufftComplex)*((Nx/2+1))*(Ny)*Nz);    
    cudaMalloc((void**) &fdx_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz);    
    cudaMalloc((void**) &fdxR_d, sizeof(cufftReal)*(Nx)*Ny*Nz);
    cudaMalloc((void**) &fdy_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz); 
    cudaMalloc((void**) &fdyR_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &g_d, sizeof(cufftReal)*Nx*Ny*Nz);    
    cudaMalloc((void**) &g_complex_d, sizeof(cufftComplex)*((Nx/2+1))*(Ny)*Nz);    
    cudaMalloc((void**) &gdx_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz);    
    cudaMalloc((void**) &gdxR_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &gdy_d, sizeof(cufftComplex)*(Nx/2+1)*Ny*Nz); 
    cudaMalloc((void**) &gdyR_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &scaler, sizeof(float));
    cudaMalloc((void**) &multR_d, sizeof(cufftReal)*Nx*Ny*Nz);
    
    //set fcn on host
    
    /* 
    0 < kx < Nx/2
    -Ny/2+1 < ky < Ny/2
    */
    /* int fkx=1;
    int fky=2;
    int gkx= 1;
    int gky= 1; */
    
    
    //printf("\nf real, f(x,y): cos(%dx + %dy)\n", fkx, fky);
    //printf("\ng real, g(x,y): cos(%dx + %dy)\n", gkx, gky);
    for(int k=0; k<Nz; k++) {
     for(int j=0; j<Ny; j++) {
      for(int i=0; i<Nx; i++) {
      
      
      x[i] = 2*M_PI*(float)(i-Nx/2)/Nx;
      y[j] = 2*M_PI*(float)(j-Ny/2)/Ny;
      int index = i + Nx*j + Ny*Nx*k;
      
           
      f[index]= fcos*cos( fkx*x[i] + fky*y[j]) + fsin*sin(fkx*x[i] + fky*y[j]);
      g[index]= gcos*cos( gkx*x[i] + gky*y[j]) + gsin*sin(gkx*x[i] + gky*y[j]);
      
      
      //printf("f(%.2fPI,%.2fPI)=%.3f: %d ", x[i]/M_PI, y[j]/M_PI, f[index], index);
      
      }
      //printf("\n");
     }
    } 
    //printf("\n");
    cudaMemcpy(f_d, f, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d, g, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
    
    //getf(f, f_d, x, y, Nx, Ny, Nz);
    
    
    int block_size_x=4; int block_size_y=2; 

    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid(Nx/dimBlock.x, Ny/dimBlock.y);  
    
    cufftHandle plan;
    cufftHandle plan2;
    int n[2] = {Ny, Nx};
    
    
    cufftPlanMany(&plan, 2,n,NULL,1,0,NULL,1,0,CUFFT_R2C,Nz);
    cufftPlanMany(&plan2,2,n,NULL,1,0,NULL,1,0,CUFFT_C2R,Nz);
    
    
    //cufftPlan2d(&plan2, Ny, Nx, CUFFT_C2R);
    
    cufftExecR2C(plan, f_d, f_complex_d);
    cufftExecR2C(plan, g_d, g_complex_d);
    
    //printf("f complex, F(kx,ky):\n");
    //getfC(f_complex, f_complex_d, Nx, Ny, Nz);
    //printf("\n");
    
    //return 0;
    
    kInit(kx, ky, kx_d, ky_d, Nx, Ny, Nz);
    
    
    
    deriv<<<dimGrid, dimBlock>>> (f_complex_d, fdx_d, fdy_d, g_complex_d, gdx_d, gdy_d,
                                                      kx_d, ky_d, Nx, Ny, Nz);
    
    
    //printf("deriv_x complex, F'_x(kx,ky):\n");
    //getfC(fdx, fdx_d, Nx, Ny, Nz);    
    
    //printf("\nderiv_y complex, F'_y(kx,ky):\n");
    //getfC(fdy, fdy_d, Nx, Ny, Nz);
    
    
          
    cufftExecC2R(plan2, fdx_d, fdxR_d);
    cufftExecC2R(plan2, fdy_d, fdyR_d);
    cufftExecC2R(plan2, gdx_d, gdxR_d);
    cufftExecC2R(plan2, gdy_d, gdyR_d);
     
  
    
    scaler = (float)1/(Nx*Nx*Ny*Ny);
    
    bracket<<<dimGrid,dimBlock>>> (multR_d, fdxR_d, fdyR_d, gdxR_d, gdyR_d, scaler, Nx, Ny, Nz);
    
    cudaMemcpy(multR, multR_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    
    //printf("\nderiv_x real, f'_x(x,y): \n");
    //getf(fdxR, fdxR_d, x, y, Nx, Ny, Nz);
    
    //printf("\nderiv_y real, f'_y(x,y): \n");
    //getf(fdyR, fdyR_d, x, y, Nx, Ny, Nz);
    
    //printf("\npoisson bracket result (real): (df/dx)(dg/dy)-(df/dy)(dg/dx)\n");
    //getf(multR, multR_d, x, y, Nx, Ny, Nz);
    
    //printf("\n");
    cufftDestroy(plan);
    cufftDestroy(plan2);
    
    
    cudaFree(kx_d), cudaFree(ky_d);
    cudaFree(f_d), cudaFree(fdxR_d), cudaFree(fdyR_d);
    cudaFree(g_d), cudaFree(gdxR_d), cudaFree(gdyR_d);
    cudaFree(f_complex_d), cudaFree(fdx_d), cudaFree(fdy_d);
    cudaFree(g_complex_d), cudaFree(gdx_d), cudaFree(gdy_d);
    //cudaFree(scaler);
    cudaFree(multR_d);
    
    return multR;
}

void getf(cufftReal* f, cufftReal* f_d, float* x, float* y, int Nx, int Ny, int Nz) {
  cudaMemcpy(f, f_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
  
  for(int k=0; k<Nz; k++) {  
   for(int j=0; j<Ny; j++) {
    for(int i=0; i<Nx; i++) {
      int index = i + Nx*j + Nx*Ny*k;
      printf("f(%.2fPI,%.2fPI)=%.3f ", x[i]/M_PI, y[j]/M_PI, f[index]);     
      }
      printf("\n");
    } printf("\n");
   } 
  }  
    
void getfC(cufftComplex* f, cufftComplex* f_d, int Nx, int Ny, int Nz) {
  for(int index = 0; index<(Nx/2+1)*(Ny)*(Nz); index++) {
    f[index].x = 0;
    f[index].y = 0;
  }  
  
  cudaMemcpy(f, f_d, sizeof(cufftComplex)*(Nx/2+1)*(Ny)*Nz, cudaMemcpyDeviceToHost);
    
    for(int k=0; k<Nz; k++) {
     for(int j=0; j<Ny/2+1; j++) { 
      for(int i=0; i<Nx/2+1; i++) {  
	int index = i + (Nx/2+1)*(j) + (Nx/2+1)*(Ny)*k;
	
	printf("F(%d,%d)=%.3f+i*%.3f: %d ", i, j, -f[index].x*2/(Nx*Ny), -f[index].y*2/(Nx*Ny), index);
      }
      printf("\n");
     }  
     for(int j=-Ny/2+1; j<0; j++) {
      for(int i=0; i<Nx/2+1; i++) {
        int index = (i) + (Nx/2+1)*(j+Ny) + (Nx/2+1)*(Ny)*k;
	
	printf("F(%d,%d)=%.3f+i*%.3f: %d ", i, j, -f[index].x*2/(Nx*Ny), -f[index].y*2/(Nx*Ny), index);
      }
        
      printf("\n");
     }
     printf("\n");
    } 
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
  
  for(int k=0; k<Nz; k++) {
   for(int j=0; j<Ny/2+1; j++) {
    for(int i=0; i<Nx/2+1; i++) {
      int index = i + (Nx/2+1)*j +(Ny)*(Nx/2+1)*k;
      kx[index] = 0;
      ky[index] = 0;
      //printf("kx[%d]=%f\n", index, kx[index]);
    }
   }
   for(int j=-Ny/2+1; j<0; j++) {
    for(int i=0; i<Nx/2+1; i++) {
      int index = i + (Nx/2+1)*(Ny+j) + (Nx/2+1)*(Ny)*k;
      kx[index] = 0;
      ky[index] = 0;
      //printf("kx[%d]=%f\n", index, kx[index]);
    }
   }
  }  
  cudaMemcpy(kx, kx_d, sizeof(float)*(Nx/2+1)*Ny*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(ky, ky_d, sizeof(float)*(Nx/2+1)*Ny*Nz, cudaMemcpyDeviceToHost);
  
  for(int k=0; k<Nz; k++) {
   for(int j=0; j<Ny/2+1; j++) {
    for(int i=0; i<Nx/2+1; i++) {
      int index = i + (Nx/2+1)*j +(Ny)*(Nx/2+1)*k;
      
      //printf("ky[%d]=%f\n", index, ky[index]);
    }
   }
   for(int j=-Ny/2+1; j<0; j++) {
    for(int i=0; i<Nx/2+1; i++) {
      int index = i + (Nx/2+1)*(Ny+j) + (Nx/2+1)*(Ny)*k;
     
      //printf("ky[%d]=%f\n", index, ky[index]);
    }
   }
  }  
  
  
}          

