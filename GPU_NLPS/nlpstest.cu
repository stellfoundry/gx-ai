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

//void getfcn(cufftReal* fcn, cufftReal* fcn_d, int Nx, int Ny, int Nz);
//void getfcnC(cufftComplex* fcn, cufftComplex* fcn_d, int Nx, int Ny, int Nz);

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

void getfcn(cufftReal* fcn_d, int Nx, int Ny, int Nz) {
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

__global__ void scale(cufftReal *f, float a, int N, int N, int N);
__global__ void zeroC(cufftComplex *f, int N, int N, int N);
__global__ void zero(cufftReal *f, int N, int N, int N);


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
    cufftReal *f_d, *g_d, *nlps_d, *fdxR_d, *fdyR_d, *gdxR_d, *gdyR_d;
    cufftComplex *f_complex_d, *g_complex_d, *nlps_complex_d;
    
    float scaler;
    float *kx_d, *ky_d;
    cudaMalloc((void**) &ky_d, sizeof(float)*(Ny/2+1));                                 
    cudaMalloc((void**) &kx_d, sizeof(float)*(Nx));     
    cudaMalloc((void**) &f_d, sizeof(cufftReal)*Nx*Ny*Nz); 
    cudaMalloc((void**) &g_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &f_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);
    cudaMalloc((void**) &g_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);
    cudaMalloc((void**) &fdxR_d, sizeof(cufftReal)*(Ny)*Nx*Nz);
    cudaMalloc((void**) &fdyR_d, sizeof(cufftReal)*Ny*Nx*Nz);
    cudaMalloc((void**) &gdxR_d, sizeof(cufftReal)*Ny*Nx*Nz);
    cudaMalloc((void**) &gdyR_d, sizeof(cufftReal)*Ny*Nx*Nz);
    cudaMalloc((void**) &nlps_complex_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
    cudaMalloc((void**) &nlps_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &scaler, sizeof(float));
    
    /*int block_size_x=2; int block_size_y=2; 
    int dimGridx, dimGridy;

    dim3 dimBlock(block_size_x, block_size_y);
    if(Ny/dimBlock.x == 0) {dimGridx = 1;}
    else dimGridx = Nx/dimBlock.x;
    if(Nx/dimBlock.y == 0) {dimGridy = 1;}
    else dimGridy = Ny/dimBlock.y;
    dim3 dimGrid(dimGridx, dimGridy); */ 
    
    int xy = 512/Nz;
    int blockxy = sqrt(xy);
    //dimBlock = threadsPerBlock, dimGrid = numBlocks
    dim3 dimBlock(blockxy,blockxy,Nz);
    dim3 dimGrid(Nx/dimBlock.x+1,Ny/dimBlock.y+1,1);
    //if(dimGrid.x == 0) {dimGrid.x = 1;}
    //if(dimGrid.y == 0) {dimGrid.y = 1;}
    //if(dimGrid.z == 0) {dimGrid.z = 1;}    
    
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

    					    
	
    kInit<<<dimGrid, dimBlock>>> (kx_d, ky_d, Ny, Nx, Nz);
    
    
    
    						    
    
    zero<<<dimGrid, dimBlock>>> (fdxR_d, Ny, Nx, Nz);
    zero<<<dimGrid, dimBlock>>> (fdyR_d, Ny, Nx, Nz);
    zero<<<dimGrid, dimBlock>>> (gdxR_d, Ny, Nx, Nz);
    zero<<<dimGrid, dimBlock>>> (gdyR_d, Ny, Nx, Nz);
    zero<<<dimGrid, dimBlock>>> (nlps_d, Ny, Nx, Nz);
    zeroC<<<dimGrid, dimBlock>>> (nlps_complex_d, Ny, Nx, Nz);
    for(int index=0; index<Nx*Ny*Nz; index++) {
      nlps[index] = 0;
    } 
    
    
        
    nlps_complex_d= NLPS(f_complex_d, fdxR_d, fdyR_d, g_complex_d, gdxR_d, gdyR_d, kx_d, ky_d, Ny, Nx, Nz, 0);
    
    
    
    //nlps_complex_d= NLPS(f_complex_d, fdxR_d, fdyR_d, g_complex_d, gdxR_d, gdyR_d, kx_d, ky_d, Ny, Nx, Nz, 1);
    
    //nlps_complex_d= NLPS(f_complex_d, fdxR_d, fdyR_d, g_complex_d, gdxR_d, gdyR_d, kx_d, ky_d, Ny, Nx, Nz, 2);    
    
    
    cufftExecC2R(plan2, nlps_complex_d, nlps_d);
    
    
    
    scaler = (float)1/(Nx*Ny);
    
    scale<<<dimGrid,dimBlock>>>(nlps_d, scaler, Ny, Nx, Nz);
    
    cudaFree(fdxR_d); cudaFree(fdyR_d); cudaFree(gdxR_d); cudaFree(gdyR_d); 
    cudaFree(nlps_complex_d); cudaFree(f_complex_d); cudaFree(g_complex_d);
    cudaFree(f_d); cudaFree(g_d); cudaFree(kx_d); cudaFree(ky_d);

    cudaMemcpy(nlps, nlps_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    
    cudaFree(nlps_d);
    
    
    
    return nlps;
}

   
    
__global__ void scale(cufftReal* nlps, float scaler, int Ny, int Nx, int Nz)
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
   if(idy<(Ny) && idx<Nx && idz<Nz) {
    int index = idy + (Ny)*idx + Nx*(Ny)*idz;
    
    nlps[index] = nlps[index]*scaler;  
    
   }
   
}      

__global__ void zeroC(cufftComplex* f, int Ny, int Nx, int Nz) 
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    f[index].x = 0;
    f[index].y = 0;
    }
  
}    

__global__ void zero(cufftReal* f, int Ny, int Nx, int Nz) 
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
   if(idy<(Ny) && idx<Nx && idz<Nz) {
    int index = idy + (Ny)*idx + Nx*(Ny)*idz;
    
    f[index] = 0;
    
    }
  
}    

    
    
    
      
