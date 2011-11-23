#include <stdio.h>

 // kInit, derivB, derivC, scale, scaleReal

__global__ void kInit(float* kz, int Nz)
{
  //int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  for(int idx=0; idx<Nz; idx++) {
    if(idx<(Nz/2+1)) 
      kz[idx] = idx;
    else
      kz[idx] = idx - Nz;
  } 
  
  /* for(int i=0; i<(Nz/2+1); i++) {
    kz[i] = i;
  } */
  
  
} 

__global__ void zderiv(cufftComplex* a, cufftComplex* b, float* kz,
                                               int Ny, int Nx, int Nz)
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  
  for(int k = 0; k<Nz; k++) {
   if(idy<(Ny/2+1) && idx<Nx) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*k;
    
    
    
    //b(ky,kx,kz)= i*kz*a(ky,kx,kz)
    b[index].x = -kz[k]*a[index].y;
    b[index].y = kz[k]*a[index].x;
    
   }
  }
} 

__global__ void zhilbert(cufftComplex* a, cufftComplex* c, float* kz,
                                               int Ny, int Nx, int Nz)
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  
  for(int k = 0; k<Nz; k++) {
   if(idy<(Ny/2+1) && idx<Nx) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*k;
    
    //c(ky,kx,kz)= |kz|*a(ky,kx,kz)
    c[index].x = abs(kz[k])*a[index].x;
    c[index].y = abs(kz[k])*a[index].y;
    
   }
  }
}   					       
    
__global__ void scale(cufftComplex* b, float scaler, int Ny, int Nx, int Nz)
{
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  for(int k=0; k<Nz; k++) {
   if(idy<Ny/2+1 && idx<Nx) {
    int index = idy + (Ny/2+1)*idx + (Ny/2+1)*(Nx)*k;
    
    b[index].x = scaler*b[index].x;
    b[index].y = scaler*b[index].y;

   }
  }
}  

__global__ void scaleReal(cufftReal* b, float scaler, int Ny, int Nx, int Nz)
{
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  for(int k=0; k<Nz; k++) {
   if(idy<Ny && idx<Nx) {
    int index = idy + Ny*idx + Ny*Nx*k;
      
    b[index] = scaler*b[index];
    
   }
  } 
}    
