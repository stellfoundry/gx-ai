#include <stdio.h>

 // kInit, derivB, derivC, scale, scaleReal

__global__ void kInit(float* kz, int Nz)
{
  int idz = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  
   if(idz<Nz) {
    if(idz<(Nz/2+1)) 
      kz[idz] = idz;
    else
      kz[idz] = idz - Nz;
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
  
  
  for(int idz=0; idz<Nz; idz++) {
   if(idy<(Ny/2+1) && idx<Nx) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
    
    //b(ky,kx,kz)= i*kz*a(ky,kx,kz)
    b[index].x = -kz[idz]*a[index].y;
    b[index].y = kz[idz]*a[index].x;
    
    }
  }
} 

__global__ void zhilbert(cufftComplex* a, cufftComplex* c, float* kz,
                                               int Ny, int Nx, int Nz)
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  
  
  
  for(int idz=0; idz<Nz; idz++) {
   if(idy<(Ny/2+1) && idx<Nx) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    //c(ky,kx,kz)= |kz|*a(ky,kx,kz)
    c[index].x = abs(kz[idz])*a[index].x;
    c[index].y = abs(kz[idz])*a[index].y;
    }
   
  }
}   					       
    
__global__ void scale(cufftComplex* b, float scaler, int Ny, int Nx, int Nz)
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  
  
  
  for(int idz=0; idz<Nz; idz++) {
   if(idy<(Ny/2+1) && idx<Nx) {
    int index = idy + (Ny/2+1)*idx + (Ny/2+1)*(Nx)*idz;
    
    b[index].x = scaler*b[index].x;
    b[index].y = scaler*b[index].y;

   }
  }
}  

__global__ void scaleReal(cufftReal* b, float scaler, int Ny, int Nx, int Nz)
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  
  
  
  for(int idz=0; idz<Nz; idz++) {
   if(idy<(Ny) && idx<Nx) {
    int index = idy + Ny*idx + Ny*Nx*idz;
      
    b[index] = scaler*b[index];
   } 
   
  } 
}    
