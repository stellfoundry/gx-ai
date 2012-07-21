//SWITCHED X AND Y -> a(ky,kx,z)
/***** LINES CHANGED FOR X <-> Y MARKED BY '//' *******/

#include <stdio.h>
#include "constants.h"


__global__ void deriv(cufftComplex* f, cufftComplex* fdx, cufftComplex* fdy, float* kx, float* ky)                        
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
  if(Nz<=64) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
     //df/dx
     fdy[index].x = -ky[idy]*f[index].y;			
     fdy[index].y =  ky[idy]*f[index].x;			
    
     //df/dy
     fdx[index].x = -kx[idx]*f[index].y;			
     fdx[index].y =  kx[idx]*f[index].x;   
   }
  } 
  else {
   for(int i=0; i<Nz/64; i++) { 
    if(idy<(Ny/2+1) && idx<Nx && idz<64) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*64*i;
    
    //df/dx
    fdy[index].x = -ky[idy]*f[index].y;			
    fdy[index].y =  ky[idy]*f[index].x;			
    
    //df/dy
    fdx[index].x = -kx[idx]*f[index].y;			
    fdx[index].y =  kx[idx]*f[index].x;			
    }
   }
  } 
}  

__global__ void mask(cufftComplex* mult) 
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=64) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
    if( (idy<(Ny/3+1) || (idx<(Nx/3+1) || idx>(2*Nx/3+1))) == false) {
      mult[index].x = 0;
      mult[index].y = 0;
    }  
   }
  }
  else {
   for(int i=0; i<Nz/64; i++) {
    if(idy<(Ny/2+1) && idx<Nx && idz<64) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*64*i;
    
    
     if( (idy<(Ny/3+1) || (idx<(Nx/3+1) || idx>(2*Nx/3+1))) == false) {
       mult[index].x = 0;
       mult[index].y = 0;
     }  
    }
   }
  }
    
}      
  
  

__global__ void bracket(cufftReal* mult, cufftReal* fdx, cufftReal* fdy, 
                      cufftReal* gdx, cufftReal* gdy, float scaler)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=64) {
   if(idy<(Ny) && idx<Nx && idz<Nz ) {
    unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz;
    
    
    mult[index] = scaler*( (fdx[index])*(gdy[index]) - (fdy[index])*(gdx[index]) );  
   }
  }
  else {
   for(int i=0; i<Nz/64; i++) {
    if(idy<(Ny) && idx<Nx && idz<64 ) {
    unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz + Nx*Ny*64*i;
    
    
    mult[index] = scaler*( (fdx[index])*(gdy[index]) - (fdy[index])*(gdx[index]) );  
    }
   }
  } 
 
}  
     					      
__global__ void kInit(float* kx, float* ky) 
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  
  
  
    if(idy<Ny/2+1 && idx<Nx) {
      
      ky[idy] = idy;
      
      if(idx<Nx/2+1) {					
	kx[idx] = idx;					
      } else {						
	kx[idx] = idx - Nx;				
      }
    }
      
}       
     
__global__ void scale(cufftComplex* f, float scaler)
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=64) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    f[index].x = f[index].x*scaler;  
    f[index].y = f[index].y*scaler;  
    
   }
  }
  else {
   for(int i=0; i<Nz/64; i++) {
    if(idy<(Ny) && idx<Nx && idz<64) {
    int index = idy + (Ny)*idx + Nx*(Ny)*idz + Nx*Ny*64*i;
    
    f[index].x = f[index].x*scaler;  
    f[index].y = f[index].y*scaler;  
    
    }
   }
  }    
}                               

__global__ void scale(cufftReal* nlps, float scaler)
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=64) {
   if(idy<(Ny) && idx<Nx && idz<Nz) {
    int index = idy + (Ny)*idx + Nx*(Ny)*idz;
    
    nlps[index] = nlps[index]*scaler;  
    
   }
  }
  else {
   for(int i=0; i<Nz/64; i++) {
    if(idy<(Ny) && idx<Nx && idz<64) {
    int index = idy + (Ny)*idx + Nx*(Ny)*idz + Nx*Ny*64*i;
    
    nlps[index] = nlps[index]*scaler;  
    
    }
   }
  }    
}      
    
    
    
    		      
