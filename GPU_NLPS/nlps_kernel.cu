//SWITCHED X AND Y -> a(ky,kx,z)

#include <stdio.h>

__global__ void deriv(cufftComplex* f, cufftComplex* fdx, cufftComplex* fdy, 
                      cufftComplex* g, cufftComplex* gdx, cufftComplex* gdy,  
                      float* kx, float* ky, int Nx, int Ny, int Nz) 
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  
  for(int k = 0; k<Nz; k++) {
   if(idx<(Nx/2+1) && idy<Ny) {
    int index = idx + (Nx/2+1)*idy + Ny*(Nx/2+1)*k;
    
    
    
    //df/dx
    fdx[index].x = -kx[idx]*f[index].y;
    fdx[index].y =  kx[idx]*f[index].x;
    
    //df/dy
    fdy[index].x = -ky[idy]*f[index].y;
    fdy[index].y =  ky[idy]*f[index].x;
    
    //dg/dx
    gdx[index].x = -kx[idx]*g[index].y;
    gdx[index].y =  kx[idx]*g[index].x;
    
    //dg/dx
    gdy[index].x = -ky[idy]*g[index].y;
    gdy[index].y =  ky[idy]*g[index].x;
    
    
    /* //da/dx
    
    fdx[index].x = -(index%(Nx/2+1))*f[index].y;  
    fdx[index].y = (index%(Nx/2+1))*f[index].x;
    
    //da/dy
    fdy[index].x = -(index / (Nx/2+1))*f[index].y;
    fdy[index].y = (index / (Nx/2+1))*f[index].x; */
  }
 }
}  

__global__ void bracket(cufftReal* mult, cufftReal* fdx, cufftReal* fdy, 
                      cufftReal* gdx, cufftReal* gdy, float scaler, int Nx, int Ny, int Nz)
{
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  for(int k=0; k<Nz; k++) {
   if(idx<Nx && idy<Ny) {
    int index = idx + (Nx)*idy + Nx*(Ny)*k;
    
    mult[index] = scaler*( fdx[index]*gdy[index] - fdy[index]*gdx[index] ); 
    
    
  }
 }
}  
     					      
__global__ void kInit(float* kx, float* ky, int Nx, int Ny, int Nz) 
{
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  
    if(idx<Nx/2+1 && idy<Ny) {
      
      if(idy<Ny/2+1) {
        kx[idx] = idx;
	ky[idy] = idy;
      } else {
        kx[idx] = idx;
	ky[idy] = idy - Ny;
      }
    }
      
}       
     
                           
    
    
    
    
    		      
