#include <stdio.h>

__global__ void deriv(cufftComplex* f, cufftComplex* fdx, cufftComplex* fdy, 
                      cufftComplex* g, cufftComplex* gdx, cufftComplex* gdy,  
                      float* kx, float* ky, int Nx, int Ny, int Nz) 
{
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  
  
  for(int k = 0; k<Nz; k++) {
   if(idx<(Nx/2+1) && idy<Ny) {
    int index = idx + (Nx/2+1)*idy + Ny*(Nx/2+1)*k;
    
    fdx[index].x = 0;
    fdx[index].y = 0;
    fdy[index].x = 0;
    fdy[index].y = 0; 
    
    //df/dx
    fdx[index].x = -kx[index]*f[index].y;
    fdx[index].y =  kx[index]*f[index].x;
    
    //df/dy
    fdy[index].x = -ky[index]*f[index].y;
    fdy[index].y =  ky[index]*f[index].x;
    
    //dg/dx
    gdx[index].x = -kx[index]*g[index].y;
    gdx[index].y =  kx[index]*g[index].x;
    
    //dg/dy
    gdy[index].x = -ky[index]*g[index].y;
    gdy[index].y =  ky[index]*g[index].x;
    
    
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
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  
  for(int k=0; k<Nz; k++) {
   if(idx<Nx && idy<Ny) {
    int index = idx + (Nx)*idy + Ny*(Nx)*k;
    
    mult[index] = scaler*( fdx[index]*gdy[index] - fdy[index]*gdx[index] ); 
    
    
  }
 }
}  
     					      

     
                           
    
    
    
    
    		      
