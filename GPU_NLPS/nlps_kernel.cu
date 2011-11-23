//SWITCHED X AND Y -> a(ky,kx,z)
/***** LINES CHANGED FOR X <-> Y MARKED BY '//' *******/

#include <stdio.h>

__global__ void deriv(cufftComplex* f, cufftComplex* fdx, cufftComplex* fdy, 
                      cufftComplex* g, cufftComplex* gdx, cufftComplex* gdy,  
                      float* kx, float* ky, int Ny, int Nx, int Nz) 
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  
  for(int k = 0; k<Nz; k++) {
   if(idy<(Ny/2+1) && idx<Nx) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*k;
    
    
    
    //df/dx
    fdy[index].x = -ky[idy]*f[index].y;			// only 'ky' changed!!
    fdy[index].y =  ky[idy]*f[index].x;			//
    
    //df/dy
    fdx[index].x = -kx[idx]*f[index].y;			//
    fdx[index].y =  kx[idx]*f[index].x;			//
    
    //dg/dx
    gdy[index].x = -ky[idy]*g[index].y;			//
    gdy[index].y =  ky[idy]*g[index].x;			//
    
    //dg/dy
    gdx[index].x = -kx[idx]*g[index].y;			//
    gdx[index].y =  kx[idx]*g[index].x;			//
    
    
    
  }
 }
}  

__global__ void bracket(cufftReal* mult, cufftReal* fdx, cufftReal* fdy, 
                      cufftReal* gdx, cufftReal* gdy, float scaler, int Ny, int Nx, int Nz)
{
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  for(int k=0; k<Nz; k++) {
   if(idy<Ny && idx<Nx) {
    int index = idy + (Ny)*idx + Ny*(Nx)*k;
    
    mult[index] = scaler*( fdx[index]*gdy[index] - fdy[index]*gdx[index] ); 
    
    
    
    
  }
 }
}  
     					      
__global__ void kInit(float* kx, float* ky, int Ny, int Nx, int Nz) 
{
  int idy = blockIdx.y*blockDim.y+threadIdx.y;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  
  
    if(idy<Ny/2+1 && idx<Nx) {
      
      if(idx<Nx/2+1) {
        ky[idy] = idy;					// idx, idy, Ny, Nx not changed!!
	kx[idx] = idx;					//
      } else {
        ky[idy] = idy;					//	
	kx[idx] = idx - Nx;				//
      }
    }
      
}       
     
                           
    
    
    
    
    		      
