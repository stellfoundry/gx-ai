//SWITCHED X AND Y -> a(ky,kx,z)
/***** LINES CHANGED FOR X <-> Y MARKED BY '//' *******/

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
    fdx[index].x = -ky[idx]*f[index].y;			// only 'ky' changed!!
    fdx[index].y =  ky[idx]*f[index].x;			//
    
    //df/dy
    fdy[index].x = -kx[idy]*f[index].y;			//
    fdy[index].y =  kx[idy]*f[index].x;			//
    
    //dg/dx
    gdx[index].x = -ky[idx]*g[index].y;			//
    gdx[index].y =  ky[idx]*g[index].x;			//
    
    //dg/dy
    gdy[index].x = -kx[idy]*g[index].y;			//
    gdy[index].y =  kx[idy]*g[index].x;			//
    
    
    
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
        ky[idx] = idx;					// idx, idy, Nx, Ny not changed!!
	kx[idy] = idy;					//
      } else {
        ky[idx] = idx;					//	
	kx[idy] = idy - Ny;				//
      }
    }
      
}       
     
                           
    
    
    
    
    		      
