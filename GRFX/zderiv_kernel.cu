__global__ void zderiv(cufftComplex* result, cufftComplex* f, float* kz)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idx<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
    
      //result(ky,kx,kz)= i*kz*f(ky,kx,kz)
      result[index].x = -kz[idz]*f[index].y;
      result[index].y = kz[idz]*f[index].x;    
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
	result[index].x = -kz[idz]*f[index].y;
        result[index].y = kz[idz]*f[index].x;  
      }
    }
  }    	
} 				       
    
__global__ void scale(cufftComplex* b, float scaler)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + (Ny/2+1)*(Nx)*idz;
    
      b[index].x = scaler*b[index].x;
      b[index].y = scaler*b[index].y;
    }
  }
    
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
	b[index].x = scaler*b[index].x;
        b[index].y = scaler*b[index].y; 
      }
    }
  }    	
} 

__global__ void scaleReal(cufftReal* b, float scaler)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
  if(Nz<=zThreads) {
    if(idy<(Ny) && idx<Nx && idz<Nz) {
      unsigned int index = idy + Ny*idx + Ny*Nx*idz;
      
      b[index] = scaler*b[index];
    }
  } 
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
	b[index] = scaler*b[index]; 
      }
    }
  }     
}    
