__global__ void zderiv(cufftComplex* f, float* kz)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
    
      //result(ky,kx,kz)= i*kz*f(ky,kx,kz)
      f[index].x = -kz[idz]*f[index].y;
      f[index].y = kz[idz]*f[index].x;    
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
	f[index].x = -kz[idz]*f[index].y;
        f[index].y = kz[idz]*f[index].x;  
      }
    }
  }    	
} 
  			        	  
__global__ void mask_Z(cufftComplex* mult) 
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
    if( idy>(Ny-1)/3 || ( idx>(Nx-1)/3 && idx<2*(Nx)/3+1 ) || ( idz>(Nz-1)/3 && idz<2*(Nz)/3+1 ) ) {
      mult[index].x = 0;
      mult[index].y = 0;
    }  
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
    
    
     if( idy>(Ny-1)/3 || ( idx>(Nx-1)/3 && idx<2*(Nx)/3+1 ) || ( idz>(Nz-1)/3 && idz<2*(Nz)/3+1 ) ) {
       mult[index].x = 0;
       mult[index].y = 0;
     }  
    }
   }
  }
    
}   

