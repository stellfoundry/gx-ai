__global__ void deriv(cufftComplex* f, cufftComplex* fdx, cufftComplex* fdy, float* kx, float* ky)                        
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
     //df/dy
     fdy[index].x = -ky[idy]*f[index].y;			
     fdy[index].y =  ky[idy]*f[index].x;			
    
     //df/dx
     fdx[index].x = -kx[idx]*f[index].y;			
     fdx[index].y =  kx[idx]*f[index].x;   
   }
  } 
  else {
   for(int i=0; i<Nz/zThreads; i++) { 
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
    
    //df/dy
    fdy[index].x = -ky[idy]*f[index].y;			
    fdy[index].y =  ky[idy]*f[index].x;			
    
    //df/dx
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
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
    if( idy>(Ny-1)/3 || ( idx>(Nx-1)/3 && idx<2*(Nx)/3+1 ) ) {
      mult[index].x = 0;
      mult[index].y = 0;
    }  
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
    
    
     if( idy>(Ny-1)/3 || ( idx>(Nx-1)/3 && idx<2*(Nx)/3+1 ) ) {
       mult[index].x = 0;
       mult[index].y = 0;
     }  
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
  

__global__ void bracket(cufftReal* mult, cufftReal* fdx, cufftReal* fdy, 
                      cufftReal* gdx, cufftReal* gdy, float scaler)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
   if(idy<(Ny) && idx<Nx && idz<Nz ) {
    unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz;
    
    
    mult[index] = scaler*( (fdx[index])*(gdy[index]) - (fdy[index])*(gdx[index]) );  
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny) && idx<Nx && idz<zThreads ) {
    unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz + Nx*Ny*zThreads*i;
    
    
    mult[index] = scaler*( (fdx[index])*(gdy[index]) - (fdy[index])*(gdx[index]) );  
    }
   }
  } 
 
}  



