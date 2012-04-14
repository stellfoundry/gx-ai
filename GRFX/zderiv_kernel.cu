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


__global__ void coveringCopy(cufftComplex* g, int nLinks, int nChains, int* ky, int* kx, cufftComplex* f) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int p = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int n = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(i<Nz && p<nLinks && n<nChains) {
    unsigned int j= i + p*Nz + 2*n*Nz*nLinks;
    g[j].x = f[ky[p+nLinks*n] + (Ny/2+1)*kx[p+nLinks*n] + i*Nx*(Ny/2+1)].x;  
    g[j].y = f[ky[p+nLinks*n] + (Ny/2+1)*kx[p+nLinks*n] + i*Nx*(Ny/2+1)].y;    
    g[j+Nz*nLinks].x = -f[ky[p+nLinks*n] + (Ny/2+1)*kx[p+nLinks*n] + i*Nx*(Ny/2+1)].x;
    g[j+Nz*nLinks].y = -f[ky[p+nLinks*n] + (Ny/2+1)*kx[p+nLinks*n] + i*Nx*(Ny/2+1)].y;
  }
}  

__global__ void coveringCopyBack(cufftComplex* f, int nLinks, int nChains, int* ky, int* kx, cufftComplex* g) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int p = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int n = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(i<Nz && p<nLinks && n<nChains) {
    unsigned int j= i + p*Nz + 2*n*Nz*nLinks;
    f[ky[p+nLinks*n] + (Ny/2+1)*kx[p+nLinks*n] + i*Nx*(Ny/2+1)].x = g[j].x;  
    f[ky[p+nLinks*n] + (Ny/2+1)*kx[p+nLinks*n] + i*Nx*(Ny/2+1)].y = g[j].y;   
  }
}    
    

__global__ void zderiv_covering(cufftComplex* f, int nLinks, int nChains, float* kz)
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int p = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int n = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(i<2*Nz && p<nLinks && n<nChains) {
    unsigned int index= i + 2*p*Nz + n*2*Nz*nLinks;
    f[index].x = -kz[i + 2*p*Nz]*f[index].y;
    f[index].y = kz[i + 2*p*Nz]*f[index].x;
  }
}    

__global__ void scale_covering(cufftComplex* f, int nLinks, int nChains, float scaler) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int p = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int n = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(i<Nz && p<nLinks && n<nChains) {
    unsigned int index= i + p*Nz + 2*n*Nz*nLinks;  
    f[index].x = scaler*f[index].x;
    f[index].y = scaler*f[index].y;
  }
}

__global__ void kzInitCovering(float* kz, int nLinks)
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int p = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;  
  
  if(i<2*Nz && p<nLinks) {
    int index = i + 2*p*Nz;
    if(index < (2*Nz*nLinks)/2+1) 
      kz[index] = (float) index/(nLinks*X0);
    else
      kz[index] = (float) (index-2*Nz*nLinks)/(nLinks*X0); // need X0 -> Z0
  }
}
      
  
  
  

