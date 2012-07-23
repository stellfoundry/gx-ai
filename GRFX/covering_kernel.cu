//kernels for use with the covering space z derivative routines

__global__ void coveringCopy(cufftComplex* g, int nLinks, int nChains, int* ky, int* kx, cufftComplex* f) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks<=zThreads) {
    if(i<Nz && p<nLinks && n<nChains) {
      unsigned int j= i + p*Nz + n*Nz*nLinks;
      unsigned int kidx= p + nLinks*n;
      unsigned int fidx= ky[kidx] + (Ny/2+1)*kx[kidx] + i*Nx*(Ny/2+1);
      g[j].x = f[fidx].x;  
      g[j].y = f[fidx].y;    
    }
  }  
  else {  
    for(int a=0; a<nLinks/zThreads; a++) {
      if(i<Nz && n<nChains && p<zThreads) {
        unsigned int j = i + p*Nz + a*zThreads*Nz + n*Nz*zThreads*nLinks;
	unsigned int kidx = p + a*zThreads + n*zThreads*nLinks;
	unsigned int fidx = ky[kidx] + (Ny/2+1)*kx[kidx] + i*Nx*(Ny/2+1);
	g[j].x = f[fidx].x;  
        g[j].y = f[fidx].y;
      }
    }
  }    	 
}  

__global__ void coveringCopyBack(cufftComplex* f, int nLinks, int nChains, int* ky, int* kx, cufftComplex* g) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks <= zThreads) {
    if(i<Nz && p<nLinks && n<nChains) {
      unsigned int j= i + p*Nz + n*Nz*nLinks;
      unsigned int kidx= p + nLinks*n;
      unsigned int fidx= ky[kidx] + (Ny/2+1)*kx[kidx] + i*Nx*(Ny/2+1);
      f[fidx].x = g[j].x;  
      f[fidx].y = g[j].y;   
    }
  }
  else {
    for(int a=0; a<nLinks/zThreads; a++) {
      if(i<Nz && p<zThreads && n<nChains) {
        unsigned int j = i + p*Nz + a*zThreads*Nz + n*Nz*zThreads*nLinks;
	unsigned int kidx = p + a*zThreads + n*zThreads*nLinks;
        unsigned int fidx= ky[kidx] + (Ny/2+1)*kx[kidx] + i*Nx*(Ny/2+1);
        f[fidx].x = g[j].x;  
        f[fidx].y = g[j].y;   
      } 
    }
  }    
}    
    

__global__ void zderiv_covering(cufftComplex* f, int nLinks, int nChains, float* kz)
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks <= zThreads) {
    if(i<Nz && p<nLinks && n<nChains) {
      unsigned int index= i + p*Nz + n*Nz*nLinks;
      unsigned int kidx= i + p*Nz;
      f[index].x = -kz[kidx]*f[index].y;
      f[index].y = kz[kidx]*f[index].x;
    }
  }
  else {
    for(int a=0; a<nLinks/zThreads; a++) {
      if(i<Nz && p<zThreads && n<nChains) { 
        unsigned int index = i + p*Nz + a*zThreads*Nz + n*Nz*zThreads*nLinks;
	unsigned int kidx = i + p*Nz + a*zThreads*Nz;
	f[index].x = -kz[kidx]*f[index].y;
        f[index].y = kz[kidx]*f[index].x; 
      }
    }
  }    	  
}    

__global__ void scale_covering(cufftComplex* f, int nLinks, int nChains, float scaler) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks <= zThreads) {
    if(i<Nz && p<nLinks && n<nChains) {
      unsigned int index= i + p*Nz + n*Nz*nLinks;  
      f[index].x = scaler*f[index].x;
      f[index].y = scaler*f[index].y;
    }
  }
  else {
    for(int a=0; a<nLinks/zThreads; a++) {
      if(i<Nz && p<zThreads && n<nChains) {
        unsigned int index = i + p*Nz + a*zThreads*Nz + n*Nz*zThreads*nLinks;
	f[index].x = scaler*f[index].x;
        f[index].y = scaler*f[index].y;
      }
    }
  }    
}

__global__ void zeroCovering(cufftComplex* f, int nLinks, int nChains) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks <= zThreads) {
    if(i<Nz && p<nLinks && n<nChains) {
      unsigned int index= i + p*Nz + n*Nz*nLinks;
      f[index].x = 0;
      f[index].y = 0;
    }
  }
  else {
    for(int a=0; a<nLinks/zThreads; a++) {
      if(i<Nz && p<zThreads && n<nChains) { 
        unsigned int index = i + p*Nz + a*zThreads*Nz + n*Nz*zThreads*nLinks;
	f[index].x = 0;
        f[index].y = 0; 
      }
    }
  }    	  
}    

__global__ void mask_Z_covering(cufftComplex* f, int nLinks, int nChains) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks <= zThreads) {
    if(i<Nz && p<nLinks && n<nChains) {
      unsigned int index= i + p*Nz + n*Nz*nLinks;
      unsigned int idz= i + p*Nz;
      if( idz>(Nz-1)/3 && idz<2*(Nz)/3+1 ) {
        f[index].x = 0;
        f[index].y = 0;
      }
    }
  }
  else {
    for(int a=0; a<nLinks/zThreads; a++) {
      if(i<Nz && p<zThreads && n<nChains) { 
        unsigned int index = i + p*Nz + a*zThreads*Nz + n*Nz*zThreads*nLinks;
	unsigned int idz= i + p*Nz + a*zThreads*Nz;
	if( idz>(Nz-1)/3 && idz<2*(Nz)/3+1 ) {
	  f[index].x = 0;
          f[index].y = 0;
	} 
      }
    }
  }    	  
}    


__global__ void kzInitCovering(float* kz, int nLinks)
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;  
  

  if(nLinks <= zThreads) {
    if(i<Nz && p<nLinks) {
      int index = i + p*Nz;
      if(index < (Nz*nLinks)/2) 
        kz[index] = (float) index/(nLinks*Z0);
      else
        kz[index] = (float) (index-Nz*nLinks)/(nLinks*Z0);
    }	

  }
  else {
    for(int a=0; a<nLinks/zThreads; a++) {
      if(i<Nz && p<zThreads) {  
        int index = i + p*Nz + a*zThreads*Nz;
	if(index < (Nz*nLinks)/2) 
          kz[index] = (float) index/(nLinks*Z0);
        else
          kz[index] = (float) (index-Nz*nLinks)/(nLinks*Z0);
      }
    }
  }    	    
}
      
