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
      
  
  
  

