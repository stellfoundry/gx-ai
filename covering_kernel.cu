//kernels for use with the covering space z derivative routines

__global__ void coveringCopy(cufftComplex* g, int nLinks, int nChains, int* ky, int* kx, cufftComplex* f, int icovering) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks<=zthreads) {
    if(i<nz && p<nLinks && n<nChains) {
      unsigned int j= i + p*nz + n*nz*nLinks*icovering; 
      unsigned int j2= (nz*nLinks*icovering - (i+p*nz)) + n*nz*nLinks*icovering;
      unsigned int kidx= p + nLinks*n;
      unsigned int fidx= ky[kidx] + (ny/2+1)*kx[kidx] + i*nx*(ny/2+1);
      // if(ky[kidx] == 0) { //&& kx[kidx] > 6) {
//         g[j].x = 0;
// 	g[j].y = 0;
//       } else{	        
	g[j] = f[fidx]; 
        if(icovering == 2) {
          g[j2].x = -g[j].x;
	  g[j2].y = -g[j].y;
        }
        
      //} 
         
    }
  }  
  else {  
    for(int a=0; a<nLinks/zthreads; a++) {
      if(i<nz && n<nChains && p<zthreads) {
        unsigned int P = p+a*zthreads;
	unsigned int j = i + P*nz + n*nz*nLinks*icovering;
        unsigned int j2 = (nz*nLinks*icovering - (i+p*nz)) + n*nz*nLinks*icovering;
	unsigned int kidx = P + n*nLinks;
	unsigned int fidx = ky[kidx] + (ny/2+1)*kx[kidx] + i*nx*(ny/2+1);
	//if(ky[kidx] == 0 && kx[kidx] > 9) {
          //g[j].x = 0;
	  //g[j].y = 0;
	//} else{	        
	g[j] = f[fidx]; 
        if(icovering == 2) {
          g[j2].x = -g[j].x;
	  g[j2].y = -g[j].y;
        }
	//} 
      }
    }
  }    	 
}  

__global__ void coveringCopyBack(cufftComplex* f, int nLinks, int nChains, int* ky, int* kx, cufftComplex* g) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks <= zthreads) {
    if(i<nz && p<nLinks && n<nChains) {
      unsigned int j= i + p*nz + n*nz*nLinks;
      unsigned int kidx= p + nLinks*n;
      unsigned int fidx= ky[kidx] + (ny/2+1)*kx[kidx] + i*nx*(ny/2+1);
      f[fidx].x = g[j].x;  
      f[fidx].y = g[j].y;   
    }
  }
  else {
    for(int a=0; a<nLinks/zthreads; a++) {
      if(i<nz && p<zthreads && n<nChains) {
        unsigned int P = p+a*zthreads;
	unsigned int j = i + P*nz + n*nz*nLinks;
	unsigned int kidx = P + n*nLinks;
        unsigned int fidx= ky[kidx] + (ny/2+1)*kx[kidx] + i*nx*(ny/2+1);
        f[fidx].x = g[j].x;  
        f[fidx].y = g[j].y;   
      } 
    }
  }    
}    
    

__global__ void zderiv_covering(cufftComplex* f, int nLinks, int nChains, float* kz, int icovering)
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks*icovering <= zthreads) {
    if(i<nz && p<icovering*nLinks && n<nChains) {
      unsigned int index= i + p*nz + n*nz*nLinks*icovering;
      unsigned int kidx= i + p*nz;
      
      cuComplex tmp;
      tmp.x = -kz[kidx]*f[index].y;
      tmp.y = kz[kidx]*f[index].x;      
      f[index] = tmp;
      
    }
  }
  else {
    for(int a=0; a<nLinks*icovering/zthreads; a++) {
      if(i<nz && p<zthreads && n<nChains) { 
        unsigned int P = p+a*zthreads;
	unsigned int index = i + P*nz + n*nz*nLinks*icovering;
	unsigned int kidx = i + P*nz;
		
	cuComplex tmp;
	tmp.x = -kz[kidx]*f[index].y;
	tmp.y = kz[kidx]*f[index].x;      
	f[index] = tmp; 
      }
    }
  }    	  
}

__global__ void zderiv_abs_covering(cufftComplex* f, int nLinks, int nChains, float* kz, int icovering)
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks*icovering <= zthreads) {
    if(i<nz && p<nLinks*icovering && n<nChains) {
      unsigned int index= i + p*nz + n*nz*nLinks*icovering;
      unsigned int kidx= i + p*nz;
      f[index] = abs(kz[kidx])*f[index];
    }
  }
  else {
    for(int a=0; a<nLinks*icovering/zthreads; a++) {
      if(i<nz && p<zthreads && n<nChains) { 
        unsigned int P = p+a*zthreads;
	unsigned int index = i + P*nz + n*nz*nLinks*icovering;
	unsigned int kidx = i + P*nz;
	f[index] = abs(kz[kidx])*f[index];
      }
    }
  }    	  
}
    
__global__ void scale_covering(cufftComplex* f, int nLinks, int nChains, float scaler) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks <= zthreads) {
    if(i<nz && p<nLinks && n<nChains) {
      unsigned int index= i + p*nz + n*nz*nLinks;  
      f[index] = scaler*f[index];
    }
  }
  else {
    for(int a=0; a<nLinks/zthreads; a++) {
      if(i<nz && p<zthreads && n<nChains) {
        unsigned int P = p+a*zthreads;
	unsigned int index = i + P*nz + n*nz*nLinks;
	f[index] = scaler*f[index];
      }
    }
  }    
}

__global__ void zeroCovering(cufftComplex* f, int nLinks, int nChains, int icovering) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks*icovering <= zthreads) {
    if(i<nz && p<nLinks*icovering && n<nChains) {
      unsigned int index= i + p*nz + n*nz*nLinks*icovering;
      f[index].x = 0;
      f[index].y = 0;
    }
  }
  else {
    for(int a=0; a<nLinks*icovering/zthreads; a++) {
      if(i<nz && p<zthreads && n<nChains) { 
        unsigned int P = p+a*zthreads;
	unsigned int index = i + P*nz + n*nz*nLinks*icovering;
	f[index].x = 0;
        f[index].y = 0; 
      }
    }
  }    	  
}    

__global__ void coveringBounds(cuComplex* f, int nLinks, int nChains, int* ky)
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks <= zthreads) {
    if( i==0 && p==0 && n<nChains) {
      unsigned int index= i + p*nz + n*nz*nLinks;
      unsigned int kidx= p + nLinks*n;
      if( ky[kidx] != 0 ) {      
	f[index].x = 0;
	f[index].y = 0;
      }
    }
  }
  else {
    for(int a=0; a<nLinks/zthreads; a++) {
      if( (i==0) && p<zthreads && n<nChains) { 
        unsigned int P = p+a*zthreads;
	unsigned int index = i + P*nz + n*nz*nLinks;
	unsigned int kidx= P + nLinks*n;
	if(ky[kidx] != 0 && (p==0 && i==0) ) {      
	  f[index].x = 0;
	  f[index].y = 0;
	}
      }
    }
  }    	  
}    

__global__ void mask_Z_covering(cufftComplex* f, int nLinks, int nChains) 
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nLinks <= zthreads) {
    if(i<nz && p<nLinks && n<nChains) {
      unsigned int index= i + p*nz + n*nz*nLinks;
      unsigned int idz= i + p*nz;
      if( idz>(nz*nLinks-1)/3 && idz<2*(nz*nLinks/3)+1 ) {
        f[index].x = 0;
        f[index].y = 0;
      }
    }
  }
  else {
    for(int a=0; a<nLinks/zthreads; a++) {
      if(i<nz && p<zthreads && n<nChains) { 
        unsigned int P = p+a*zthreads;
	unsigned int index = i + P*nz + n*nz*nLinks;
	unsigned int idz= i + P*nz;
	if( idz>(nz-1)/3 && idz<2*(nz)/3+1 ) {
	  f[index].x = 0;
          f[index].y = 0;
	} 
      }
    }
  }    	  
}    


__global__ void kzInitCovering(float* kz, int nLinks, bool NO_ZDERIV_COVERING, int icovering)
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int p = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;  
  

  if(nLinks*icovering <= zthreads) {
    if(i<nz && p<nLinks*icovering) {
      int index = i + p*nz;
      if(index < (nz*nLinks*icovering)/2+1) 
        kz[index] = (float) index/(Zp_d*nLinks*icovering);
      else
        kz[index] = (float) (index-nz*nLinks*icovering)/(Zp_d*nLinks*icovering);
	
      if(NO_ZDERIV_COVERING) kz[index] = 0;	
    }	
    
    

  }
  else {
    for(int a=0; a<nLinks*icovering/zthreads; a++) {
      if(i<nz && p<zthreads) {  
        unsigned int P = p+a*zthreads;
	int index = i + P*nz;
	if(index < (nz*nLinks*icovering)/2+1) 
          kz[index] = (float) index/(Zp_d*nLinks*icovering);
        else
          kz[index] = (float) (index-nz*nLinks*icovering)/(Zp_d*nLinks*icovering);
	  
	if(NO_ZDERIV_COVERING) kz[index] = 0;  
      }
    }
    
  }    	    
}

//only called for first class (nLinks=1). assumes first ntheta0 chains are ky=0, as currently set up. 
// g = g(ky=0, kx, kz)
__global__ void reality_covering(cufftComplex* g)
{
  unsigned int i = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int n = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int p = 0;  //only 1 link
  
  int ntheta0 = 1 + 2*((nx-1)/3); 
  int nLinks = 1;
  if(i<(nz/2+1) && n<((ntheta0+1)/2) && i!=0 && n!=0) {
    unsigned int index1 = i + p*nz + n*nz*nLinks;
    unsigned int index2 = (nz-i) + p*nz + (ntheta0 - n)*nz*nLinks;
    g[index2].x = g[index1].x;
    g[index2].y = -g[index1].y;
  }
  if(i==0 && n<((ntheta0+1)/2) && n!=0) {
    unsigned int index1 = i + p*nz + n*nz*nLinks;
    unsigned int index2 = i + p*nz + (ntheta0 - n)*nz*nLinks;
    g[index2].x = g[index1].x;
    g[index2].y = -g[index1].y;
  } 
 
}  
      
