//kernels used for reduction routines
__global__ void sum(cufftComplex* result, cufftComplex* a)
{
  //shared mem size = 8*8*8*sizeof(cufftComplex)
  extern __shared__ cufftComplex result_s[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 8*8*8
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
    
  if(tid<8*8*8) {
    result_s[tid].x = 0;
    result_s[tid].y = 0;
    
    result_s[tid] = a[blockIdx.x*blockDim.x*blockDim.y*blockDim.z+tid];
    __syncthreads();
    
    for(int s=(blockDim.x*blockDim.y*blockDim.z)/2; s>0; s>>=1) {
      if(tid<s) {
        result_s[tid].x += result_s[tid+s].x;	
	result_s[tid].y += result_s[tid+s].y;
      }
      __syncthreads();
    }
    
    if(tid==0) {
      result[blockIdx.x].x = result_s[0].x;
      result[blockIdx.x].y = result_s[0].y;
    }   
  }
}

__global__ void maximum(cufftComplex* result, cufftComplex* a)
{
  //shared mem size = 8*8*8*sizeof(cufftComplex)
  extern __shared__ cufftComplex result_s[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 8*8*8
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
    
  if(tid<8*8*8) {
    result_s[tid].x = 0;
    result_s[tid].y = 0;
    
    result_s[tid] = a[blockIdx.x*blockDim.x*blockDim.y*blockDim.z+tid];
    __syncthreads();
    
    for(int s=(blockDim.x*blockDim.y*blockDim.z)/2; s>0; s>>=1) {
      if(tid<s) {
        if(result_s[tid+s].x*result_s[tid+s].x+result_s[tid+s].y*result_s[tid+s].y >
	        result_s[tid].x*result_s[tid].x+result_s[tid].y*result_s[tid].y) {
				
	  result_s[tid].x = result_s[tid+s].x;
	  result_s[tid].y = result_s[tid+s].y;	
	
	}  
      }
      __syncthreads();
    }
    
    if(tid==0) {
      result[blockIdx.x].x = result_s[0].x; 
      result[blockIdx.x].y = result_s[0].y;  
    }   
  }
}


__global__ void zeroPadded(cufftComplex* a) 
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idx<Nx && idy<Ny && idz<Nz) {
      int index = idy + Ny*idx + Nx*Ny*idz;
      
      a[index].x = 0;
      a[index].y = 0;
    }   
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      
      if(idx<Nx && idy<Ny && idz<Nz) {
        int index = idy + Ny*idx + Nx*Ny*idz + Nx*Ny*zThreads*i;
	
        a[index].x = 0;
	a[index].y = 0;
      }
    }
  }        
}  

__global__ void cleanPadded(cufftComplex* a) 
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    int index= idy + Ny*idx + Nx*Ny*idz;
    if(index > Nx*(Ny/2+1)*Nz-1 && idx<Nx && idy<Ny && idz<Nz) {
    
      a[index].x = 0;
      a[index].y = 0;
    }   
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      int index = idy + Ny*idx + Nx*Ny*idz + Nx*Ny*zThreads*i;
      if(index > Nx*(Ny/2+1)*Nz-1 && idx<Nx && idy<Ny && idz<Nz) {
    
        a[index].x = 0;
	a[index].y = 0;
      }
    }
  }        
}  

//copies f(ky[i]) into fky
__global__ void kycopy(cufftComplex* fky, cufftComplex* f, int i) {
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
    
  
  if(idy<Nz && idx<Nx) {
    unsigned int index = idy + (Nz)*idx;
    fky[index].x = f[i + index*(Ny/2+1)].x;
    fky[index].y = f[i + index*(Ny/2+1)].y;
  }
}      

//gets rid of duplications from fft ***for diagnostics only
__global__ void fixFFT(cufftComplex* f)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      if((idy==0 || idy==Ny/2+1) && idx>Nx/2+1) {
        f[index].x = 0;
	f[index].y = 0;
      }
      
      if((idy==0 || idy==Ny/2+1) && idx==Nx/2+1) {
        f[index].x = .5*f[index].x;
	f[index].y = .5*f[index].y;
      }
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
	if((idy==0 || idy==Ny/2+1) && idx>Nx/2+1) {
          f[index].x = 0;
	  f[index].y = 0;
        }
      
        if((idy==0 || idy==Ny/2+1) && idx==Nx/2+1) {
          f[index].x = .5*f[index].x;
	  f[index].y = .5*f[index].y;
        }
      }
    }
  }
}        	
      
      
      
      
      
      
      
