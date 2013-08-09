//kernels used for reduction routines
__global__ void sum(cufftComplex* result, cufftComplex* a)
{
  //shared mem size = 512*sizeof(cufftComplex)
  extern __shared__ cufftComplex result_s[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 512
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
    
  if(tid<512) {
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
  //shared mem size = 512*sizeof(cufftComplex)
  extern __shared__ cufftComplex result_s[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 512
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
    
  if(tid<512) {
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
  
  if(nz<=zthreads) {
    if(idx<nx && idy<ny && idz<nz) {
      int index = idy + ny*idx + nx*ny*idz;
      
      a[index].x = 0;
      a[index].y = 0;
    }   
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      
      if(idx<nx && idy<ny && idz<nz) {
        int index = idy + ny*idx + nx*ny*idz + nx*ny*zthreads*i;
	
        a[index].x = 0;
	a[index].y = 0;
      }
    }
  }        
}  

__global__ void cleanPadded(cufftComplex* a, int nx, int ny, int nz) 
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nz<=zthreads) {
    int index= idy + ny*idx + nx*ny*idz;
    if(index > nx*(ny/2+1)*nz-1 && idx<nx && idy<ny && idz<nz) {
    
      a[index].x = 0;
      a[index].y = 0;
    }   
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      int index = idy + ny*idx + nx*ny*idz + nx*ny*zthreads*i;
      if(index > nx*(ny/2+1)*nz-1 && idx<nx && idy<ny && idz<nz) {
    
        a[index].x = 0;
	a[index].y = 0;
      }
    }
  }        
}  

//copies f(ky[i]) into fky
__global__ void kycopy(cufftComplex* fky, cufftComplex* f, int i) {
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
    
  if(nz<=zthreads) {
    if(idz<nz && idx<nx) {
      unsigned int idxz = idx + (nx)*idz;
      fky[idxz].x = f[i + idxz*(ny/2+1)].x;
      fky[idxz].y = f[i + idxz*(ny/2+1)].y;
    }
  }
  
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if( idz<zthreads && idx<nx ) {
	unsigned int IDZ = idz+i*zthreads;
	unsigned int idxz = idx + (nx)*IDZ;
	fky[idxz].x = f[i + idxz*(ny/2+1)].x;
	fky[idxz].y = f[i + idxz*(ny/2+1)].y;
      }
    }
  }
}
       

__global__ void kxkycopy(float* f_z, cuComplex* f, int i)
{
  unsigned int idz = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  
  if(idz<nz) {
    f_z[idz] = f[i + nx*(ny/2+1)*idz].x;
  }
}  

//copies f(kx,ky,i) into f_i(kx,ky)
__global__ void zcopy(float* fkxky_z, cuComplex* f, int i) 
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
    
  
  if(idy<ny/2+1 && idx<nx) {
    unsigned int index = idy + (ny/2+1)*idx;
    fkxky_z[index] = f[index + nx*(ny/2+1)*i].x;   
  }
}  

//copy into f(kx) for each z
__global__ void zcopyX_Y0(float* fkx_z, cuComplex* f, int i) 
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
    
  
  if(idx<nx) {
    //only use ky=0 mode
    unsigned int index = 0 + (ny/2+1)*idx;
    fkx_z[index] = f[index + nx*(ny/2+1)*i].x;   
  }
}

//normalize FFTs
__global__ void fixFFT(cufftComplex* f)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      /*
      if((idy==0 || idy==ny/2) && idx>nx/2) {
        f[index].x = 0;
	f[index].y = 0;
      }
      
      if((idy==0 || idy==ny/2) && idx==nx/2) {
        f[index].x = .5*f[index].x;
	f[index].y = .5*f[index].y;
      }
      
      
      if(idy!=0) {
        f[index] = .5*f[index];
      }
      */
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	/*
	if((idy==0 || idy==ny/2) && idx>nx/2) {
          f[index].x = 0;
	  f[index].y = 0;
        }
      
        if((idy==0 || idy==ny/2) && idx==nx/2) {
          f[index].x = .5*f[index].x;
	  f[index].y = .5*f[index].y;
        }
	*/
	
	
	if(idy==0) {
          f[index] = .5*f[index];
        }
	
	
      }
    }
  }
}        	

__global__ void fixFFT(float* f2)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      /*
      if((idy==0 || idy==ny/2) && idx>nx/2) {
        f[index] = 0;
      }
      
      if((idy==0 || idy==ny/2) && idx==nx/2) {
        f[index] = .5*f[index];
      }
      
      if(idy==0) f[index] = .5*f[index];
      */
      
      if(idy==0) f2[index] = .25*f2[index];
      else f2[index] = .5*f2[index];
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	/*
	if((idy==0 || idy==ny/2) && idx>nx/2) {
          f[index] = 0;
        }
      
        if((idy==0 || idy==ny/2) && idx==nx/2) {
          f[index] = .5*f[index];
        }
	*/
	if(idy==0) f2[index] = .25*f2[index];
        else f2[index] = .5*f2[index];
      }
    }
  }
}        	      
      
__global__ void assign(float* f, float f_i, int i)
{
  f[i] = f_i;
}      


__global__ void sumZ(float* sum_XY, cuComplex* f)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idx < nx && idy < ny/2+1 ) {
    unsigned int index = idy + (ny/2+1)*idx;
    sum_XY[index] = 0;
    for(int i=0; i<nz; i++) {
      sum_XY[index] = sum_XY[index] + f[index + nx*(ny/2+1)*i].x;
    } 
  }
}

__global__ void sumZ(float* sum_XY, float* f)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idx < nx && idy < ny/2+1 ) {
    unsigned int index = idy + (ny/2+1)*idx;
    sum_XY[index] = 0;
    for(int i=0; i<nz; i++) {
      sum_XY[index] = sum_XY[index] + f[index + nx*(ny/2+1)*i];
    } 
  }
}

__global__ void sumZ_Ky0(float* sum_X, float* f)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idx < nx ) {
    unsigned int index = 0 + (ny/2+1)*idx;
    sum_X[idx] = 0;
    //if( idx!=0 ) {
      for(int i=0; i<nz; i++) {
	sum_X[idx] = sum_X[idx] + f[index + nx*(ny/2+1)*i];
      } 
    //}
  }
  
    
}      
  
__global__ void sumXZ(float* sum_y, float* f)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idy < ny/2+1) {
    sum_y[idy] = 0;
    for(int i=0; i<nz; i++) {
      for(int j=0; j<nx; j++) {
        sum_y[idy] = sum_y[idy] + f[idy + (ny/2+1)*j + nx*(ny/2+1)*i];
      }
    }
  }
}    

__global__ void sumXZ(float* sum_y, float* f, int a)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idy < ny/2+1) {
    sum_y[idy] = 0;
    for(int i=0; i<nz; i++) {
      for(int j=a; j<a+1; j++) {
        sum_y[idy] = sum_y[idy] + f[idy + (ny/2+1)*j + nx*(ny/2+1)*i];
      }
    }
  }
} 
 
// sum_kx[ f(kx,ky) ]    
__global__ void sumX(float* sum_y, float* f, int a)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idy < ny/2+1) {
    sum_y[idy] = 0;
    for(int i=a; i<a+1; i++) {
      sum_y[idy] = sum_y[idy] + f[idy + (ny/2+1)*i];
    }
  }
}

__global__ void sumX(float* sum_y, float* f)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idy < ny/2+1) {
    sum_y[idy] = 0;
    for(int i=0; i<nx; i++) {
      sum_y[idy] = sum_y[idy] + f[idy + (ny/2+1)*i];
    }
  }
}

__global__ void sumY_neq_0(float* sum_x, float* f)
{
  unsigned int idx = get_idx();
  
  if(idx < nx) {
    sum_x[idx] = 0;
    for(int i=1; i<ny/2+1; i++) {   //start at i=1 to skip ky=0
      sum_x[idx] = sum_x[idx] + f[i + (ny/2+1)*idx];
    }
  }
}
