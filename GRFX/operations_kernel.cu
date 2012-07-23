//zeroes complex arrays
__global__ void zeroC(cufftComplex* f) 
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    f[index].x = 0;
    f[index].y = 0;
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
    
    f[index].x = 0;
    f[index].y = 0;
    }
   }
  }    
}
    
//zeroes real arrays
__global__ void zero(cufftReal* f) 
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
   if(idy<(Ny) && idx<Nx && idz<Nz) {
    int index = idy + (Ny)*idx + Nx*(Ny)*idz;
    
    f[index] = 0;
    
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny) && idx<Nx && idz<zThreads) {
    int index = idy + (Ny)*idx + Nx*(Ny)*idz + Nx*Ny*zThreads*i;
    
    f[index] = 0;
    
    }
   }
  }     
}    

//add (a=1) or subtract (a=-1) complex arrays
__global__ void addsubt(cufftComplex* result, cufftComplex* f, cufftComplex* g, int a)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      
      if(a == 1) {
        result[index].x = f[index].x + g[index].x;
	result[index].y = f[index].y + g[index].y;
      }	
      if(a == -1) {
        result[index].x = f[index].x - g[index].x;
	result[index].y = f[index].y - g[index].y;
      }	
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        if(a == 1) {
          result[index].x = f[index].x + g[index].x;
	  result[index].y = f[index].y + g[index].y;
	  }
        if(a == -1) {
          result[index].x = f[index].x - g[index].x;
	  result[index].y = f[index].y - g[index].y;
	  }
      }
    }
  }
}          

//multiply a complex array by a scaler
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

//multiply a real array by a scaler
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
      if(idy<(Ny) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz + Nx*(Ny)*zThreads*i;
	
	b[index] = scaler*b[index]; 
      }
    }
  }     
}    


//multiply by kPerp^2
__global__ void multKPerp(cufftComplex* fK, cufftComplex* f, float* kPerp2, int a)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      unsigned int kidx = idy + (Ny/2+1)*idx;
      
      if(a == 1) {
        fK[index].x = f[index].x * kPerp2[kidx];
	fK[index].y = f[index].y * kPerp2[kidx];
      }	
      if(a == -1) {
        fK[index].x = f[index].x * -kPerp2[kidx];
        fK[index].y = f[index].y * -kPerp2[kidx];
      }	 
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	unsigned int kidx = idy + (Ny/2+1)*idx;
	
        if(a == 1) {
          fK[index].x = f[index].x * kPerp2[kidx];
	  fK[index].y = f[index].y * kPerp2[kidx];
	  }
        if(a == -1) {
          fK[index].x = f[index].x * -kPerp2[kidx];
	  fK[index].y = f[index].y * -kPerp2[kidx];
	  }
        
      }
    }
  }
}       

//multiply by kx
__global__ void multKx(cufftComplex* fK, cufftComplex* f, float* kx) 
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      fK[index].x = f[index].x * kx[idx];
      fK[index].y = f[index].y * kx[idx];
      		 
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        fK[index].x = f[index].x * kx[idx];
        fK[index].y = f[index].y * kx[idx];
        
      }
    }
  }
}   

//multiply by ky
__global__ void multKy(cufftComplex* fK, cufftComplex* f, float* ky) 
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      fK[index].x = f[index].x * ky[idy];
      fK[index].y = f[index].y * ky[idy];
      		 
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        fK[index].x = f[index].x * ky[idy];
        fK[index].y = f[index].y * ky[idy];
        
      }
    }
  }
}        
 
//squares a complex
__global__ void squareComplex(cufftComplex* f)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      f[index].x = f[index].x*f[index].x + f[index].y * f[index].y;
      f[index].y = 0;
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        f[index].x = f[index].x*f[index].x + f[index].y * f[index].y;
	f[index].y = 0;
      }
    }
  }
}    

//fixes roundoff errors after fft
__global__ void roundoff(cufftComplex* f, float max)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      if( abs(f[index].x) < max)
        f[index].x = 0.0f;
      if( abs(f[index].y) < max)
        f[index].y = 0.0f;
	
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        if( abs(f[index].x) < max)
          f[index].x = 0.0f;
        if( abs(f[index].y) < max)
          f[index].y = 0.0f;	
      }
    }
  }

}     
