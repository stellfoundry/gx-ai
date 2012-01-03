__global__ void kInit(float* kx, float* ky, float* kz) 
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
  if(idy<Ny/2+1 && idx<Nx) {
      
    ky[idy] = (float) idy;
      
    if(idx<Nx/2+1) {					
      kx[idx] = (float) idx;					
    } else {						
      kx[idx] = (float) (idx - Nx);				
    }
  }
  
  
  
  if(Nz<=zThreads) { 
    if(idz<Nz) {
      if(idz<(Nz/2+1))
        kz[idz] = idz;
      else
        kz[idz] = idz - Nz;
    }	
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idz<zThreads) {
	int IDZ = idz + zThreads*i;
	if(IDZ<(Nz/2+1))
	  kz[IDZ] = IDZ;
	else
	  kz[IDZ] = IDZ - Nz;
      }
    }
  }  	    

}     

__global__ void kPerpInit(float* kPerp2, float* kx, float* ky)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idy<(Ny/2+1) && idx<Nx) {
    unsigned int index = idy + (Ny/2+1)*idx;
      
    kPerp2[index] = - kx[idx]*kx[idx] - ky[idy]*ky[idy]; 	 
    
  }
  
  //kPerp2[0] = 1;

}   

__global__ void kPerpInvInit(float* kPerp2Inv, float* kPerp2)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idy<(Ny/2+1) && idx<Nx) {
    unsigned int index = idy + (Ny/2+1)*idx;
      
    kPerp2Inv[index] = (float) 1.0f / (2*kPerp2[index]); 	 
    
  }
  kPerp2Inv[0] = 0;

}   

__global__ void multKPerp(cufftComplex* fK, cufftComplex* f, float* kPerp2, int a)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      if(a == 1) {
        fK[index].x = f[index].x * kPerp2[idy + (Ny/2+1)*idx];
	fK[index].y = f[index].y * kPerp2[idy + (Ny/2+1)*idx];
      }	
      if(a == -1) {
        fK[index].x = f[index].x * -kPerp2[idy + (Ny/2+1)*idx];
        fK[index].y = f[index].y * -kPerp2[idy + (Ny/2+1)*idx];
      }	 
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        if(a == 1) {
          fK[index].x = f[index].x * kPerp2[idy + (Ny/2+1)*idx];
	  fK[index].y = f[index].y * kPerp2[idy + (Ny/2+1)*idx];
	  }
        if(a == -1) {
          fK[index].x = f[index].x * -kPerp2[idy + (Ny/2+1)*idx];
	  fK[index].y = f[index].y * -kPerp2[idy + (Ny/2+1)*idx];
	  }
        
      }
    }
  }

}       

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


__global__ void step(cufftComplex* fNew, cufftComplex* fOld,
                     cufftComplex* ZDeriv, cufftComplex* brackets,
		     float dt, int a)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      //for zp
      if(a == 1) {
        fNew[index].x = fOld[index].x + dt*(ZDeriv[index].x - brackets[index].x);
	fNew[index].y = fOld[index].y + dt*(ZDeriv[index].y - brackets[index].y);
	}
      //for zm
      if(a == -1) {
        fNew[index].x = fOld[index].x - dt*(ZDeriv[index].x + brackets[index].x);
	fNew[index].y = fOld[index].y - dt*(ZDeriv[index].y + brackets[index].y);
	}
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        if(a == 1) {
          fNew[index].x = fOld[index].x + dt*(ZDeriv[index].x - brackets[index].x);
	  fNew[index].y = fOld[index].y + dt*(ZDeriv[index].y - brackets[index].y);
	  }
        if(a == -1) {
          fNew[index].x = fOld[index].x - dt*(ZDeriv[index].x + brackets[index].x);
	  fNew[index].y = fOld[index].y - dt*(ZDeriv[index].y + brackets[index].y);
	  }
      }
    }
  }

}     

__global__ void damping(cufftComplex* bracket, cufftComplex* zp, cufftComplex* zm,
                        float* kPerp2, float NuEta, int a)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      
      if(a == 1) {
        bracket[index].x = bracket[index].x - NuEta*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].x+zm[index].x);
	bracket[index].y = bracket[index].y - NuEta*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].y+zm[index].y);
	}
      
      if(a == -1) {
        bracket[index].x = bracket[index].x + NuEta*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].x-zm[index].x);
	bracket[index].y = bracket[index].y + NuEta*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].y-zm[index].y);
	}
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        if(a == 1) {
          bracket[index].x = bracket[index].x - NuEta*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].x+zm[index].x);
	  bracket[index].y = bracket[index].y - NuEta*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].y+zm[index].y);
	}
      
        if(a == -1) {
          bracket[index].x = bracket[index].x + NuEta*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].x-zm[index].x);
	  bracket[index].y = bracket[index].y + NuEta*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].y-zm[index].y);
	}
      }
    }
  }

}       			  

__global__ void sum(cufftComplex* result, cufftComplex* a)
{
  //shared mem size = 8*8*8*sizeof(cufftComplex)
  extern __shared__ cufftComplex result_s[];
  //tid up to blockDim.x*blockDim.y*blockDim.z = 8*8*8
  int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
  
  
  if(tid<8*8*8) {
    result_s[tid].x = 0;
    
     
    result_s[tid] = a[blockIdx.x*blockDim.x*blockDim.y*blockDim.z+tid];
    __syncthreads();
    
    for(int s=(blockDim.x*blockDim.y*blockDim.z)/2; s>0; s>>=1) {
      if(tid<s) {
      
        result_s[tid].x += result_s[tid+s].x;
        
	
      }
      __syncthreads();
    }
    
    if(tid==0) {
      result[blockIdx.x].x = result_s[0].x;
      
    }
    __syncthreads();
    
  }
  
}

__global__ void zeroPadded(cufftComplex* a) 
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


__global__ void clean(cufftComplex* f)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
    if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
      unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
      
      if( abs(f[index].x) < .00001*Nx*(Ny/2+1))
        f[index].x = 0.0f;
      if( abs(f[index].y) < .00001*Nx*(Ny/2+1))
        f[index].y = 0.0f;
	
    }
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
        unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
	
        if( abs(f[index].x) < .00001*Nx*(Ny/2+1))
          f[index].x = 0.0f;
        if( abs(f[index].y) < .00001*Nx*(Ny/2+1))
          f[index].y = 0.0f;	
      }
    }
  }

}     

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
      
      	
       	  


