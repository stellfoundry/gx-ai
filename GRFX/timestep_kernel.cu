__device__ cufftComplex complexMult(cufftComplex a, cufftComplex b) {
  cufftComplex c;
  c.x = a.x*b.x-a.y*b.y;
  c.y = a.y*b.x + b.y*a.x;
  return c;
}  

__device__ cufftComplex complexAdd(cufftComplex a, cufftComplex b) {
  cufftComplex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

__device__ float complexNorm(cufftComplex a) {
  float c;
  c = sqrt(a.x*a.x + a.y*a.y);
  return c;
}  
  
__global__ void kInit(float* kx, float* ky, float* kz) 
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
  if(idy<Ny/2+1 && idx<Nx) {
      
    ky[idy] = (float) idy/X0;
      
    if(idx<Nx/2+1) {					
      kx[idx] = (float) idx/X0;					
    } else {						
      kx[idx] = (float) (idx - Nx)/X0;				
    }
  }
  
  
  
  if(Nz<=zThreads) { 
    if(idz<Nz) {
      if(idz<(Nz/2+1))
        kz[idz] = (float) idz/X0;
      else
        kz[idz] = (float) (idz - Nz)/X0;
    }	
  }
  else {
    for(int i=0; i<Nz/zThreads; i++) {
      if(idz<zThreads) {
	int IDZ = idz + zThreads*i;
	if(IDZ<(Nz/2+1))
	  kz[IDZ] = (float) IDZ/X0;
	else
	  kz[IDZ] = (float) (IDZ - Nz)/X0;
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
        bracket[index].x = bracket[index].x - NuEta*kPerp2[idy+(Ny/2+1)*idx]*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].x+zm[index].x);
	bracket[index].y = bracket[index].y - NuEta*kPerp2[idy+(Ny/2+1)*idx]*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].y+zm[index].y);
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
          bracket[index].x = bracket[index].x - NuEta*kPerp2[idy+(Ny/2+1)*idx]*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].x+zm[index].x);
	  bracket[index].y = bracket[index].y - NuEta*kPerp2[idy+(Ny/2+1)*idx]*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].y+zm[index].y);
	}
      
        if(a == -1) {
          bracket[index].x = bracket[index].x + NuEta*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].x-zm[index].x);
	  bracket[index].y = bracket[index].y + NuEta*kPerp2[idy+(Ny/2+1)*idx]*(zp[index].y-zm[index].y);
	}
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


//gets rid of duplications from fft
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
      
      	
       	  


