__global__ void step(cufftComplex* fNew, cufftComplex* fOld,
                     cufftComplex* ZDeriv, cufftComplex* brackets,
		     float dt, int a)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
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
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
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
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      unsigned int kidx = idy + (ny/2+1)*idx;
      
      
      if(a == 1) {
        //bracket[index] = cuCsubf( bracket[index], cuCscalef( NuEta*kPerp2[idy+(ny/2+1)*idx]*kPerp2[idy+(ny/2+1)*idx], cuCaddf(zp[index],zm[index]) ) );
	bracket[index].x = bracket[index].x - NuEta*kPerp2[kidx]*kPerp2[kidx]*(zp[index].x+zm[index].x);
	bracket[index].y = bracket[index].y - NuEta*kPerp2[kidx]*kPerp2[kidx]*(zp[index].y+zm[index].y);
	}
      
      if(a == -1) {
        bracket[index].x = bracket[index].x + NuEta*kPerp2[kidx]*(zp[index].x-zm[index].x);
	bracket[index].y = bracket[index].y + NuEta*kPerp2[kidx]*(zp[index].y-zm[index].y);
	}
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	unsigned int kidx = idy * (ny/2+1)*idx;
	
        if(a == 1) {
          bracket[index].x = bracket[index].x - NuEta*kPerp2[kidx]*kPerp2[kidx]*(zp[index].x+zm[index].x);
	  bracket[index].y = bracket[index].y - NuEta*kPerp2[kidx]*kPerp2[kidx]*(zp[index].y+zm[index].y);
	}
      
        if(a == -1) {
          bracket[index].x = bracket[index].x + NuEta*kPerp2[kidx]*(zp[index].x-zm[index].x);
	  bracket[index].y = bracket[index].y + NuEta*kPerp2[kidx]*(zp[index].y-zm[index].y);
	}
      }
    }
  }

}     

