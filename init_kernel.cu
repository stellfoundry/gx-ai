__global__ void kInit(float* kx, float* ky, float* kz, float* kx_abs, bool NO_ZDERIV, float qsf, float shat) 
{
  unsigned int idy = get_idy();
  int idx = iget_idx();
  int idz = iget_idz();
  
  
  if(idy<ny/2+1 && idx<nx) {
      
    ky[idy] = (float) idy/Y0_d;
      
    if(idx<nx/2+1) {					
      kx[idx] = (float) idx/X0_d;					
    } else {						
      kx[idx] = (float) (idx - nx)/X0_d;				
    }

    kx_abs[idx] = abs(kx[idx]);
  }
  
  
  
  if(nz<=zthreads) { 
    if(idz<nz) {
      if(idz<(nz/2+1))
        kz[idz] = (float) idz/Zp_d;
      else
        kz[idz] = (float) (idz - nz)/Zp_d;
      if(qsf<0.) kz[idz] = shat;
      if(NO_ZDERIV) kz[idz] = 0;
    }	
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idz<zthreads) {
	int IDZ = idz + zthreads*i;
	if(IDZ<(nz/2+1))
	  kz[IDZ] = (float) IDZ/Zp_d;
	else
	  kz[IDZ] = (float) (IDZ - nz)/Zp_d;
        if(qsf<0.) kz[idz] = shat;
	if(NO_ZDERIV) kz[IDZ] = 0;  
      }
    }
   
  }  	    
}     


__global__ void bmagInit(float* bmag, float* bmagInv)
{
  unsigned int idz = get_idz();
  
  if(nz<zthreads) {
    if(idz<nz) {
      bmagInv[idz] = 1. / bmag[idz];
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idz<zthreads) {
	int IDZ = idz + zthreads*i;
	
	bmagInv[IDZ] = 1. / bmag[IDZ];
      }
    }
  }    
}

__global__ void jacobianInit(float* jacobian, float drhodpsi, float gradpar, float* bmag)
{
  unsigned int idz = get_idz();
  
  if(nz<zthreads) {
    if(idz<nz) {
      jacobian[idz] = 1. / abs(drhodpsi*gradpar*bmag[idz]);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idz<zthreads) {
	int IDZ = idz + zthreads*i;
	
	jacobian[IDZ] = 1. / abs(drhodpsi*gradpar*bmag[IDZ]);
      }
    }
  }    
}

//for NLPS test
__global__ void kPerpInit(float* kPerp2, float* kx, float* ky)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idy<(ny/2+1) && idx<nx) {
    unsigned int index = idy + (ny/2+1)*idx;
      
    kPerp2[index] = - kx[idx]*kx[idx] - ky[idy]*ky[idy]; 	 
    
  }
}   

__global__ void kPerpInvInit(float* kPerp2Inv, float* kPerp2)
{
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  
  if(idy<(ny/2+1) && idx<nx) {
    unsigned int index = idy + (ny/2+1)*idx;
      
    kPerp2Inv[index] = (float) 1.0f / (2*kPerp2[index]); 	 
    
  }
  kPerp2Inv[0] = 0;

}   


