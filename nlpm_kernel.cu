__global__ void nlpm_shear1(float* nu, float* Phi2ZF, float dnlpm, float* kx, 
		float rho, float* ky, float shat, float* gds2, float* gds21, float*gds22, float* bmagInv)
{
  unsigned int idz = get_idz();
  unsigned int idy = 0;
  unsigned int idx = get_idx();
  
  if(nz<=zthreads) {
    if(idz<nz && idx<nx) {
      
      unsigned int idxz = idx + nx*idz;
      
      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
      nu[idxz] = abs(kx[idx])*abs(flr(bidx))*sqrt(Phi2ZF[idx]);
            
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;
	unsigned int idxz = idx + nx*IDZ;

	
	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        nu[idxz] = abs(kx[idx])*abs(flr(bidx))*sqrt(Phi2ZF[idx]);
	
	
      }
    }
  }
}


__global__ void nlpm_shear2(float* nu, float* Phi2ZF, float dnlpm, float* kx, 
		float rho, float* ky, float shat, float* gds2, float* gds21, float*gds22, float* bmagInv)
{
  unsigned int idz = get_idz();
  unsigned int idy = 0;
  unsigned int idx = get_idx();
  
  if(nz<=zthreads) {
    if(idz<nz && idx<nx) {
      
      unsigned int idxz = idx + nx*idz;
      
      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
      nu[idxz] = pow(kx[idx],2)*pow(flr(bidx),2)*Phi2ZF[idx];
            
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;
	unsigned int idxz = idx + nx*IDZ;

	
	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        nu[idxz] = pow(kx[idx],2)*pow(flr(bidx),2)*Phi2ZF[idx];
	
	
      }
    }
  }
}

__global__ void nlpm(cuComplex* res, cuComplex* field, float* ky, float* nu_nlpm, float dnlpm) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = field[index]*abs(ky[idy])*dnlpm*nu_nlpm[idz];
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = field[index]*abs(ky[idy])*dnlpm*nu_nlpm[IDZ];
      }
    }
  }
}


__global__ void nlpm_filter(cuComplex* field, float* nu_nlpm, float* ky, float dt_loc, float dnlpm)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      field[index] = field[index]/( 1. + dt_loc*dnlpm*nu_nlpm[idz]*ky[idy] );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	field[index] = field[index]/( 1. + dt_loc*dnlpm*nu_nlpm[IDZ]*ky[idy] );
      }
    }
  }

}



