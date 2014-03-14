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

__global__ void get_ky0kx1_rms(float* Phi_zf_kx1, float* Phi2)
{
  int ikx1 = round(X0_d); //determine the index of the kx=1 mode
  if(ikx1 > (nx-1)/3) ikx1=(nx-1)/3; //if kx=1 is not in the box, use the highest kx 
  int iky0 = 0;
  *Phi_zf_kx1 = sqrt(Phi2[iky0 + (ny/2+1)*ikx1]);
}

__global__ void get_kx1_rms(float* Phi_zf_kx1, float* Phi2_zonal)
{
  int ikx1 = round(X0_d); //determine the index of the kx=1 mode
  if(ikx1 > (nx-1)/3) ikx1=(nx-1)/3; //if kx=1 is not in the box, use the highest kx 
  *Phi_zf_kx1 = sqrt(Phi2_zonal[ikx1]);
}

__global__ void get_Dnlpm(float* Dnlpm, float Phi_zf_kx1, float low_cutoff, float high_cutoff, float nu)
{
  //float low_cutoff= .04; //smallest value of phi_zf that D_nlpm is an effect
  //float high_cutoff = .08; //past this value D=1
  float d = (Phi_zf_kx1 - low_cutoff)/(high_cutoff-low_cutoff);
  if(d<0) d=0.; // 0 < D_nlpm < 1
  if(d>1) d=1.;
  *Dnlpm = d;
}

__global__ void get_Dnlpm_quadratic(float* Dnlpm, float Phi_zf_kx1)
{
  *Dnlpm = Phi_zf_kx1;
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

__global__ void nlpm_filter(cuComplex* field, float* nu_nlpm, float* ky, float dt_loc, float dnlpm, float kxfac)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;


      field[index] = field[index]/( 1. + dt_loc*kxfac*(dnlpm)*nu_nlpm[idz]*ky[idy] );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	

	field[index] = field[index]/( 1. + dt_loc*kxfac*(dnlpm)*nu_nlpm[IDZ]*ky[idy] );
      }
    }
  }

}

__global__ void nlpm_filter(cuComplex* field, float* nu_nlpm, float* ky, float dt_loc, float* Dnlpm, float kxfac)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;


      field[index] = field[index]/( 1. + dt_loc*(*Dnlpm)*kxfac*nu_nlpm[idz]*ky[idy] );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	

	field[index] = field[index]/( 1. + dt_loc*(*Dnlpm)*kxfac*nu_nlpm[IDZ]*ky[idy] );
      }
    }
  }

}



