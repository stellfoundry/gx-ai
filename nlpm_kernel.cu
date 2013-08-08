__global__ void shear1()
{

}


__global__ void shear2(cuComplex* shear, float dnlpm, float* kx, float rho, float* ky, float shat, float* gds2, float* gds21, float*gds22,
								float* bmagInv, cuComplex* Phi)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy==0 && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      shear[index].x = pow( kx[idx]*flr(bidx) , 2)*(Phi[index].x*Phi[index].x + Phi[index].y*Phi[index].y);
      shear[index].y = 0;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy==0 && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	shear[index].x = pow( kx[idx]*flr(bidx) , 2)*(Phi[index].x*Phi[index].x + Phi[index].y*Phi[index].y);
        shear[index].y = 0;
      }
    }
  }
	
  
}

__global__ void filter2(cuComplex* filter, float* shear, float* ky, float dt_loc, float dnlpm)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      filter[index].x = 1./( 1. + dt_loc*dnlpm*sqrt(shear[idz])*ky[idy] );
      filter[index].y = 0;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	filter[index].x = 1./(1. + dt_loc*dnlpm*sqrt(shear[IDZ])*ky[idy]);
	filter[index].y = 0;
      }
    }
  }

}




