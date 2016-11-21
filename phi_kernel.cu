__global__ void phi_n(cuComplex* res, cuComplex* phi, float tprim, float rho, float fprim,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * ( fprim*sgam0(bidx) + tprim * flr(bidx) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * ( fprim*sgam0(bidx) + tprim * flr(bidx) );
      }
    }
  }    	    
}  


__global__ void phi_u(cuComplex* res, cuComplex* Phi, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = Phi[index] * sgam0( bidx );      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = Phi[index] * sgam0(bidx);
      }
    }
  }    	    
}    

__global__ void phi_u_inverse(cuComplex* res, cuComplex* Phi, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = Phi[index] / sgam0( bidx );      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = Phi[index] / sgam0(bidx);
      }
    }
  }    	    
}    

__global__ void phi_u_NL(cuComplex* res, cuComplex* Phi, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, int iflr)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(iflr==1) res[index] = Phi[index] * exp( -bidx/2. );      
      else if(iflr==20) res[index] = Phi[index] * nwgt(bidx);
      else res[index] = Phi[index] * sgam0(bidx);
      //if(iflr==2) res[index] = Phi[index] / ( 1 + bidx/2. );      
      //if(iflr==4) res[index] = Phi[index] * sgam0(bidx) * sgam0(bidx) * exp(bidx/2.);
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(iflr==0) res[index] = Phi[index] * sgam0(bidx);
        if(iflr==1) res[index] = Phi[index] * exp( -bidx/2. );      
      }
    }
  }    	    
}    
__global__ void phi_tperp_nlpm(cuComplex* res, cuComplex* Phi, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, int iflr)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(iflr==1) res[index] = Phi[index] * (1. + bidx / 2.);      
      else if (iflr==2) res[index] = Phi[index] / (1. + bidx / 2.) + (bidx*Phi[index]) / pow(1. + bidx / 2.,2);
      else if (iflr==3) res[index] = Phi[index] * expf(-bidx/2.) * (1 + bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(iflr==1) res[index] = Phi[index] * (1. + bidx / 2.);      
        else if (iflr==2) res[index] = Phi[index] / (1. + bidx / 2.) + (bidx*Phi[index]) / pow(1. + bidx / 2.,2);
        else if (iflr==3) res[index] = Phi[index] * expf(-bidx/2.) * (1 + bidx);
      }
    }
  }    	    
}    
__global__ void phi_u_low_b(cuComplex* res, cuComplex* Phi, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, int iflr)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(iflr==1) res[index] = Phi[index] * (1. - bidx / 2.);      
      else if (iflr==2) res[index] = Phi[index] / (1. + bidx / 2.);      
      else if (iflr==3) res[index] = Phi[index] * expf(-bidx/2.);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(iflr==1) res[index] = Phi[index] * (1. - bidx / 2.);      
        else if (iflr==2) res[index] = Phi[index] / (1. + bidx / 2.);      
        else if (iflr==3) res[index] = Phi[index] * expf(-bidx/2.);
      }
    }
  }    	    
}    
__global__ void phi_u_force(cuComplex* res, float phiext, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(idy == 0 && idx!=0) {
        res[index].x = phiext * sgam0( bidx );///g0(bidx);
        //res[index].y = phiext * sgam0( bidx )/g0(bidx);
        res[index].y = 0.;      
      }
      else {
        res[index].x = 0.;
        res[index].y = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(idy == 0 && idx!=0) {
          res[index].x = phiext * sgam0( bidx );
          res[index].y = 0.;      
        }
        else {
          res[index].x = 0.;
          res[index].y = 0.;
        }
      }
    }
  }    	    
}    
__global__ void phi_tpar(cuComplex* res, cuComplex* phi, float tprim, float rho,
			 float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * sgam0(bidx) * tprim;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * sgam0(bidx) * tprim;
      }
    }
  }    	    
}  

__global__ void phi_tperp(cuComplex* res, cuComplex* phi, float tprim, float rho, float fprim,
			  float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy(); //blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * ( tprim*( sgam0(bidx) + flr2(bidx) ) + fprim*flr(bidx) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * ( tprim*( sgam0(bidx) + flr2(bidx) ) + fprim*flr(bidx) );
      }
    }
  }    	    
}

//also known as phi_qperp
__global__ void phi_flr(cuComplex* res, cuComplex* phi, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * flr(bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * flr(bidx);
      }
    }
  }    	    
}    

__global__ void phi_flr_NL(cuComplex* res, cuComplex* phi, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, int iflr)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(iflr==1) res[index] = phi[index] * (-bidx/2. * exp( -bidx/2. ));
      else if(iflr==10) res[index] = phi[index] * flr(bidx) / sgam0(bidx) * exp(-bidx/2.);
      else if(iflr==11) res[index] = phi[index] * flr(bidx) / sgam0(bidx) * exp(-bidx);
      else if(iflr==20) res[index] = phi[index] * twgt(bidx);
      else res[index] = phi[index] * flr(bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(iflr==0) res[index] = phi[index] * flr(bidx);
        if(iflr==1) res[index] = phi[index] * (-bidx/2. * exp( -bidx/2. ));
      }
    }
  }    	    
}    

__global__ void phi_flr(cuComplex* res, cuComplex* phi, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, float fac)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * flr(bidx) * fac;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * flr(bidx) * fac;
      }
    }
  }    	    
}    
__global__ void phi_flr_low_b(cuComplex* res, cuComplex* phi, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, int iflr)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(iflr==1) res[index] = phi[index] * (-bidx / 2.);      
      else if (iflr==2) res[index] = phi[index]*(-bidx/2.) / pow(1. + bidx / 2.,2);      
      else if (iflr==3) res[index] = phi[index] *(-bidx/2.)*expf(-bidx/2.);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        res[index] = phi[index] *(-bidx/2.);
      }
    }
  }    	    
}    
__global__ void phi_flr_low_b_abs(cuComplex* res, cuComplex* phi, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, int iflr)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      float phi_abs = sqrt( pow(phi[index].x,2) + pow(phi[index].y,2) );

      if(iflr==1) {res[index].x = phi_abs * (-bidx / 2.); res[index].y = 0.;}     
      else if (iflr==2) res[index] = phi[index]*(-bidx/2.) / pow(1. + bidx / 2.,2);      
      else if (iflr==3) res[index] = phi[index] *(-bidx/2.)*expf(-bidx/2.);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        res[index] = phi[index] *(-bidx/2.);
      }
    }
  }    	    
}    
__global__ void phi_flr_zonal_abs(cuComplex* res, float* phi_zf_abs, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(idy==0) {
        res[index].x = abs(phi_zf_abs[idx]) * abs(flr(bidx));
      } else {
        res[index].x = 0.;
      }
      res[index].y = 0.;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(idy==0) {
          res[index].x = abs(phi_zf_abs[idx]) * abs(flr(bidx));
        } else {
          res[index].x = 0.;
        }
        res[index].y = 0.;
      }
    }
  }    	    
}    

__global__ void phi_flr_zonal_complex(cuComplex* res, cuComplex* phi_zf_complex, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(idy==0) {
        res[index] = phi_zf_complex[idx] * flr(bidx);
      } else {
        res[index].x = 0.;
        res[index].y = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(idy==0) {
          res[index] = phi_zf_complex[idx] * flr(bidx);
        } else {
          res[index].x = 0.;
          res[index].y = 0.;
        }
      }
    }
  }    	    
}    

__global__ void phi_flr_squared(cuComplex* res, cuComplex* phi, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, float fac)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * flr(bidx)*flr(bidx)*fac;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        res[index] = phi[index] * flr(bidx)*flr(bidx)*fac;
      }
    }
  }    	    
}    

__global__ void phi_flr_squared_zonal_abs(cuComplex* res, float* phi_zf_abs, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(idy==0) {
        res[index].x = abs(phi_zf_abs[idx]) * abs(flr(bidx)*flr(bidx));
      } else {
        res[index].x = 0.;
      }
      res[index].y = 0.;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(idy==0) {
          res[index].x = abs(phi_zf_abs[idx]) * abs(flr(bidx)*flr(bidx));
        } else {
          res[index].x = 0.;
        }
        res[index].y = 0.;
      }
    }
  }    	    
}    
__global__ void phi_flr_force(cuComplex* res, float phiext, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(idy==0 && idx!=0) {
        res[index].x = phiext * flr(bidx);///g0(bidx);
        //res[index].y = phiext * flr(bidx)/g0(bidx);
        res[index].y = 0.;
      }
      else {
        res[index].x = 0.;
        res[index].y = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(idy==0 && idx!=0) {
          res[index].x = phiext * flr(bidx);
          res[index].y = 0.;
        }
        else {
          res[index].x = 0.;
          res[index].y = 0.;
        }
      }
    }
  }    	    
}    

__global__ void phi_flr2(cuComplex* res, cuComplex* phi, float rho,
			 float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * flr2(bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * flr2(bidx);
      }
    }
  }    	    
}    


__global__ void phi_flr2_NL(cuComplex* res, cuComplex* phi, float rho,
			 float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, int iflr)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(iflr==1) res[index] = phi[index] * bidx/4.*(bidx-4.)*exp( -bidx/2. );
      else res[index] = phi[index] * flr2(bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(iflr==0) res[index] = phi[index] * flr2(bidx);
        if(iflr==1) res[index] = phi[index] * bidx/4.*(bidx-4.)*exp( -bidx/2. );
      }
    }
  }    	    
}    


__global__ void phi_flr2_flr(cuComplex* res, cuComplex* phi, float rho,
			 float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * (flr2(bidx) - flr(bidx) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        res[index] = phi[index] * (flr2(bidx) - flr(bidx) );
      }
    }
  }    	    
}    
__global__ void phi_flr2_low_b(cuComplex* res, cuComplex* phi, float rho,
			 float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, int iflr)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(iflr==1) res[index] = phi[index] * (-bidx);      
      else if (iflr==2) res[index] = phi[index]*(-bidx) / pow(1. + bidx / 2.,3);      
      else if (iflr==3) res[index] = phi[index] *(-bidx)*expf(-bidx/2.)*(1-bidx/4.);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * (-bidx);
      }
    }
  }    	    
}    
__global__ void phi_nd(cuComplex* res, cuComplex* phi, float zt, float rho,
		       float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * ( zt * (2.*sgam0(bidx) + flr(bidx)) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * ( zt * (2.*sgam0(bidx) + flr(bidx)) );
      }
    }
  }    	    
}    

__global__ void phi_nd_force(cuComplex* res, float phiext, float zt, float rho,
		       float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(idy==0 && idx!=0) {
        res[index].x = phiext * ( zt * (2.*sgam0(bidx) + flr(bidx)) );///g0(bidx);
        //res[index].y = phiext * ( zt * (2.*sgam0(bidx) + flr(bidx)) )/g0(bidx);
        res[index].y = 0.;
      }
      else {
        res[index].x = 0.;
        res[index].y = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(idy==0 && idx!=0) {
	  res[index].x = phiext * ( zt * (2.*sgam0(bidx) + flr(bidx)) );
	  res[index].y = 0.; 
        }
        else {
          res[index].x = 0.;
          res[index].y = 0.;
        }
      }
    }
  }    	    
}    

__global__ void phi_tpard(cuComplex* res, cuComplex* phi, float zt, float rho,
			  float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * ( zt * 2. * sgam0(bidx) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * ( zt * 2. * sgam0(bidx) );
      }
    }
  }    	    
}    

__global__ void phi_tpard_force(cuComplex* res, float phiext, float zt, float rho,
			  float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(idy==0 && idx!=0) {
        res[index].x = phiext * ( zt * 2. * sgam0(bidx) );///g0(bidx);
        //res[index].y = phiext * ( zt * 2. * sgam0(bidx) )/g0(bidx);
        res[index].y = 0.;
      }
      else {
       res[index].x = 0.;
       res[index].y = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(idy==0 && idx!=0) {
          res[index].x = phiext * ( zt * 2. * sgam0(bidx) );
          res[index].y = 0.;
        }
        else {
          res[index].x = 0.;
          res[index].y = 0.;
        }
      }
    }
  }    	    
}    

__global__ void phi_tperpd_force(cuComplex* res, float phiext, float zt, float rho,
			   float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(idy==0 && idx!=0) { 
        res[index].x = phiext * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) );///g0(bidx);
        //res[index].y = phiext * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) )/g0(bidx);
        res[index].y = 0.;
      }
      else {
        res[index].x = 0.;
        res[index].y = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(idy==0 && idx!=0) { 
          res[index].x = phiext * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) );
          res[index].y = 0.; 
        }
        else {
          res[index].x = 0.;
          res[index].y = 0.;
        }
      }
    }
  }    	    
} 
__global__ void phi_tperpd(cuComplex* res, cuComplex* phi, float zt, float rho,
			   float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) );
      }
    }
  }    	    
} 

__global__ void phi_qperpb(cuComplex* res, cuComplex* phi, float rho,
			   float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * (flr2(bidx) - flr(bidx));
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * (flr2(bidx) - flr(bidx));
      }
    }
  }    	    
}      

__global__ void phi_qperpb_force(cuComplex* res, float phiext, float rho,
			   float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(idy==0 && idx!=0) {
        res[index].x = phiext * (flr2(bidx) - flr(bidx));///g0(bidx);
        //res[index].y = phiext * (flr2(bidx) - flr(bidx))/g0(bidx);
        res[index].y = 0.;
      }
      else {
        res[index].x = 0.;
        res[index].y = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        if(idy==0 && idx!=0) {
          res[index].x = phiext * (flr2(bidx) - flr(bidx));
          res[index].y = 0.;
        }
        else {
          res[index].x = 0.;
          res[index].y = 0.;
        }  
      }
    }
  }    	    
}      
  

__global__ void phi_NLPM_dens(cuComplex* res, cuComplex* phi, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, int iflr)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(iflr==1) res[index] = phi[index] * 0.0625*pow(bidx,2.)*exp(-0.5*bidx);
      else res[index] = phi[index] * pow(flr(bidx),2)/sgam0(bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        res[index] = phi[index] * 0.0625*pow(bidx,2.)*exp(-0.5*bidx);
      }
    }
  }    	    
}    

__global__ void phi_NLPM_tprp(cuComplex* res, cuComplex* phi, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * -0.03125*bidx*pow(bidx-4.,2.)*exp(-0.5*bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
        res[index] = phi[index] * -0.03125*bidx*pow(bidx-4.,2.)*exp(-0.5*bidx);
      }
    }
  }    	    
}    
