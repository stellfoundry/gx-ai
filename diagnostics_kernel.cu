__global__ void growthRate(cuComplex *omega, cuComplex *phinew, cuComplex *phiold, float dt)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  
  //i_dt  i/dt
  cuComplex i_dt;
  i_dt.x = 0;
  i_dt.y = 1/dt;
  
  
  if(idy<(ny/2+1) && idx<nx) 
  {
    unsigned int index = idy + (ny/2+1)*idx;
    
    cuComplex ratio = phinew[index+nx*(ny/2+1)*((int)(.6*nz+1))] / phiold[index+nx*(ny/2+1)*((int)(.6*nz+1))];
    
    cuComplex log;
    log.x = logf(cuCabsf(ratio));
    log.y = atan2f(ratio.y,ratio.x);
    omega[index] = log*i_dt;
  }
}

__global__ void normalize(cuComplex *f, cuComplex *Phi, float norm)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz)
    {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      unsigned int kidx_z = idy + (ny/2+1)*idx + nx*(ny/2+1)*(nz/2);
      

      cuComplex c_k;
      c_k.x = cos(atan2f(Phi[kidx_z].y,Phi[kidx_z].x))/cuCabsf(Phi[kidx_z]);
      c_k.y = -sin(atan2f(Phi[kidx_z].y,Phi[kidx_z].x))/cuCabsf(Phi[kidx_z]);

      f[index] = (c_k * f[index]) * norm;
    }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
     if(idy<(ny/2+1) && idx<nx && idz<zthreads)
     {
       unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
       unsigned int kidx_z = idy + (ny/2+1)*idx + nx*(ny/2+1)*(nz/2);
       
       cuComplex c_k;
       c_k.x = cos(atan2f(Phi[kidx_z].y,Phi[kidx_z].x))/cuCabsf(Phi[kidx_z]);
       c_k.y = -sin(atan2f(Phi[kidx_z].y,Phi[kidx_z].x))/cuCabsf(Phi[kidx_z]);

       f[index] = (c_k * f[index]) * norm;
     }  
   }
 }    
}

__global__ void get_kperp(float* kperp, int z, float rho, float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  int idz = nz/2;

    if(idy<(ny/2+1) && idx<nx)
    {
      unsigned int index = idy + (ny/2+1)*idx;
      
      kperp[index] = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
    }
  
  
}
  
__global__ void calcReIm(cuComplex* result, cuComplex* a, cuComplex* b)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      result[index].x = a[index].x*b[index].x + a[index].y*b[index].y;
      result[index].y = 0;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	result[index].x = a[index].x*b[index].x + a[index].y*b[index].y;
	result[index].y = 0;
      }		
    }
  }         
}     

__global__ void zCorrelation(float* zcorr, cuComplex* Phi)
{
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idz<nz) {
      unsigned int index;
      unsigned int idyz = idy + (ny/2+1)*idz;
      unsigned int idxy_z0;
      
      zcorr[idyz] = 0;
      float fac;
      cuComplex phi2;
      for(int i=0; i<nx; i++) {
        index = idy + (ny/2+1)*i + nx*(ny/2+1)*idz;
	idxy_z0 = idy + (ny/2+1)*i + nx*(ny/2+1)*(nz/2);
        if(idy==0) fac = 0.25;
	else fac = 0.5;
	phi2 = Phi[idxy_z0] * cuConjf( Phi[index] );
	zcorr[idyz] = zcorr[idyz] + fac * phi2.x;
      }
      
      
    }
  }
  /*
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	unsigned int idxy_z0 = idy + (ny/2+1)*idx + nx*(ny/2+1)*(nz/2);
	unsigned int idxy = idy + (ny/2+1)*idx;
        
        corrNum_XYZ[index].x = Phi[idxy_z0].x*Phi[index].x + Phi[idxy_z0].y*Phi[index].y;
	corrNum_XYZ[index].y = 0;
        corrDen_XY[idxy] = Phi[idxy_z0].x*Phi[idxy_z0].x + Phi[idxy_z0].y*Phi[idxy_z0].y;
      }		
    }
  }  */       
}     

__global__ void zCorrelation_part2(float* corr_YZ, float* corrNum_YZ, float* corrDen_Y, cuComplex* corrNum_XYZ, float* corrDen_XY)
{
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idz<nz) {
      unsigned int idyz = idy + (ny/2+1)*idz;
      
      //sum numerator and denominator over kx
      corrNum_YZ[idyz] = 0.;
      corrDen_Y[idy] = 0.;
      for(int i=0; i<nx; i++) {
        corrNum_YZ[idyz] = (double) corrNum_YZ[idyz] + corrNum_XYZ[idy + (ny/2+1)*i + nx*(ny/2+1)*idz].x;
	corrDen_Y[idy] = (double) corrDen_Y[idy] + corrDen_XY[idy + (ny/2+1)*i];
      }
      
      
      if(idy==0) {
        corr_YZ[idyz] = (double) corrNum_YZ[idyz] / corrDen_Y[idy];
      }
      else {
        corr_YZ[idyz] = (double) .5*corrNum_YZ[idyz] / corrDen_Y[idy];
      }
      
    } 
  }  
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idz<zthreads)
      {
        unsigned int IDZ = idz + zthreads*i;
	unsigned int idyz = idy + (ny/2+1)*IDZ;

	//sum numerator and denominator over kx
	corrNum_YZ[idyz] = 0;
	corrDen_Y[idy] = 0;
	for(int i=0; i<nx; i++) {
          corrNum_YZ[idyz] = corrNum_YZ[idyz] + corrNum_XYZ[idy + (ny/2+1)*i + nx*(ny/2+1)*IDZ].x;
	  corrDen_Y[idy] = corrDen_Y[idy] + corrDen_XY[idy + (ny/2+1)*i];
	}

	if(idy==0) {
          corr_YZ[idyz] = corrNum_YZ[idyz] / corrDen_Y[idy];
	}
	else {
          corr_YZ[idyz] = .5*corrNum_YZ[idyz] / corrDen_Y[idy];
	}
      } 
    }  
  }
}	

__global__ void corr_length_1(float* corr_length1, float* phi_corr_J, float* phi_corr_norm, float* z)
{
  unsigned int idy = get_idy();

  if(idy<(ny/2+1)) {
    float z2_phi_corr = 0.;
    float z_phi_corr = 0.;
    for(int idz=0; idz<nz; idz++) {
      unsigned int idyz = idy + (ny/2+1)*idz;
      z2_phi_corr = z2_phi_corr + z[idz]*z[idz]*phi_corr_J[idyz]; 
      z_phi_corr = z_phi_corr + z[idz]*phi_corr_J[idyz];
    }
    z2_phi_corr = z2_phi_corr / phi_corr_norm[idy];
    z_phi_corr = pow( z_phi_corr / phi_corr_norm[idy] , 2);
    
    corr_length1[idy] = sqrt( abs(z2_phi_corr - z_phi_corr) );
    
  }

}

__global__ void corr_length_3(float* corr_length3, float* phi_corr_norm, float* phi_corr_z0, float fluxDenInv)
{
  unsigned int idy = get_idy();
  
  if(idy<(ny/2+1)) {
    double tmp = (double) phi_corr_norm[idy] / phi_corr_z0[idy];
    corr_length3[idy] = tmp;
  }
}

__global__ void corr_length_4(float* corr_length4, float* phi_corr_J, float* phi_corr_norm, float* z)
{
  unsigned int idy = get_idy();

  if(idy<(ny/2+1)) {
    float z_phi_corr = 0.;
    for(int idz=0; idz<nz; idz++) {
      unsigned int idyz = idy + (ny/2+1)*idz;
      z_phi_corr = z_phi_corr + abs(z[idz])*phi_corr_J[idyz];
    }
    
    corr_length4[idy] = z_phi_corr / phi_corr_norm[idy];    
  }

}


__global__ void volflux(float* flux, cuComplex* f, cuComplex* g, float* jacobian, float fluxDenInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx) {
      unsigned int index;
      unsigned int idxy = idy + (ny/2+1)*idx;
      
      flux[idxy] = 0;
      float fac;
      cuComplex fg;
      if(idy==0) fac = 0.25;
      else fac = 0.5;
      for(int i = 0; i<nz; i++) {
        index = idxy + nx*(ny/2+1)*i;
	fg = f[index] * cuConjf( g[index] )*jacobian[i];
	flux[idxy] = flux[idxy] + fac*fg.x;
      }      
      flux[idxy] = flux[idxy] * fluxDenInv;

    }
  }
}

__global__ void volflux_zonal(float* flux, cuComplex* f, cuComplex* g, float* jacobian, float fluxDenSqInv)
{
  unsigned int idx = get_idx();
  
  
  
    if(idx<nx) {
      unsigned int index;
      unsigned int idxy0 = 0 + (ny/2+1)*idx;
      
      flux[idx] = 0;
      cuComplex fz;
      cuComplex gz;
      cuComplex fsum;
      cuComplex gsum;
      cuComplex fg;
      fsum.x = 0.;
      fsum.y = 0.;
      gsum.x = 0.;
      gsum.y = 0.;
      float fac = .25;  //only ky=0 modes      
      for(int i = 0; i<nz; i++) {
        index = idxy0 + nx*(ny/2+1)*i;
	fz = f[index]*jacobian[i];
	gz = cuConjf( g[index] )*jacobian[i];
	fsum = fsum + fz;
	gsum = gsum + gz;
      }    
      fg = fsum*gsum;        
      flux[idx] = fac*fg.x*fluxDenSqInv;

    }
  
}

__global__ void volflux_varenna(cuComplex* T_fsa_X, cuComplex* T, float* jacobian, float fluxDenInv)
{
  unsigned int idx = get_idx();
  unsigned int idy = 0;
  if(idx<nx) 
  {
    T_fsa_X[idx].x = 0.;
    T_fsa_X[idx].y = 0.;
    for(int idz=0; idz<nz; idz++) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      T_fsa_X[idx] = T_fsa_X[idx] + T[index]*jacobian[idz];
    }
    T_fsa_X[idx] = T_fsa_X[idx]*fluxDenInv;
  }
}

__global__ void volflux_part2(float* flux_XY, cuComplex* tmp, float fluxDenInv)
{      
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();

  if(idy<(ny/2+1) && idx<nx) {
      
      unsigned int idxy = idy + (ny/2+1)*idx;
      
      flux_XY[idxy] = 0;
      for(int i=0; i<nz; i++) {
        flux_XY[idxy] = flux_XY[idxy] + tmp[idxy + i*nx*(ny/2+1)].x;  
      }  
      
      if(idy == 0) {
        flux_XY[idxy] = flux_XY[idxy]*fluxDenInv;
      } else {
        flux_XY[idxy] = .5*flux_XY[idxy]*fluxDenInv;
      }
    
  }
}

__global__ void expect_k(float* kphi2_XY, float* phi2_XY, float* k) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  
  if(idy<(ny/2+1) && idx<nx) {
      unsigned int idxy = idy + (ny/2+1)*idx;
      
      //ignore the ky=0 modes
      if(idy == 0) {
        kphi2_XY[idxy] = 0;
	phi2_XY[idxy] = 0;
      }
      else {
        kphi2_XY[idxy] = phi2_XY[idxy] / k[idy];
      }
      		     
  }
}       

__global__ void getPhiVal(float* val, cuComplex* Phi, int iky, int ikx, int z) 
{
  val[0] = Phi[iky + (ny/2+1)*ikx + nx*(ny/2+1)*z].x;
}
  
__global__ void getky0(float* f_kxky0, float* f_kxky)
{
  unsigned int idx = get_idx();
  
  if(idx<nx) {
    unsigned int idx_y0 = 0 + (ny/2+1)*idx;
    f_kxky0[idx] = f_kxky[idx_y0];
  }
} 

__global__ void get_z0(cuComplex* f_z0, cuComplex* f)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  
  if(idx<nx && idy<ny/2+1) {
    unsigned int idxy_z0 = idy + (ny/2+1)*idx + (nz/2)*nx*(ny/2+1);
    unsigned int idxy = idy + (ny/2+1)*idx;
    
    f_z0[idxy] = f[idxy_z0];
  }
}
    
__global__ void get_z0(float* f_z0, float* f, int nx, int ny, int nz) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  
  if(idx<nx && idy<(ny/2+1)) {
    unsigned int idxy_z0 = idy + (ny/2+1)*idx + (nz/2)*nx*(ny/2+1);
    unsigned int idxy = idy + (ny/2+1)*idx;
    
    f_z0[idxy] = f[idxy_z0];
  }
}  
    
