__global__ void growthRate(cuComplex *omega, cuComplex *phinew, cuComplex *phiold, float dt)
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  
  //i_dt = i/dt
  cuComplex i_dt;
  i_dt.x = 0;
  i_dt.y = 1./dt;
  
  
  if(idy<(ny/2+1) && idx<nx) 
  {
    unsigned int index = idy + (ny/2+1)*idx;
    
    cuComplex ratio = phinew[index+nx*(ny/2+1)*((int)(.5*nz))] / phiold[index+nx*(ny/2+1)*((int)(.5*nz))];
    
    cuComplex log;
    log.x = logf(cuCabsf(ratio));
    log.y = atan2f(ratio.y,ratio.x);
    omega[index] = log*i_dt;
  }
}

__global__ void normalize(cuComplex *f, cuComplex *fnorm, float norm)
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
      c_k.x = cos(atan2f(fnorm[kidx_z].y,fnorm[kidx_z].x))/cuCabsf(fnorm[kidx_z]);
      c_k.y = -sin(atan2f(fnorm[kidx_z].y,fnorm[kidx_z].x))/cuCabsf(fnorm[kidx_z]);

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
       c_k.x = cos(atan2f(fnorm[kidx_z].y,fnorm[kidx_z].x))/cuCabsf(fnorm[kidx_z]);
       c_k.y = -sin(atan2f(fnorm[kidx_z].y,fnorm[kidx_z].x))/cuCabsf(fnorm[kidx_z]);

       f[index] = (c_k * f[index]) * norm;
     }  
   }
 }    
}

__global__ void normalize_covering(cuComplex *f, cuComplex *fnorm, float norm, int nLinks, int nChains)
{
  unsigned int i = get_idx();
  unsigned int n = get_idy();
  unsigned int p = get_idz();
  
    if(i<nz && p<nLinks && n<nChains) 
    {
      unsigned int index = i + p*nz + n*nz*nLinks;
      unsigned int index_0 = nz*nLinks/2 + n*nz*nLinks;
      

      cuComplex c_k;
      c_k.x = cos(atan2f(fnorm[index_0].y,fnorm[index_0].x))/cuCabsf(fnorm[index_0]);
      c_k.y = -sin(atan2f(fnorm[index_0].y,fnorm[index_0].x))/cuCabsf(fnorm[index_0]);

      f[index] = (c_k * f[index]) * norm;
    }
  
}

__global__ void prevent_overflow_by_mode(cuComplex* result,cuComplex* b, float lim, float scaler)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + (ny/2+1)*(nx)*idz;
      unsigned int idxyz0 = idy + (ny/2+1)*idx + (ny/2+1)*(nx)*(nz/2);
    
      if( abs(b[idxyz0].x) > lim || abs(b[idxyz0].y) > lim ) result[index] = scaler*b[index]; 
    }
  }
    
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
        unsigned int idxyz0 = idy + (ny/2+1)*idx + (ny/2+1)*(nx)*(nz/2);
	
        if( abs(b[idxyz0].x) > lim || abs(b[idxyz0].y) > lim ) result[index] = scaler*b[index]; 
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
        if(idy==0) fac = 1.;
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
      if(idy==0) fac = 1.;
      else fac = 0.5;
      for(int i = 1; i<nz; i++) {
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
      float fac = 1.;  //only ky=0 modes      
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

__global__ void volflux_zonal_rms(float* flux, cuComplex* f, cuComplex* g, float* jacobian, float fluxDenSqInv)
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
      float fac = 1.;  //only ky=0 modes      
      for(int i = 0; i<nz; i++) {
        index = idxy0 + nx*(ny/2+1)*i;
	fz = f[index]*jacobian[i];
	gz = cuConjf( g[index] )*jacobian[i];
	fsum = fsum + fz;
	gsum = gsum + gz;
      }    
      fg = fsum*gsum;        
      flux[idx] = sqrt(fac*fg.x*fluxDenSqInv);

    }
  
}

__global__ void field_line_avg(cuComplex* favg, cuComplex* f, float* jacobian, float fluxDenInv)
{  
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  if(idx<nx && idy<ny/2+1) 
  {
    unsigned int idxy = idy + (ny/2+1)*idx;
    favg[idxy].x = 0.;
    favg[idxy].y = 0.;
    for(int idz=0; idz<nz; idz++) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      favg[idxy] = favg[idxy] + f[index]*jacobian[idz];
    }
    favg[idxy] = favg[idxy]*fluxDenInv;
  }
}

__global__ void field_line_avg_xyz(cuComplex* favg, cuComplex* f, float* jacobian, float fluxDenInv)
{  
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int IDZ = get_idz();
  if(idx<nx && idy<ny/2+1 && IDZ<nz) 
  {
    unsigned int idxy = idy + (ny/2+1)*idx;
    favg[idxy].x = 0.;
    favg[idxy].y = 0.;
    for(int idz=0; idz<nz; idz++) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      favg[idxy] = favg[idxy] + f[index]*jacobian[idz];
    }
    favg[idxy+nx*(ny/2+1)*IDZ] = favg[idxy]*fluxDenInv;
  }
}


__global__ void volflux_zonal_complex(cuComplex* T_fsa_X, cuComplex* T, float* jacobian, float fluxDenInv)
{
  unsigned int idx = get_idx();
  unsigned int idy = 0;
  if(idx<nx) 
  {
    T_fsa_X[idx].x = 0.;
    T_fsa_X[idx].y = 0.;
    unsigned int idxy = idy + (ny/2+1)*idx;
    for(int idz=0; idz<nz; idz++) {
      T_fsa_X[idx].x = fma(T[idxy+idz*nx*(ny/2+1)].x,jacobian[idz], T_fsa_X[idx].x);
      T_fsa_X[idx].y = fma(T[idxy+idz*nx*(ny/2+1)].y,jacobian[idz], T_fsa_X[idx].y);
    }
    __syncthreads();
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
	//phi2_XY[idxy] = 0;
      }
      else {
        kphi2_XY[idxy] = phi2_XY[idxy] / k[idy];
      }
      		     
  }
}       

__global__ void getModeVal(float* val, cuComplex* f, int iky, int ikx, int z) 
{
  val[0] = f[iky + (ny/2+1)*ikx + nx*(ny/2+1)*z].x;
}

__global__ void getModeValReal(float* val, cuComplex* f, int iky, int ikx, int z) 
{
  val[0] = f[iky + (ny/2+1)*ikx + nx*(ny/2+1)*z].x;
}
__global__ void getModeValImag(float* val, cuComplex* f, int iky, int ikx, int z) 
{
  val[0] = f[iky + (ny/2+1)*ikx + nx*(ny/2+1)*z].y;
}
__global__ void getModeValSq(float* val, cuComplex* f, int iky, int ikx, int z) 
{
  cuComplex a = f[iky + (ny/2+1)*ikx + nx*(ny/2+1)*z]*cuConjf(f[iky + (ny/2+1)*ikx + nx*(ny/2+1)*z]);
  val[0] = a.x;
}

__global__ void getPhiVal(float* val, cuComplex* Phi, int iky, int ikx, int z) 
{
  val[0] = Phi[iky + (ny/2+1)*ikx + nx*(ny/2+1)*z].x;
}
  
__global__ void getPhiVal(float* val, float* Phi, int iky, int ikx, int z) 
{
  val[0] = Phi[iky + (ny/2+1)*ikx + nx*(ny/2+1)*z];
}

__global__ void getPhiVal(float* val, cuComplex* Phi, int ikx) 
{
  val[0] = Phi[ikx].x;
}

__global__ void getPhiVal(float* val, float* Phi, int ikx) 
{
  val[0] = Phi[ikx];
}
__global__ void getky0z0(float* f_kxky0, cuComplex* f_kxky)
{
  unsigned int idx = get_idx();
  
  if(idx<nx) {
    unsigned int idx_y0 = 0 + (ny/2+1)*idx + nx*(ny/2+1)*(nz/2);
    f_kxky0[idx] = f_kxky[idx_y0].x;
  }
} 

__global__ void getky0(float* f_kxky0, cuComplex* f_kxky)
{
  unsigned int idx = get_idx();
  
  if(idx<nx) {
    unsigned int idx_y0 = 0 + (ny/2+1)*idx;
    f_kxky0[idx] = f_kxky[idx_y0].x;
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


__global__ void get_real_X(float* R, cuComplex* C)

{
  unsigned int idx = get_idx();
  
  if(idx<nx) {
    R[idx] = C[idx].x;
  }
} 

 
__global__ void PSdiagnostic_odd(cuComplex* Qps, float psfac, float* kx, float* gds22, float
				qsf, float eps, float* bmagInv, cuComplex* T, float shat, float rho)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  float shatInv;
  if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  } else {
    shatInv = 1.;
  }
  
  if(nz<=zthreads) {
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      //psfac is 3 or 1 depending on whether using Qpar or Qprp, respectively
       
      if(idy==0) {
	//double check signs... k_r = -kx for ky=0?
		
	cuComplex tmp;
	tmp.x = -psfac*(-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho*T[index].y;
	tmp.y = psfac*(-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho*T[index].x;
	Qps[index] = tmp;
      }
      else {
        Qps[index].x = Qps[index].y = 0;
      }
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny/2+1 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
	if(idy==0) {
	  //double check signs... k_r = -kx for ky=0?
	  cuComplex tmp;
	  tmp.x = -psfac*(-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*rho*T[idx].y;
	  tmp.y = psfac*(-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*rho*T[idx].x;
	  Qps[index] = tmp;
        }
	else {
	  Qps[index].x = Qps[index].y = 0.;
	}
      }
    }
  }
}


__global__ void PSdiagnostic_odd_fsa(cuComplex* Qps, float psfac, float* kx, float* gds22, float
				qsf, float eps, float* bmagInv, cuComplex* T_fluxsurfavg, float shat, float rho)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  float shatInv;
  if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  } else {
    shatInv = 1.;
  }
  
  if(nz<=zthreads) {
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      //psfac is 3 or 1 depending on whether using Qpar or Qprp, respectively
       
      if(idy==0) {
	//double check signs... k_r = -kx for ky=0?
		
	cuComplex tmp;
	tmp.x = -psfac*(-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho*T_fluxsurfavg[idx].y;
	tmp.y = psfac*(-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho*T_fluxsurfavg[idx].x;
	Qps[index] = tmp;
      }
      else {
        Qps[index].x = Qps[index].y = 0.;
      }
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny/2+1 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
	if(idy==0) {
	  //double check signs... k_r = -kx for ky=0?
	  cuComplex tmp;
	  tmp.x = -psfac*(-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*rho*T_fluxsurfavg[idx].y;
	  tmp.y = psfac*(-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*rho*T_fluxsurfavg[idx].x;
	  Qps[index] = tmp;
        }
	else {
	  Qps[index].x = Qps[index].y = 0.;
	}
      }
    }
  }
}


__global__ void PSdiagnostic_even(cuComplex* Qps, float psfac, float* kx, float* gds22, float
				qsf, float eps, float* bmagInv, cuComplex* T, float shat, float rho)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  float shatInv;
  if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  } else {
    shatInv = 1.;
  }
  
  if(nz<=zthreads) {
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      //psfac is 3 or 1 depending on whether using Qpar or Qprp, respectively
       
      if(idy==0) {
	//double check signs... k_r = -kx for ky=0?
		
	cuComplex tmp;
	tmp.x = psfac*pow((-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho,2)*T[index].x;
	tmp.y = psfac*pow((-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho,2)*T[index].y;
	Qps[index] = tmp;
      }
      else {
        Qps[index].x = Qps[index].y = 0;
      }
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny/2+1 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
	if(idy==0) {
	  //double check signs... k_r = -kx for ky=0?
	  cuComplex tmp;
	  tmp.x = psfac*pow((-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*rho,2)*T[idx].x;
	  tmp.y = psfac*pow((-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*rho,2)*T[idx].y;
	  Qps[index] = tmp;
        }
	else {
	  Qps[index].x = Qps[index].y = 0.;
	}
      }
    }
  }
}


__global__ void PSdiagnostic_even_fsa(cuComplex* Qps, float psfac, float* kx, float* gds22, float
				qsf, float eps, float* bmagInv, cuComplex* T_fluxsurfavg, float shat, float rho)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  float shatInv;
  if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  } else {
    shatInv = 1.;
  }
  
  if(nz<=zthreads) {
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      //psfac is 3 or 1 depending on whether using Qpar or Qprp, respectively
       
      if(idy==0) {
	//double check signs... k_r = -kx for ky=0?
		
	cuComplex tmp;
	tmp.x = psfac*pow((-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho,2)*T_fluxsurfavg[idx].x;
	tmp.y = psfac*pow((-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho,2)*T_fluxsurfavg[idx].y;
	Qps[index] = tmp;
      }
      else {
        Qps[index].x = Qps[index].y = 0.;
      }
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny/2+1 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
	if(idy==0) {
	  //double check signs... k_r = -kx for ky=0?
	  cuComplex tmp;
	  tmp.x = psfac*pow((-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*rho,2)*T_fluxsurfavg[idx].x;
	  tmp.y = psfac*pow((-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*rho,2)*T_fluxsurfavg[idx].y;
	  Qps[index] = tmp;
        }
	else {
	  Qps[index].x = Qps[index].y = 0.;
	}
      }
    }
  }
}

