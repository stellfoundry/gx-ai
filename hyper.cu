__global__ void get_iso_shear(float* shear_rate_nz, cuComplex* Phi, float* kx, float* ky) {
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idx<nx && idy<(ny/2+1) && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      float fac;
      cuComplex s;
      if(idy==0) fac = 0.25;
      else fac = 0.5;
      float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
      s =  Phi[index]*cuConjf(Phi[index])*pow(kperp2,2)*fac;
      shear_rate_nz[index] = s.x;
    }
  }
}      

__global__ void hyper_filter_iso(cuComplex* field, float* shear_rate_nz, float* kx, float* ky, float kperp4_max_Inv, float dt_loc, float D_hyper)
{  
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
      field[index] = field[index]/( 1. + dt_loc*D_hyper*shear_rate_nz[idz]*pow(kperp2,2)*kperp4_max_Inv );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
        float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
	
        field[index] = field[index]/( 1. + dt_loc*D_hyper*shear_rate_nz[IDZ]*pow(kperp2,2)*kperp4_max_Inv );
      }
    }
  }

}

void iso_shear(float* shear_rate_nz_tmpZ, float* shear_rate_nz_tmpXYZ, cuComplex* Phi) {
  get_iso_shear<<<dimGrid,dimBlock>>>(shear_rate_nz_tmpXYZ, Phi,kx,ky);
  sumReduc_Partial(shear_rate_nz_tmpZ, shear_rate_nz_tmpXYZ, Nx*(Ny/2+1)*Nz, Nz, false);
  sqrtZ<<<dimGrid,dimBlock>>>(shear_rate_nz_tmpZ, shear_rate_nz_tmpZ);
}

void filterHyper_iso(int is, fields_struct * fields_d, 
		 float* tmpXYZ, float* shear_rate_nz, float dt_loc)
{
  cuComplex* Phi = fields_d->phi;
  cuComplex* Dens = fields_d->dens[is];
  cuComplex* Upar = fields_d->upar[is];
  cuComplex* Tpar = fields_d->tpar[is];
  cuComplex* Tprp = fields_d->tprp[is];
  cuComplex* Qpar = fields_d->qpar[is];
  cuComplex* Qprp = fields_d->qprp[is];
  iso_shear(shear_rate_nz, tmpXYZ, Phi); 
  hyper_filter_iso<<<dimGrid,dimBlock>>>(Dens, shear_rate_nz, kx, ky, kperp4_max_Inv, dt_loc,D_hyper);
  hyper_filter_iso<<<dimGrid,dimBlock>>>(Upar, shear_rate_nz, kx, ky, kperp4_max_Inv, dt_loc,D_hyper);
  hyper_filter_iso<<<dimGrid,dimBlock>>>(Tpar, shear_rate_nz, kx, ky, kperp4_max_Inv, dt_loc,D_hyper);
  hyper_filter_iso<<<dimGrid,dimBlock>>>(Tprp, shear_rate_nz, kx, ky, kperp4_max_Inv, dt_loc,D_hyper);
  hyper_filter_iso<<<dimGrid,dimBlock>>>(Qpar, shear_rate_nz, kx, ky, kperp4_max_Inv, dt_loc,D_hyper);
  hyper_filter_iso<<<dimGrid,dimBlock>>>(Qprp, shear_rate_nz, kx, ky, kperp4_max_Inv, dt_loc,D_hyper);
}

//////////////////////////////////////////////////////////////////


__global__ void get_aniso_shear_nz(float* shear_rate_nz, cuComplex* Phi, float* kx, float* ky) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idx<nx && idy<(ny/2+1) && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      float fac;
      cuComplex s;
      float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
      if(idy==0) fac = 0.;   //for nonzonal components, zero ky=0 modes so that they are not included in sum
      else fac = 0.5;
      s =  Phi[index]*cuConjf(Phi[index])*pow(kperp2,2)*fac;
      shear_rate_nz[index] = s.x;  //shearing rate due to non-zonal modes (on non-zonal modes)
    }
  }
}      

__global__ void get_aniso_shear_z(float* shear_rate_z, cuComplex* Phi, float* kx, float* ky) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idx<nx && idy<(ny/2+1) && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      float fac;
      cuComplex s;
      float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
      if(idy==0) fac = 1.;
      else fac = 0.; //for zonal component, zero ky!=0 modes so that they are not included in sum
      s =  Phi[index]*cuConjf(Phi[index])*pow(kperp2,2)*fac;
      shear_rate_z[index] = s.x;  //shearing rate due to zonal modes (on non-zonal modes)
    }
  }
}      
__global__ void get_aniso_shear_z_nz(float* shear_rate_z_nz, cuComplex* Phi, float* kx, float* ky) 
{
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idx<nx && idy<(ny/2+1) && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      float fac;
      cuComplex s;
      if(idy==0) fac = 0.;   //for nonzonal components, zero ky=0 modes so that they are not included in sum
      else fac = 0.5;
      s = Phi[index]*cuConjf(Phi[index])*pow(ky[idy],4)*fac;
      shear_rate_z_nz[index] = s.x;  //shearing rate due to non-zonal modes (on zonal modes)

    }
  }
}      

__global__ void finish_get_aniso_shear(float* shear_rate_nz, float* shear_rate_z, float* shear_rate_z_nz)
{
  unsigned int idz = get_idz();
  
  if(idz<nz) {
    float omega_osc = 0.4;
    shear_rate_nz[idz] = 0.5* ( -omega_osc + sqrt( pow(omega_osc,2) + 2*shear_rate_nz[idz] ) );   
    shear_rate_z[idz] = 0.5* ( -omega_osc + sqrt( pow(omega_osc,2) + 2*shear_rate_z[idz] ) );   
    shear_rate_z_nz[idz] = 0.5* ( -omega_osc + sqrt( pow(omega_osc,2) + 2*shear_rate_z_nz[idz] ) );   
  }
}

__global__ void hyper_filter_aniso(cuComplex* field, float* shear_rate_nz, float* shear_rate_z, float* shear_rate_z_nz,
                float* kx, float* ky, float kperp4_max_Inv, float kx4_max_Inv, float ky_max_Inv, float dt_loc, float D_hyper)
{  
  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
      if(idy==0) { 
        field[index] = field[index]/( 1. + dt_loc*D_hyper*shear_rate_z_nz[idz]*pow(kx[idx],4)*kx4_max_Inv );
      }
      else {
        field[index] = field[index]/( 1. + dt_loc*D_hyper*
           ( shear_rate_nz[idz]*pow(kperp2,2)*kperp4_max_Inv + shear_rate_z[idz]*pow(kx[idx],4)*kx4_max_Inv*ky[idy]*ky_max_Inv ));
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
        float kperp2 = kx[idx]*kx[idx] + ky[idy]*ky[idy];
	
        if(idy==0) {
          field[index] = field[index]/( 1. + dt_loc*D_hyper*shear_rate_z_nz[idz]*pow(kx[idx],4)*kx4_max_Inv );
        }
        else {
          field[index] = field[index]/( 1. + dt_loc*D_hyper*
             ( shear_rate_nz[idz]*pow(kperp2,2)*kperp4_max_Inv + shear_rate_z[idz]*pow(kx[idx],4)*kx4_max_Inv*ky[idy]*ky_max_Inv ));
        }
      }
    }
  }

}

void aniso_shear(float* shear_rate_nz, float* shear_rate_z, float* shear_rate_z_nz, float* tmpXYZ, cuComplex* Phi) {
  get_aniso_shear_nz<<<dimGrid,dimBlock>>>(tmpXYZ, Phi,kx,ky);
  sumReduc_Partial(shear_rate_nz, tmpXYZ, Nx*(Ny/2+1)*Nz, Nz, false);
  
  get_aniso_shear_z<<<dimGrid,dimBlock>>>(tmpXYZ, Phi,kx,ky);
  sumReduc_Partial(shear_rate_z, tmpXYZ, Nx*(Ny/2+1)*Nz, Nz, false);
  
  get_aniso_shear_z_nz<<<dimGrid,dimBlock>>>(tmpXYZ, Phi,kx,ky);
  sumReduc_Partial(shear_rate_z_nz, tmpXYZ, Nx*(Ny/2+1)*Nz, Nz, false);
  
  finish_get_aniso_shear<<<dimGrid,dimBlock>>>(shear_rate_nz,shear_rate_z,shear_rate_z_nz); 
}

void filterHyper_aniso(int is, fields_struct * fields_d,		 float* tmpXYZ, hyper_struct * hyper, float dt_loc)
{
  cuComplex* Phi = fields_d->phi;
  cuComplex* Dens = fields_d->dens[is];
  cuComplex* Upar = fields_d->upar[is];
  cuComplex* Tpar = fields_d->tpar[is];
  cuComplex* Tprp = fields_d->tprp[is];
  cuComplex* Qpar = fields_d->qpar[is];
  cuComplex* Qprp = fields_d->qprp[is];
  float* shear_rate_nz = hyper->shear_rate_nz;
  float* shear_rate_z = hyper->shear_rate_z;
  float* shear_rate_z_nz = hyper->shear_rate_z_nz;
  aniso_shear(shear_rate_nz, shear_rate_z, shear_rate_z_nz, tmpXYZ, Phi); 
  hyper_filter_aniso<<<dimGrid,dimBlock>>>(Dens, shear_rate_nz, shear_rate_z, shear_rate_z_nz, kx, ky,
  			                   kperp4_max_Inv, kx4_max_Inv, ky_max_Inv, dt_loc,D_hyper);
  hyper_filter_aniso<<<dimGrid,dimBlock>>>(Upar, shear_rate_nz, shear_rate_z, shear_rate_z_nz, kx, ky,
                    			   kperp4_max_Inv, kx4_max_Inv, ky_max_Inv, dt_loc,D_hyper);
  hyper_filter_aniso<<<dimGrid,dimBlock>>>(Tpar, shear_rate_nz, shear_rate_z, shear_rate_z_nz, kx, ky,
              			           kperp4_max_Inv, kx4_max_Inv, ky_max_Inv, dt_loc,D_hyper);
  hyper_filter_aniso<<<dimGrid,dimBlock>>>(Tprp, shear_rate_nz, shear_rate_z, shear_rate_z_nz, kx, ky,
               			           kperp4_max_Inv, kx4_max_Inv, ky_max_Inv, dt_loc,D_hyper);
  hyper_filter_aniso<<<dimGrid,dimBlock>>>(Qpar, shear_rate_nz, shear_rate_z, shear_rate_z_nz, kx, ky,
                		           kperp4_max_Inv, kx4_max_Inv, ky_max_Inv, dt_loc,D_hyper);
  hyper_filter_aniso<<<dimGrid,dimBlock>>>(Qprp, shear_rate_nz, shear_rate_z, shear_rate_z_nz, kx, ky,
                   			   kperp4_max_Inv, kx4_max_Inv, ky_max_Inv, dt_loc,D_hyper);
}

