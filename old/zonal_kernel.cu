__global__ void zonal_tpard(cuComplex* zonal_tmp, cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp, cuComplex* phi_tpard,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, int dens_switch, bool tpar_omegad_corrections)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx;
      if(tpar_omegad_corrections) cidx = c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, rho);
      else cidx = 0.;
      float c2 = __powf(cidx,2.);
      float c4 = __powf(cidx,4.);

      zonal_tmp[index] = __expf(-.5*c2) * ( Dens[index]*dens_switch*(4. - 7.*c2 + c4) + Tpar[index]*(7. - 13.*c2 + 2.*c4) + Tprp[index]*(1. - c2) ) 
                             - 2.*Dens[index] - Tpar[index] - Tprp[index] + phi_tpard[index];
      
    }
  }
}


__global__ void zonal_tperpd(cuComplex* zonal_tmp, cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp, cuComplex* phi_tperpd,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, int dens_switch, bool tperp_omegad_corrections)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx; 
      if(tperp_omegad_corrections) cidx = c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, rho);
      else cidx = 0.;
      float c2 = __powf(cidx,2.);

      zonal_tmp[index] = __expf(-.5*c2) * ( Dens[index]*dens_switch*(3. - c2) + Tpar[index]*(1. - c2) + Tprp[index]*(5. - c2) ) 
                            - 2.*Dens[index] - Tpar[index] - Tprp[index] + phi_tperpd[index];
      
    }
  }
}

__global__ void zonal_qpar_gradpar(cuComplex* zonal_tmp, cuComplex* Dens, cuComplex* Tpar,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, int dens_switch, float Beta_par, bool qpar_gradpar_corrections)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx;
      if(qpar_gradpar_corrections) c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, rho);
      else cidx = 0.;
      float c2 = __powf(cidx,2.);
      float c4 = __powf(cidx,4.);

      zonal_tmp[index] = __expf(-.5*c2) * ( Dens[index]*dens_switch + 2.*Tpar[index] ) * (3. - 6.*c2 + c4) + Beta_par*Tpar[index] - 3.*Dens[index] - 3.*Tpar[index];
      
    }
  }
}

__global__ void zonal_qpar_bgrad(cuComplex* zonal_tmp, cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, int dens_switch, bool qpar_bgrad_corrections)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx;
      if(qpar_bgrad_corrections) cidx = c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, rho);
      else cidx = 0.;
      float c2 = __powf(cidx,2.);
      float c4 = __powf(cidx,4.);

      zonal_tmp[index] = __expf(-.5*c2) * ( Dens[index]*dens_switch*(3.*c2 - c4) + Tpar[index]*(-3. - 9.*c2 - 2.*c4) + 3.*Tprp[index]*(1. - c2) ) + 3.*Tpar[index] - 3.*Tprp[index];
      
    }
  }
}

__global__ void zonal_qperp_gradpar(cuComplex* zonal_tmp, cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp, cuComplex* phi_flr,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, int dens_switch, float zt, bool qperp_gradpar_corrections)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx;
      if(qperp_gradpar_corrections) cidx = c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, rho);
      else cidx = 0.;
      float c2 = __powf(cidx,2.);

      zonal_tmp[index] = __expf(-.5*c2) * ( Dens[index]*dens_switch + Tpar[index] + Tprp[index]) * (1. - c2) - Dens[index] - Tpar[index] + zt*phi_flr[index];
      
    }
  }
}

__global__ void zonal_qperp_bgrad(cuComplex* zonal_tmp, cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp, cuComplex* phi_qperpb,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, int dens_switch, float zt, bool qperp_bgrad_corrections)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx;
      if(qperp_bgrad_corrections) cidx = c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, rho);
      else cidx = 0.;
      float c2 = __powf(cidx,2.);

      zonal_tmp[index] = __expf(-.5*c2) * 2. * ( Dens[index]*dens_switch*c2 + Tpar[index]*(c2 - 1.) + Tprp[index]*(1. + c2) ) + Tpar[index] - Tprp[index] + zt*phi_qperpb[index];
      
    }
  }
}

__global__ void zonal_qpar0(cuComplex* Qpar0, cuComplex* Qpar, cuComplex* Dens, cuComplex* Tpar, 
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, int q0_dens_switch, bool qpar0)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;      
       
      if(idy==0 && idx!=0) {
                
        float cidx;
        if(qpar0) cidx = c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, rho);
        else cidx = 0.;
        float c2 = __powf(cidx,2.);
        float c3 = __powf(cidx,3.);
 
        cuComplex tmp; 
        tmp.x = Qpar[index].x + __expf(-.5*c2) * 
( Dens[index].y*q0_dens_switch + Tpar[index].y ) * (3.*cidx - c3);            
        tmp.y = Qpar[index].y - __expf(-.5*c2) * 
( Dens[index].x*q0_dens_switch + Tpar[index].x ) * (3.*cidx - c3);            
        Qpar0[index] = tmp;
      } 
      else {
        Qpar0[index] = Qpar[index];
      }
    }
  }
}

__global__ void zonal_qprp0(cuComplex* Qprp0, cuComplex* Qprp, cuComplex* Dens, cuComplex* Tprp, 
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, int q0_dens_switch, bool qprp0)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;      
       
      if(idy==0 && idx!=0) {
                
        float cidx;
        if(qprp0) cidx = c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, rho);
        else cidx = 0.;
        float c2 = __powf(cidx,2.);   

        cuComplex tmp; 
        tmp.x = Qprp[index].x + __expf(-.5*c2) * 
( Dens[index].y*q0_dens_switch + Tprp[index].y ) * (cidx);            
        tmp.y = Qprp[index].y - __expf(-.5*c2) *
( Dens[index].x*q0_dens_switch + Tprp[index].x ) * (cidx);            
        Qprp0[index] = tmp;
      } 
      else {
        Qprp0[index] = Qprp[index];
      }
    }
  }
}

__global__ void zonal_qpar0_fsa(cuComplex* Qpar0, cuComplex* Qpar, cuComplex* Dens, cuComplex* Tpar_fsa, 
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, int q0_dens_switch, bool qpar0)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;      
       
      if(idy==0 && idx!=0) {
                
        float cidx;
        if(qpar0) cidx = c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, rho);
        else cidx = 0.;
        float c2 = __powf(cidx,2.);
        float c3 = __powf(cidx,3.);

        cuComplex tmp; 
        tmp.x = Qpar[index].x + __expf(-.5*c2) * 
( Dens[index].y*q0_dens_switch + Tpar_fsa[idx].y ) * (3.*cidx - c3);            
        tmp.y = Qpar[index].y - __expf(-.5*c2) * 
( Dens[index].x*q0_dens_switch + Tpar_fsa[idx].x ) * (3.*cidx - c3);            
        Qpar0[index] = tmp;
      } 
      else {
        Qpar0[index] = Qpar[index];
      }
    }
  }
}

__global__ void zonal_qprp0_fsa(cuComplex* Qprp0, cuComplex* Qprp, cuComplex* Dens, cuComplex* Tprp_fsa, 
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, int q0_dens_switch, bool qprp0)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;      
       
      if(idy==0 && idx!=0) {
                
        float cidx;
        if(qprp0) cidx = c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, rho);
        else cidx = 0.;
        float c2 = __powf(cidx,2.);
   
        cuComplex tmp; 
        tmp.x = Qprp[index].x + __expf(-.5*c2) * 
( Dens[index].y*q0_dens_switch + Tprp_fsa[idx].y ) * (cidx);            
        tmp.y = Qprp[index].y - __expf(-.5*c2) *
( Dens[index].x*q0_dens_switch + Tprp_fsa[idx].x ) * (cidx);            
        Qprp0[index] = tmp;
      } 
      else {
        Qprp0[index] = Qprp[index];
      }
    }
  }
}

__global__ void RH_equilibrium_init(cuComplex* Dens, cuComplex* Upar, cuComplex* Tpar, cuComplex* Tprp, cuComplex* Qpar, cuComplex* Qprp, 
                                      float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, specie s)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx = 0.; // c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, s.rho);
      float c2 = __powf(cidx,2.);
      float c4 = __powf(cidx,4.);

      Dens[index].x = s.dens*__expf(-c2/2.);
      Dens[index].y = 0.;

      Upar[index].x = 0.;
      Upar[index].y = -cidx*s.dens*__expf(-c2/2.);

      Tpar[index].x = __expf(-c2/2.) * ( s.temp*(1. - c2) );
      Tpar[index].y = 0.;

      Tprp[index].x = __expf(-c2/2.) * s.temp;
      Tprp[index].y = 0.;

      Qpar[index].x = 0.;
      Qpar[index].y = __expf(-c2/2.) * ( s.dens* s.temp ) * cidx * ( c2 - 3. );
      
      Qprp[index].x = 0.;
      Qprp[index].y = __expf(-c2/2.) * ( s.dens*s.temp ) * (-cidx);
    }
  }
}

__global__ void RH_eq_tpard(cuComplex* r_tmp, cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp, cuComplex* phi_tpard,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, specie s)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx = 0.; // c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, s.rho);
      float c2 = __powf(cidx,2.);
      float c4 = __powf(cidx,4.);

      r_tmp[index].x = __expf(-c2/2.) * ( (s.dens*s.temp*s.temp)*( 3. - 6.*c2 + c4 ) + (s.dens*s.temp*s.temp)*(1. - c2) ) - 2*Dens[index].x - Tpar[index].x - Tprp[index].x + phi_tpard[index].x;
      r_tmp[index].y = -2.*Dens[index].y - Tpar[index].y - Tprp[index].y + phi_tpard[index].y;
      
    }
  }
}

__global__ void RH_eq_tperpd(cuComplex* r_tmp, cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp, cuComplex* phi_tperpd,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, specie s)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx = 0.; // c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, s.rho);
      float c2 = __powf(cidx,2.);
      float c4 = __powf(cidx,4.);

      r_tmp[index].x = __expf(-c2/2.) * ( (s.dens*s.temp*s.temp)*(1. - c2) + (2.*(s.dens + 2.*s.temp)) ) - 2*Dens[index].x - Tpar[index].x - Tprp[index].x + phi_tperpd[index].x;
      r_tmp[index].y = -2.*Dens[index].y - Tpar[index].y - Tprp[index].y + phi_tperpd[index].y;
      
    }
  }
}

__global__ void RH_eq_qpar_gradpar(cuComplex* r_tmp, cuComplex* Dens, cuComplex* Tpar,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, specie s)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx = 0.; // c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, s.rho);
      float c2 = __powf(cidx,2.);
      float c4 = __powf(cidx,4.);

      r_tmp[index].x = __expf(-c2/2.) * ( (s.dens*s.temp*s.temp)*( 3. - 6.*c2 + c4 ) ) - 3.*Dens[index].x - 3.*Tpar[index].x;
      r_tmp[index].y = -3.*Dens[index].y - 3.*Tpar[index].y;
      
    }
  }
}

__global__ void RH_eq_qpar_bgrad(cuComplex* r_tmp, cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, specie s)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx = 0.; // c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, s.rho);
      float c2 = __powf(cidx,2.);
      float c4 = __powf(cidx,4.);

      r_tmp[index].x = __expf(-c2/2.) * ( -(s.dens*s.temp*s.temp)*( 3. - 6.*c2 + c4 ) + 3.*(s.dens*s.temp*s.temp)*(1. - c2) ) + 3.*Tpar[index].x - 3.*Tprp[index].x;
      r_tmp[index].y = 3.*Tpar[index].y - 3.*Tprp[index].y;
      
    }
  }
}

__global__ void RH_eq_qpard(cuComplex* s_tmp, cuComplex* Upar, cuComplex* Qpar, cuComplex* Qprp, 
                            float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, specie s)
{

  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx = 0.; // c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, s.rho);
      float c2 = __powf(cidx,2.);
      float c3 = __powf(cidx,3.);
      float c5 = __powf(cidx,5.);
 
      s_tmp[index].x = 6.*Upar[index].x - 3.*Qpar[index].x - 3.*Qprp[index].x;
      s_tmp[index].y = __expf(-c2/2.) * ( s.dens*s.temp*s.temp*(-18.*cidx + 11.*c3 - c5) ) + 6.*Upar[index].y - 3.*Qpar[index].y - 3.*Qprp[index].y; 

    }
  }
}
    
__global__ void RH_eq_qperp_gradpar(cuComplex* r_tmp, cuComplex* Dens, cuComplex* Tpar, cuComplex* phi_flr,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, specie s)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx = 0.; // c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, s.rho);
      float c2 = __powf(cidx,2.);

      r_tmp[index].x = __expf(-.5*c2) * ( s.dens*s.temp*s.temp) * (1. - c2) - Dens[index].x - Tpar[index].x+ s.zt*phi_flr[index].x;
      r_tmp[index].y = - Dens[index].y - Tpar[index].y + s.zt*phi_flr[index].y;
    }
  }
}

__global__ void RH_eq_qperp_bgrad(cuComplex* r_tmp, cuComplex* Tpar, cuComplex* Tprp, cuComplex* phi_qperpb,
                              float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, specie s)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx = 0.; // c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, s.rho);
      float c2 = __powf(cidx,2.);
      float c4 = __powf(cidx,4.);

      r_tmp[index].x = __expf(-c2/2.) * ( -2.*(s.dens*s.temp*s.temp)*(1. - c2) + (2.*(s.dens + 2.*s.temp)) ) + Tpar[index].x - Tprp[index].x + s.zt*phi_qperpb[index].x;
      r_tmp[index].y = Tpar[index].y - Tprp[index].y + s.zt*phi_qperpb[index].y;
      
    }
  }
}

__global__ void RH_eq_qperpd(cuComplex* s_tmp, cuComplex* Upar, cuComplex* Qpar, cuComplex* Qprp, 
                            float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, specie s)
{

  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float cidx = 0.; // c(kx[idx], gds22[idz], qsf, eps, bmagInv[idz], shat, s.rho);
      float c2 = __powf(cidx,2.);
      float c3 = __powf(cidx,3.);
      float c5 = __powf(cidx,5.);
 
      s_tmp[index].x = Upar[index].x - Qpar[index].x - Qprp[index].x;
      //s_tmp[index].y = __expf(-c2/2.) * ( s.dens*(-5.*cidx + c3) + s.temp*(-10.*cidx +2.*c3) ) + Upar[index].y - Qpar[index].y - Qprp[index].y; 
      s_tmp[index].y = __expf(-c2/2.) * ( s.dens*s.temp*s.temp*(-5.*cidx + c3) ) + Upar[index].y - Qpar[index].y - Qprp[index].y; 

    }
  }
}

__global__ void cattoTpar0(cuComplex* Tpar0, cuComplex* Phi_fluxsurfavg, float* kx, float* gds22, float qsf, float eps, float* bmagInv, float shat, float rho, float shaping_ps)
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

       
      if(idy==0) {
        //Tpar0 = -2*1.6*eps^(3/2)*k_perp^2*rho_pol^2*<<<Phi>>>
        Tpar0[index] = -2.*shaping_ps*__powf(eps,1.5)*__powf(kx[idx]*shatInv*qsf/eps*bmagInv[idz]*rho,2.)*gds22[idz]*Phi_fluxsurfavg[idx];
      }
      else {
        Tpar0[index].x = 0.;
        Tpar0[index].y = 0.;
      }
      
      
    }
  }
}
