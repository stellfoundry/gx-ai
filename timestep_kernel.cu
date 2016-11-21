// these kernels contain all linear terms except grad_parallel (ZDerivCovering) terms, which involve FFTs and  must be evaluated separately

__global__ void density_linear_terms(cuComplex* dens_field, cuComplex* phi, cuComplex* dens, cuComplex* tpar, cuComplex* tprp,
                      float* kx, float* ky, float shat, float rho, float vt, float tprim, float fprim, float zt,  
                      float *gds2, float *gds21, float *gds22, float *bmagInv,
                      float* gb,float* gb0,float* cv, float* cv0)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      cuComplex phi_n = phi[index] * ( fprim*sgam0(bidx) + tprim * flr(bidx) );  

      cuComplex iomegastar = iOmegaStar(ky[idy]);
      
      cuComplex phi_nd = phi[index] * ( zt * (2.*sgam0(bidx) + flr(bidx)) ); 

      cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]);

      dens_field[index] = iomegastar*phi_n - iomegad*(phi_nd + 2.*dens[index] + tpar[index] + tprp[index]); 

    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

        unsigned int IDZ = idz + zthreads*i;

        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;

        double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
  
        cuComplex phi_n = phi[index] * ( fprim*sgam0(bidx) + tprim * flr(bidx) );  

        cuComplex iomegastar = iOmegaStar(ky[idy]);
        
        cuComplex phi_nd = phi[index] * ( zt * (2.*sgam0(bidx) + flr(bidx)) ); 

        cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]);

        dens_field[index] = iomegastar*phi_n - iomegad*(phi_nd + 2.*dens[index] + tpar[index] + tprp[index]); 

      }
    }
  }
}

__global__ void upar_linear_terms(cuComplex* upar_field, cuComplex* phi, cuComplex* apar, cuComplex* aparNew, cuComplex* dens, cuComplex* upar, 
		      cuComplex* tprp, cuComplex* qpar, cuComplex* qprp,
                      float* kx, float* ky, float shat, float rho, float vt, float tprim, float fprim, float zt, float* bgrad,
                      float *gds2, float *gds21, float *gds22, float *bmagInv,
                      float* gb,float* gb0,float* cv, float* cv0, float beta, float dt, int is)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      cuComplex phi_flr = phi[index] * flr(bidx);

      cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]);

      upar_field[index] = vt*(dens[index] + tprp[index] + zt*phi_flr)*bgrad[idz] - iomegad*( 4.*upar[index] + qpar[index] + qprp[index]); 

      if(beta>0.) {
        cuComplex iomegastar = iOmegaStar(ky[idy]);
        cuComplex apar_term = apar[index] * ( fprim*sgam0(bidx) + tprim *(sgam0(bidx)+flr(bidx)) );  
        cuComplex dApardt_sgam0 = sgam0(bidx) * ( aparNew[index] - apar[index] ) / dt;    

        upar_field[index] = upar_field[index] - vt*iomegastar*apar_term;
        if(is!=nspecies-1) upar_field[index] = upar_field[index] + vt*zt*dApardt_sgam0;  // for ions, add in dApar/dt term
        // for electrons, we are dropping dUpar/dt term, so upar equation becomes apar equation, and we don't need dApar/dt here
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

        unsigned int IDZ = idz + zthreads*i;

        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;

        double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
  
        cuComplex phi_flr = phi[index] * flr(bidx);

        cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]);

        upar_field[index] = vt*(dens[index] + tprp[index] + zt*phi_flr)*bgrad[IDZ] - iomegad*( 4.*upar[index] + qpar[index] + qprp[index]); 

        if(beta>0.) {
          cuComplex iomegastar = iOmegaStar(ky[idy]);
          cuComplex apar_term = apar[index] * ( fprim*sgam0(bidx) + tprim *(sgam0(bidx)+flr(bidx)) );  
          cuComplex dApardt_sgam0 = sgam0(bidx) * ( aparNew[index] - apar[index] ) / dt;    

          upar_field[index] = upar_field[index] - vt*iomegastar*apar_term + vt*zt*dApardt_sgam0;
        }

      }
    }
  }
}

__global__ void tpar_linear_terms(cuComplex* tpar_field, cuComplex* phi, cuComplex* dens, cuComplex* upar, 
                      cuComplex* tpar, cuComplex* tprp, cuComplex* qprp,
                      float* kx, float* ky, float shat, float rho, float vt, float tprim, float fprim, float zt, float* bgrad,
                      float *gds2, float *gds21, float *gds22, float *bmagInv,
                      float* gb,float* gb0,float* cv, float* cv0,
                      float nu_ss, cuComplex nu1, cuComplex nu2, cuComplex mu1, cuComplex mu2, bool varenna,
                      cuComplex* rparpar, cuComplex* rparprp, bool higher_order_moments)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      cuComplex phi_tpar = phi[index] * sgam0(bidx) * tprim;

      cuComplex iomegastar = iOmegaStar(ky[idy]);

      cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]);

      cuComplex phi_tpard = phi[index] * ( zt * 2. * sgam0(bidx) );
 
      float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]));

      cuComplex n1, n2;

      if(varenna && ky[idy]==0.) {
        n1 = mu1;
        n2 = mu2;
        abs_omegad = 0.;
      } else {
        n1 = nu1;
        n2 = nu2;
      }

      if(higher_order_moments && ky[idy]==0. ) {
        tpar_field[index] = iomegastar*phi_tpar - iomegad*(phi_tpard + rparpar[index] + rparprp[index]
			      - 2.*dens[index] - tpar[index] - tprp[index])
                            + 2.*vt*(qprp[index] + upar[index])*bgrad[idz] 
                            + (2.*nu_ss/3.)*(tpar[index] - tprp[index]);
      } else {
        tpar_field[index] = iomegastar*phi_tpar - iomegad*(phi_tpard + (6.+2.*n1.y)*tpar[index] + 2.*dens[index] + 2.*n2.y*tprp[index])
                        + 2.*vt*(qprp[index] + upar[index])*bgrad[idz] + abs_omegad*(2.*n1.x*tpar[index] + 2.*n2.x*tprp[index])
                        + (2.*nu_ss/3.)*(tpar[index] - tprp[index]);
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

        unsigned int IDZ = idz + zthreads*i;

        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;

        double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
  
        cuComplex phi_tpar = phi[index] * sgam0(bidx) * tprim;

        cuComplex iomegastar = iOmegaStar(ky[idy]);

        cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]);

        cuComplex phi_tpard = phi[index] * ( zt * 2. * sgam0(bidx) );
 
        float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]));

        cuComplex n1, n2;

        if(varenna && ky[idy]==0.) {
          n1 = mu1;
          n2 = mu2;
          abs_omegad = 0.;
        } else {
          n1 = nu1;
          n2 = nu2;
        }

        tpar_field[index] = iomegastar*phi_tpar - iomegad*(phi_tpard + (6.+2.*n1.y)*tpar[index] + 2.*dens[index] + 2.*n2.y*tprp[index])
                          + 2.*vt*(qprp[index] + upar[index])*bgrad[IDZ] + abs_omegad*(2.*n1.x*tpar[index] + 2.*n2.x*tprp[index])
                          + (2.*nu_ss/3.)*(tpar[index] - tprp[index]);

      }
    }
  }
}

__global__ void tprp_linear_terms(cuComplex* tprp_field, cuComplex* phi, cuComplex* dens, cuComplex* upar, cuComplex* tpar, cuComplex* tprp,
                      float* kx, float* ky, float shat, float rho, float vt, float tprim, float fprim, float zt, float* bgrad,
                      float *gds2, float *gds21, float *gds22, float *bmagInv,
                      float* gb,float* gb0,float* cv, float* cv0,
                      float nu_ss, cuComplex nu3, cuComplex nu4, cuComplex mu3, cuComplex mu4, bool varenna,
		      cuComplex* rparprp, cuComplex* rprpprp, bool higher_order_moments)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      cuComplex phi_tperp = phi[index] * ( tprim*( sgam0(bidx) + flr2(bidx) ) + fprim*flr(bidx) );

      cuComplex iomegastar = iOmegaStar(ky[idy]);

      cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]);

      cuComplex phi_tperpd = phi[index] * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) );
 
      float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]));

      cuComplex n3, n4;

      if(varenna && ky[idy]==0.) {
        n3 = mu3;
        n4 = mu4;
        abs_omegad = 0.;
      } else {
        n3 = nu3;
        n4 = nu4;
      }

      if(higher_order_moments && ky[idy]==0. ) {
        tprp_field[index] = iomegastar*phi_tperp - iomegad*(phi_tperpd + rparprp[index] + rprpprp[index] 
			      - 2.*dens[index] - tpar[index] - tprp[index])
                            - vt*upar[index]*bgrad[idz] 
                            + (nu_ss/3.)*(tprp[index] - tpar[index]);
      } else {
        tprp_field[index] = iomegastar*phi_tperp - iomegad*(phi_tperpd + (4.+2.*n4.y)*tprp[index] + dens[index] + 2.*n3.y*tpar[index])
                        - vt*upar[index]*bgrad[idz] + abs_omegad*(2.*n3.x*tpar[index] + 2.*n4.x*tprp[index])
                        + (nu_ss/3.)*(tprp[index] - tpar[index]);
      }

    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

        unsigned int IDZ = idz + zthreads*i;

        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;

        double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
  
        cuComplex phi_tperp = phi[index] * ( tprim*( sgam0(bidx) + flr2(bidx) ) + fprim*flr(bidx) );

        cuComplex iomegastar = iOmegaStar(ky[idy]);

        cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]);

        cuComplex phi_tperpd = phi[index] * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) );
 
        float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]));

        cuComplex n3, n4;

        if(varenna && ky[idy]==0.) {
          n3 = mu3;
          n4 = mu4;
          abs_omegad = 0.;
        } else {
          n3 = nu3;
          n4 = nu4;
        }

        tprp_field[index] = iomegastar*phi_tperp - iomegad*(phi_tperpd + (4.+2.*n4.y)*tprp[index] + dens[index] + 2.*n3.y*tpar[index])
                          - vt*upar[index]*bgrad[IDZ] + abs_omegad*(2.*n3.x*tpar[index] + 2.*n4.x*tprp[index])
                          + (nu_ss/3.)*(tprp[index] - tpar[index]);
      }
    }
  }
}

__global__ void qpar_linear_terms(cuComplex* qpar_field, cuComplex* apar, cuComplex* dens, cuComplex* upar, cuComplex* tpar, 
		      cuComplex* tprp, cuComplex* qpar, cuComplex* qprp,
                      float* kx, float* ky, float shat, float rho, float vt, float tprim, float fprim, float zt, float* bgrad, float Beta_par,
                      float *gds2, float *gds21, float *gds22, float *bmagInv,
                      float* gb,float* gb0,float* cv, float* cv0,
                      float nu_ss, cuComplex nu5, cuComplex nu6, cuComplex nu7, cuComplex mu5, cuComplex mu6, cuComplex mu7, bool varenna,
		      cuComplex* r_terms, cuComplex* sparpar, cuComplex* sparprp, bool higher_order_moments, float beta)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]);

      float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]));
     
      cuComplex n5, n6, n7;

      float fac = 0.;

      if(varenna && ky[idy]==0.) {
        n5 = mu5;
        n6 = mu6;
        n7 = mu7;
        abs_omegad = 0.;
        fac = 1.;
      } else {
        n5 = nu5;
        n6 = nu6;
        n7 = nu7;
      }

      if(higher_order_moments && ky[idy]==0. ) {
        qpar_field[index] = vt*( r_terms[index] + 3.*tpar[index] - 3.*tprp[index])*bgrad[idz] 
		        - iomegad*((sparpar[index]-15.*upar[index]) + (sparprp[index]-3.*upar[index])
                           - 3.*qpar[index] - 3.*qprp[index] + 6.*upar[index])
			+ nu_ss * qpar[index];
      } else {
        qpar_field[index] = //vt*(Beta_par*tpar[index])*bgrad[idz] 
                        //+ vt*fac*(tpar[index]-tprp[index])*bgrad[idz] 
		        //- iomegad*( (-3.+n6.y)*qpar[index] + (-3.+n7.y)*qprp[index] + (6.+n5.y)*upar[index] )
		        - iomegad*( (8.+n6.y)*qpar[index] + (n7.y)*qprp[index] + (6.+n5.y)*upar[index] )
	                + abs_omegad*( n5.x*upar[index] + n6.x*qpar[index] + n7.x*qprp[index] )
			+ nu_ss * qpar[index];
      }
      
      if(beta>0.) {
        cuComplex iomegastar = iOmegaStar(ky[idy]);
        cuComplex apar_term = apar[index] * ( 3.*tprim*sgam0(bidx) );  
     
        qpar_field[index] = qpar_field[index] - vt*iomegastar*apar_term;
      }


    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

        unsigned int IDZ = idz + zthreads*i;

        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;

        double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);

        cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]);
  
        float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]));
       
        cuComplex n5, n6, n7;
  
        if(varenna && ky[idy]==0.) {
          n5 = mu5;
          n6 = mu6;
          n7 = mu7;
          abs_omegad = 0.;
        } else {
          n5 = nu5;
          n6 = nu6;
          n7 = nu7;
        }
  
        qpar_field[index] = //-iomegad*( (-3.+n6.y)*qpar[index] + (-3.+n7.y)*qprp[index] + (6.+n5.y)*upar[index] )
		          - iomegad*( (8.+n6.y)*qpar[index] + (n7.y)*qprp[index] + (6.+n5.y)*upar[index] )
  	                  + abs_omegad*( n5.x*upar[index] + n6.x*qpar[index] + n7.x*qprp[index] )
  		          + nu_ss * qpar[index];
        if(beta>0.) {
          cuComplex iomegastar = iOmegaStar(ky[idy]);
          cuComplex apar_term = apar[index] * ( 3.*tprim*sgam0(bidx) );  
     
          qpar_field[index] = qpar_field[index] - vt*iomegastar*apar_term;
        }

      }
    }
  }
}

__global__ void qprp_linear_terms(cuComplex* qprp_field, cuComplex* phi, cuComplex* apar, cuComplex* aparNew,
                      cuComplex* dens, cuComplex* upar, cuComplex* tpar, 
                      cuComplex* tprp, cuComplex* qpar, cuComplex* qprp,
                      float* kx, float* ky, float shat, float rho, float vt, float tprim, float fprim, float zt, float* bgrad,
                      float *gds2, float *gds21, float *gds22, float *bmagInv,
                      float* gb,float* gb0,float* cv, float* cv0,
                      float nu_ss, cuComplex nu8, cuComplex nu9, cuComplex nu10, cuComplex mu8, cuComplex mu9, cuComplex mu10, bool varenna,
		      cuComplex* r_terms, cuComplex* sparprp, cuComplex* sprpprp, bool higher_order_moments, float beta, float dt)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      cuComplex phi_qperpb = phi[index] * (flr2(bidx) - flr(bidx));

      cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]);

      float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]));

      cuComplex n8, n9, n10;
 
      float fac = 1.;

      if(varenna && ky[idy]==0.) {
        n8 = mu8;
        n9 = mu9;
        n10 = mu10;
        abs_omegad = 0.;
        //fac = 0.;
      } else {
        n8 = nu8;
        n9 = nu9;
        n10 = nu10;
      }

      if(higher_order_moments && ky[idy]==0. ) {
        qprp_field[index] = vt*( zt*phi_qperpb + r_terms[index] + tpar[index] - tprp[index] )*bgrad[idz] 
                        - iomegad*( (sparprp[index]-3.*upar[index]) + (sprpprp[index]-2.*upar[index])
			  - qpar[index] - qprp[index] + upar[index] )
                        + nu_ss*qprp[index];
      } else { 
        qprp_field[index] = vt*( zt*phi_qperpb + tprp[index] - fac*tpar[index] )*bgrad[idz] 
                        //- iomegad*( (-1.+n9.y)*qpar[index] + (-1.+n10.y)*qprp[index] + (1.+n8.y)*upar[index] )
                        - iomegad*( (n9.y)*qpar[index] + (6.+n10.y)*qprp[index] + (1.+n8.y)*upar[index] )
                        + abs_omegad*( n8.x*upar[index] + n9.x*qpar[index] + n10.x*qprp[index]  )
                        + nu_ss*qprp[index];
      }

      if(beta>0.) {
        cuComplex iomegastar = iOmegaStar(ky[idy]);
        cuComplex apar_term1 = apar[index] * ( tprim*(sgam0(bidx)+flr2(bidx)) + (fprim + tprim)*flr(bidx) );  
        cuComplex apar_term2 = apar[index] * ( flr(bidx) );
        cuComplex dApardt_flr = flr(bidx) * ( aparNew[index] - apar[index] ) / dt;    

        qprp_field[index] = qprp_field[index] - vt*iomegastar*apar_term1 + vt*zt*iomegad*apar_term2 + vt*zt*dApardt_flr;
        // double check sign of iomegad term. - is consistent with gryffin code. + seems to be consistent with snyder thesis. 
        // gryffin also leaves off the vt*zt factors
      }

    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

        unsigned int IDZ = idz + zthreads*i;

        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;

        double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
  
        cuComplex phi_qperpb = phi[index] * (flr2(bidx) - flr(bidx));

        cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]);

        float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]));

        cuComplex n8, n9, n10;

        if(varenna && ky[idy]==0.) {
          n8 = mu8;
          n9 = mu9;
          n10 = mu10;
          abs_omegad = 0.;
        } else {
          n8 = nu8;
          n9 = nu9;
          n10 = nu10;
        }

        qprp_field[index] = vt*( zt*phi_qperpb + tprp[index] - tpar[index] )*bgrad[IDZ] 
                          - iomegad*( (-1.+n9.y)*qpar[index] + (-1.+n10.y)*qprp[index] + (1.+n8.y)*upar[index] )
                          + abs_omegad*( n8.x*upar[index] + n9.x*qpar[index] + n10.x*qprp[index]  )
                          + nu_ss*qprp[index];
      }
    }
  }
}

__global__ void electron_density_linear_terms(cuComplex* dens_field, cuComplex* phi, cuComplex* dens, cuComplex* tpar, cuComplex* tprp,
                      float* kx, float* ky, float shat, float* gb,float* gb0,float* cv, float* cv0, float ti_ov_te, float fprim)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      cuComplex iomegastar_e = iOmegaStar(ky[idy]);
      cuComplex iomegad_e = -1.*iOmegaD(kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]);

      dens_field[index] = fprim*iomegastar_e*phi[index] - iomegad_e*(-2.*phi[index] + 2.*dens[index]/ti_ov_te + tpar[index] + tprp[index]); 

    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

        unsigned int IDZ = idz + zthreads*i;

        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;

        cuComplex iomegastar_e = iOmegaStar(ky[idy]);
        cuComplex iomegad_e = -1.*iOmegaD(kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]);

        dens_field[index] = fprim*iomegastar_e*phi[index] - iomegad_e*(-2.*phi[index] + 2.*dens[index]/ti_ov_te + tpar[index] + tprp[index]); 

      }
    }
  }
}

__global__ void electron_momentum_linear_terms(cuComplex* apar_field, cuComplex* apar, cuComplex* tpar, cuComplex* tprp,
                      float* kx, float* ky, float shat, float fprim, float* bgrad,
                      float *gds2, float *gds21, float *gds22, float *bmagInv,
                      float* gb,float* gb0,float* cv, float* cv0, float ti_ov_te, float beta_e, float nu_ei)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      cuComplex iomegastar_e = iOmegaStar(ky[idy])/ti_ov_te;

      double bidx = b(1., kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]); // just kperp^2, not (kperp rho)^2

      apar_field[index] = iomegastar_e*(fprim)*apar[index] - nu_ei*(-2.*bidx)*apar[index]/(ti_ov_te * beta_e);// + (tpar[index] - tprp[index])*bgrad[idz];
      //NOTE: when using Snyder's Te closure, tpar = tprp, so mirror force term drops out

    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

        unsigned int IDZ = idz + zthreads*i;

        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;

        cuComplex iomegastar_e = iOmegaStar(ky[idy])/ti_ov_te;
        
        double bidx = b(1., kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]); // just kperp^2, not (kperp rho)^2

        apar_field[index] = (tpar[index] - tprp[index])*bgrad[idz] + iomegastar_e*(fprim)*apar[index] - nu_ei*(-2.*bidx)*apar[index]/(ti_ov_te * beta_e);

      }
    }
  }
}

__global__ void apar_semi_implicit_gradpar_term(cuComplex* result, cuComplex* phi, cuComplex* dens, cuComplex* densNew, 
                      float* kx, float* ky, float shat, 
                      float *gds2, float *gds21, float *gds22, float *bmagInv,
                      float ti_ov_te, specie* s)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      float pfilter2 = 0.;    
      double bidx;
      
//      for(int i=0; i<nspecies-1; i++) { // electron contribution to this sum is negligible.
//        bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
//        pfilter2 = pfilter2 + s[i].dens*s[i].z*s[i].zt*( 1. - g0(bidx) );
//      }
//
//      if(pfilter2!=0.) result[index] = -(1./pfilter2 + 1./ti_ov_te)*(dens[index] + densNew[index])/2. + dens[index]/pfilter2 + phi[index];
//      else { result[index].x = 0.; result[index].y = 0.; }

      result[index] = -1./ti_ov_te*(dens[index]+densNew[index])/2. + phi[index];

    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

        unsigned int IDZ = idz + zthreads*i;

        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;

      float pfilter2 = 0.;    
      double bidx;
      
      for(int i=0; i<nspecies-1; i++) {
        bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
        pfilter2 = pfilter2 + s[i].dens*s[i].z*s[i].zt*( 1. - g0(bidx) );
      }

      result[index] = -(1./pfilter2 + 1./ti_ov_te)*(dens[index] + densNew[index])/2. + dens[index]/pfilter2 + phi[index];

      }
    }
  }
}

__global__ void electron_temperature_closure(cuComplex* result, cuComplex* apar, float* ky, float tprim, float ti_ov_te)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

//      cuComplex iomegastar_e = iOmegaStar(ky[idy])/ti_ov_te;
//
//      result[index] = tprim*iomegastar_e*apar[index];

      float omegastar_e = ky[idy]/ti_ov_te;

      result[index] = tprim*omegastar_e*apar[index];
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

        unsigned int IDZ = idz + zthreads*i;

        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;

        cuComplex iomegastar_e = iOmegaStar(ky[idy])/ti_ov_te;

        result[index] = tprim*iomegastar_e*apar[index];

      }
    }
  }
}
//__global__ void electron_tpar_linear_terms(cuComplex* tpar_field, cuComplex* phi, cuComplex* dens, cuComplex* upar, 
//                      cuComplex* tpar, cuComplex* tprp, cuComplex* qprp,
//                      float* kx, float* ky, float shat, float rho, float vt, float tprim, float fprim, float zt, float* bgrad,
//                      float *gds2, float *gds21, float *gds22, float *bmagInv,
//                      float* gb,float* gb0,float* cv, float* cv0,
//                      float nu_ss, cuComplex n1, cuComplex n2, cuComplex mu1, cuComplex mu2, bool varenna,
//                      cuComplex* rparpar, cuComplex* rparprp, bool higher_order_moments)
//{
//  unsigned int idy = get_idy();
//  unsigned int idx = get_idx();
//  unsigned int idz = get_idz();
//
//    if(idy<(ny/2+1) && idx<nx && idz<nz) {
//
//      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
//
//      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
//
//      cuComplex phi_tpar = phi[index] * sgam0(bidx) * tprim;
//
//      cuComplex iomegastar_e = iOmegaStar(ky[idy])/ti_ov_te;
//
//      cuComplex iomegad_e = (-1./ti_ov_te)*iOmegaD(kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]);
//
//      cuComplex phi_tpard = phi[index] * ( zt * 2. * sgam0(bidx) );
// 
//      float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]));
//
//        tpar_field[index] = iomegastar_e*tprim*phi[index] 
//			- iomegad_e*( -2.*phi[index] + (6.+2.*n1.y)*tpar[index] + 2.*dens[index]/ti_ov_te + 2.*n2.y*tprp[index])
//                        + 2.*vt*(qprp[index] + upar[index])*bgrad[idz] + abs_omegad*(2.*n1.x*tpar[index] + 2.*n2.x*tprp[index])
//                        + (2.*nu_ss/3.)*(tpar[index] - tprp[index]);
//    }
//}
//
//__global__ void electron_tprp_linear_terms(cuComplex* tprp_field, cuComplex* phi, cuComplex* dens, cuComplex* upar, cuComplex* tpar, cuComplex* tprp,
//                      float* kx, float* ky, float shat, float rho, float vt, float tprim, float fprim, float zt, float* bgrad,
//                      float *gds2, float *gds21, float *gds22, float *bmagInv,
//                      float* gb,float* gb0,float* cv, float* cv0,
//                      float nu_ss, cuComplex nu3, cuComplex nu4, cuComplex mu3, cuComplex mu4, bool varenna,
//		      cuComplex* rparprp, cuComplex* rprpprp, bool higher_order_moments)
//{
//  unsigned int idy = get_idy();
//  unsigned int idx = get_idx();
//  unsigned int idz = get_idz();
//
//  if(nz<=zthreads) {
//    if(idy<(ny/2+1) && idx<nx && idz<nz) {
//
//      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
//
//      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
//
//      cuComplex phi_tperp = phi[index] * ( tprim*( sgam0(bidx) + flr2(bidx) ) + fprim*flr(bidx) );
//
//      cuComplex iomegastar = iOmegaStar(ky[idy]);
//
//      cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]);
//
//      cuComplex phi_tperpd = phi[index] * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) );
// 
//      float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]));
//
//      cuComplex n3, n4;
//
//      if(varenna && ky[idy]==0.) {
//        n3 = mu3;
//        n4 = mu4;
//        abs_omegad = 0.;
//      } else {
//        n3 = nu3;
//        n4 = nu4;
//      }
//
//      if(higher_order_moments && ky[idy]==0. ) {
//        tprp_field[index] = iomegastar*phi_tperp - iomegad*(phi_tperpd + rparprp[index] + rprpprp[index] 
//			      - 2.*dens[index] - tpar[index] - tprp[index])
//                            - vt*upar[index]*bgrad[idz] 
//                            + (nu_ss/3.)*(tprp[index] - tpar[index]);
//      } else {
//        tprp_field[index] = iomegastar*phi_tperp - iomegad*(phi_tperpd + (4.+2.*n4.y)*tprp[index] + dens[index] + 2.*n3.y*tpar[index])
//                        - vt*upar[index]*bgrad[idz] + abs_omegad*(2.*n3.x*tpar[index] + 2.*n4.x*tprp[index])
//                        + (nu_ss/3.)*(tprp[index] - tpar[index]);
//      }
//
//    }
//  }
//  else {
//    for(int i=0; i<nz/zthreads; i++) {
//      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
//
//        unsigned int IDZ = idz + zthreads*i;
//
//        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
//
//        double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
//  
//        cuComplex phi_tperp = phi[index] * ( tprim*( sgam0(bidx) + flr2(bidx) ) + fprim*flr(bidx) );
//
//        cuComplex iomegastar = iOmegaStar(ky[idy]);
//
//        cuComplex iomegad = rho*vt*iOmegaD(kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]);
//
//        cuComplex phi_tperpd = phi[index] * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) );
// 
//        float abs_omegad = abs(omegaD(rho*vt,kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]));
//
//        cuComplex n3, n4;
//
//        if(varenna && ky[idy]==0.) {
//          n3 = mu3;
//          n4 = mu4;
//          abs_omegad = 0.;
//        } else {
//          n3 = nu3;
//          n4 = nu4;
//        }
//
//        tprp_field[index] = iomegastar*phi_tperp - iomegad*(phi_tperpd + (4.+2.*n4.y)*tprp[index] + dens[index] + 2.*n3.y*tpar[index])
//                          - vt*upar[index]*bgrad[IDZ] + abs_omegad*(2.*n3.x*tpar[index] + 2.*n4.x*tprp[index])
//                          + (nu_ss/3.)*(tprp[index] - tpar[index]);
//      }
//    }
//  }
//}

__global__ void implicit_step_par(float* fNew, float* f, float par_fac, float* nu, float dt) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
     
      fNew[index] = f[index] / (1. + dt*par_fac*nu[index]);
    }
  }

}

__global__ void implicit_step_prp(float* fNew, float* f_impl, float prp_fac_expl, float* f_expl, float prp_fac_impl, float* nu, float dt) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {

      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
     
      fNew[index] = (f_impl[index]-dt*prp_fac_expl*nu[index]*f_expl[index]) / (1. + dt*prp_fac_impl*nu[index]);
    }
  }

}
