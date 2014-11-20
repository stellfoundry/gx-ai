//inline void shear0(float* nu, float* nu_tmpXZ, float* Phi2ZF, specie s) {
//  nlpm_shear0<<<dimGrid,dimBlock>>>(nu_tmpXZ, Phi2ZF, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv, nlpm_zonal_kx1_only);
//  sumReduc_Partial(nu, nu_tmpXZ, Nx*Nz, Nz, false);
//}

inline void shear0(float* nu, float* nu_tmpZ, float* nu_tmpXZ, float* Phi2ZF, specie s) {
  nlpm_shear0<<<dimGrid,dimBlock>>>(nu_tmpXZ, Phi2ZF, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv, nlpm_zonal_kx1_only);
  sumReduc_Partial(nu, nu_tmpXZ, Nx*Nz, Nz, false);
}

//for when dorland_phase is complex
inline void shear0(cuComplex* nu, cuComplex* nu_CtmpXZ, cuComplex* PhiZF_CtmpX, specie s) {
  nlpm_shear0<<<dimGrid,dimBlock>>>(nu_CtmpXZ, PhiZF_CtmpX, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv, nlpm_zonal_kx1_only);
  sumReduc_Partial_complex(nu, nu_CtmpXZ, Nx*Nz, Nz, false); 
}

inline void shear0_ifac(cuComplex* nu, cuComplex* nu_CtmpXZ, cuComplex* PhiZF_CtmpX, specie s) {
  nlpm_shear0_ifac<<<dimGrid,dimBlock>>>(nu_CtmpXZ, PhiZF_CtmpX, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv, nlpm_zonal_kx1_only);
  sumReduc_Partial_complex(nu, nu_CtmpXZ, Nx*Nz, Nz, false); 
}

//inline void shear1(float* nu, float* nu_tmpXZ, float* Phi2ZF, specie s) {  
//  nlpm_shear1<<<dimGrid,dimBlock>>>(nu_tmpXZ, Phi2ZF, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv, nlpm_zonal_kx1_only);
//  sumReduc_Partial(nu, nu_tmpXZ, Nx*Nz, Nz, false); 
//}

inline void shear1(float* nu, float* nu_tmpZ, float* nu_tmpXZ, float* Phi2ZF, specie s) {  
  nlpm_shear1<<<dimGrid,dimBlock>>>(nu_tmpXZ, Phi2ZF, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv, nlpm_zonal_kx1_only);
  sumReduc_Partial(nu, nu_tmpXZ, Nx*Nz, Nz, false); 
}

//inline void shear2(float* nu, float* nu_tmpXZ, float* Phi2ZF, specie s) {
//  float val;
//  nlpm_shear2<<<dimGrid,dimBlock>>>(nu_tmpXZ, Phi2ZF, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv, nlpm_zonal_kx1_only);
//  //val = maxReduc(nu_tmpXZ, Nx*Nz, false);
//  //printf("In shear2: max nu(x,z) = %e\n", val);
//  sumReduc_Partial(nu, nu_tmpXZ, Nx*Nz, Nz, false); 
//  //val = maxReduc(nu, Nz, false);
//  //printf("In shear2: after sumReduc_Partial, max nu(z) = %e\n", val);
//  sqrtZ<<<dimGrid,dimBlock>>>(nu, nu);  
//  //val = maxReduc(nu, Nz, false);
//  //printf("In shear2: after sqrtZ, max nu(z) = %e\n", val);
//}  

inline void shear2(float* nu, float* nu_tmpZ, float* nu_tmpXZ, float* Phi2ZF, specie s) {
  //float val;
  nlpm_shear2<<<dimGrid,dimBlock>>>(nu_tmpXZ, Phi2ZF, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv, nlpm_zonal_kx1_only);
  //val = maxReduc(nu_tmpXZ, Nx*Nz, false);
  //printf("In shear2: max nu(x,z) = %e\n", val);
  sumReduc_Partial(nu_tmpZ, nu_tmpXZ, Nx*Nz, Nz, false); 
  //val = maxReduc(nu_tmpZ, Nz, false);
  //printf("In shear2: after sumReduc_Partial, max nu(z) = %e\n", val);
  sqrtZ<<<dimGrid,dimBlock>>>(nu, nu_tmpZ);  
  //val = maxReduc(nu, Nz, false);
  //printf("In shear2: after sqrtZ, max nu(z) = %e\n", val);
}  

typedef void (*nlpm_switch)(float*, float*, float*, float*, specie);
nlpm_switch shear[] = {shear0, shear1, shear2};
    
inline void get_nu_nlpm(float* nu_nlpm, cuComplex* Phi, float* Phi2ZF_tmpX, float* nu_nlpm_tmpXZ, float* tmpZ, specie s)
{
  
  // get zonal flow component of Phi
  volflux_zonal<<<dimGrid,dimBlock>>>(Phi2ZF_tmpX, Phi, Phi, jacobian, 1./(fluxDen*fluxDen) );
  
  shear[inlpm](nu_nlpm, tmpZ, nu_nlpm_tmpXZ, Phi2ZF_tmpX, s);  

}

inline void get_dorland_nu_nlpm(float* nu_abs_nlpm, float* nu1_nlpm, float* nu22_nlpm, cuComplex* Phi, float* Phi2ZF_tmpX, float* nu_nlpm_tmpXZ, specie s)
{

  // get zonal flow component of Phi
  volflux_zonal<<<dimGrid,dimBlock>>>(Phi2ZF_tmpX, Phi, Phi, jacobian, 1./(fluxDen*fluxDen) );

  shear[inlpm](nu_abs_nlpm, nu1_nlpm, nu_nlpm_tmpXZ, Phi2ZF_tmpX, s); //dissipative term is the same
 // shear2_tmp(nu_abs_nlpm, nu1_nlpm, nu_nlpm_tmpXZ, Phi2ZF_tmpX, s); //dissipative term is the same
  

  if(dorland_nlpm_phase) shear[0](nu22_nlpm, nu1_nlpm, nu_nlpm_tmpXZ, Phi2ZF_tmpX, s);

  if(dorland_nlpm_phase && dorland_phase_ifac!=0) {
    add_scaled<<<dimGrid,dimBlock>>>(nu1_nlpm, .4, nu_abs_nlpm, dorland_phase_ifac*-.6, nu22_nlpm, 1, 1, Nz);
    add_scaled<<<dimGrid,dimBlock>>>(nu22_nlpm, 1.6, nu_abs_nlpm, dorland_phase_ifac*-1.3, nu22_nlpm, 1, 1, Nz);
  }
  else if(dorland_nlpm_phase && dorland_phase_ifac==0) {
    add_scaled<<<dimGrid,dimBlock>>>(nu1_nlpm, .4, nu_abs_nlpm, -.6, nu22_nlpm, 1, 1, Nz);
    add_scaled<<<dimGrid,dimBlock>>>(nu22_nlpm, 1.6, nu_abs_nlpm, -1.3, nu22_nlpm, 1, 1, Nz);
  }
  else {
    zero<<<dimGrid,dimBlock>>>(nu22_nlpm, 1, 1, Nz);
    add_scaled<<<dimGrid,dimBlock>>>(nu1_nlpm, .4, nu_abs_nlpm, 0., nu22_nlpm, 1, 1, Nz);
    add_scaled<<<dimGrid,dimBlock>>>(nu22_nlpm, 1.6, nu_abs_nlpm, 0., nu22_nlpm, 1, 1, Nz);
  }

}

//for when dorland_phase is complex
inline void get_dorland_nu_nlpm(float* nu_abs_nlpm, cuComplex* nu1_nlpm, cuComplex* nu22_nlpm, float* tmpZ, cuComplex* Phi, float* Phi2ZF_tmpX, cuComplex* PhiZF_CtmpX, float* nu_nlpm_tmpXZ, cuComplex* nu_nlpm_CtmpXZ, specie s)
{
  
  // get zonal flow component of Phi
  volflux_zonal<<<dimGrid,dimBlock>>>(Phi2ZF_tmpX, Phi, Phi, jacobian, 1./(fluxDen*fluxDen) );
  volflux_zonal_complex<<<dimGrid,dimBlock>>>(PhiZF_CtmpX, Phi, jacobian, 1./fluxDen);
  
  shear[inlpm](nu_abs_nlpm, tmpZ, nu_nlpm_tmpXZ, Phi2ZF_tmpX, s); //dissipative term is the same as non-dorland nlpm
   
  if(dorland_nlpm_phase && dorland_phase_ifac!=0) shear0(nu22_nlpm, nu_nlpm_CtmpXZ, PhiZF_CtmpX, s);
  else if(dorland_nlpm_phase && dorland_phase_ifac==0) shear0_ifac(nu22_nlpm, nu_nlpm_CtmpXZ, PhiZF_CtmpX, s);

  if(dorland_nlpm_phase && dorland_phase_ifac!=0) {
    add_scaled<<<dimGrid,dimBlock>>>(nu1_nlpm, .4, nu_abs_nlpm, dorland_phase_ifac*-.6, nu22_nlpm, 1, 1, Nz);
    add_scaled<<<dimGrid,dimBlock>>>(nu22_nlpm, 1.6, nu_abs_nlpm, dorland_phase_ifac*-1.3, nu22_nlpm, 1, 1, Nz);
  }
  else if(dorland_nlpm_phase && dorland_phase_ifac==0) {
    add_scaled<<<dimGrid,dimBlock>>>(nu1_nlpm, .4, nu_abs_nlpm, -.6, nu22_nlpm, 1, 1, Nz);
    add_scaled<<<dimGrid,dimBlock>>>(nu22_nlpm, 1.6, nu_abs_nlpm, -1.3, nu22_nlpm, 1, 1, Nz);
  }
  else {
    zeroC<<<dimGrid,dimBlock>>>(nu22_nlpm, 1, 1, Nz);
    add_scaled<<<dimGrid,dimBlock>>>(nu1_nlpm, .4, nu_abs_nlpm, 0., nu22_nlpm, 1, 1, Nz);
    add_scaled<<<dimGrid,dimBlock>>>(nu22_nlpm, 1.6, nu_abs_nlpm, 0., nu22_nlpm, 1, 1, Nz);
  }

}

inline void filterNLPM(cuComplex* Phi, cuComplex* Dens, cuComplex* Upar, cuComplex* Tpar,
                cuComplex* Tprp, cuComplex* Qpar, cuComplex* Qprp,
                float* Phi2ZF_tmpX, float* tmpXZ, float* filter_tmpYZ, float* nu_nlpm, float* nu1_nlpm, float* nu22_nlpm,
                specie s, float dt_loc, float* Dnlpm_d, float Phi_zf_kx1, float Phi_zf_rms, cuComplex* tmp)
{
  float nu1max, nu22max;
  if(dorland_nlpm) {
    if(strcmp(nlpm_option,"constant") == 0 || (strcmp(nlpm_option,"cutoff") == 0 && Phi_zf_rms>low_cutoff)) {
      get_dorland_nu_nlpm(nu_nlpm, nu1_nlpm, nu22_nlpm, Phi, Phi2ZF_tmpX, tmpXZ, s);
      //nu1max = maxReduc(nu1_nlpm, Nz, false);
      //nu22max = maxReduc(nu22_nlpm, Nz, false);
      //printf("nu1max = %e, nu22max = %e\n", nu1max, nu22max);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tpar, nu1_nlpm, ky, dt_loc, dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tprp, nu22_nlpm, ky, dt_loc, dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qpar, nu1_nlpm, ky, dt_loc, dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qprp, nu22_nlpm, ky, dt_loc, dnlpm, kxfac);
    }
    if(strcmp(nlpm_option,"cutoff") == 0 && Phi_zf_rms<low_cutoff) {
      get_dorland_nu_nlpm(nu_nlpm, nu1_nlpm, nu22_nlpm, Phi, Phi2ZF_tmpX, tmpXZ, s);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tpar, nu1_nlpm, ky, dt_loc, dnlpm*Phi_zf_rms/low_cutoff, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tprp, nu22_nlpm, ky, dt_loc, dnlpm*Phi_zf_rms/low_cutoff, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qpar, nu1_nlpm, ky, dt_loc, dnlpm*Phi_zf_rms/low_cutoff, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qprp, nu22_nlpm, ky, dt_loc, dnlpm*Phi_zf_rms/low_cutoff, kxfac);
    }
    if(strcmp(nlpm_option,"quadratic") == 0) {
      get_dorland_nu_nlpm(nu_nlpm, nu1_nlpm, nu22_nlpm, Phi, Phi2ZF_tmpX, tmpXZ, s);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tpar, nu1_nlpm, ky, dt_loc, Phi_zf_rms*dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tprp, nu22_nlpm, ky, dt_loc, Phi_zf_rms*dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qpar, nu1_nlpm, ky, dt_loc, Phi_zf_rms*dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qprp, nu22_nlpm, ky, dt_loc, Phi_zf_rms*dnlpm, kxfac);
    }
  }
  else {
    get_nu_nlpm(nu_nlpm, Phi, Phi2ZF_tmpX, tmpXZ, nu1_nlpm, s);
    if(strcmp(nlpm_option,"cutoff") == 0) {
      get_Dnlpm<<<1,1>>>(Dnlpm_d, Phi_zf_kx1, low_cutoff, high_cutoff, s.nu_ss, dnlpm_max);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tpar, nu_nlpm, ky, dt_loc, Dnlpm_d, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tprp, nu_nlpm, ky, dt_loc, Dnlpm_d, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qpar, nu_nlpm, ky, dt_loc, Dnlpm_d, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qprp, nu_nlpm, ky, dt_loc, Dnlpm_d, kxfac);
    }
    if(strcmp(nlpm_option,"constant") == 0) {
      nlpm_filter<<<dimGrid,dimBlock>>>(Tpar, nu_nlpm, ky, dt_loc, dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tprp, nu_nlpm, ky, dt_loc, dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qpar, nu_nlpm, ky, dt_loc, dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qprp, nu_nlpm, ky, dt_loc, dnlpm, kxfac);
    }
    if(strcmp(nlpm_option,"quadratic") == 0) {
      get_Dnlpm_quadratic<<<1,1>>>(Dnlpm_d, Phi_zf_kx1);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tpar, nu_nlpm, ky, dt_loc, Dnlpm_d, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tprp, nu_nlpm, ky, dt_loc, Dnlpm_d, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qpar, nu_nlpm, ky, dt_loc, Dnlpm_d, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qprp, nu_nlpm, ky, dt_loc, Dnlpm_d, kxfac);
    }
  }
  //cudaMemcpy(&Dnlpm_h, Dnlpm_d, sizeof(float), cudaMemcpyDeviceToHost);
  //printf("Dnlpm=%f\n", Dnlpm_h);      
  //nlpm_filter<<<dimGrid,dimBlock>>>(Dens, nu_nlpm, ky, dt_loc, dnlpm);        
  //nlpm_filter<<<dimGrid,dimBlock>>>(Upar, nu_nlpm, ky, dt_loc, dnlpm);        
}

//for when dorland_phase is complex
inline void filterNLPM(cuComplex* Phi, cuComplex* Dens, cuComplex* Upar, cuComplex* Tpar,
		cuComplex* Tprp, cuComplex* Qpar, cuComplex* Qprp, 
		float* Phi2ZF_tmpX, cuComplex* PhiZF_CtmpX, float* tmpXZ, cuComplex* CtmpXZ, float* filter_tmpYZ, float* nu_nlpm, cuComplex* nu1_nlpm, cuComplex* nu22_nlpm,
		float* tmpZ, specie s, float dt_loc, float* Dnlpm_d, float Phi_zf_kx1, float kx2Phi_zf_rms)
{
  if(!nlpm_kxdep) {
    if(strcmp(nlpm_option,"constant") == 0 || (strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms>low_cutoff)) {
      get_dorland_nu_nlpm(nu_nlpm, nu1_nlpm, nu22_nlpm, tmpZ, Phi, Phi2ZF_tmpX, PhiZF_CtmpX, tmpXZ, CtmpXZ, s);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tpar, nu1_nlpm, ky, dt_loc, dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tprp, nu22_nlpm, ky, dt_loc, dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qpar, nu1_nlpm, ky, dt_loc, dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qprp, nu22_nlpm, ky, dt_loc, dnlpm, kxfac);
    }
    if(strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms<low_cutoff) {
      get_dorland_nu_nlpm(nu_nlpm, nu1_nlpm, nu22_nlpm, tmpZ, Phi, Phi2ZF_tmpX, PhiZF_CtmpX, tmpXZ, CtmpXZ, s);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tpar, nu1_nlpm, ky, dt_loc, dnlpm*kx2Phi_zf_rms/low_cutoff, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tprp, nu22_nlpm, ky, dt_loc, dnlpm*kx2Phi_zf_rms/low_cutoff, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qpar, nu1_nlpm, ky, dt_loc, dnlpm*kx2Phi_zf_rms/low_cutoff, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qprp, nu22_nlpm, ky, dt_loc, dnlpm*kx2Phi_zf_rms/low_cutoff, kxfac);
    }
    if(strcmp(nlpm_option,"quadratic") == 0) {
      get_dorland_nu_nlpm(nu_nlpm, nu1_nlpm, nu22_nlpm, tmpZ, Phi, Phi2ZF_tmpX, PhiZF_CtmpX, tmpXZ, CtmpXZ, s);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tpar, nu1_nlpm, ky, dt_loc, kx2Phi_zf_rms*dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Tprp, nu22_nlpm, ky, dt_loc, kx2Phi_zf_rms*dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qpar, nu1_nlpm, ky, dt_loc, kx2Phi_zf_rms*dnlpm, kxfac);
      nlpm_filter<<<dimGrid,dimBlock>>>(Qprp, nu22_nlpm, ky, dt_loc, kx2Phi_zf_rms*dnlpm, kxfac);
    }
  }
  else {
    // get zonal flow component of Phi
    //Phi2ZF_tmpX = |Phi_zf(kx)|_rms
    volflux_zonal_rms<<<dimGrid,dimBlock>>>(Phi2ZF_tmpX, Phi, Phi, jacobian, 1./(fluxDen*fluxDen) );
    //complex Phi_zf(kx)
    volflux_zonal_complex<<<dimGrid,dimBlock>>>(PhiZF_CtmpX, Phi, jacobian, 1./fluxDen);
    
    nlpm_filter_kxdep<<<dimGrid,dimBlock>>>(Tpar, ky, kx, dt_loc, dnlpm, kxfac, .4, Phi2ZF_tmpX, .6, PhiZF_CtmpX, s.rho, shat, gds2, gds21, gds22, bmagInv);
    nlpm_filter_kxdep<<<dimGrid,dimBlock>>>(Tprp, ky, kx, dt_loc, dnlpm, kxfac, 1.6, Phi2ZF_tmpX, 1.3, PhiZF_CtmpX, s.rho, shat, gds2, gds21, gds22, bmagInv);
    nlpm_filter_kxdep<<<dimGrid,dimBlock>>>(Qpar, ky, kx, dt_loc, dnlpm, kxfac, .4, Phi2ZF_tmpX, .6, PhiZF_CtmpX, s.rho, shat, gds2, gds21, gds22, bmagInv);
    nlpm_filter_kxdep<<<dimGrid,dimBlock>>>(Qprp, ky, kx, dt_loc, dnlpm, kxfac, 1.6, Phi2ZF_tmpX, 1.3, PhiZF_CtmpX, s.rho, shat, gds2, gds21, gds22, bmagInv);
    
  }
}


