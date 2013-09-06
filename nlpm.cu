void shear1(float* nu, float* nu_tmpXZ, float* Phi2ZF, specie s) {  
  nlpm_shear1<<<dimGrid,dimBlock>>>(nu_tmpXZ, Phi2ZF, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv);
  sumReduc_Partial(nu, nu_tmpXZ, Nx*Nz, Nz, false); 
}

void shear2(float* nu, float* nu_tmpXZ, float* Phi2ZF, specie s) {
  nlpm_shear2<<<dimGrid,dimBlock>>>(nu_tmpXZ, Phi2ZF, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv);
  sumReduc_Partial(nu, nu_tmpXZ, Nx*Nz, Nz, false); 
  sqrtZ<<<dimGrid,dimBlock>>>(nu, nu);  
}  

typedef void (*nlpm_switch)(float*, float*, float*, specie);
nlpm_switch shear[] = {shear1, shear2};
    

void get_nu_nlpm(float* nu_nlpm, cuComplex* Phi, float* Phi2ZF_tmpX, float* nu_nlpm_tmpXZ, specie s)
{
  
  // get zonal flow component of Phi
  volflux_zonal<<<dimGrid,dimBlock>>>(Phi2ZF_tmpX, Phi, Phi, jacobian, 1./(fluxDen*fluxDen) );
    
  shear[inlpm-1](nu_nlpm, nu_nlpm_tmpXZ, Phi2ZF_tmpX, s);  
     
}

void filterNLPM(cuComplex* Phi, cuComplex* Dens, cuComplex* Upar, cuComplex* Tpar,
		cuComplex* Tprp, cuComplex* Qpar, cuComplex* Qprp, 
		float* tmpX, float* tmpXZ, float* filter_tmpYZ, float* nu_nlpm, 
		specie s, float dt_loc)
{
  
  get_nu_nlpm(nu_nlpm, Phi, tmpX, tmpXZ, s);  
  nlpm_filter<<<dimGrid,dimBlock>>>(Tpar, nu_nlpm, ky, dt_loc, dnlpm);
  nlpm_filter<<<dimGrid,dimBlock>>>(Tprp, nu_nlpm, ky, dt_loc, dnlpm); 	
  nlpm_filter<<<dimGrid,dimBlock>>>(Qpar, nu_nlpm, ky, dt_loc, dnlpm); 	
  nlpm_filter<<<dimGrid,dimBlock>>>(Qprp, nu_nlpm, ky, dt_loc, dnlpm); 	
  //nlpm_filter<<<dimGrid,dimBlock>>>(Dens, nu_nlpm, ky, dt_loc, dnlpm); 	
  //nlpm_filter<<<dimGrid,dimBlock>>>(Upar, nu_nlpm, ky, dt_loc, dnlpm); 	
}
