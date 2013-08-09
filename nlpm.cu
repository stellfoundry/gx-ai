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

