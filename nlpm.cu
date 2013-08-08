//looped over species outside call to NPLM
void NLPM(cuComplex* Phi, cuComplex* Dens, cuComplex* Upar, cuComplex* Tpar, cuComplex* Tprp, cuComplex* Qpar, cuComplex* Qprp, 
          float* shear_tmpX, float* shear_tmpZ, cuComplex* shear_tmp, cuComplex* filter_tmp, specie s, float dt_loc)
{
  float dnlpm = 1.;
  
  zeroC<<<dimGrid,dimBlock>>>(shear_tmp);
  
  //if(inlpm == 1)
    //shear1<<<dimGrid,dimBlock>>>(shear_tmp, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv, Phi);
  //if(inlpm == 2)
  
  shear2<<<dimGrid,dimBlock>>>(shear_tmp, dnlpm, kx, s.rho, ky, shat, gds2, gds21, gds22, bmagInv, Phi);
  
  //we have shear(kx,ky=0,z). now we need to sum over kx. to do this we must break the shear(kx,z) array into
  //shear(kx) and reduce over kx for each z.
  
  fixFFT<<<dimGrid,dimBlock>>>(shear_tmp);
  
  float shear;
  
  for(int i=0; i<Nz; i++) {
    //copy into shear(kx) for each z
    zcopyX_Y0<<<dimGrid,dimBlock>>>(shear_tmpX, shear_tmp, i);
    //reduce shear(kx) 
    shear = sumReduc(shear_tmpX, Nx, false);
    //assign shear to ith element of shear(z) (without copying)
    assign<<<1,1>>> (shear_tmpZ, shear, i);    
  }
  
  
  filter2<<<dimGrid,dimBlock>>>(filter_tmp, shear_tmpZ, ky, dt_loc, dnlpm);
  
  multdiv<<<dimGrid,dimBlock>>>(Tpar,Tpar,filter_tmp,1);
  multdiv<<<dimGrid,dimBlock>>>(Tprp,Tprp,filter_tmp,1);
  multdiv<<<dimGrid,dimBlock>>>(Qpar,Qpar,filter_tmp,1);
  multdiv<<<dimGrid,dimBlock>>>(Qprp,Qprp,filter_tmp,1);
  multdiv<<<dimGrid,dimBlock>>>(Upar,Upar,filter_tmp,1);
  multdiv<<<dimGrid,dimBlock>>>(Dens,Dens,filter_tmp,1);
  

}
