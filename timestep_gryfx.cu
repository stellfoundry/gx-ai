void timestep(cuComplex *Dens, cuComplex *DensOld, cuComplex *DensNew,
              cuComplex *Upar, cuComplex *UparOld, cuComplex *UparNew,
              cuComplex *Tpar, cuComplex *TparOld, cuComplex *TparNew,
              cuComplex *Qpar, cuComplex *QparOld, cuComplex *QparNew,
              cuComplex *Tprp, cuComplex *TprpOld, cuComplex *TprpNew,
              cuComplex *Qprp, cuComplex *QprpOld, cuComplex *QprpNew,
	      cuComplex *Phi, int** kxCover, int** kyCover, cuComplex** g_covering, float** kz_covering, specie s, float dt,
	      cuComplex *dens_field, cuComplex *upar_field, cuComplex *tpar_field,
	      cuComplex *qpar_field, cuComplex *tprp_field, cuComplex *qprp_field, 
	      cuComplex *phi_tmp, cuComplex *nlps_tmp, cuComplex* omegaStar_tmp, cuComplex* qps_tmp,
	      cuComplex *fields_over_B_tmp, cuComplex *B_gradpar_tmp, 
	      cuComplex *gradpar_tmp, cuComplex *omegaD_tmp, cuComplex *sum_tmp,
	      cuComplex *fields_over_B2_tmp, cuComplex *B2_gradpar_tmp, cuComplex * bgrad_tmp, 
	      cuComplex* hyper_tmp, cuComplex* nlpm_tmp,
	      float *gradparB_tmpZ, cufftHandle* plan_covering,
	      float* nu_nlpm, float* Phi2ZF_tmpX, float* tmpXZ)
{
  
  /*
  //calculate nu_nlpm for this timestep... to be used in each field equation
  if(!LINEAR && NLPM) {
    get_nu_nlpm(nu_nlpm, Phi, Phi2ZF_tmpX, tmpXZ, s);
  }
  */
  
  
  //NOTE ABOUT TEMPORARY ARRAYS:
  //all variables _tmp are the same array
  //all variables _field are the same array       
  
  cudaMemset(phi_tmp, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

  ////////////////////////////////////////     
  //DENSITY
  
  cudaMemset(dens_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 
 
  if(!LINEAR) {
    phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);          
    NLPS(nlps_tmp, phi_tmp, DensOld, kx, ky);    
    accum <<<dimGrid, dimBlock>>> (dens_field, nlps_tmp, 1);
    // +{phi_u,Dens}
    
    phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);      
    NLPS(nlps_tmp, phi_tmp, TprpOld, kx, ky);   
    accum <<<dimGrid, dimBlock>>> (dens_field, nlps_tmp, 1);  
    // +{phi_flr,Tprp}
  }
    
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, UparOld, bmagInv);    
  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, kxCover, kyCover, g_covering, kz_covering, "",plan_covering);  
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag); 
  add_scaled <<<dimGrid, dimBlock>>> (dens_field, 1., dens_field, s.vt, B_gradpar_tmp); 
  // +vt*B*gradpar(Upar/B)
  
  phi_n <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.tprim, s.rho, s.fprim, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  iOmegaStar <<<dimGrid, dimBlock>>> (omegaStar_tmp, phi_tmp, ky); 
  accum <<<dimGrid, dimBlock>>> (dens_field, omegaStar_tmp, 1);
  // +iOmegaStar*phi_n
  
  phi_nd <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.zt, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., phi_tmp, 2., DensOld, 1., TparOld, 1., TprpOld);  
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (dens_field, omegaD_tmp, -1);
  // -iOmegaD*(phi_nd + 2*Dens + Tpar + Tprp)
  
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, DensOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(dens_field, 1., dens_field, nu_hyper, hyper_tmp);
    // + nu_hyper * ((kperp/kperp_max)**(2*p_hyper)) * Dens
  }
  
  /*
  if(!LINEAR && NLPM) {
    nlpm<<<dimGrid,dimBlock>>>(nlpm_tmp, DensOld, ky, nu_nlpm, dnlpm);
    add_scaled<<<dimGrid,dimBlock>>>(dens_field, 1., dens_field, 1., nlpm_tmp);
    // + nu_nlpm*|ky|*Dens
  }
  */
  
  //step
  add_scaled <<<dimGrid, dimBlock>>> (DensNew, 1., Dens, -dt, dens_field);
  
  if(SMAGORINSKY) SmagorinskyDiffusion<<<dimGrid,dimBlock>>>(DensNew, DensNew, diffusion, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  
  //DensNew = Dens - dt*[ {phi_u, Dens} + {phi_flr,Tprp} + vt*B*gradpar(Upar/B) + iOmegaStar*phi_n - iOmegaD*(phi_nd + 2*Dens + Tpar + Tprp) ]
  
  ////////////////////////////////////////
  //UPAR

  cudaMemset(upar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
  if(!LINEAR) {
    phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, UparOld, kx, ky);
    accum <<<dimGrid, dimBlock>>> (upar_field, nlps_tmp, 1);
    // + {phi_u, Upar}
    
    phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, QprpOld, kx, ky);    
    accum <<<dimGrid, dimBlock>>> (upar_field, nlps_tmp, 1);
    // + {phi_flr, Qprp}
  }
  
  addsubt <<<dimGrid, dimBlock>>> (sum_tmp, DensOld, TparOld, 1);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, sum_tmp, bmagInv);
  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, kxCover, kyCover,g_covering, kz_covering, "",plan_covering);
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
  add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt, B_gradpar_tmp);
  // + vt*B*gradpar( (Dens+Tpar)/B )
  
  phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  ZDerivCovering(gradpar_tmp, phi_tmp, kxCover, kyCover, g_covering, kz_covering,"",plan_covering);
  add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt*s.zt, gradpar_tmp);
  // + vt*zt*gradpar(phi_u)
  
  phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., DensOld, 1., TprpOld, s.zt, phi_tmp);
  multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,sum_tmp,bgrad);
  add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt, bgrad_tmp);
  // + vt*( (Dens + Tprp + zt*phi_flr) )*bgrad
  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., QparOld, 1., QprpOld, 4., UparOld);
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (upar_field, omegaD_tmp, -1);
  // - iOmegaD*(Qpar + Qprp + 4*Upar)
    
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, UparOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(upar_field, 1., upar_field, nu_hyper, hyper_tmp);
    // + nu_hyper * (kperp**(2*p_hyper)) * Upar
  }
  
  /*
  if(!LINEAR && NLPM) {
    nlpm<<<dimGrid,dimBlock>>>(nlpm_tmp, UparOld, ky, nu_nlpm, dnlpm);
    add_scaled<<<dimGrid,dimBlock>>>(upar_field, 1., upar_field, 1., nlpm_tmp);
    // + nu_nlpm*|ky|*Upar
  }
  */
  
  //step
  add_scaled <<<dimGrid, dimBlock>>> (UparNew, 1., Upar, -dt, upar_field);
  
  if(SMAGORINSKY) SmagorinskyDiffusion<<<dimGrid,dimBlock>>>(UparNew, UparNew, diffusion, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);

  //UparNew = Upar - dt * [ {phi_u, Upar} + {phi_flr, Qprp} + vt*B*gradpar( (Dens+Tpar)/B ) + vt*zt*gradpar(phi_u) 
  //                        + vt*( (Dens + Tprp + zt*phi_flr) ) * Bgrad - iOmegaD*(Qpar + Qprp + 4*Upar) ]
  
  
  ////////////////////////////////////////
  //TPAR

  cudaMemset(tpar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
  if(!LINEAR) {
    phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, TparOld, kx, ky);
    accum <<<dimGrid, dimBlock>>> (tpar_field, nlps_tmp, 1);
    // + {phi_u,Tpar}
  }
    
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., QparOld, 2., UparOld);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, sum_tmp, bmagInv);
  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, kxCover, kyCover,g_covering, kz_covering, "",plan_covering);
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
  add_scaled <<<dimGrid, dimBlock>>> (tpar_field, 1., tpar_field, s.vt, B_gradpar_tmp);
  // + vt*B*gradpar( (Qpar + 2*Upar)/B )
  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., QprpOld, 1., UparOld);
  multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,sum_tmp,bgrad);
  add_scaled<<<dimGrid,dimBlock>>>(tpar_field, 1., tpar_field, 2*s.vt, bgrad_tmp);
  // + 2*vt*(Qprp + Upar)*bgrad
  
  phi_tpar <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.tprim, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  iOmegaStar <<<dimGrid, dimBlock>>> (omegaStar_tmp, phi_tmp, ky);
  accum <<<dimGrid, dimBlock>>> (tpar_field, omegaStar_tmp, 1);
  // + iOmegaStar*phi_tpar
  
  phi_tpard <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.zt, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., phi_tmp, 6.+2*nu[1].y, TparOld, 2., DensOld, 2*nu[2].y, TprpOld);  
  if(varenna) {
    add_scaled_Ky0 <<<dimGrid, dimBlock>>> (sum_tmp, 1., sum_tmp, -(6.+2*nu[1].y) + (6.+2*mu[1].y), TparOld, -(2*nu[2].y) + (2*mu[2].y), TprpOld);
  }
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (tpar_field, omegaD_tmp, -1);
  // - iOmegaD*( phi_tpard + (6+2*nu1.y)*Tpar + 2*Dens + 2*nu2.y*Tprp )
    
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 2*nu[1].x, TparOld, 2*nu[2].x, TprpOld);  
  if(varenna) {
    add_scaled_Ky0 <<<dimGrid, dimBlock>>> (sum_tmp, 1.,sum_tmp, -(2*nu[1].x) + (2*mu[1].x), TparOld, -(2*nu[2].x) + (2*mu[2].x), TprpOld);
  }
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna);
  accum <<<dimGrid, dimBlock>>> (tpar_field, omegaD_tmp, 1);
  // + |OmegaD|*(2*nu1.x*Tpar + 2*nu2.x*Tprp)

  add_scaled <<<dimGrid, dimBlock>>> (tpar_field, 1., tpar_field, 2*s.nu_ss/3, TparOld, -2*s.nu_ss/3, TprpOld);
  // + (2*nu_ss/3)*(Tpar - Tprp)
  
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, TparOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(tpar_field, 1., tpar_field, nu_hyper, hyper_tmp);
    // + nu_hyper * (kperp**(2*p_hyper)) * Tpar
  }
  
  /*
  if(!LINEAR && NLPM) {
    nlpm<<<dimGrid,dimBlock>>>(nlpm_tmp, TparOld, ky, nu_nlpm, dnlpm);
    add_scaled<<<dimGrid,dimBlock>>>(tpar_field, 1., tpar_field, 1., nlpm_tmp);
    // + nu_nlpm*|ky|*Tpar
  }
  */
  
  //step
  add_scaled <<<dimGrid, dimBlock>>> (TparNew, 1., Tpar, -dt, tpar_field);
  
  if(SMAGORINSKY) SmagorinskyDiffusion<<<dimGrid,dimBlock>>>(TparNew, TparNew, diffusion, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  
  
  //TparNew = Tpar - dt * [ ( 2*vt*(Qprp + Upar) ) * Bgrad + {phi_u, Tpar} + vt*B*gradpar( (Qpar+2*Upar)/B ) + iOmegaStar*phi_tpar  
  //                 - iOmegaD*( phi_tpard + (6+2*nu1.y)*Tpar + 2*Dens + 2*nu2.y*Tprp ) + |omegaD|*(2*nu1.x*Tpar + 2*nu2.x*Tprp) + (2*nu_ss/3)*(Tpar - Tprp) ]
  
  ////////////////////////////////////////
  //TPERP
  
  cudaMemset(tprp_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      
  if(!LINEAR) {
    phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, TprpOld, kx, ky);    
    accum <<<dimGrid, dimBlock>>> (tprp_field, nlps_tmp, 1);
    // + {phi_u, Tprp}

    
    phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, DensOld, kx, ky);    
    accum <<<dimGrid, dimBlock>>> (tprp_field, nlps_tmp, 1);
    // + {phi_flr, Dens}
    
    
    phi_flr2 <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, TprpOld, kx, ky);    
    accum <<<dimGrid, dimBlock>>> (tprp_field, nlps_tmp, 1);
    // + {phi_flr2, Tprp}
  }
  
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, QprpOld, bmagInv);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B2_tmp, fields_over_B_tmp, bmagInv);
  ZDerivCovering(gradpar_tmp, fields_over_B2_tmp, kxCover, kyCover,g_covering, kz_covering, "",plan_covering);
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
  multZ <<<dimGrid, dimBlock>>> (B2_gradpar_tmp, B_gradpar_tmp, bmag);
  add_scaled <<<dimGrid, dimBlock>>> (tprp_field, 1., tprp_field, s.vt, B2_gradpar_tmp);
  // + vt*B2*gradpar( Qprp/B2 )
  
  multZ<<<dimGrid,dimBlock>>>(bgrad_tmp, UparOld, bgrad);
  add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, -s.vt, bgrad_tmp);
  // - vt*bgrad*Upar
  
  phi_tperp <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.tprim, s.rho, s.fprim, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  iOmegaStar <<<dimGrid, dimBlock>>> (omegaStar_tmp, phi_tmp, ky);
  accum <<<dimGrid, dimBlock>>> (tprp_field, omegaStar_tmp, 1);
  // + iOmegaStar(phi_tperp)
  
  
  phi_tperpd <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.zt, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., phi_tmp, 4.+2*nu[4].y, TprpOld, 1., DensOld, 2*nu[3].y, TparOld);  
  if(varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>> (sum_tmp, 1., sum_tmp, -(4.+2*nu[4].y) + (4.+2*mu[4].y), TprpOld, -(2*nu[3].y) + (2*mu[3].y), TparOld);
  }
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (tprp_field, omegaD_tmp, -1);
  //-iOmegaD*[ phi_tperpd + (4+2*nu4.y)*Tprp + Dens + (2*nu3.y)*Tpar ]
  
  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 2*nu[3].x, TparOld, 2*nu[4].x, TprpOld);  
  if(varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>> (sum_tmp, 1., sum_tmp, -(2*nu[3].x) + (2*mu[3].x), TparOld, -(2*nu[4].x) + (2*mu[4].x), TprpOld);
  }
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna);
  accum <<<dimGrid, dimBlock>>> (tprp_field, omegaD_tmp, 1);
  // + |OmegaD|*( (2*nu3.x)*Tpar + (2*nu4.x)*Tprp )
  
  
  add_scaled <<<dimGrid, dimBlock>>> (tprp_field, 1., tprp_field, -s.nu_ss/3, TparOld, s.nu_ss/3, TprpOld);
  // + (nu_ss/3)*(Tprp-Tpar)
  
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, TprpOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, nu_hyper, hyper_tmp);
    // + nu_hyper * (kperp**(2*p_hyper)) * Tprp
  }
  
  /*
  if(!LINEAR && NLPM) {
    nlpm<<<dimGrid,dimBlock>>>(nlpm_tmp, TprpOld, ky, nu_nlpm, dnlpm);
    add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, 1., nlpm_tmp);
    // + nu_nlpm*|ky|*Tprp
  }
  */
  
  //step
  add_scaled <<<dimGrid, dimBlock>>> (TprpNew, 1., Tprp, -dt, tprp_field);
  
  if(SMAGORINSKY) SmagorinskyDiffusion<<<dimGrid,dimBlock>>>(TprpNew, TprpNew, diffusion, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);

  //TprpNew = Tprp - dt * [ {phi_u, Tprp} + {phi_flr, Dens} + {phi_flr2, Tprp} + vt*B2*gradpar((Qprp+Upar)/B2) - vt*B*gradpar( Upar/B) + iOmegaStar*phi_tperp
  //                - iOmegaD*( phi_tperpd + (4+2*nu4.y)*Tprp + Dens + (2*nu3.y)*Tpar ) + |omegaD|*( (2*nu3.x)*Tpar + (2*nu4.x)*Tprp ) + (nu_ss/3)*(Tprp-Tpar) ]
  
  ////////////////////////////////////////
  //QPAR
  
  cudaMemset(qpar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 

  
  if(!LINEAR) {  
    phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);    
    NLPS(nlps_tmp, phi_tmp, QparOld, kx, ky);    
    accum <<<dimGrid, dimBlock>>> (qpar_field, nlps_tmp, 1);
    // + {phi_u, Qpar}
  }
  
  
  ZDerivCovering(gradpar_tmp, TparOld, kxCover, kyCover, g_covering, kz_covering,"",plan_covering);  
  add_scaled <<<dimGrid, dimBlock>>> (qpar_field, 1., qpar_field, s.vt*(3+Beta_par), gradpar_tmp);  
  if(varenna) {
    PfirschSchluter<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, 3., kx, gds22, qsf, eps, bmagInv, TparOld, shat);  //defined in operations_kernel.cu  
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs",plan_covering);
  }
  else {
    ZDerivCovering(gradpar_tmp, QparOld, kxCover, kyCover,g_covering, kz_covering, "abs",plan_covering);
  }  
  add_scaled <<<dimGrid, dimBlock>>> (qpar_field, 1., qpar_field, s.vt*sqrt(2)*D_par, gradpar_tmp, s.nu_ss, QparOld);  
  // + vt*sqrt(2)*D_par*|gradpar|(Qpar - Qpar0) + nu_ss*Qpar  
  
  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, -3.+nu[6].y, QparOld, -3.+nu[7].y, QprpOld, 6.+nu[5].y, UparOld);  
  if(varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, -(-3.+nu[6].y) + (-3.+mu[6].y), QparOld, -(-3.+nu[7].y) + (-3.+mu[7].y),QprpOld, -(6.+nu[5].y) + (6.+mu[5].y),UparOld);
  }
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (qpar_field, omegaD_tmp, -1);
  // - iOmegaD*( (-3+nu6.y)*Qpar + (-3+nu7.y)*Qprp + (6+nu5.y)*Upar )

  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, nu[5].x, UparOld, nu[6].x, QparOld, nu[7].x, QprpOld);  
  if(varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, -(nu[5].x) + (mu[5].x), UparOld, -(nu[6].x) + (mu[6].x), QparOld, -(nu[7].x) + (mu[7].x), QprpOld);
  }
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna);
  accum <<<dimGrid, dimBlock>>> (qpar_field, omegaD_tmp, 1);
  // + |omegaD|*(nu5.x*Upar + nu6.x*Qpar + nu7.x*Qprp)
  
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, QparOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(qpar_field, 1., qpar_field, nu_hyper, hyper_tmp);
    // + nu_hyper * (kperp**(2*p_hyper)) * Qpar
  }
  
  /*
  if(!LINEAR && NLPM) {
    nlpm<<<dimGrid,dimBlock>>>(nlpm_tmp, QparOld, ky, nu_nlpm, dnlpm);
    add_scaled<<<dimGrid,dimBlock>>>(qpar_field, 1., qpar_field, 1., nlpm_tmp);
    // + nu_nlpm*|ky|*Qpar
  }
  */
  
  //step
  add_scaled <<<dimGrid, dimBlock>>> (QparNew, 1., Qpar, -dt, qpar_field);
  
  if(SMAGORINSKY) SmagorinskyDiffusion<<<dimGrid,dimBlock>>>(QparNew, QparNew, diffusion, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);

  //QparNew = Qpar - dt * [ {phi_u, Qpar} + vt*(3+Beta_par)*gradpar(Tpar) + vt*sqrt(2)*D_par*|gradpar|(Qpar - Qpar0)  
  //                - iOmegaD*( (-3+nu6.y)*Qpar + (-3+nu7.y)*Qprp + (6+nu5.y)*Upar ) + |omegaD|*(nu5.x*Upar + nu6.x*Qpar + nu7.x*Qprp) + nu_ss*Qpar ]
  
  ////////////////////////////////////////
  //QPERP
  
  cudaMemset(qprp_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 
  

  if(!LINEAR) {
    phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);    
    NLPS(nlps_tmp, phi_tmp, QprpOld, kx, ky);
    accum <<<dimGrid, dimBlock>>> (qprp_field, nlps_tmp, 1);
    // + {phi_u, Qprp}
    
    phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, UparOld, kx, ky);
    accum <<<dimGrid, dimBlock>>> (qprp_field, nlps_tmp, 1);
    // + {phi_flr, Upar}
    
    phi_flr2 <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, QprpOld, kx, ky);
    accum <<<dimGrid, dimBlock>>> (qprp_field, nlps_tmp, 1);
    // + {phi_flr2, Qprp}
  }
  
  
  phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, s.zt, phi_tmp, 1., TprpOld);  
  ZDerivCovering(gradpar_tmp, sum_tmp, kxCover, kyCover, g_covering, kz_covering,"",plan_covering);  
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt, gradpar_tmp);
  // + vt*gradpar( zt*phi_flr + Tprp )
  
  
  if(varenna) {
    PfirschSchluter<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, 1., kx, gds22, qsf, eps, bmagInv, TprpOld, shat);  //defined in operations_kernel.cu  
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering);
  }
  else {
    ZDerivCovering(gradpar_tmp, QprpOld, kxCover, kyCover,g_covering, kz_covering, "abs",plan_covering);
  }
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt*sqrt(2)*D_prp, gradpar_tmp);
  // + vt*sqrt(2)*D_prp*|gradpar|(Qprp - Qprp0)
  
  phi_qperpb <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., TprpOld, -1., TparOld, s.zt, phi_tmp);
  multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,sum_tmp,bgrad);
  add_scaled<<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt, bgrad_tmp);
  // + vt*( zt*phi_qperpb + Tprp - Tpar )*bgrad
    
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, -1.+nu[9].y, QparOld, -1.+nu[10].y, QprpOld, 1.+nu[8].y, UparOld);  
  if(varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>> (sum_tmp, 1., sum_tmp, -(-1.+nu[9].y) + (-1.+mu[9].y), QparOld, -(-1.+nu[10].y) + (-1.+mu[10].y), QprpOld, -(1.+nu[8].y) + 1.+mu[8].y,UparOld);
  }  
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (qprp_field, omegaD_tmp, -1);
  // - iOmegaD*( (-1+nu9.y)*Qpar + (-1+nu10.y)*Qprp + (1+nu8.y)*Upar )
  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, nu[8].x, UparOld, nu[9].x, QparOld, nu[10].x, QprpOld);  
  if(varenna) {
    add_scaled_Ky0 <<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, -(nu[8].x) + (mu[8].x), UparOld, -(nu[9].x) + (mu[9].x), QparOld, -(nu[10].x) + (mu[10].x), QprpOld);
  }
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna);
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, 1., omegaD_tmp, s.nu_ss, QprpOld);
  // + |omegaD|*(nu8.x*Upar + nu9.x*Qpar + nu10.x*Qprp) + nu_ss*Qprp
  
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, QprpOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(qprp_field, 1., qprp_field, nu_hyper, hyper_tmp);
    // + nu_hyper * (kperp**(2*p_hyper)) * Qprp
  }
  
  /*
  if(!LINEAR && NLPM) {
    nlpm<<<dimGrid,dimBlock>>>(nlpm_tmp, QprpOld, ky, nu_nlpm, dnlpm);
    add_scaled<<<dimGrid,dimBlock>>>(qprp_field, 1., qprp_field, 1., nlpm_tmp);
    // + nu_nlpm*|ky|*Qprp
  }
  */
  
  //step
  add_scaled <<<dimGrid, dimBlock>>> (QprpNew, 1., Qprp, -dt, qprp_field);
  
  if(SMAGORINSKY) SmagorinskyDiffusion<<<dimGrid,dimBlock>>>(QprpNew, QprpNew, diffusion, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
   
  //QprpNew = Qprp - dt * [ vt*( zt*phi_qperpb + Tprp - Tpar )*bgrad + {phi_u, Qprp} + {phi_flr, Upar} + {phi_flr2, Qprp} + vt*gradpar( zt*phi_flr + Tprp ) 
  //    + vt*sqrt(2)*D_prp*|gradpar|(Qprp-Qprp0) - iOmegaD*( (-1+nu9.y)*Qpar + (-1+nu10.y)*Qprp + (1+nu8.y)*Upar ) + |omegaD|*(nu8.x*Upar + nu9.x*Qpar + nu10.x*Qprp) + nu_ss*Qprp ]

}

