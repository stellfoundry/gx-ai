inline void linear_timestep(
  int is,
  int first_half_step,
  everything_struct * ev_h,
  everything_struct * ev_hd,
  everything_struct * ev_d 
)
{
cuComplex *Dens;
cuComplex *Upar;
cuComplex *Tpar;
cuComplex *Qpar;
cuComplex *Tprp;
cuComplex *Qprp;
cuComplex *Phi;// = ev_hd->fields.phi;

cuComplex *DensOld; cuComplex *DensNew;
cuComplex *UparOld; cuComplex *UparNew;
cuComplex *TparOld; cuComplex *TparNew;
cuComplex *QparOld; cuComplex *QparNew;
cuComplex *TprpOld; cuComplex *TprpNew;
cuComplex *QprpOld; cuComplex *QprpNew;


double dt = ev_h->time.dt;

if (first_half_step==1){
  //First half of RK2
  dt = dt/2.0;
  if(!LINEAR){
    Dens = ev_hd->fields.dens1[is];
    Upar = ev_hd->fields.upar1[is];
    Tpar = ev_hd->fields.tpar1[is];
    Tprp = ev_hd->fields.tprp1[is];
    Qpar = ev_hd->fields.qpar1[is];
    Qprp = ev_hd->fields.qprp1[is];
  }
  else {
    Dens = ev_hd->fields.dens[is];
    Upar = ev_hd->fields.upar[is];
    Tpar = ev_hd->fields.tpar[is];
    Tprp = ev_hd->fields.tprp[is];
    Qpar = ev_hd->fields.qpar[is];
    Qprp = ev_hd->fields.qprp[is];
  }
  DensOld = ev_hd->fields.dens[is];
  UparOld = ev_hd->fields.upar[is];
  TparOld = ev_hd->fields.tpar[is];
  TprpOld = ev_hd->fields.tprp[is];
  QparOld = ev_hd->fields.qpar[is];
  QprpOld = ev_hd->fields.qprp[is];
  DensNew = ev_hd->fields.dens1[is];
  UparNew = ev_hd->fields.upar1[is];
  TparNew = ev_hd->fields.tpar1[is];
  TprpNew = ev_hd->fields.tprp1[is];
  QparNew = ev_hd->fields.qpar1[is];
  QprpNew = ev_hd->fields.qprp1[is];
  Phi = ev_hd->fields.phi;
}
else {
  if(!LINEAR){
    Dens = ev_hd->fields.dens[is];
    Upar = ev_hd->fields.upar[is];
    Tpar = ev_hd->fields.tpar[is];
    Tprp = ev_hd->fields.tprp[is];
    Qpar = ev_hd->fields.qpar[is];
    Qprp = ev_hd->fields.qprp[is];
  }
  else {
    Dens = ev_hd->fields.dens[is];
    Upar = ev_hd->fields.upar[is];
    Tpar = ev_hd->fields.tpar[is];
    Tprp = ev_hd->fields.tprp[is];
    Qpar = ev_hd->fields.qpar[is];
    Qprp = ev_hd->fields.qprp[is];
  }
  DensNew = ev_hd->fields.dens[is];
  UparNew = ev_hd->fields.upar[is];
  TparNew = ev_hd->fields.tpar[is];
  TprpNew = ev_hd->fields.tprp[is];
  QparNew = ev_hd->fields.qpar[is];
  QprpNew = ev_hd->fields.qprp[is];
  DensOld = ev_hd->fields.dens1[is];
  UparOld = ev_hd->fields.upar1[is];
  TparOld = ev_hd->fields.tpar1[is];
  TprpOld = ev_hd->fields.tprp1[is];
  QparOld = ev_hd->fields.qpar1[is];
  QprpOld = ev_hd->fields.qprp1[is];
  Phi = ev_hd->fields.phi1;
}

int** kxCover = ev_hd->grids.kxCover;
int** kyCover = ev_hd->grids.kyCover;
cuComplex** g_covering = ev_hd->grids.g_covering;
float** kz_covering = ev_hd->grids.kz_covering;
specie s = ev_h->pars.species[is];

cuComplex *dens_field = ev_hd->fields.field;
cuComplex *upar_field = ev_hd->fields.field;
cuComplex *tpar_field = ev_hd->fields.field;
cuComplex *qpar_field = ev_hd->fields.field;
cuComplex *tprp_field = ev_hd->fields.field;
cuComplex *qprp_field = ev_hd->fields.field;

cuComplex *phi_tmp = ev_hd->tmp.CXYZ;
//cuComplex *nlps_tmp = ev_hd->tmp.CXYZ;
cuComplex* omegaStar_tmp = ev_hd->tmp.CXYZ;
cuComplex* qps_tmp = ev_hd->tmp.CXYZ;

cuComplex *fields_over_B_tmp = ev_hd->tmp.CXYZ;
cuComplex *B_gradpar_tmp = ev_hd->tmp.CXYZ;

cuComplex *gradpar_tmp = ev_hd->tmp.CXYZ;
cuComplex *omegaD_tmp = ev_hd->tmp.CXYZ;
cuComplex *sum_tmp = ev_hd->tmp.CXYZ;

cuComplex *fields_over_B2_tmp = ev_hd->tmp.CXYZ;
cuComplex *B2_gradpar_tmp = ev_hd->tmp.CXYZ;
cuComplex * bgrad_tmp = ev_hd->tmp.CXYZ;

//cuComplex* hyper_tmp = ev_hd->tmp.CXYZ;
//cuComplex* nlpm_tmp = ev_hd->tmp.CXYZ;
cuComplex* Tpar0_tmp = ev_hd->tmp.CXYZ;

//float *gradparB_tmpZ = ev_hd->tmp.Z;
cufftHandle* plan_covering = ev_h->ffts.plan_covering;

//float* nu_nlpm = ev_hd->nlpm.nu;
//float* Phi2ZF_tmpX = ev_hd->tmp.X;
//float* tmpXZ = ev_hd->tmp.XZ;
cuComplex* fluxsurfavg_CtmpX = ev_hd->tmp.CX;
//cuComplex* fluxsurfavg_CtmpX2 = ev_hd->tmp.CX2;
  
  /*
  //calculate nu_nlpm for this timestep... to be used in each field equation
  if(!LINEAR && NLPM) {
    get_nu_nlpm(nu_nlpm, Phi, Phi2ZF_tmpX, tmpXZ, s);
  }
  */
  
  float ps_fac;
  
  //NOTE ABOUT TEMPORARY ARRAYS:
  //all variables _tmp are the same array
  //all variables _field are the same array       
  
  cudaMemset(phi_tmp, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  //zeroC<<<dimGrid,dimBlock>>>(phi_tmp);

  ////////////////////////////////////////     
  //DENSITY
  
  cudaMemset(dens_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);  
  //zeroC<<<dimGrid,dimBlock>>>(dens_field);
    
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


  if(init == FORCE) {
    phi_nd_force <<<dimGrid, dimBlock>>> (phi_tmp, phiext, s.zt, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, phi_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
    accum <<<dimGrid, dimBlock>>> (dens_field, omegaD_tmp, -1);
  } 

  //add_scaled_Ky0<<<dimGrid,dimBlock>>>(dens_field, 1., dens_field, 1.);

  // -iOmegaD*(phi_nd + 2*Dens + Tpar + Tprp)
 
/* 
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, DensOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(dens_field, 1., dens_field, nu_hyper, hyper_tmp);
    // + nu_hyper * ((kperp/kperp_max)**(2*p_hyper)) * Dens
  }
*/  
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
  //zeroC<<<dimGrid,dimBlock>>>(upar_field);
  
  addsubt <<<dimGrid, dimBlock>>> (sum_tmp, DensOld, TparOld, 1);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, sum_tmp, bmagInv);
  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, kxCover, kyCover,g_covering, kz_covering, "",plan_covering);
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
  add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt, B_gradpar_tmp);
  // + vt*B*gradpar( (Dens+Tpar)/B )
  
  phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  ZDerivCovering(gradpar_tmp, phi_tmp, kxCover, kyCover, g_covering, kz_covering,"",plan_covering);
  add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt*s.zt, gradpar_tmp);
 
  if(init == FORCE) {
    phi_u_force<<<dimGrid,dimBlock>>>(phi_tmp, phiext, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    ZDerivCovering(gradpar_tmp, phi_tmp, kxCover, kyCover, g_covering, kz_covering,"",plan_covering);
    add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt*s.zt, gradpar_tmp);
  } 
  // + vt*zt*gradpar(phi_u)
  
  phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., DensOld, 1., TprpOld, s.zt, phi_tmp);
  multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,sum_tmp,bgrad);
  add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt, bgrad_tmp);

  if(init == FORCE) {
    phi_flr_force <<<dimGrid, dimBlock>>> (phi_tmp, phiext, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,phi_tmp,bgrad);
    add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt, bgrad_tmp);
  } 
  // + vt*( (Dens + Tprp + zt*phi_flr) )*bgrad
  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., QparOld, 1., QprpOld, 4., UparOld);
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (upar_field, omegaD_tmp, -1);
  // - iOmegaD*(Qpar + Qprp + 4*Upar)
    
/*
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, UparOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(upar_field, 1., upar_field, nu_hyper, hyper_tmp);
    // + nu_hyper * (kperp**(2*p_hyper)) * Upar
  }
*/
  
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
  //zeroC<<<dimGrid,dimBlock>>>(tpar_field);
  
    
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
  if(varenna && abs(ivarenna)==5) {
    add_scaled_Ky0 <<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, mu[0].y, DensOld);
  }
  if(new_varenna) {
    //add_scaled_Ky0 <<<dimGrid, dimBlock>>> (sum_tmp, 1., sum_tmp, -(2*nu[1].y), TparOld, -(2*nu[2].y), TprpOld);
    zonal_tpard <<<dimGrid, dimBlock>>>(sum_tmp, DensOld, TparOld, TprpOld, phi_tmp, kx, gds22, qsf, eps, bmagInv, shat, s.rho, zonal_dens_switch, tpar_omegad_corrections); 
  }    
  if(init == RH_equilibrium) {
    RH_eq_tpard <<<dimGrid,dimBlock>>>(sum_tmp, DensOld, TparOld, TprpOld, phi_tmp, kx, gds22, qsf, eps, bmagInv, shat, s);
  }
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (tpar_field, omegaD_tmp, -1);
  // - iOmegaD*( phi_tpard + (6+2*nu1.y)*Tpar + 2*Dens + 2*nu2.y*Tprp )

  if(init == FORCE) {
    phi_tpard_force<<<dimGrid,dimBlock>>>(phi_tmp, phiext, s.zt, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, phi_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
    accum <<<dimGrid, dimBlock>>> (tpar_field, omegaD_tmp, -1);
  } 
    
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 2*nu[1].x, TparOld, 2*nu[2].x, TprpOld);  
  if(varenna) {
    add_scaled_Ky0 <<<dimGrid, dimBlock>>> (sum_tmp, 1.,sum_tmp, -(2*nu[1].x) + (2*mu[1].x), TparOld, -(2*nu[2].x) + (2*mu[2].x), TprpOld);
  }
  if(new_catto) {
    volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, Phi, jacobian, 1./fluxDen);
    cattoTpar0<<<dimGrid,dimBlock>>>(Tpar0_tmp, fluxsurfavg_CtmpX, kx, gds22, qsf, eps, bmagInv, shat, s.rho, shaping_ps);
    add_scaled_Ky0 <<<dimGrid, dimBlock>>> (sum_tmp, 1., sum_tmp, -(2*nu[1].x), Tpar0_tmp, -(2*nu[2].x), TprpOld);
  } 
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna, new_varenna);
  accum <<<dimGrid, dimBlock>>> (tpar_field, omegaD_tmp, 1);
  // + |OmegaD|*(2*nu1.x*Tpar + 2*nu2.x*Tprp)

  add_scaled <<<dimGrid, dimBlock>>> (tpar_field, 1., tpar_field, 2*s.nu_ss/3, TparOld, -2*s.nu_ss/3, TprpOld);
  // + (2*nu_ss/3)*(Tpar - Tprp)
 
/* 
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, TparOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(tpar_field, 1., tpar_field, nu_hyper, hyper_tmp);
    // + nu_hyper * (kperp**(2*p_hyper)) * Tpar
  }
*/  
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
  //zeroC<<<dimGrid,dimBlock>>>(tprp_field);
      
  
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
  if(new_varenna) {  
    //add_scaled_Ky0<<<dimGrid,dimBlock>>> (sum_tmp, 1., sum_tmp, -(2*nu[4].y), TprpOld, -(2*nu[3].y), TparOld);
    zonal_tperpd <<<dimGrid, dimBlock>>>(sum_tmp, DensOld, TparOld, TprpOld, phi_tmp, kx, gds22, qsf, eps, bmagInv, shat, s.rho, zonal_dens_switch, tperp_omegad_corrections); 
  }    
  if(init == RH_equilibrium) {
    RH_eq_tperpd <<<dimGrid,dimBlock>>>(sum_tmp, DensOld, TparOld, TprpOld, phi_tmp, kx, gds22, qsf, eps, bmagInv, shat, s);
  }
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (tprp_field, omegaD_tmp, -1);
  //-iOmegaD*[ phi_tperpd + (4+2*nu4.y)*Tprp + Dens + (2*nu3.y)*Tpar ]
  
  if(init == FORCE) {
    phi_tperpd_force <<<dimGrid, dimBlock>>> (phi_tmp, phiext, s.zt, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, phi_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
    accum <<<dimGrid, dimBlock>>> (tprp_field, omegaD_tmp, -1);
  } 

  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 2*nu[3].x, TparOld, 2*nu[4].x, TprpOld);  
  if(varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>> (sum_tmp, 1., sum_tmp, -(2*nu[3].x) + (2*mu[3].x), TparOld, -(2*nu[4].x) + (2*mu[4].x), TprpOld);
  }
  if(new_catto) {
    //Tprp0 = 0
    add_scaled_Ky0<<<dimGrid,dimBlock>>> (sum_tmp, 1., sum_tmp, -(2*nu[3].x), TparOld);
  }
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna, new_varenna);
  accum <<<dimGrid, dimBlock>>> (tprp_field, omegaD_tmp, 1);
  // + |OmegaD|*( (2*nu3.x)*Tpar + (2*nu4.x)*Tprp )
  
  
  add_scaled <<<dimGrid, dimBlock>>> (tprp_field, 1., tprp_field, -s.nu_ss/3, TparOld, s.nu_ss/3, TprpOld);
  // + (nu_ss/3)*(Tprp-Tpar)
 
/* 
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, TprpOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, nu_hyper, hyper_tmp);
    // + nu_hyper * (kperp**(2*p_hyper)) * Tprp
  }
*/  
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
  //zeroC<<<dimGrid,dimBlock>>>(qpar_field);

  
  if(new_varenna) {
    scale<<<dimGrid,dimBlock>>>(gradpar_tmp, TparOld, (3.+Beta_par));
    zonal_qpar_gradpar<<<dimGrid,dimBlock>>>(gradpar_tmp, DensOld, TparOld, kx, gds22, qsf, eps, bmagInv, shat, s.rho, zonal_dens_switch, Beta_par, qpar_gradpar_corrections); 
    if (init == RH_equilibrium) {
      RH_eq_qpar_gradpar<<<dimGrid,dimBlock>>>(gradpar_tmp, DensOld, TparOld, kx, gds22, qsf, eps, bmagInv, shat, s);
    }
    ZDerivCovering(gradpar_tmp, gradpar_tmp, kxCover, kyCover, g_covering, kz_covering,"",plan_covering);  
    add_scaled <<<dimGrid, dimBlock>>> (qpar_field, 1., qpar_field, s.vt, gradpar_tmp);  
  }
  else {
    ZDerivCovering(gradpar_tmp, TparOld, kxCover, kyCover, g_covering, kz_covering,"",plan_covering);  
    add_scaled <<<dimGrid, dimBlock>>> (qpar_field, 1., qpar_field, s.vt*(3+Beta_par), gradpar_tmp);  
  }

  if(new_catto) {
    volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, Phi, jacobian, 1./fluxDen);
    ps_fac = -shaping_ps*pow(eps,1.5)*3.;
    PfirschSchluter_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, ps_fac, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
       
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering);
  }
  else if(varenna && (abs(ivarenna)==1 || abs(ivarenna)==3 || abs(ivarenna)==5)  && eps!=0. && varenna_fsa==true) {
    if(ivarenna>0) {
      volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, TparOld, jacobian, 1./fluxDen);
      ps_fac = 3.;
      PfirschSchluter_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, ps_fac, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
    }
    if(ivarenna<0) {
      volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, Phi, jacobian, 1./fluxDen);
      ps_fac = -shaping_ps*pow(eps,1.5)*3.;
      PfirschSchluter_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, ps_fac, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
    }      
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering);
  }
  else if(varenna && (abs(ivarenna)==1 || abs(ivarenna)==3 || abs(ivarenna)==5) && eps!=0. && varenna_fsa==false) { 
    if(ivarenna>0) {
      ps_fac = 3.;
      PfirschSchluter<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, ps_fac, kx, gds22, qsf, eps, bmagInv, TparOld, shat, s.rho);  //defined in operations_kernel.cu  
    }
    if(ivarenna<0) {
      ps_fac = -shaping_ps*pow(eps,1.5)*3.;
      PfirschSchluter<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, ps_fac, kx, gds22, qsf, eps, bmagInv, Phi, shat, s.rho);  //defined in operations_kernel.cu  
    }      
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering);
  }
  else if(new_varenna && new_varenna_fsa == true && init != RH_equilibrium) {
    volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, TparOld, jacobian, 1./fluxDen);
    zonal_qpar0_fsa<<<dimGrid, dimBlock>>>(qps_tmp, QparOld, DensOld, fluxsurfavg_CtmpX, kx, gds22, qsf, eps, bmagInv, shat, s.rho, q0_dens_switch, qpar0_switch);
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering); 
  }
  else if(new_varenna && new_varenna_fsa == false && init != RH_equilibrium) {
    zonal_qpar0<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, DensOld, TparOld, kx, gds22, qsf, eps, bmagInv, shat, s.rho, q0_dens_switch, qpar0_switch);
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering); 
  }
  else if( init == RH_equilibrium ) {
    zeroC<<<dimGrid,dimBlock>>>(gradpar_tmp);
  }
  else {
    ZDerivCovering(gradpar_tmp, QparOld, kxCover, kyCover,g_covering, kz_covering, "abs",plan_covering);
  }  
  add_scaled <<<dimGrid, dimBlock>>> (qpar_field, 1., qpar_field, s.vt*sqrt(2)*D_par, gradpar_tmp, s.nu_ss, QparOld);  
  // + vt*sqrt(2)*D_par*|gradpar|(Qpar - Qpar0) + nu_ss*Qpar  
  
  if(new_varenna) {
    zonal_qpar_bgrad<<<dimGrid,dimBlock>>>(bgrad_tmp, DensOld, TparOld, TprpOld, kx, gds22, qsf, eps, bmagInv, shat, s.rho, zonal_dens_switch, qpar_bgrad_corrections); 
    if(init == RH_equilibrium) {
      RH_eq_qpar_bgrad<<<dimGrid,dimBlock>>>(bgrad_tmp, DensOld, TparOld, TprpOld, kx, gds22, qsf, eps, bmagInv, shat, s);
    }
    multZ<<<dimGrid,dimBlock>>>(bgrad_tmp, bgrad_tmp, bgrad);
    add_scaled_Ky0<<<dimGrid, dimBlock>>> (qpar_field, 1., qpar_field, s.vt, bgrad_tmp);
  }
 
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, -3.+nu[6].y, QparOld, -3.+nu[7].y, QprpOld, 6.+nu[5].y, UparOld);  
  if(varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, -(-3.+nu[6].y) + (-3.+mu[6].y), QparOld, -(-3.+nu[7].y) + (-3.+mu[7].y),QprpOld, -(6.+nu[5].y) + (6.+mu[5].y),UparOld);
  }
  if(new_varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, -(nu[6].y), QparOld, -(nu[7].y), QprpOld, -(nu[5].y), UparOld); //effectively nu=0 for ky=0
  }
  if(new_catto) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, -(nu[6].y), QparOld, -(nu[7].y), QprpOld, -(nu[5].y), UparOld); //effectively nu=0 for ky=0
  }
  if(init == RH_equilibrium) {
    RH_eq_qpard<<<dimGrid,dimBlock>>>(sum_tmp, UparOld, QparOld, QprpOld, kx, gds22, qsf, eps, bmagInv, shat, s);
  }

  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (qpar_field, omegaD_tmp, -1);
  // - iOmegaD*( (-3+nu6.y)*Qpar + (-3+nu7.y)*Qprp + (6+nu5.y)*Upar )

  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, nu[5].x, UparOld, nu[6].x, QparOld, nu[7].x, QprpOld);  
  if(varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, -(nu[5].x) + (mu[5].x), UparOld, -(nu[6].x) + (mu[6].x), QparOld, -(nu[7].x) + (mu[7].x), QprpOld);
  }
  if(new_catto) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, -(nu[5].x), UparOld, -(nu[6].x), QparOld, -(nu[7].x), QprpOld); //effectively nu=0 for ky=0
  }
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna, new_varenna);
  accum <<<dimGrid, dimBlock>>> (qpar_field, omegaD_tmp, 1);
  // + |omegaD|*(nu5.x*Upar + nu6.x*Qpar + nu7.x*Qprp)
 
/* 
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, QparOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(qpar_field, 1., qpar_field, nu_hyper, hyper_tmp);
    // + nu_hyper * (kperp**(2*p_hyper)) * Qpar
  }
*/  
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
  //zeroC<<<dimGrid,dimBlock>>>(qprp_field);
  
  phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, s.zt, phi_tmp, 1., TprpOld);  
  if(new_varenna) {
    zonal_qperp_gradpar<<<dimGrid,dimBlock>>>(sum_tmp, DensOld, TparOld, TprpOld, phi_tmp, kx, gds22, qsf, eps, bmagInv, shat, s.rho, zonal_dens_switch, s.zt, qperp_gradpar_corrections);
  }
  if(init == RH_equilibrium) {
    RH_eq_qperp_gradpar<<<dimGrid, dimBlock>>>(sum_tmp, DensOld, TparOld, phi_tmp, kx, gds22, qsf, eps, bmagInv, shat, s);
  }
  ZDerivCovering(gradpar_tmp, sum_tmp, kxCover, kyCover, g_covering, kz_covering,"",plan_covering);  
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt, gradpar_tmp);
  
  if(init == FORCE) {
    phi_flr_force <<<dimGrid, dimBlock>>> (phi_tmp, phiext, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    ZDerivCovering(gradpar_tmp, phi_tmp, kxCover, kyCover, g_covering, kz_covering,"",plan_covering);  
    add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt, gradpar_tmp);
  } 
  // + vt*gradpar( zt*phi_flr + Tprp )

  if(new_catto) {
    volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, Phi, jacobian, 1./fluxDen);
    ps_fac = -shaping_ps*pow(eps,1.5)*1.;
    PfirschSchluter_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, ps_fac, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
         
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering);
  }  
  else if(varenna && (abs(ivarenna)==1 || abs(ivarenna)==3 || abs(ivarenna)==5) && eps!=0. && varenna_fsa==true) {
    if(ivarenna>0) {
      volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, TprpOld, jacobian, 1./fluxDen);
      ps_fac = 1.;
      PfirschSchluter_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, ps_fac, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
    }
    if(ivarenna<0) {
      volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, Phi, jacobian, 1./fluxDen);
      ps_fac = -shaping_ps*pow(eps,1.5)*1.;
      PfirschSchluter_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, ps_fac, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
    }      
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering);
  }
  else  if(varenna && (abs(ivarenna)==1 || abs(ivarenna)==3 || abs(ivarenna)==5) && eps!=0. && varenna_fsa==false) {
    if(ivarenna>0) {
      ps_fac = 1.;
      PfirschSchluter<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, ps_fac, kx, gds22, qsf, eps, bmagInv, TprpOld, shat, s.rho);  //defined in operations_kernel.cu  
    }
    if(ivarenna<0) {
      ps_fac = -shaping_ps*pow(eps,1.5)*1.;
      PfirschSchluter<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, ps_fac, kx, gds22, qsf, eps, bmagInv, Phi, shat, s.rho);  //defined in operations_kernel.cu  
    }      
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering);
  }
  else if(new_varenna && new_varenna_fsa == true && init != RH_equilibrium) {
    volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, TprpOld, jacobian, 1./fluxDen);
    zonal_qprp0_fsa<<<dimGrid, dimBlock>>>(qps_tmp, QprpOld, DensOld, fluxsurfavg_CtmpX, kx, gds22, qsf, eps, bmagInv, shat, s.rho, q0_dens_switch, qprp0_switch);
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering); 
  }
  else if(new_varenna && new_varenna_fsa == false && init != RH_equilibrium) {
    zonal_qprp0<<<dimGrid, dimBlock>>>(qps_tmp, QprpOld, DensOld, TprpOld, kx, gds22, qsf, eps, bmagInv, shat, s.rho, q0_dens_switch, qprp0_switch);
    ZDerivCovering(gradpar_tmp, qps_tmp, kxCover, kyCover,g_covering, kz_covering, "abs", plan_covering); 
  }
  else if (init == RH_equilibrium) {
    zeroC<<<dimGrid,dimBlock>>>(gradpar_tmp);
  }
  else {
    ZDerivCovering(gradpar_tmp, QprpOld, kxCover, kyCover,g_covering, kz_covering, "abs",plan_covering);
  }
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt*sqrt(2)*D_prp, gradpar_tmp);
  // + vt*sqrt(2)*D_prp*|gradpar|(Qprp - Qprp0)
  
  phi_qperpb <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., TprpOld, -1., TparOld, s.zt, phi_tmp);
  if(new_varenna) {
    zonal_qperp_bgrad <<<dimGrid,dimBlock>>>(sum_tmp, DensOld, TparOld, TprpOld, phi_tmp, kx, gds22, qsf, eps, bmagInv, shat, s.rho, zonal_dens_switch, s.zt, qperp_bgrad_corrections);
  }
  if(init == RH_equilibrium) {
    RH_eq_qperp_bgrad <<<dimGrid,dimBlock>>>(sum_tmp, TparOld, TprpOld, phi_tmp, kx, gds22, qsf, eps, bmagInv, shat, s); 
  }
  multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,sum_tmp,bgrad);
  add_scaled<<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt, bgrad_tmp);
  
  if(init == FORCE) {
    phi_qperpb_force <<<dimGrid, dimBlock>>> (phi_tmp, phiext, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,phi_tmp,bgrad);
    add_scaled<<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt, bgrad_tmp);
  } 
  // + vt*( zt*phi_qperpb + Tprp - Tpar )*bgrad
    
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, -1.+nu[9].y, QparOld, -1.+nu[10].y, QprpOld, 1.+nu[8].y, UparOld);  
  if(varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>> (sum_tmp, 1., sum_tmp, -(-1.+nu[9].y) + (-1.+mu[9].y), QparOld, -(-1.+nu[10].y) + (-1.+mu[10].y), QprpOld, -(1.+nu[8].y) + 1.+mu[8].y,UparOld);
  }  
  if(new_varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>> (sum_tmp, 1., sum_tmp, -(nu[9].y), QparOld, -(nu[10].y), QprpOld, -(nu[8].y), UparOld);  //effectively nu=0
  }  
  if(new_catto) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>> (sum_tmp, 1., sum_tmp, -(nu[9].y), QparOld, -(nu[10].y), QprpOld, -(nu[8].y), UparOld);  //effectively nu=0
  }  
  if(init == RH_equilibrium) {
    RH_eq_qperpd<<<dimGrid,dimBlock>>>(sum_tmp, UparOld, QparOld, QprpOld, kx, gds22, qsf, eps, bmagInv, shat, s);
  }
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (qprp_field, omegaD_tmp, -1);
  // - iOmegaD*( (-1+nu9.y)*Qpar + (-1+nu10.y)*Qprp + (1+nu8.y)*Upar )
  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, nu[8].x, UparOld, nu[9].x, QparOld, nu[10].x, QprpOld);  
  if(varenna) {
    add_scaled_Ky0 <<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, -(nu[8].x) + (mu[8].x), UparOld, -(nu[9].x) + (mu[9].x), QparOld, -(nu[10].x) + (mu[10].x), QprpOld);
  }
  if(new_catto) {
    add_scaled_Ky0 <<<dimGrid,dimBlock>>>(sum_tmp, 1., sum_tmp, -(nu[8].x), UparOld, -(nu[9].x), QparOld, -(nu[10].x), QprpOld); //effectively nu=0 for ky=0
  }
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna,new_varenna);
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, 1., omegaD_tmp, s.nu_ss, QprpOld);
  // + |omegaD|*(nu8.x*Upar + nu9.x*Qpar + nu10.x*Qprp) + nu_ss*Qprp
 
/* 
  if(HYPER) {
    hyper_dissipation<<<dimGrid,dimBlock>>>(hyper_tmp, QprpOld, p_hyper, kx, ky, kperp2_max_Inv);
    add_scaled<<<dimGrid,dimBlock>>>(qprp_field, 1., qprp_field, nu_hyper, hyper_tmp);
    // + nu_hyper * (kperp**(2*p_hyper)) * Qprp
  }
*/  
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
//



inline void nonlinear_timestep(
  int is,
  int first_half_step,
  everything_struct * ev_h,
  everything_struct * ev_hd,
  everything_struct * ev_d) 
{

cuComplex *Dens = ev_hd->fields.dens[is];
cuComplex *Upar = ev_hd->fields.upar[is];
cuComplex *Tpar = ev_hd->fields.tpar[is];
cuComplex *Qpar = ev_hd->fields.qpar[is];
cuComplex *Tprp = ev_hd->fields.tprp[is];
cuComplex *Qprp = ev_hd->fields.qprp[is];

cuComplex *Phi;// = ev_hd->fields.phi;

cuComplex* NLdens_ky0_d = ev_hd->hybrid.dens[is];
cuComplex* NLupar_ky0_d = ev_hd->hybrid.upar[is];
cuComplex* NLtpar_ky0_d = ev_hd->hybrid.tpar[is];
cuComplex* NLtprp_ky0_d = ev_hd->hybrid.tprp[is];
cuComplex* NLqpar_ky0_d = ev_hd->hybrid.qpar[is];
cuComplex* NLqprp_ky0_d = ev_hd->hybrid.qprp[is];

double* dt_full = &ev_h->time.dt;
int counter = ev_h->time.counter;

specie s = ev_h->pars.species[is];

cuComplex *dens_field = ev_hd->fields.field;
cuComplex *upar_field = ev_hd->fields.field;
cuComplex *tpar_field = ev_hd->fields.field;
cuComplex *qpar_field = ev_hd->fields.field;
cuComplex *tprp_field = ev_hd->fields.field;
cuComplex *qprp_field = ev_hd->fields.field;
 
cuComplex *phi_tmp = ev_hd->tmp.CXYZ;
cuComplex *nlps_tmp = ev_hd->tmp.CXYZ;
float* Phi_zf_rms_tmpX = ev_hd->tmp.X;
cuComplex* Phi_zf_CtmpX = ev_hd->tmp.CX;
cuComplex *field_h = ev_h->fields.field;
float kx2Phi_zf_rms_in = ev_h->nlpm.kx2Phi_zf_rms;
float kx2Phi_zf_rms_avg = ev_h->nlpm.kx2Phi_zf_rms_avg;

cuComplex *DensOld; cuComplex *DensNew;
cuComplex *UparOld; cuComplex *UparNew;
cuComplex *TparOld; cuComplex *TparNew;
cuComplex *QparOld; cuComplex *QparNew;
cuComplex *TprpOld; cuComplex *TprpNew;
cuComplex *QprpOld; cuComplex *QprpNew;

if (first_half_step==1){
  DensOld = ev_hd->fields.dens[is];
  UparOld = ev_hd->fields.upar[is];
  TparOld = ev_hd->fields.tpar[is];
  TprpOld = ev_hd->fields.tprp[is];
  QparOld = ev_hd->fields.qpar[is];
  QprpOld = ev_hd->fields.qprp[is];
  DensNew = ev_hd->fields.dens1[is];
  UparNew = ev_hd->fields.upar1[is];
  TparNew = ev_hd->fields.tpar1[is];
  TprpNew = ev_hd->fields.tprp1[is];
  QparNew = ev_hd->fields.qpar1[is];
  QprpNew = ev_hd->fields.qprp1[is];
  Phi = ev_hd->fields.phi;
}
else {
  DensNew = ev_hd->fields.dens[is];
  UparNew = ev_hd->fields.upar[is];
  TparNew = ev_hd->fields.tpar[is];
  TprpNew = ev_hd->fields.tprp[is];
  QparNew = ev_hd->fields.qpar[is];
  QprpNew = ev_hd->fields.qprp[is];
  DensOld = ev_hd->fields.dens1[is];
  UparOld = ev_hd->fields.upar1[is];
  TparOld = ev_hd->fields.tpar1[is];
  TprpOld = ev_hd->fields.tprp1[is];
  QparOld = ev_hd->fields.qpar1[is];
  QprpOld = ev_hd->fields.qprp1[is];
  Phi = ev_hd->fields.phi1;
}


  float kx2Phi_zf_rms;
  if(nlpm_cutoff_avg) kx2Phi_zf_rms = kx2Phi_zf_rms_avg;
  else kx2Phi_zf_rms = kx2Phi_zf_rms_in;

  cudaMemset(phi_tmp, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

  if(nlpm_nlps) {
    // get zonal flow component of Phi
    //Phi2ZF_tmpX = |Phi_zf(kx)|_rms
    volflux_zonal_rms<<<dimGrid,dimBlock>>>(Phi_zf_rms_tmpX, Phi, Phi, jacobian, 1./(fluxDen*fluxDen) );
    //complex Phi_zf(kx)
    volflux_zonal_complex<<<dimGrid,dimBlock>>>(Phi_zf_CtmpX, Phi, jacobian, 1./fluxDen);
  }
  ////////////////////////////////////////     
  //DENSITY
  
  cudaMemset(dens_field, 0., sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);  

    phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);          
    if(first_half_step) cfl_flag = true; //only calculate cfl condition and change dt in first bracket of half step
    NLPS(nlps_tmp, phi_tmp, DensOld, kx, ky);    
    cfl_flag = false;

#ifndef GS2_zonal
    //if not running in GS2, change the timestep here
    //if running in GS2, timestep is changed less frequently, with a reset in run_gryfx
    if(first_half_step) {
      *dt_full = dt_cfl;
    }
#endif

    double dt;

    if(first_half_step) dt = *dt_full/2.;
    else dt = *dt_full;

    accum <<<dimGrid, dimBlock>>> (dens_field, nlps_tmp, 1);
    // +{phi_u,Dens}
    
    phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);      
    NLPS(nlps_tmp, phi_tmp, TprpOld, kx, ky);   
    accum <<<dimGrid, dimBlock>>> (dens_field, nlps_tmp, 1);  
    // +{phi_flr,Tprp}

    if(first_half_step && counter==0) fieldWrite(dens_field, field_h, "NLdens_t0.field", filename);

#ifdef GS2_zonal
      
      getky0_nopad<<<dimGrid,dimBlock>>>(NLdens_ky0_d, dens_field);

#endif

  if(secondary_test) scale<<<dimGrid,dimBlock>>>(dens_field, dens_field, NLdensfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (DensNew, 1., Dens, -dt, dens_field);
  
  ////////////////////////////////////////
  //UPAR

  cudaMemset(upar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
    phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, UparOld, kx, ky);
    accum <<<dimGrid, dimBlock>>> (upar_field, nlps_tmp, 1);
    // + {phi_u, Upar}

    
    phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, QprpOld, kx, ky);    
    accum <<<dimGrid, dimBlock>>> (upar_field, nlps_tmp, 1);
    // + {phi_flr, Qprp}

#ifdef GS2_zonal
  
      getky0_nopad<<<dimGrid,dimBlock>>>(NLupar_ky0_d, upar_field);

#endif

  if(secondary_test) scale<<<dimGrid,dimBlock>>>(upar_field, upar_field, NLuparfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (UparNew, 1., Upar, -dt, upar_field);

  ////////////////////////////////////////
  //TPAR

  cudaMemset(tpar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
    phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, TparOld, kx, ky);
    accum <<<dimGrid, dimBlock>>> (tpar_field, nlps_tmp, 1);
    // + {phi_u,Tpar}

    if(nlpm_nlps) {
      phi_flr_zonal_abs<<<dimGrid,dimBlock>>>(phi_tmp, Phi_zf_rms_tmpX, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPM_NLPS(nlps_tmp, phi_tmp, TparOld, kx_abs, ky);
      if(strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms<high_cutoff && kx2Phi_zf_rms>low_cutoff) {
        add_scaled<<<dimGrid,dimBlock>>>(tpar_field, 1., tpar_field, .4*dnlpm*(kx2Phi_zf_rms-low_cutoff)/(high_cutoff-low_cutoff), nlps_tmp);
      }
      else if((strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms>=high_cutoff) || strcmp(nlpm_option,"constant")==0) {
        add_scaled<<<dimGrid,dimBlock>>>(tpar_field, 1., tpar_field, .4*dnlpm, nlps_tmp);
      }
      
      phi_flr_zonal_complex<<<dimGrid,dimBlock>>>(phi_tmp, Phi_zf_CtmpX, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPM_NLPS(nlps_tmp, phi_tmp, TparOld, kx, ky);  
      if(strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms<high_cutoff && kx2Phi_zf_rms>low_cutoff) {
        add_scaled<<<dimGrid,dimBlock>>>(tpar_field, 1., tpar_field, -.6*dnlpm*(kx2Phi_zf_rms-low_cutoff)/(high_cutoff-low_cutoff), nlps_tmp);
      }
      else if((strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms>=high_cutoff) || strcmp(nlpm_option,"constant")==0) {
        add_scaled<<<dimGrid,dimBlock>>>(tpar_field, 1., tpar_field, -.6*dnlpm, nlps_tmp);
      }
    }

#ifdef GS2_zonal
  
      getky0_nopad<<<dimGrid,dimBlock>>>(NLtpar_ky0_d, tpar_field);

#endif

  if(secondary_test) scale<<<dimGrid,dimBlock>>>(tpar_field, tpar_field, NLtparfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (TparNew, 1., Tpar, -dt, tpar_field);

  ////////////////////////////////////////
  //TPERP
  
  cudaMemset(tprp_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      
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

    if(nlpm_nlps) {
      phi_flr_zonal_abs<<<dimGrid,dimBlock>>>(phi_tmp, Phi_zf_rms_tmpX, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPM_NLPS(nlps_tmp, phi_tmp, TprpOld, kx_abs, ky);
      if(strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms<high_cutoff && kx2Phi_zf_rms>low_cutoff) {
        add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, 1.6*dnlpm*(kx2Phi_zf_rms-low_cutoff)/(high_cutoff-low_cutoff), nlps_tmp);
      }
      else if((strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms>=high_cutoff) || strcmp(nlpm_option,"constant")==0) {
        add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, 1.6*dnlpm, nlps_tmp);
      }
      
      phi_flr_zonal_complex<<<dimGrid,dimBlock>>>(phi_tmp, Phi_zf_CtmpX, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPM_NLPS(nlps_tmp, phi_tmp, TprpOld, kx, ky);  
      if(strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms<high_cutoff && kx2Phi_zf_rms>low_cutoff) {
        add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, -1.3*dnlpm*(kx2Phi_zf_rms-low_cutoff)/(high_cutoff-low_cutoff), nlps_tmp);
      }
      else if((strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms>=high_cutoff) || strcmp(nlpm_option,"constant")==0) {
        add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, -1.3*dnlpm, nlps_tmp);
      }
    }
#ifdef GS2_zonal
  
      getky0_nopad<<<dimGrid,dimBlock>>>(NLtprp_ky0_d, tprp_field);

#endif

  if(secondary_test) scale<<<dimGrid,dimBlock>>>(tprp_field, tprp_field, NLtprpfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (TprpNew, 1., Tprp, -dt, tprp_field);

  ////////////////////////////////////////
  //QPAR
  
  cudaMemset(qpar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 

    phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);    
    NLPS(nlps_tmp, phi_tmp, QparOld, kx, ky);    
    accum <<<dimGrid, dimBlock>>> (qpar_field, nlps_tmp, 1);
    // + {phi_u, Qpar}

    if(nlpm_nlps) {
      phi_flr_zonal_abs<<<dimGrid,dimBlock>>>(phi_tmp, Phi_zf_rms_tmpX, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPM_NLPS(nlps_tmp, phi_tmp, QparOld, kx_abs, ky);
      if(strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms<high_cutoff && kx2Phi_zf_rms>low_cutoff) {
        add_scaled<<<dimGrid,dimBlock>>>(qpar_field, 1., qpar_field, .4*dnlpm*(kx2Phi_zf_rms-low_cutoff)/(high_cutoff-low_cutoff), nlps_tmp);
      }
      else if((strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms>=high_cutoff) || strcmp(nlpm_option,"constant")==0) {
        add_scaled<<<dimGrid,dimBlock>>>(qpar_field, 1., qpar_field, .4*dnlpm, nlps_tmp);
      }      

      phi_flr_zonal_complex<<<dimGrid,dimBlock>>>(phi_tmp, Phi_zf_CtmpX, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPM_NLPS(nlps_tmp, phi_tmp, QparOld, kx, ky);  
      if(strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms<high_cutoff && kx2Phi_zf_rms>low_cutoff) {
        add_scaled<<<dimGrid,dimBlock>>>(qpar_field, 1., qpar_field, -.6*dnlpm*(kx2Phi_zf_rms-low_cutoff)/(high_cutoff-low_cutoff), nlps_tmp);
      }
      else if((strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms>=high_cutoff) || strcmp(nlpm_option,"constant")==0) {
        add_scaled<<<dimGrid,dimBlock>>>(qpar_field, 1., qpar_field, -.6*dnlpm, nlps_tmp);
      }      
    }
#ifdef GS2_zonal
  
      getky0_nopad<<<dimGrid,dimBlock>>>(NLqpar_ky0_d, qpar_field);

#endif

  if(secondary_test) scale<<<dimGrid,dimBlock>>>(qpar_field, qpar_field, NLqparfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (QparNew, 1., Qpar, -dt, qpar_field);

  ////////////////////////////////////////
  //QPERP
  
  cudaMemset(qprp_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 
  
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

    if(nlpm_nlps) {
      phi_flr_zonal_abs<<<dimGrid,dimBlock>>>(phi_tmp, Phi_zf_rms_tmpX, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPM_NLPS(nlps_tmp, phi_tmp, QprpOld, kx_abs, ky);
      if(strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms<high_cutoff && kx2Phi_zf_rms>low_cutoff) {
        add_scaled<<<dimGrid,dimBlock>>>(qprp_field, 1., qprp_field, 1.6*dnlpm*(kx2Phi_zf_rms-low_cutoff)/(high_cutoff-low_cutoff), nlps_tmp);
      }
      else if((strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms>=high_cutoff) || strcmp(nlpm_option,"constant")==0) {
        add_scaled<<<dimGrid,dimBlock>>>(qprp_field, 1., qprp_field, 1.6*dnlpm, nlps_tmp);
      }      

      phi_flr_zonal_complex<<<dimGrid,dimBlock>>>(phi_tmp, Phi_zf_CtmpX, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPM_NLPS(nlps_tmp, phi_tmp, QprpOld, kx, ky);  
      if(strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms<high_cutoff && kx2Phi_zf_rms>low_cutoff) {
        add_scaled<<<dimGrid,dimBlock>>>(qprp_field, 1., qprp_field, -1.3*dnlpm*(kx2Phi_zf_rms-low_cutoff)/(high_cutoff-low_cutoff), nlps_tmp);
      }
      else if((strcmp(nlpm_option,"cutoff") == 0 && kx2Phi_zf_rms>=high_cutoff) || strcmp(nlpm_option,"constant")==0) {
        add_scaled<<<dimGrid,dimBlock>>>(qprp_field, 1., qprp_field, -1.3*dnlpm, nlps_tmp);
      }      
    }
#ifdef GS2_zonal
  
      getky0_nopad<<<dimGrid,dimBlock>>>(NLqprp_ky0_d, qprp_field);

#endif

  if(secondary_test) scale<<<dimGrid,dimBlock>>>(qprp_field, qprp_field, NLqprpfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (QprpNew, 1., Qprp, -dt, qprp_field);


}


