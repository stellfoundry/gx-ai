void linear_timestep(
  int is,
  int first_half_step,
  everything_struct * ev_h,
  everything_struct * ev_hd,
  everything_struct * ev_d 
)
{
#ifdef PROFILE
PUSH_RANGE("gryfx linear timestep",3);

PUSH_RANGE("setting up",1);
#endif


cuComplex *Dens;
cuComplex *Upar;
cuComplex *Tpar;
cuComplex *Qpar;
cuComplex *Tprp;
cuComplex *Qprp;
cuComplex *Phi;// = ev_hd->fields.phi;
cuComplex *AparOld;
cuComplex *AparNew;

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
  AparOld = ev_hd->fields.apar;
  AparNew = ev_hd->fields.apar1;
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
  AparOld = ev_hd->fields.apar1;
  AparNew = ev_hd->fields.apar;
}

//int** kxCover = ev_hd->grids.kxCover;
//int** kyCover = ev_hd->grids.kyCover;
//cuComplex** g_covering = ev_hd->grids.g_covering;
//cuComplex** g_covering_d = ev_d->grids.g_covering;
//float** kz_covering = ev_hd->grids.kz_covering;
specie s = ev_h->pars.species[is];

cuComplex *dens_field = ev_hd->fields.field;
cuComplex *upar_field = ev_hd->fields.field;
cuComplex *tpar_field = ev_hd->fields.field;
cuComplex *qpar_field = ev_hd->fields.field;
cuComplex *tprp_field = ev_hd->fields.field;
cuComplex *qprp_field = ev_hd->fields.field;

cuComplex *phi_tmp = ev_hd->tmp.CXYZ;
cuComplex *apar_tmp = ev_hd->tmp.CXYZ;
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

bool higher_order_moments = ev_h->pars.higher_order_moments;

char filename[500];
  
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

#ifdef PROFILE
POP_RANGE;
#endif

  ////////////////////////////////////////     
  //DENSITY
#ifdef PROFILE
PUSH_RANGE("density",5);
#endif
  
  cudaMemset(dens_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);  
    
  //if(ev_h->pars.fapar > 0.) {
  //  // for electromagnetic ions, the 'upar' and 'qprp' evolved by the code are actually 
  //  // 'upar' = upar + vt*zt*apar_u
  //  // 'qprp' = qprp + vt*zt*apar_flr
  //  // here we subtract off the apar parts
  //  // we will add them back later
  //  phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  //  add_scaled<<<dimGrid,dimBlock>>>(UparOld, 1., UparOld, -s.vt*s.zt, apar_tmp);
  //  
  //  phi_flr <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  //  add_scaled<<<dimGrid,dimBlock>>>(QprpOld, 1., QprpOld, -s.vt*s.zt, apar_tmp);
  //}

  density_linear_terms<<<dimGrid,dimBlock>>>(dens_field, Phi, DensOld, TparOld, TprpOld, 
                                kx, ky, shat, s.rho, s.vt, s.tprim, s.fprim, s.zt,
                                gds2, gds21, gds22, bmagInv,
                                gbdrift, gbdrift0, cvdrift, cvdrift0);
  // +iOmegaStar*phi_n - iOmegaD*(phi_nd + 2*Dens + Tpar + Tprp)

/*
  phi_n <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.tprim, s.rho, s.fprim, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  iOmegaStar <<<dimGrid, dimBlock>>> (omegaStar_tmp, phi_tmp, ky); 
  accum <<<dimGrid, dimBlock>>> (dens_field, omegaStar_tmp, 1);
  // +iOmegaStar*phi_n
  
  phi_nd <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.zt, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., phi_tmp, 2., DensOld, 1., TparOld, 1., TprpOld);  
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (dens_field, omegaD_tmp, -1);
  // - iOmegaD*(phi_nd + 2*Dens + Tpar + Tprp)
*/

  if(init == FORCE) {
    phi_nd_force <<<dimGrid, dimBlock>>> (phi_tmp, phiext, s.zt, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, phi_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
    accum <<<dimGrid, dimBlock>>> (dens_field, omegaD_tmp, -1);
  } 
 
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, UparOld, bmagInv);    
  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "",plan_covering);  
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag); 
  add_scaled <<<dimGrid, dimBlock>>> (dens_field, 1., dens_field, s.vt, B_gradpar_tmp); 
  // +vt*B*gradpar(Upar/B)

  //step
  add_scaled <<<dimGrid, dimBlock>>> (DensNew, 1., Dens, -dt, dens_field);
  
  //DensNew = Dens - dt*[ {phi_u, Dens} + {phi_flr,Tprp} + vt*B*gradpar(Upar/B) + iOmegaStar*phi_n - iOmegaD*(phi_nd + 2*Dens + Tpar + Tprp) ]
  
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("upar",3);
#endif

  ////////////////////////////////////////
  //UPAR

  cudaMemset(upar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
  upar_linear_terms<<<dimGrid,dimBlock>>>(upar_field, Phi, AparOld, AparNew, DensOld, UparOld, TprpOld, QparOld, QprpOld,
                                kx, ky, shat, s.rho, s.vt, s.tprim, s.fprim, s.zt, bgrad,
                                gds2, gds21, gds22, bmagInv,
                                gbdrift, gbdrift0, cvdrift, cvdrift0, ev_h->pars.fapar, dt);
  // + vt*( (Dens + Tprp + zt*phi_flr) )*bgrad - iOmegaD*(Qpar + Qprp + 4*Upar)

/*
  phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., DensOld, 1., TprpOld, s.zt, phi_tmp);
  multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,sum_tmp,bgrad);
  add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt, bgrad_tmp);
  // + vt*( (Dens + Tprp + zt*phi_flr) )*bgrad

  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., QparOld, 1., QprpOld, 4., UparOld);
  iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
  accum <<<dimGrid, dimBlock>>> (upar_field, omegaD_tmp, -1);
  // - iOmegaD*(Qpar + Qprp + 4*Upar)
*/
    
  if(init == FORCE) {
  //  phi_flr_force <<<dimGrid, dimBlock>>> (phi_tmp, phiext, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  //  multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,phi_tmp,bgrad);
  //  add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt, bgrad_tmp);
  } 

  addsubt <<<dimGrid, dimBlock>>> (sum_tmp, DensOld, TparOld, 1);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, sum_tmp, bmagInv);
  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "",plan_covering);
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
  add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt, B_gradpar_tmp);
  // + vt*B*gradpar( (Dens+Tpar)/B )
  
  phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  ZDerivCovering(gradpar_tmp, phi_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids,"",plan_covering);
  add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt*s.zt, gradpar_tmp);
  // + vt*zt*gradpar(phi_u)
 
  if(init == FORCE) {
    phi_u_force<<<dimGrid,dimBlock>>>(phi_tmp, phiext, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    ZDerivCovering(gradpar_tmp, phi_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids,"",plan_covering);
    add_scaled <<<dimGrid, dimBlock>>> (upar_field, 1., upar_field, s.vt*s.zt, gradpar_tmp);
  } 
  
  //if(ev_h->pars.fapar > 0.) {
  //  // for electromagnetic ions, the 'upar' and 'qprp' evolved by the code are actually 
  //  // 'upar' = upar + vt*zt*apar_u
  //  // 'qprp' = qprp + vt*zt*apar_flr
  //  // here we add back the apar part for upar before advancing 'upar'
  //  phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  //  add_scaled<<<dimGrid,dimBlock>>>(UparOld, 1., UparOld, s.vt*s.zt, apar_tmp);
  //}
  //step
  add_scaled <<<dimGrid, dimBlock>>> (UparNew, 1., Upar, -dt, upar_field);
  
  //UparNew = Upar - dt * [ {phi_u, Upar} + {phi_flr, Qprp} + vt*B*gradpar( (Dens+Tpar)/B ) + vt*zt*gradpar(phi_u) 
  //                        + vt*( (Dens + Tprp + zt*phi_flr) ) * Bgrad - iOmegaD*(Qpar + Qprp + 4*Upar) ]
  
  
  ////////////////////////////////////////
  //TPAR
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("tpar",4);
#endif

  cudaMemset(tpar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

  //if(ev_h->pars.fapar > 0.) {
  //  // for electromagnetic ions, the 'upar' and 'qprp' evolved by the code are actually 
  //  // 'upar' = upar + vt*zt*apar_u
  //  // 'qprp' = qprp + vt*zt*apar_flr
  //  // here we subtract off the apar parts
  //  // we will add them back later
  //  phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  //  add_scaled<<<dimGrid,dimBlock>>>(UparOld, 1., UparOld, -s.vt*s.zt, apar_tmp);
  //}
  
  if(higher_order_moments) {
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(omegaD_tmp, ev_hd->hybrid.dens[0]); // this is rparpar
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, ev_hd->hybrid.upar[0]); // this is rparprp
  }
  tpar_linear_terms<<<dimGrid,dimBlock>>>(tpar_field, Phi, DensOld, UparOld, TparOld, TprpOld, QprpOld,
                                kx, ky, shat, s.rho, s.vt, s.tprim, s.fprim, s.zt, bgrad,
                                gds2, gds21, gds22, bmagInv,
                                gbdrift, gbdrift0, cvdrift, cvdrift0,
                                s.nu_ss, nu[1], nu[2], mu[1], mu[2], varenna,
                                omegaD_tmp, ev_hd->tmp.CXYZ2, higher_order_moments);
  // + ( 2*vt*(Qprp + Upar) ) * Bgrad + iOmegaStar*phi_tpar - iOmegaD*( phi_tpard + (6+2*nu1.y)*Tpar + 2*Dens + 2*nu2.y*Tprp ) 
  // + |omegaD|*(2*nu1.x*Tpar + 2*nu2.x*Tprp) + (2*nu_ss/3)*(Tpar - Tprp) 

/*
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
  
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., QprpOld, 1., UparOld);
  multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,sum_tmp,bgrad);
  add_scaled<<<dimGrid,dimBlock>>>(tpar_field, 1., tpar_field, 2*s.vt, bgrad_tmp);
  // + 2*vt*(Qprp + Upar)*bgrad
    
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 2*nu[1].x, TparOld, 2*nu[2].x, TprpOld);  
  if(varenna) {
    add_scaled_Ky0 <<<dimGrid, dimBlock>>> (sum_tmp, 1.,sum_tmp, -(2*nu[1].x) + (2*mu[1].x), TparOld, -(2*nu[2].x) + (2*mu[2].x), TprpOld);
  }
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna, new_varenna);
  accum <<<dimGrid, dimBlock>>> (tpar_field, omegaD_tmp, 1);
  // + |OmegaD|*(2*nu1.x*Tpar + 2*nu2.x*Tprp)

  add_scaled <<<dimGrid, dimBlock>>> (tpar_field, 1., tpar_field, 2*s.nu_ss/3, TparOld, -2*s.nu_ss/3, TprpOld);
  // + (2*nu_ss/3)*(Tpar - Tprp)
*/
 
  if(init == FORCE) {
    phi_tpard_force<<<dimGrid,dimBlock>>>(phi_tmp, phiext, s.zt, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, phi_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
    accum <<<dimGrid, dimBlock>>> (tpar_field, omegaD_tmp, -1);
  } 
    
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., QparOld, 2., UparOld);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, sum_tmp, bmagInv);
  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "",plan_covering);
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
  add_scaled <<<dimGrid, dimBlock>>> (tpar_field, 1., tpar_field, s.vt, B_gradpar_tmp);
  // + vt*B*gradpar( (Qpar + 2*Upar)/B )
  
  //step
  add_scaled <<<dimGrid, dimBlock>>> (TparNew, 1., Tpar, -dt, tpar_field);
  
  //TparNew = Tpar - dt * [ ( 2*vt*(Qprp + Upar) ) * Bgrad + {phi_u, Tpar} + vt*B*gradpar( (Qpar+2*Upar)/B ) + iOmegaStar*phi_tpar  
  //                 - iOmegaD*( phi_tpard + (6+2*nu1.y)*Tpar + 2*Dens + 2*nu2.y*Tprp ) + |omegaD|*(2*nu1.x*Tpar + 2*nu2.x*Tprp) + (2*nu_ss/3)*(Tpar - Tprp) ]
  
  ////////////////////////////////////////
  //TPERP
  
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("tprp",5);
#endif
  cudaMemset(tprp_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
  if(higher_order_moments) {
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(omegaD_tmp, ev_hd->hybrid.upar[0]); // this is rparprp
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, ev_hd->hybrid.tpar[0]); // this is rprpprp
  }
  tprp_linear_terms<<<dimGrid,dimBlock>>>(tprp_field, Phi, DensOld, UparOld, TparOld, TprpOld,
                                kx, ky, shat, s.rho, s.vt, s.tprim, s.fprim, s.zt, bgrad,
                                gds2, gds21, gds22, bmagInv,
                                gbdrift, gbdrift0, cvdrift, cvdrift0,
                                s.nu_ss, nu[3], nu[4], mu[3], mu[4], varenna,
                                omegaD_tmp, ev_hd->tmp.CXYZ2, higher_order_moments);
/*    
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
*/
  
  if(init == FORCE) {
    phi_tperpd_force <<<dimGrid, dimBlock>>> (phi_tmp, phiext, s.zt, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    iOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, phi_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0);
    accum <<<dimGrid, dimBlock>>> (tprp_field, omegaD_tmp, -1);
  } 
/*
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 2*nu[3].x, TparOld, 2*nu[4].x, TprpOld);  
  if(varenna) {
    add_scaled_Ky0<<<dimGrid,dimBlock>>> (sum_tmp, 1., sum_tmp, -(2*nu[3].x) + (2*mu[3].x), TparOld, -(2*nu[4].x) + (2*mu[4].x), TprpOld);
  }
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna, new_varenna);
  accum <<<dimGrid, dimBlock>>> (tprp_field, omegaD_tmp, 1);
  // + |OmegaD|*( (2*nu3.x)*Tpar + (2*nu4.x)*Tprp )
  
  add_scaled <<<dimGrid, dimBlock>>> (tprp_field, 1., tprp_field, -s.nu_ss/3, TparOld, s.nu_ss/3, TprpOld);
  // + (nu_ss/3)*(Tprp-Tpar)

  multZ<<<dimGrid,dimBlock>>>(bgrad_tmp, UparOld, bgrad);
  add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, -s.vt, bgrad_tmp);
  // - vt*bgrad*Upar
*/
  
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, QprpOld, bmagInv);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B2_tmp, fields_over_B_tmp, bmagInv);
  ZDerivCovering(gradpar_tmp, fields_over_B2_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "",plan_covering);
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
  multZ <<<dimGrid, dimBlock>>> (B2_gradpar_tmp, B_gradpar_tmp, bmag);
  add_scaled <<<dimGrid, dimBlock>>> (tprp_field, 1., tprp_field, s.vt, B2_gradpar_tmp);
  // + vt*B2*gradpar( Qprp/B2 )
  
  //step
  add_scaled <<<dimGrid, dimBlock>>> (TprpNew, 1., Tprp, -dt, tprp_field);
  
  //TprpNew = Tprp - dt * [ {phi_u, Tprp} + {phi_flr, Dens} + {phi_flr2, Tprp} + vt*B2*gradpar((Qprp+Upar)/B2) - vt*B*gradpar( Upar/B) + iOmegaStar*phi_tperp
  //                - iOmegaD*( phi_tperpd + (4+2*nu4.y)*Tprp + Dens + (2*nu3.y)*Tpar ) + |omegaD|*( (2*nu3.x)*Tpar + (2*nu4.x)*Tprp ) + (nu_ss/3)*(Tprp-Tpar) ]
  
  ////////////////////////////////////////
  //QPAR


#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("qpar",1);
#endif
  
  cudaMemset(qpar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 

  if(higher_order_moments) {
    //replace_ky0_nopad<<<dimGrid,dimBlock>>>(qpar_field, ev_hd->hybrid.upar[0]); // this is rparprp

    replace_ky0_nopad<<<dimGrid,dimBlock>>>(bgrad_tmp, ev_hd->hybrid.dens[0]); // this is rparpar
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(qpar_field, ev_hd->hybrid.upar[0]); // this is rparprp
    add_scaled<<<dimGrid,dimBlock>>>(qpar_field, -1., bgrad_tmp, 3., qpar_field);//, -Beta_par, TparOld, sqrt(2.)*D_par, QparOld);    

    replace_ky0_nopad<<<dimGrid,dimBlock>>>(omegaD_tmp, ev_hd->hybrid.tprp[0]); // this is sparpar
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, ev_hd->hybrid.qpar[0]); // this is sparprp
  }
  qpar_linear_terms<<<dimGrid,dimBlock>>>(qpar_field, AparOld, DensOld, UparOld, TparOld, TprpOld, QparOld, QprpOld,
                                kx, ky, shat, s.rho, s.vt, s.tprim, s.fprim, s.zt, bgrad,
                                gds2, gds21, gds22, bmagInv,
                                gbdrift, gbdrift0, cvdrift, cvdrift0,
                                s.nu_ss, nu[5], nu[6], nu[7], mu[5], mu[6], mu[7], varenna,
                                qpar_field, omegaD_tmp, ev_hd->tmp.CXYZ2, higher_order_moments, ev_h->pars.fapar);
/*  
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
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna, new_varenna);
  accum <<<dimGrid, dimBlock>>> (qpar_field, omegaD_tmp, 1);
  // + |omegaD|*(nu5.x*Upar + nu6.x*Qpar + nu7.x*Qprp)

  add_scaled <<<dimGrid, dimBlock>>> (qpar_field, 1., qpar_field, s.nu_ss, QparOld);
  // + (nu_ss)*(Qpar)
*/

if(higher_order_moments) {
  replace_ky0_nopad<<<dimGrid,dimBlock>>>(gradpar_tmp, ev_hd->hybrid.dens[0]); // this is rparpar
  //add_scaled<<<dimGrid,dimBlock>>>(sum_tmp, 1., gradpar_tmp, -3., DensOld, -3., TparOld);
  add_scaled<<<dimGrid,dimBlock>>>(sum_tmp, 0., gradpar_tmp, 0., DensOld, 3., TparOld);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, sum_tmp, bmagInv);
  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids,"",plan_covering);  
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
  add_scaled <<<dimGrid, dimBlock>>> (qpar_field, 1., qpar_field, s.vt, B_gradpar_tmp);  
} else {
  //if(varenna) {
  //  add_scaled<<<dimGrid,dimBlock>>>(gradpar_tmp, 2., DensOld, 2.+(3.+Beta_par), TparOld);
  //  ZDerivCovering(gradpar_tmp, gradpar_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids,"",plan_covering);  
  //  add_scaled<<<dimGrid,dimBlock>>>(qpar_field, 1., qpar_field, s.vt, gradpar_tmp);
  //} else {
    ZDerivCovering(gradpar_tmp, TparOld, &ev_h->grids,&ev_hd->grids,&ev_d->grids,"",plan_covering);  
    add_scaled <<<dimGrid, dimBlock>>> (qpar_field, 1., qpar_field, s.vt*(3.+Beta_par), gradpar_tmp);  
  //}
  if(varenna && (abs(ivarenna)==1 || abs(ivarenna)==3)  && eps!=0. && varenna_fsa==true) {
    if(ivarenna>0) {
      volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, TparOld, jacobian, 1./fluxDen);
      ps_fac = 3.;
      PfirschSchluter_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, ps_fac, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
    }
    if(ivarenna<0) {
      //volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, Phi, jacobian, 1./fluxDen);
      //ps_fac = -shaping_ps*pow(eps,1.5)*3.;
      //PfirschSchluter_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, ps_fac, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
      add_scaled<<<dimGrid,dimBlock>>>(qps_tmp, 1., DensOld, 1., TparOld);
      volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, qps_tmp, jacobian, 1./fluxDen);
      new_varenna_zf_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
    }      
    ZDerivCovering(gradpar_tmp, qps_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "abs", plan_covering);
  }
  else if(varenna && (abs(ivarenna)==1 || abs(ivarenna)==3) && eps!=0. && varenna_fsa==false) { 
    if(ivarenna>0) {
      ps_fac = 3.;
      PfirschSchluter<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, ps_fac, kx, gds22, qsf, eps, bmagInv, TparOld, shat, s.rho);  //defined in operations_kernel.cu  
    }
    if(ivarenna<0) {
      //ps_fac = -shaping_ps*pow(eps,1.5)*3.;
      //PfirschSchluter<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, ps_fac, kx, gds22, qsf, eps, bmagInv, Phi, shat, s.rho);  //defined in operations_kernel.cu  
      new_varenna_zf<<<dimGrid,dimBlock>>>(qps_tmp, QparOld, kx, gds22, qsf, eps, bmagInv, TparOld, shat, s.rho);  //defined in operations_kernel.cu  
    }      
    ZDerivCovering(gradpar_tmp, qps_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "abs", plan_covering);
  }
  else {
    ZDerivCovering(gradpar_tmp, QparOld, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "abs",plan_covering);
  }  
  add_scaled <<<dimGrid, dimBlock>>> (qpar_field, 1., qpar_field, s.vt*sqrt(2)*D_par, gradpar_tmp);  
  // + vt*sqrt(2)*D_par*|gradpar|(Qpar - Qpar0) 
}
  //step
  add_scaled <<<dimGrid, dimBlock>>> (QparNew, 1., Qpar, -dt, qpar_field);

  //QparNew = Qpar - dt * [ {phi_u, Qpar} + vt*(3+Beta_par)*gradpar(Tpar) + vt*sqrt(2)*D_par*|gradpar|(Qpar - Qpar0)  
  //                - iOmegaD*( (-3+nu6.y)*Qpar + (-3+nu7.y)*Qprp + (6+nu5.y)*Upar ) + |omegaD|*(nu5.x*Upar + nu6.x*Qpar + nu7.x*Qprp) + nu_ss*Qpar ]
  
  ////////////////////////////////////////
  //QPERP
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("qprp",2);
#endif
  
  cudaMemset(qprp_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 

  if(higher_order_moments) {
    //replace_ky0_nopad<<<dimGrid,dimBlock>>>(qprp_field, ev_hd->hybrid.tpar[0]); // this is rprpprp

    replace_ky0_nopad<<<dimGrid,dimBlock>>>(bgrad_tmp, ev_hd->hybrid.upar[0]); // this is rparprp
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(qprp_field, ev_hd->hybrid.tpar[0]); // this is rprpprp
    add_scaled<<<dimGrid,dimBlock>>>(qprp_field, -2., bgrad_tmp, 1., qprp_field, -1., TparOld, 0., DensOld, 0., TprpOld);    

    replace_ky0_nopad<<<dimGrid,dimBlock>>>(omegaD_tmp, ev_hd->hybrid.qpar[0]); // this is sparprp
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, ev_hd->hybrid.qprp[0]); // this is sprpprp
  }
  qprp_linear_terms<<<dimGrid,dimBlock>>>(qprp_field, Phi, AparOld, AparNew, DensOld, UparOld, TparOld, TprpOld, QparOld, QprpOld,
                                kx, ky, shat, s.rho, s.vt, s.tprim, s.fprim, s.zt, bgrad,
                                gds2, gds21, gds22, bmagInv,
                                gbdrift, gbdrift0, cvdrift, cvdrift0,
                                s.nu_ss, nu[8], nu[9], nu[10], mu[8], mu[9], mu[10], varenna,
                                qprp_field, omegaD_tmp, ev_hd->tmp.CXYZ2, higher_order_moments, ev_h->pars.fapar, dt);
/*
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
  absOmegaD <<<dimGrid, dimBlock>>> (omegaD_tmp, sum_tmp, s.rho, s.vt, kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0,varenna,new_varenna);
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, 1., omegaD_tmp, s.nu_ss, QprpOld);
  // + |omegaD|*(nu8.x*Upar + nu9.x*Qpar + nu10.x*Qprp) + nu_ss*Qprp
*/

  if(init == FORCE) {
    phi_qperpb_force <<<dimGrid, dimBlock>>> (phi_tmp, phiext, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    multZ<<<dimGrid,dimBlock>>>(bgrad_tmp,phi_tmp,bgrad);
    add_scaled<<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt, bgrad_tmp);
  } 
  
if(higher_order_moments) {
  phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  ZDerivCovering(gradpar_tmp, phi_tmp, &ev_h->grids, &ev_hd->grids, &ev_d->grids, "", plan_covering);
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.zt*s.vt, gradpar_tmp);

  add_scaled<<<dimGrid,dimBlock>>>(sum_tmp, 1., DensOld, 1., TparOld);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, sum_tmp, bmagInv);
  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids,"",plan_covering);  
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, -s.vt, B_gradpar_tmp);  
  
  //replace_ky0_nopad<<<dimGrid,dimBlock>>>(gradpar_tmp, ev_hd->hybrid.upar[0]); // this is rparprp
  add_scaled<<<dimGrid,dimBlock>>>(gradpar_tmp, 1., DensOld, 1., TparOld, 1., TprpOld);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, gradpar_tmp, bmagInv);
  multZ <<<dimGrid, dimBlock>>> (fields_over_B2_tmp, fields_over_B_tmp, bmagInv);
  ZDerivCovering(gradpar_tmp, fields_over_B2_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "",plan_covering);
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
  multZ <<<dimGrid, dimBlock>>> (B2_gradpar_tmp, B_gradpar_tmp, bmag);
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt, B2_gradpar_tmp);

  //replace_ky0_nopad<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, ev_hd->hybrid.upar[0]); // this is rparprp
  //add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, s.zt, phi_tmp, 1., ev_hd->tmp.CXYZ2, -1., DensOld, -1., TparOld);
  //add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, s.zt, phi_tmp, 1., TprpOld, 0., TparOld);
} else {
  phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, s.zt, phi_tmp, 1., TprpOld);  
  ZDerivCovering(gradpar_tmp, sum_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids,"",plan_covering);  
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt, gradpar_tmp);
}
  
  if(init == FORCE) {
    phi_flr_force <<<dimGrid, dimBlock>>> (phi_tmp, phiext, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    ZDerivCovering(gradpar_tmp, phi_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids,"",plan_covering);  
    add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt, gradpar_tmp);
  } 
  // + vt*gradpar( zt*phi_flr + Tprp )

if(!higher_order_moments) {
  if(varenna && (abs(ivarenna)==1 || abs(ivarenna)==3) && eps!=0. && varenna_fsa==true) {
    if(ivarenna>0) {
      volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, TprpOld, jacobian, 1./fluxDen);
      ps_fac = 1.;
      PfirschSchluter_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, ps_fac, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
    }
    if(ivarenna<0) {
      //volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, Phi, jacobian, 1./fluxDen);
      //ps_fac = -shaping_ps*pow(eps,1.5)*1.;
      //PfirschSchluter_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, ps_fac, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
      add_scaled<<<dimGrid,dimBlock>>>(qps_tmp, 1., DensOld, 1., TprpOld);
      volflux_zonal_complex<<<dimGrid,dimBlock>>>(fluxsurfavg_CtmpX, qps_tmp, jacobian, 1./fluxDen);
      new_varenna_zf_fsa<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, kx, gds22, qsf, eps, bmagInv, fluxsurfavg_CtmpX, shat, s.rho);  //defined in operations_kernel.cu  
    }      
    ZDerivCovering(gradpar_tmp, qps_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "abs", plan_covering);
  }
  else if(varenna && (abs(ivarenna)==1 || abs(ivarenna)==3) && eps!=0. && varenna_fsa==false) {
    if(ivarenna>0) {
      ps_fac = 1.;
      PfirschSchluter<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, ps_fac, kx, gds22, qsf, eps, bmagInv, TprpOld, shat, s.rho);  //defined in operations_kernel.cu  
    }
    if(ivarenna<0) {
      //ps_fac = -shaping_ps*pow(eps,1.5)*1.;
      //PfirschSchluter<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, ps_fac, kx, gds22, qsf, eps, bmagInv, Phi, shat, s.rho);  //defined in operations_kernel.cu  
      new_varenna_zf<<<dimGrid,dimBlock>>>(qps_tmp, QprpOld, kx, gds22, qsf, eps, bmagInv, TprpOld, shat, s.rho);  //defined in operations_kernel.cu  
    }      
    ZDerivCovering(gradpar_tmp, qps_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "abs", plan_covering);
  }
  else {
    ZDerivCovering(gradpar_tmp, QprpOld, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "abs",plan_covering);
  }
  add_scaled <<<dimGrid, dimBlock>>> (qprp_field, 1., qprp_field, s.vt*sqrt(2)*D_prp, gradpar_tmp);
  // + vt*sqrt(2)*D_prp*|gradpar|(Qprp - Qprp0)
}

  //if(ev_h->pars.fapar > 0.) {
  //  // for electromagnetic ions, the 'upar' and 'qprp' evolved by the code are actually 
  //  // 'upar' = upar + vt*zt*apar_u
  //  // 'qprp' = qprp + vt*zt*apar_flr
  //  // here we add back the apar parts to upar and qprp before advancing 'qprp' and leaving the linear timestep routine
  //  phi_flr <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  //  add_scaled<<<dimGrid,dimBlock>>>(QprpOld, 1., QprpOld, s.vt*s.zt, apar_tmp);

  //  phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  //  add_scaled<<<dimGrid,dimBlock>>>(UparOld, 1., UparOld, s.vt*s.zt, apar_tmp);
  //}
  
  //step
  add_scaled <<<dimGrid, dimBlock>>> (QprpNew, 1., Qprp, -dt, qprp_field);
  
  //QprpNew = Qprp - dt * [ vt*( zt*phi_qperpb + Tprp - Tpar )*bgrad + {phi_u, Qprp} + {phi_flr, Upar} + {phi_flr2, Qprp} + vt*gradpar( zt*phi_flr + Tprp ) 
  //    + vt*sqrt(2)*D_prp*|gradpar|(Qprp-Qprp0) - iOmegaD*( (-1+nu9.y)*Qpar + (-1+nu10.y)*Qprp + (1+nu8.y)*Upar ) + |omegaD|*(nu8.x*Upar + nu9.x*Qpar + nu10.x*Qprp) + nu_ss*Qprp ]

#ifdef PROFILE
POP_RANGE;
POP_RANGE;
#endif

}
//



void nonlinear_timestep(
  int is,
  int first_half_step,
  everything_struct * ev_h,
  everything_struct * ev_hd,
  everything_struct * ev_d) 
{

#ifdef PROFILE
PUSH_RANGE("gryfx nonlinear timestep",4);
#endif
cuComplex *Dens = ev_hd->fields.dens[is];
cuComplex *Upar = ev_hd->fields.upar[is];
cuComplex *Tpar = ev_hd->fields.tpar[is];
cuComplex *Qpar = ev_hd->fields.qpar[is];
cuComplex *Tprp = ev_hd->fields.tprp[is];
cuComplex *Qprp = ev_hd->fields.qprp[is];

cuComplex *Phi;// = ev_hd->fields.phi;
cuComplex *Apar;

#ifdef GS2_zonal
cuComplex* NLdens_ky0_d = ev_hd->hybrid.dens[is];
cuComplex* NLupar_ky0_d = ev_hd->hybrid.upar[is];
cuComplex* NLtpar_ky0_d = ev_hd->hybrid.tpar[is];
cuComplex* NLtprp_ky0_d = ev_hd->hybrid.tprp[is];
cuComplex* NLqpar_ky0_d = ev_hd->hybrid.qpar[is];
cuComplex* NLqprp_ky0_d = ev_hd->hybrid.qprp[is];
#endif

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
cuComplex *apar_tmp = ev_hd->tmp.CXYZ;
cuComplex *nlps_tmp = ev_hd->tmp.CXYZ;

cuComplex *fields_over_B_tmp2 = ev_hd->tmp.CXYZ2;


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
  Apar = ev_hd->fields.apar;
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
  Apar = ev_hd->fields.apar1;
}
float dnlpm = ev_h->pars.dnlpm;
float dnlpm_dens = ev_h->pars.dnlpm_dens;
float dnlpm_tprp = ev_h->pars.dnlpm_tprp;
cuffts_struct * ffts = &ev_h->ffts;
bool nlpm_test = ev_h->pars.nlpm_test;
bool new_nlpm = ev_h->pars.new_nlpm;
bool nlpm_zonal_only = ev_h->pars.nlpm_zonal_only;
bool nlpm_vol_avg = ev_h->pars.nlpm_vol_avg;
bool low_b = ev_h->pars.low_b;
bool low_b_all = ev_h->pars.low_b_all;
int iflr = ev_h->pars.iflr;
bool hammett_nlpm_interference = ev_h->pars.hammett_nlpm_interference;
bool nlpm_abs_sgn = ev_h->pars.nlpm_abs_sgn;


cuComplex mu_nlpm1;
cuComplex mu_nlpm2;
cuComplex mu_nlpm3;
if(iflr == 1) {
mu_nlpm1.x = dnlpm * 1.21;
mu_nlpm1.y = dnlpm * -2.3;
mu_nlpm2.x = dnlpm * -.55;
mu_nlpm2.y = dnlpm * 1.99;
mu_nlpm3.x = dnlpm * .40;
mu_nlpm3.y = dnlpm * -.52;
}
if(iflr == 2) {
mu_nlpm1.x = dnlpm * 1.27;
mu_nlpm1.y = dnlpm * -2.2;
mu_nlpm2.x = dnlpm * -.6;
mu_nlpm2.y = dnlpm * 1.96;
mu_nlpm3.x = dnlpm * .44;
mu_nlpm3.y = dnlpm * -.53;
}
if(iflr == 3) {
mu_nlpm1.x = dnlpm * 1.27;
mu_nlpm1.y = dnlpm * -2.22;
mu_nlpm2.x = dnlpm * -.57;
mu_nlpm2.y = dnlpm * 1.98;
mu_nlpm3.x = dnlpm * .42;
mu_nlpm3.y = dnlpm * -.53;
}
if(!low_b) {
float fac=1.;
if(nlpm_abs_sgn) fac=1.6;
mu_nlpm1.x = dnlpm * fac * 1.307;
mu_nlpm1.y = dnlpm * -2.008;
mu_nlpm2.x = dnlpm * fac * -.546;
mu_nlpm2.y = dnlpm *  2.008;
mu_nlpm3.x = dnlpm * fac * .44;
mu_nlpm3.y = dnlpm * -.53;
//mu_nlpm1.x = dnlpm * 1.27;
//mu_nlpm1.y = dnlpm * -2.13;
//mu_nlpm2.x = dnlpm * -.6;
//mu_nlpm2.y = dnlpm * 1.96;
//mu_nlpm3.x = dnlpm * .44;
//mu_nlpm3.y = dnlpm * -.53;
}

  //float kx2Phi_zf_rms;
  //if(nlpm_cutoff_avg) kx2Phi_zf_rms = kx2Phi_zf_rms_avg;
  //else kx2Phi_zf_rms = kx2Phi_zf_rms_in;

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
  
#ifdef PROFILE
PUSH_RANGE("density",2);
#endif
  cudaMemset(dens_field, 0., sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);  
  cudaMemset(nlps_tmp, 0., sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);  

    if(low_b) phi_u_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);          
    else phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);          
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
    
    if(low_b) phi_flr_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);      
    else phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);      
    NLPS(nlps_tmp, phi_tmp, TprpOld, kx, ky);   
    accum <<<dimGrid, dimBlock>>> (dens_field, nlps_tmp, 1);  
    // +{phi_flr,Tprp}

    if(ev_h->pars.fapar > 0.) {
      if(low_b) phi_flr_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);      
      else phi_flr <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);      
      NLPS(nlps_tmp, apar_tmp, QprpOld, kx, ky);   
      accum <<<dimGrid, dimBlock>>> (dens_field, nlps_tmp, -1);  
      // -{apar_flr,Qprp}

      if(low_b) phi_u_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);          
      else phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);          
      NLPS(nlps_tmp, apar_tmp, UparOld, kx, ky);
      add_scaled<<<dimGrid,dimBlock>>>(dens_field, 1., dens_field, -s.vt, nlps_tmp);
      // -vt*B{apar_u, Upar/B}
    }

#ifdef GS2_zonal
      
      getky0_nopad<<<dimGrid,dimBlock>>>(NLdens_ky0_d, dens_field);

#endif

  //if(secondary_test) scale<<<dimGrid,dimBlock>>>(dens_field, dens_field, NLdensfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (DensNew, 1., Dens, -dt, dens_field);
  
  ////////////////////////////////////////
  //UPAR
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("upar",3);
#endif

  cudaMemset(upar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

    if(ev_h->pars.fapar > 0.) {
      if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, 1., UparOld, -s.vt*s.zt, apar_tmp);
      if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPS(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky);
    } else {
      if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPS(nlps_tmp, phi_tmp, UparOld, kx, ky);
    }
    accum <<<dimGrid, dimBlock>>> (upar_field, nlps_tmp, 1);
    // + {phi_u, Upar}

    if(ev_h->pars.fapar > 0.) {
      if(low_b_all) phi_flr_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_flr <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, 1., QprpOld, -s.vt*s.zt, apar_tmp);
      if(low_b_all) phi_flr_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPS(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky);
    }
    else {
      if(low_b_all) phi_flr_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPS(nlps_tmp, phi_tmp, QprpOld, kx, ky);    
    }
    accum <<<dimGrid, dimBlock>>> (upar_field, nlps_tmp, 1);
    // + {phi_flr, Qprp}

    if(ev_h->pars.fapar > 0.) {
      if(low_b_all) phi_flr_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_flr <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPS(nlps_tmp, apar_tmp, TprpOld, kx, ky);    
      accum <<<dimGrid, dimBlock>>> (upar_field, nlps_tmp, -1);
      // - {apar_flr, Tprp}

      if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, 1., DensOld, 1., TparOld, s.zt, phi_tmp);
      if(low_b) phi_u_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);          
      else phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);          
      NLPS(nlps_tmp, apar_tmp, ev_hd->tmp.CXYZ2, kx, ky);
      add_scaled<<<dimGrid,dimBlock>>>(upar_field, 1., upar_field, -s.vt, nlps_tmp);
      // -vt{apar_u, (Dens+Tpar + s.zt*phi_u}
    }
#ifdef GS2_zonal
  
      getky0_nopad<<<dimGrid,dimBlock>>>(NLupar_ky0_d, upar_field);

#endif

  //if(secondary_test) scale<<<dimGrid,dimBlock>>>(upar_field, upar_field, NLuparfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (UparNew, 1., Upar, -dt, upar_field);

  ////////////////////////////////////////
  //TPAR
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("tpar",4);
#endif

  cudaMemset(tpar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
    if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
    else phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, TparOld, kx, ky);
    accum <<<dimGrid, dimBlock>>> (tpar_field, nlps_tmp, 1);
    // + {phi_u,Tpar}

    if(new_nlpm) {
      if(low_b) phi_flr_low_b<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv,iflr);
      else phi_flr<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      if(nlpm_zonal_only) zonal_only<<<dimGrid,dimBlock>>>(phi_tmp);
      scale<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, TparOld, mu_nlpm3.y); 
      NLPS(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky);
      accum<<<dimGrid,dimBlock>>> (tpar_field, nlps_tmp, 1);

      if(low_b) phi_flr_low_b<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv,iflr);
      else phi_flr<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      if(nlpm_zonal_only) zonal_only<<<dimGrid,dimBlock>>>(phi_tmp);
      scale<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, TparOld, mu_nlpm3.x); 
      NLPS_abs(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky, hammett_nlpm_interference, nlpm_abs_sgn);
      accum<<<dimGrid,dimBlock>>> (tpar_field, nlps_tmp, 1);
      if(nlpm_vol_avg) {
        field_line_avg_xyz<<<dimGrid,dimBlock>>>(nlps_tmp, nlps_tmp, jacobian, 1./fluxDen);
        accum<<<dimGrid,dimBlock>>> (tpar_field, nlps_tmp, -1);
      }
    }

    if(ev_h->pars.fapar > 0.) {
      if(low_b) phi_flr_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);      
      else phi_flr <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);      
      NLPS(nlps_tmp, apar_tmp, QprpOld, kx, ky);   
      accum <<<dimGrid, dimBlock>>> (tpar_field, nlps_tmp, 1);  
      // +{apar_flr,Qprp}

      if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, 1., QparOld, 2., UparOld);
      NLPS(nlps_tmp, apar_tmp, ev_hd->tmp.CXYZ2, kx, ky);
      add_scaled<<<dimGrid,dimBlock>>>(tpar_field, 1., tpar_field, -s.vt, nlps_tmp);
      // - vt*{apar_u, Qpar + 2*Upar}
      
    }

#ifdef GS2_zonal
  
      getky0_nopad<<<dimGrid,dimBlock>>>(NLtpar_ky0_d, tpar_field);

#endif

  //if(secondary_test) scale<<<dimGrid,dimBlock>>>(tpar_field, tpar_field, NLtparfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (TparNew, 1., Tpar, -dt, tpar_field);

  ////////////////////////////////////////
  //TPERP
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("tprp",5);
#endif
  float fac;
  cudaMemset(tprp_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    if(ev_h->pars.dnlpm_tprp > 99 && !new_nlpm) {
      cudaMemcpy(tprp_field, TprpOld, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
      cufftExecC2C(ffts->XYplanZ_C2C, tprp_field, tprp_field, CUFFT_INVERSE);
      cudaMemcpy(nlps_tmp, DensOld, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
      cufftExecC2C(ffts->XYplanZ_C2C, nlps_tmp, nlps_tmp, CUFFT_INVERSE);
      fac = 2.;
      sqr_div_complex<<<dimGrid,dimBlock>>>(tprp_field, tprp_field, nlps_tmp, fac);
      cufftExecC2C(ffts->XYplanZ_C2C, tprp_field, tprp_field, CUFFT_FORWARD);
      scale<<<dimGrid,dimBlock>>>(tprp_field, tprp_field, (float) 1./(Nx*(Ny/2+1)) );
      mask<<<dimGrid,dimBlock>>>(tprp_field);
      reality<<<dimGrid,dimBlock>>>(tprp_field);
      if(low_b) {
        phi_flr_low_b<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      } else {
        phi_flr<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      }
      NLPS(nlps_tmp, phi_tmp, tprp_field, kx, ky);
      cudaMemset(tprp_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      accum<<<dimGrid,dimBlock>>> (tprp_field, nlps_tmp, 1);
    }
      
    if(low_b) phi_u_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
    else phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, TprpOld, kx, ky);    
    accum <<<dimGrid, dimBlock>>> (tprp_field, nlps_tmp, 1);
    // + {phi_u, Tprp}

    if(new_nlpm) {
      if(low_b) phi_flr_low_b<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv,iflr);
      else phi_flr<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      if(nlpm_zonal_only) {
        zonal_only<<<dimGrid,dimBlock>>>(phi_tmp);
        add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, mu_nlpm1.y+mu_nlpm2.y, DensOld, mu_nlpm1.y, TprpOld); 
      } else {
        add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, mu_nlpm1.y+mu_nlpm2.y+1., DensOld, mu_nlpm1.y, TprpOld); 
      }
      NLPS(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky);
      accum<<<dimGrid,dimBlock>>> (tprp_field, nlps_tmp, 1);

      if(low_b) phi_flr_low_b<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv,iflr);
      else phi_flr<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      if(nlpm_zonal_only) zonal_only<<<dimGrid,dimBlock>>>(phi_tmp);
      add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, mu_nlpm1.x+mu_nlpm2.x, DensOld, mu_nlpm1.x, TprpOld); 
      NLPS_abs(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky, hammett_nlpm_interference, nlpm_abs_sgn);
      accum<<<dimGrid,dimBlock>>> (tprp_field, nlps_tmp, 1);
      if(nlpm_vol_avg) {
        field_line_avg_xyz<<<dimGrid,dimBlock>>>(nlps_tmp, nlps_tmp, jacobian, 1./fluxDen);
        accum<<<dimGrid,dimBlock>>> (tprp_field, nlps_tmp, -1);
      }
    }

    if(nlpm_zonal_only || !new_nlpm) { //this term is already included in new_nlpm section above
      if(low_b) phi_flr_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPS(nlps_tmp, phi_tmp, DensOld, kx, ky);    
      accum <<<dimGrid, dimBlock>>> (tprp_field, nlps_tmp, 1);
      // + {phi_flr, Dens}
    }    

    if(low_b) phi_flr2_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
    else phi_flr2 <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    NLPS(nlps_tmp, phi_tmp, TprpOld, kx, ky);    
    add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, 1., nlps_tmp);
    // + {phi_flr2, Tprp}

    if(ev_h->pars.fapar > 0.) {
      if(low_b) phi_flr_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);      
      else phi_flr <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);      
      NLPS(nlps_tmp, apar_tmp, UparOld, kx, ky);   
      accum <<<dimGrid, dimBlock>>> (tprp_field, nlps_tmp, -1);  
      // -{apar_flr,Upar}

      if(low_b) phi_u_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);          
      else phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);          
      NLPS(nlps_tmp, apar_tmp, QparOld, kx, ky);
      add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, -s.vt, nlps_tmp);
      // -vt*B{apar_u, Qpar/B}
    }
    

#ifdef GS2_zonal
  
      getky0_nopad<<<dimGrid,dimBlock>>>(NLtprp_ky0_d, tprp_field);

#endif

  //if(secondary_test) scale<<<dimGrid,dimBlock>>>(tprp_field, tprp_field, NLtprpfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (TprpNew, 1., Tprp, -dt, tprp_field);

  ////////////////////////////////////////
  //QPAR
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("qpar",1);
#endif
  
  cudaMemset(qpar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 

    if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);    
    else phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);    
    NLPS(nlps_tmp, phi_tmp, QparOld, kx, ky);    
    accum <<<dimGrid, dimBlock>>> (qpar_field, nlps_tmp, 1);
    // + {phi_u, Qpar}
 
    if(new_nlpm) {
      if(low_b) phi_flr_low_b<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv,iflr);
      else phi_flr<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      if(nlpm_zonal_only) zonal_only<<<dimGrid,dimBlock>>>(phi_tmp);
      scale<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, QparOld, mu_nlpm3.y); 
      NLPS(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky);
      accum<<<dimGrid,dimBlock>>> (qpar_field, nlps_tmp, 1);

      if(low_b) phi_flr_low_b<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv,iflr);
      else phi_flr<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      if(nlpm_zonal_only) zonal_only<<<dimGrid,dimBlock>>>(phi_tmp);
      scale<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, QparOld, mu_nlpm3.x); 
      NLPS_abs(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky, hammett_nlpm_interference, nlpm_abs_sgn);
      accum<<<dimGrid,dimBlock>>> (qpar_field, nlps_tmp, 1);
      if(nlpm_vol_avg) {
        field_line_avg_xyz<<<dimGrid,dimBlock>>>(nlps_tmp, nlps_tmp, jacobian, 1./fluxDen);
        accum<<<dimGrid,dimBlock>>> (qpar_field, nlps_tmp, -1);
      }
    }

    if(ev_h->pars.fapar > 0.) {
      if(low_b) phi_u_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);          
      else phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);          
      NLPS(nlps_tmp, apar_tmp, TparOld, kx, ky);
      add_scaled<<<dimGrid,dimBlock>>>(tprp_field, 1., tprp_field, -s.vt*(3.+Beta_par), nlps_tmp);
      // -vt*(3.+Beta_par)*{apar_u, Tpar}
    }

#ifdef GS2_zonal
  
      getky0_nopad<<<dimGrid,dimBlock>>>(NLqpar_ky0_d, qpar_field);

#endif

  //if(secondary_test) scale<<<dimGrid,dimBlock>>>(qpar_field, qpar_field, NLqparfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (QparNew, 1., Qpar, -dt, qpar_field);

  ////////////////////////////////////////
  //QPERP
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("qprp",2);
#endif
  
  cudaMemset(qprp_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 
  
    if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);    
    else phi_u <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);    
    NLPS(nlps_tmp, phi_tmp, QprpOld, kx, ky);
    accum <<<dimGrid, dimBlock>>> (qprp_field, nlps_tmp, 1);
    // + {phi_u, Qprp}

    if(new_nlpm) {
      if(ev_h->pars.fapar > 0.) {
        if(low_b_all) phi_flr_low_b<<<dimGrid,dimBlock>>>(apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv,iflr);
        else phi_flr<<<dimGrid,dimBlock>>>(apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        // apar_tmp = apar_flr
        if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (ev_hd->tmp.CXYZ2, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);    
        else phi_u <<<dimGrid, dimBlock>>> (ev_hd->tmp.CXYZ2, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);    
        //ev_hd->tmp.CXYZ2 = apar_u
      }
      if(nlpm_zonal_only) {
        if(ev_h->pars.fapar>0.) add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, mu_nlpm1.y+mu_nlpm2.y, UparOld, -s.vt*s.zt*(mu_nlpm1.y+mu_nlpm2.y), ev_hd->tmp.CXYZ2, 
                         mu_nlpm1.y, QprpOld, -s.vt*s.zt*(mu_nlpm1.y), apar_tmp); 
        else add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, mu_nlpm1.y+mu_nlpm2.y, UparOld, mu_nlpm1.y, QprpOld); 
      } else {
        if(ev_h->pars.fapar>0.) add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, mu_nlpm1.y+mu_nlpm2.y+1., UparOld,
                         -s.vt*s.zt*(mu_nlpm1.y+mu_nlpm2.y+1.), ev_hd->tmp.CXYZ2, 
                         mu_nlpm1.y, QprpOld, -s.vt*s.zt*(mu_nlpm1.y), apar_tmp); 
        else add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, mu_nlpm1.y+mu_nlpm2.y+1., UparOld, mu_nlpm1.y, QprpOld); 
      }
      if(low_b) phi_flr_low_b<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv,iflr);
      else phi_flr<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      if(nlpm_zonal_only) zonal_only<<<dimGrid,dimBlock>>>(phi_tmp);
      NLPS(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky);
      accum<<<dimGrid,dimBlock>>> (qprp_field, nlps_tmp, 1);

      if(ev_h->pars.fapar > 0.) {
        if(low_b_all) phi_flr_low_b<<<dimGrid,dimBlock>>>(apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv,iflr);
        else phi_flr<<<dimGrid,dimBlock>>>(apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (ev_hd->tmp.CXYZ2, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);    
        else phi_u <<<dimGrid, dimBlock>>> (ev_hd->tmp.CXYZ2, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);    
        add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, mu_nlpm1.x+mu_nlpm2.x, UparOld, -s.vt*s.zt*(mu_nlpm1.x+mu_nlpm2.x), ev_hd->tmp.CXYZ2,
                         mu_nlpm1.x, QprpOld, -s.vt*s.zt*(mu_nlpm1.x), apar_tmp); 
      }
      else add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, mu_nlpm1.x+mu_nlpm2.x, UparOld, mu_nlpm1.x, QprpOld); 
      if(low_b) phi_flr_low_b<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv,iflr);
      else phi_flr<<<dimGrid,dimBlock>>>(phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      if(nlpm_zonal_only) zonal_only<<<dimGrid,dimBlock>>>(phi_tmp);
      NLPS_abs(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky, hammett_nlpm_interference, nlpm_abs_sgn);
      accum<<<dimGrid,dimBlock>>> (qprp_field, nlps_tmp, 1);
      if(nlpm_vol_avg) {
        field_line_avg_xyz<<<dimGrid,dimBlock>>>(nlps_tmp, nlps_tmp, jacobian, 1./fluxDen);
        accum<<<dimGrid,dimBlock>>> (qprp_field, nlps_tmp, -1);
      }
    }
    
    if(nlpm_zonal_only || !new_nlpm) { //this term is already included in new_nlpm section above
      if(ev_h->pars.fapar > 0.) {
        if(low_b_all) phi_u_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
        else phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, 1., UparOld, -s.vt*s.zt, apar_tmp);

        if(low_b_all) phi_flr_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
        else phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        NLPS(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky);
      }
      else {
        if(low_b_all) phi_flr_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
        else phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        NLPS(nlps_tmp, phi_tmp, UparOld, kx, ky);
      }
      accum <<<dimGrid, dimBlock>>> (qprp_field, nlps_tmp, 1);
      // + {phi_flr, Upar}
    }  

    if(ev_h->pars.fapar > 0.) {
      if(low_b_all) phi_flr_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_flr <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, 1., QprpOld, -s.vt*s.zt, apar_tmp);

      if(low_b_all) phi_flr2_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_flr2 <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPS(nlps_tmp, phi_tmp, ev_hd->tmp.CXYZ2, kx, ky);
    }
    else {
      if(low_b_all) phi_flr2_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_flr2 <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPS(nlps_tmp, phi_tmp, QprpOld, kx, ky);
    }
    add_scaled<<<dimGrid,dimBlock>>>(qprp_field, 1., qprp_field, 1., nlps_tmp);
    // + {phi_flr2, Qprp}

    if(ev_h->pars.fapar > 0.) {
      phi_qperpb <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      NLPS(nlps_tmp, apar_tmp, TprpOld, kx, ky);    
      add_scaled<<<dimGrid,dimBlock>>>(qprp_field, 1., qprp_field, -1., nlps_tmp);
      // - {apar_flr2 - apar_flr, Tprp}

      if(low_b_all) phi_flr_low_b <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);
      else phi_flr <<<dimGrid, dimBlock>>> (phi_tmp, Phi, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      add_scaled<<<dimGrid,dimBlock>>>(ev_hd->tmp.CXYZ2, 1., TprpOld, s.zt, phi_tmp);
      if(low_b) phi_u_low_b <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv, iflr);          
      else phi_u <<<dimGrid, dimBlock>>> (apar_tmp, Apar, s.rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);          
      NLPS(nlps_tmp, apar_tmp, ev_hd->tmp.CXYZ2, kx, ky);
      add_scaled<<<dimGrid,dimBlock>>>(qprp_field, 1., qprp_field, -s.vt, nlps_tmp);
      // -vt{apar_u, (Tprp + s.zt*phi_flr}
    }
#ifdef GS2_zonal
  
      getky0_nopad<<<dimGrid,dimBlock>>>(NLqprp_ky0_d, qprp_field);

#endif

  //if(secondary_test) scale<<<dimGrid,dimBlock>>>(qprp_field, qprp_field, NLqprpfac);

  //step
  add_scaled <<<dimGrid, dimBlock>>> (QprpNew, 1., Qprp, -dt, qprp_field);

#ifdef PROFILE
POP_RANGE;
POP_RANGE;
#endif

}


void linear_electron_timestep(
  int is,
  int first_half_step,
  everything_struct * ev_h,
  everything_struct * ev_hd,
  everything_struct * ev_d 
)
{
#ifdef PROFILE
PUSH_RANGE("gryfx linear timestep",3);

PUSH_RANGE("setting up",1);
#endif


cuComplex *Dens;
cuComplex *Upar;
cuComplex *Tpar;
cuComplex *Tprp;
cuComplex *Phi;// = ev_hd->fields.phi;
cuComplex *Apar;

cuComplex *DensOld; cuComplex *DensNew;
cuComplex *TparOld; cuComplex *TparNew;
cuComplex *TprpOld; cuComplex *TprpNew;
cuComplex *AparOld; cuComplex *AparNew;


double dt = ev_h->time.dt;

if (first_half_step==1){
  //First half of RK2
  dt = dt/2.0;
  if(!LINEAR){
    Dens = ev_hd->fields.dens1[is];
    Apar = ev_hd->fields.apar1;
    Tpar = ev_hd->fields.tpar1[is];
    Tprp = ev_hd->fields.tprp1[is];
  }
  else {
    Dens = ev_hd->fields.dens[is];
    Apar = ev_hd->fields.apar;
    Tpar = ev_hd->fields.tpar[is];
    Tprp = ev_hd->fields.tprp[is];
  }
  DensOld = ev_hd->fields.dens[is];
  AparOld = ev_hd->fields.apar;
  TparOld = ev_hd->fields.tpar[is];
  TprpOld = ev_hd->fields.tprp[is];
  DensNew = ev_hd->fields.dens1[is];
  AparNew = ev_hd->fields.apar1;
  TparNew = ev_hd->fields.tpar1[is];
  TprpNew = ev_hd->fields.tprp1[is];
  Phi = ev_hd->fields.phi;
  Upar = ev_hd->fields.upar[is];
}
else {
  if(!LINEAR){
    Dens = ev_hd->fields.dens[is];
    Apar = ev_hd->fields.apar;
    Tpar = ev_hd->fields.tpar[is];
    Tprp = ev_hd->fields.tprp[is];
  }
  else {
    Dens = ev_hd->fields.dens[is];
    Apar = ev_hd->fields.apar;
    Tpar = ev_hd->fields.tpar[is];
    Tprp = ev_hd->fields.tprp[is];
  }
  DensNew = ev_hd->fields.dens[is];
  AparNew = ev_hd->fields.apar;
  TparNew = ev_hd->fields.tpar[is];
  TprpNew = ev_hd->fields.tprp[is];

  DensOld = ev_hd->fields.dens1[is];
  AparOld = ev_hd->fields.apar1;
  TparOld = ev_hd->fields.tpar1[is];
  TprpOld = ev_hd->fields.tprp1[is];

  Phi = ev_hd->fields.phi1;
  Upar = ev_hd->fields.upar1[is];
}

//int** kxCover = ev_hd->grids.kxCover;
//int** kyCover = ev_hd->grids.kyCover;
//cuComplex** g_covering = ev_hd->grids.g_covering;
//cuComplex** g_covering_d = ev_d->grids.g_covering;
//float** kz_covering = ev_hd->grids.kz_covering;
specie s = ev_h->pars.species[is];

cuComplex *dens_field = ev_hd->fields.field;
cuComplex *apar_field = ev_hd->fields.field;
cuComplex *tpar_field = ev_hd->fields.field;
cuComplex *tprp_field = ev_hd->fields.field;

cuComplex *phi_tmp = ev_hd->tmp.CXYZ;
cuComplex *apar_tmp = ev_hd->tmp.CXYZ;
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

bool higher_order_moments = ev_h->pars.higher_order_moments;

char filename[500];
  
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

#ifdef PROFILE
POP_RANGE;
#endif

  if(ev_h->pars.snyder_electrons) {
    //Ttot closure
    // Te = Tpar + Tprp = 2 Tpar = 2 Tprp = tprim/ev_h->pars.ti_ov_te omega_star Apar / ( k_par )
    
    electron_temperature_closure<<<dimGrid,dimBlock>>>(gradpar_tmp, AparOld, ky, s.tprim, ev_h->pars.ti_ov_te);
    ZDerivCovering(TparOld, gradpar_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "invert",plan_covering);
    // above gives Te, so Tpar = Tprp = .5 Te
    scale<<<dimGrid,dimBlock>>>(TparOld, TparOld, .5);
    cudaMemcpy(TprpOld, TparOld, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
  }

  ////////////////////////////////////////     
  //ELECTRON DENSITY
#ifdef PROFILE
PUSH_RANGE("electron density",5);
#endif
  
  cudaMemset(dens_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);  
    
  electron_density_linear_terms<<<dimGrid,dimBlock>>>(dens_field, Phi, DensOld, TparOld, TprpOld, 
                                kx, ky, shat, gbdrift, gbdrift0, cvdrift, cvdrift0, ev_h->pars.ti_ov_te, s.fprim);
 
  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, Upar, bmagInv);    
  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "",plan_covering);  
  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag); 
  add_scaled <<<dimGrid, dimBlock>>> (dens_field, 1., dens_field, 1., B_gradpar_tmp); 

  //step
  add_scaled <<<dimGrid, dimBlock>>> (DensNew, 1., Dens, -dt, dens_field);
  
  
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("upar",3);
#endif

  ////////////////////////////////////////
  //ELECTRON MOMENTUM (EVOLVES APAR)

  cudaMemset(apar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
  electron_momentum_linear_terms<<<dimGrid,dimBlock>>>(apar_field, AparOld, TparOld, TprpOld, 
                                kx, ky, shat, s.fprim, bgrad, gds2, gds21, gds22, bmagInv,
                                gbdrift, gbdrift0, cvdrift, cvdrift0, ev_h->pars.ti_ov_te, ev_h->pars.beta, ev_h->pars.nu_ei);

  apar_semi_implicit_gradpar_term<<<dimGrid,dimBlock>>>(gradpar_tmp, Phi, DensOld, DensNew,
                                kx, ky, shat, gds2, gds21, gds22, bmagInv, ev_h->pars.ti_ov_te, ev_hd->pars.species);
  ZDerivCovering(gradpar_tmp, gradpar_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "",plan_covering);
  add_scaled <<<dimGrid, dimBlock>>> (apar_field, 1., apar_field, 1., gradpar_tmp);

  //if(ev_h->pars.me_ov_mi > 0. && ev_h->pars.snyder_electrons) {
  //  // landau damping term    
  //  ZDerivCovering(gradpar_tmp, Upar, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "abs",plan_covering);
  //  add_scaled<<<dimGrid,dimBlock>>>(apar_field, 1., apar_field, sqrt(M_PI*ev_h->pars.me_ov_mi/(2.*ev_h->pars.ti_ov_te)), gradpar_tmp);
  //}
  

  //step
  add_scaled <<<dimGrid, dimBlock>>> (AparNew, 1., Apar, -dt, apar_field);
  
  
 
  if(!ev_h->pars.snyder_electrons) {

  // tpar and tprp evolution to be determined

  }

  ////////////////////////////////////////
  //TPAR
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("tpar",4);
#endif

//  cudaMemset(tpar_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

//  tpar_linear_terms<<<dimGrid,dimBlock>>>(tpar_field, Phi, DensOld, UparOld, TparOld, TprpOld, QprpOld,
//                                kx, ky, shat, s.rho, s.vt, s.tprim, s.fprim, s.zt, bgrad,
//                                gds2, gds21, gds22, bmagInv,
//                                gbdrift, gbdrift0, cvdrift, cvdrift0,
//                                s.nu_ss, nu[1], nu[2], mu[1], mu[2], varenna,
//                                omegaD_tmp, ev_hd->tmp.CXYZ2, higher_order_moments);
//    
//  add_scaled <<<dimGrid, dimBlock>>> (sum_tmp, 1., QparOld, 2., UparOld);
//  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, sum_tmp, bmagInv);
//  ZDerivCovering(gradpar_tmp, fields_over_B_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "",plan_covering);
//  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
//  add_scaled <<<dimGrid, dimBlock>>> (tpar_field, 1., tpar_field, s.vt, B_gradpar_tmp);
//  // + vt*B*gradpar( (Qpar + 2*Upar)/B )
//  
//  //step
//  add_scaled <<<dimGrid, dimBlock>>> (TparNew, 1., Tpar, -dt, tpar_field);
  
  //TparNew = Tpar - dt * [ ( 2*vt*(Qprp + Upar) ) * Bgrad + {phi_u, Tpar} + vt*B*gradpar( (Qpar+2*Upar)/B ) + iOmegaStar*phi_tpar  
  //                 - iOmegaD*( phi_tpard + (6+2*nu1.y)*Tpar + 2*Dens + 2*nu2.y*Tprp ) + |omegaD|*(2*nu1.x*Tpar + 2*nu2.x*Tprp) + (2*nu_ss/3)*(Tpar - Tprp) ]
  
  ////////////////////////////////////////
  //TPERP
  
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("tprp",5);
#endif
//  cudaMemset(tprp_field, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
//  tprp_linear_terms<<<dimGrid,dimBlock>>>(tprp_field, Phi, DensOld, UparOld, TparOld, TprpOld,
//                                kx, ky, shat, s.rho, s.vt, s.tprim, s.fprim, s.zt, bgrad,
//                                gds2, gds21, gds22, bmagInv,
//                                gbdrift, gbdrift0, cvdrift, cvdrift0,
//                                s.nu_ss, nu[3], nu[4], mu[3], mu[4], varenna,
//                                omegaD_tmp, ev_hd->tmp.CXYZ2, higher_order_moments);
//  
//  multZ <<<dimGrid, dimBlock>>> (fields_over_B_tmp, QprpOld, bmagInv);
//  multZ <<<dimGrid, dimBlock>>> (fields_over_B2_tmp, fields_over_B_tmp, bmagInv);
//  ZDerivCovering(gradpar_tmp, fields_over_B2_tmp, &ev_h->grids,&ev_hd->grids,&ev_d->grids, "",plan_covering);
//  multZ <<<dimGrid, dimBlock>>> (B_gradpar_tmp, gradpar_tmp, bmag);
//  multZ <<<dimGrid, dimBlock>>> (B2_gradpar_tmp, B_gradpar_tmp, bmag);
//  add_scaled <<<dimGrid, dimBlock>>> (tprp_field, 1., tprp_field, s.vt, B2_gradpar_tmp);
//  // + vt*B2*gradpar( Qprp/B2 )
//  
//  //step
//  add_scaled <<<dimGrid, dimBlock>>> (TprpNew, 1., Tprp, -dt, tprp_field);
  
  //TprpNew = Tprp - dt * [ {phi_u, Tprp} + {phi_flr, Dens} + {phi_flr2, Tprp} + vt*B2*gradpar((Qprp+Upar)/B2) - vt*B*gradpar( Upar/B) + iOmegaStar*phi_tperp
  //                - iOmegaD*( phi_tperpd + (4+2*nu4.y)*Tprp + Dens + (2*nu3.y)*Tpar ) + |omegaD|*( (2*nu3.x)*Tpar + (2*nu4.x)*Tprp ) + (nu_ss/3)*(Tprp-Tpar) ]
  

#ifdef PROFILE
POP_RANGE;
POP_RANGE;
#endif
}
