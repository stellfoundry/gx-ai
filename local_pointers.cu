
  /* Host arrays*/
    float * wpfx = ev_h->outs.hflux_by_species;
    float * wpfxAvg = ev_h->outs.hflux_by_species_movav;
    float * pflxAvg = ev_h->outs.hflux_by_species_movav;
    //float * wpfx_sum = ev_h->outs.hflux_by_species_movav;
  /* Device arrays*/
    cuComplex * Phi = ev_hd->fields.phi;
    cuComplex * Phi1 = ev_hd->fields1.phi;

    cuComplex ** Dens = ev_hd->fields.dens;
    //cuComplex ** Upar = ev_hd->fields.upar;
    //cuComplex ** Tpar = ev_hd->fields.tpar;
    cuComplex ** Tprp = ev_hd->fields.tprp;
    //cuComplex ** Qpar = ev_hd->fields.qpar;
    //cuComplex ** Qprp = ev_hd->fields.qprp;
    cuComplex ** Dens1 = ev_hd->fields1.dens;
    //cuComplex ** Upar1 = ev_hd->fields1.upar;
    //cuComplex ** Tpar1 = ev_hd->fields1.tpar;
    cuComplex ** Tprp1 = ev_hd->fields1.tprp;
    //cuComplex ** Qpar1 = ev_hd->fields1.qpar;
    //cuComplex ** Qprp1 = ev_hd->fields1.qprp;

    cuComplex * field_h = ev_h->fields.field;
    cuComplex * field = ev_hd->fields.field;

    // For secondary instability calculation
//    cuComplex *phi_fixed = ev_hd->sfixed.phi;
//    cuComplex *dens_fixed = ev_hd->sfixed.dens;
//    cuComplex *upar_fixed = ev_hd->sfixed.upar;
//    cuComplex *tpar_fixed = ev_hd->sfixed.tpar;
//    cuComplex *tprp_fixed = ev_hd->sfixed.tprp;
//    cuComplex *qpar_fixed = ev_hd->sfixed.qpar;
//    cuComplex *qprp_fixed = ev_hd->sfixed.qprp;

    // EGH: Not being used yet but will be
    //float * kx_shift = ev_hd->grids.kx_shift;
    //int * jump = ev_hd->grids.jump;
//
    //cuComplex * omega = ev_hd->outs.omega;
    //float * Phi2_kxky_sum = ev_hd->outs.phi2_by_mode_movav;
    //float * Phi2_zonal_sum = ev_hd->outs.phi2_zonal_by_kx_movav;
    //float * wpfxnorm_kxky_sum = ev_hd->outs.hflux_by_mode_movav;
    //float * zCorr_sum = ev_hd->outs.par_corr_kydz_movav;
    cuComplex * tmp = ev_hd->tmp.CXYZ;
    //cuComplex * CtmpX = ev_hd->tmp.CX;
    //cuComplex *CtmpXZ = ev_hd->tmp.CXZ;
    //cuComplex * CtmpX2 = ev_hd->tmp.CX;
    //float  * tmpX = ev_hd->tmp.X;
    //float  * tmpX_h = ev_h->tmp.X;
    //float * tmpX2 = ev_hd->tmp.X2;
    //float * tmpY = ev_hd->tmp.Y;
    //float * tmpY_h = ev_h->tmp.Y;
    //float * tmpY2 = ev_hd->tmp.Y2;
    //float * tmpZ = ev_hd->tmp.Z;
    //cuComplex * CtmpZ_h = ev_h->tmp.CZ;
    //float * tmpXY = ev_hd->tmp.XY;
    //float * tmpXY_h = ev_h->tmp.XY;
    //float * tmpXY2 = ev_hd->tmp.XY2;
    //float * tmpXY3 = ev_hd->tmp.XY3;
    //float * tmpXY4 = ev_hd->tmp.XY4;
    //float * tmpXY_R = ev_hd->tmp.XY_R;
    //float * tmpXY_R_h = ev_h->tmp.XY_R;
    float * tmpXZ = ev_hd->tmp.XZ;
    //float * tmpYZ = ev_hd->tmp.YZ;
    //float * tmpYZ_h = ev_h->tmp.YZ;
    //float * tmpXYZ = ev_hd->tmp.XYZ;

    cuComplex * dens_ky0_h = ev_h->hybrid.dens_h;
    cuComplex * upar_ky0_h = ev_h->hybrid.upar_h;
    cuComplex * tpar_ky0_h = ev_h->hybrid.tpar_h;
    cuComplex * tprp_ky0_h = ev_h->hybrid.tprp_h;
    cuComplex * qpar_ky0_h = ev_h->hybrid.qpar_h;
    cuComplex * qprp_ky0_h = ev_h->hybrid.qprp_h;
    cuComplex * phi_ky0_h = ev_h->hybrid.phi;

//    cuComplex ** dens_ky0_d = ev_hd->hybrid.dens;
//    cuComplex ** upar_ky0_d = ev_hd->hybrid.upar;
//    cuComplex ** tpar_ky0_d = ev_hd->hybrid.tpar;
//    cuComplex ** tprp_ky0_d = ev_hd->hybrid.tprp;
//    cuComplex ** qpar_ky0_d = ev_hd->hybrid.qpar;
//    cuComplex ** qprp_ky0_d = ev_hd->hybrid.qprp;
//    cuComplex * phi_ky0_d = ev_hd->hybrid.phi;


    float *nu_nlpm = ev_hd->nlpm.nu;
    float *nu1_nlpm = ev_hd->nlpm.nu1;
    float *nu22_nlpm = ev_hd->nlpm.nu22;
    //cuComplex *nu1_nlpm_complex = ev_hd->nlpm.nu1_complex;
    //cuComplex *nu22_nlpm_complex = ev_hd->nlpm.nu22_complex;

    //float* Dnlpm_d = &ev_d->nlpm.D; // This is a pointer to memory on the device
                                    // because ev_d is on the device
    //float* Phi_zf_kx1_d;

    //float *shear_rate_z = ev_hd->hyper.shear_rate_z;
    //float *shear_rate_z_nz = ev_hd->hyper.shear_rate_z_nz;
    //float *shear_rate_nz = ev_hd->hyper.shear_rate_nz;  

    //Some globals

    //Host
    //Device

//
//
