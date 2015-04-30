
  /* Host arrays*/
    float * wpfx = ev_h->outs.hflux_by_species;
    float * wpfxAvg = ev_h->outs.hflux_by_species_movav;
    float * pflxAvg = ev_h->outs.hflux_by_species_movav;
    //float * wpfx_sum = ev_h->outs.hflux_by_species_movav;
  /* Device arrays*/
    cuComplex * Phi = ev_hd->fields.phi;
    cuComplex * Phi1 = ev_hd->fields.phi1;

    cuComplex ** Dens = ev_hd->fields.dens;
    cuComplex ** Dens1 = ev_hd->fields.dens1;
    cuComplex ** Upar = ev_hd->fields.upar;
    cuComplex ** Upar1 = ev_hd->fields.upar1;
    cuComplex ** Tpar = ev_hd->fields.tpar;
    cuComplex ** Tpar1 = ev_hd->fields.tpar1;
    cuComplex ** Tprp = ev_hd->fields.tprp;
    cuComplex ** Tprp1 = ev_hd->fields.tprp1;
    cuComplex ** Qpar = ev_hd->fields.qpar;
    cuComplex ** Qpar1 = ev_hd->fields.qpar1;
    cuComplex ** Qprp = ev_hd->fields.qprp;
    cuComplex ** Qprp1 = ev_hd->fields.qprp1;

    cuComplex * field = ev_hd->fields.field;

    // For secondary instability calculation
    cuComplex *phi_fixed = ev_hd->sfixed.phi;
    cuComplex *dens_fixed = ev_hd->sfixed.dens;
    cuComplex *upar_fixed = ev_hd->sfixed.upar;
    cuComplex *tpar_fixed = ev_hd->sfixed.tpar;
    cuComplex *tprp_fixed = ev_hd->sfixed.tprp;
    cuComplex *qpar_fixed = ev_hd->sfixed.qpar;
    cuComplex *qprp_fixed = ev_hd->sfixed.qprp;

    // EGH: Not being used yet but will be
    //float * kx_shift = ev_hd->grids.kx_shift;
    //int * jump = ev_hd->grids.jump;
//
    cuComplex * omega = ev_hd->outs.omega;
    float * Phi2_kxky_sum = ev_hd->outs.phi2_by_mode_movav;
    float * Phi2_zonal_sum = ev_hd->outs.phi2_zonal_by_kx_movav;
    float * wpfxnorm_kxky_sum = ev_hd->outs.hflux_by_mode_movav;
    float * zCorr_sum = ev_hd->outs.par_corr_kydz_movav;
    cuComplex * tmp = ev_hd->tmp.CXYZ;
    float  * tmpX = ev_hd->tmp.X;
    float * tmpX2 = ev_hd->tmp.X2;
    float * tmpY = ev_hd->tmp.Y;
    float * tmpY2 = ev_hd->tmp.Y2;
    float * tmpZ = ev_hd->tmp.Z;
    //cuComplex * CtmpZ_h = ev_h->tmp.CZ;
    float * tmpXY = ev_hd->tmp.XY;
    float * tmpXY2 = ev_hd->tmp.XY2;
    float * tmpXY3 = ev_hd->tmp.XY3;
    float * tmpXY4 = ev_hd->tmp.XY4;
    float * tmpXY_R = ev_hd->tmp.XY_R;
    float * tmpXZ = ev_hd->tmp.XZ;
    float * tmpYZ = ev_hd->tmp.YZ;


    //Some globals

    //Host
    //Device

//
//
