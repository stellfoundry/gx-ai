
	/* Host arrays*/
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

//		field = ev_hd->fields.field;
//
//		kx_shift = ev_hd->grids.kx_shift;
//		jump = ev_hd->grids.jump;
//
//		omega = ev_hd->outs.omega;
//		Phi2_kxky_sum = ev_hd->outs.phi2_by_mode_movav;
//		Phi2_zonal_sum = ev_hd->outs.phi2_zonal_by_kx_movav;
//		wpfxnorm_kxky_sum = ev_hd->outs.hflux_by_mode_movav;
//		zCorr_sum = ev_hd->outs.par_corr_by_ky_by_deltaz_movav;
//		tmp = ev_hd->tmp.CXYZ;
//		tmpX = ev_hd->tmp.X;
//		tmpX2 = ev_hd->tmp.X2;
//		tmpY = ev_hd->tmp.Y;
//		tmpY2 = ev_hd->tmp.Y2;
//		tmpZ = ev_hd->tmp.Z;
//		tmpXY = ev_hd->tmp.XY;
//		tmpXY2 = ev_hd->tmp.XY2;
//		tmpXY3 = ev_hd->tmp.XY3;
//		tmpXY4 = ev_hd->tmp.XY4;
//		tmpXY_R = ev_hd->tmp.XY_R;
//		tmpXZ = ev_hd->tmp.XZ;
//		tmpYZ = ev_hd->tmp.YZ;
//
//		jacobian = ev_hd->geo.jacobian;
//		bmagInv = ev_hd->geo.bmagInv;
//		bmag_complex = ev_hd->geo.bmag_complex;
//
//
//
//		float * wpfx = ev_h->outs.hflux_by_species;
//		float * wpfx_sum = ev_h->outs.hflux_by_species_movav;
