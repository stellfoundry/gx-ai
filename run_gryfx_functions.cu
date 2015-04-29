
void calculate_additional_geo_arrays(
    int Nz,
    float * kz,
    float * tmpZ,
		input_parameters_struct * pars, 
		cuda_dimensions_struct * cdims, 
		geometry_coefficents_struct * geo_d, 
		geometry_coefficents_struct * geo_h){

  bmagInit <<< cdims->dimGrid,cdims->dimBlock >>> (geo_d->bmag, geo_d->bmagInv);
  if(pars->igeo==0) 
		jacobianInit <<< cdims->dimGrid, cdims->dimBlock >>> (
			geo_d->jacobian,pars->drhodpsi,geo_d->gradpar,geo_d->bmag
			);
  else {
    //EGH I think this is wrong, should be line below
jacobianInit <<< cdims->dimGrid, cdims->dimBlock >>> (
			geo_d->jacobian,pars->drhodpsi,geo_d->gradpar,geo_d->bmag
			);
    //copy from geo output
    //cudaMemcpy(geo_d->jacobian, geo_h->jacobian, sizeof(float)*grids->Nz, cudaMemcpyHostToDevice);
    
    //calculate bgrad
      //calculate bgrad = d/dz ln(B(z)) = 1/B dB/dz
     if(pars->debug) printf("calculating bgrad\n");
    cudaMemset(geo_d->bmag_complex, 0, sizeof(cuComplex)*(Nz/2+1));
    //NB This function also sets bmag_complex
    ZDerivB(geo_d->bgrad, geo_d->bmag, geo_d->bmag_complex, kz);
    multdiv<<<cdims->dimGrid,cdims->dimBlock>>>(geo_d->bgrad, geo_d->bgrad, geo_d->bmagInv, 1, 1, Nz, 1);
		/*
*/
  }  
    
  
  //for flux calculations
  multdiv <<< cdims->dimGrid,cdims->dimBlock >>> (tmpZ, geo_d->jacobian, geo_d->grho,1,1,Nz,1);
  geo_h->fluxDen = geo_d->fluxDen = sumReduc(tmpZ,Nz,false);
	
	/* Globals: to be deleted eventually */
	fluxDen = geo_d->fluxDen;
  if(DEBUG) getError("run_gryfx.cu, after init"); 
}

void initialize_grids(input_parameters_struct * pars, grids_struct * grids, grids_struct * grids_h, cuda_dimensions_struct * cdims){
	cudaMemcpy(grids->z, grids_h->z, sizeof(float)*grids->Nz, cudaMemcpyHostToDevice);
  kInit  <<< cdims->dimGrid, cdims->dimBlock >>> (grids->kx, grids->ky, grids->kz, grids->kx_abs, NO_ZDERIV);

  grids_h->kx_max = (float) ((int)((grids_h->Nx-1)/3))/pars->x0;
  grids_h->ky_max = (float) ((int)((grids_h->Ny-1)/3))/pars->y0;
  grids_h->kperp2_max = pow(grids_h->kx_max,2) + pow(grids_h->ky_max,2);
  grids_h->kx4_max = pow(grids_h->kx_max,4);
  grids_h->ky4_max = pow(grids_h->ky_max,4);
  grids_h->ky_max_Inv = 1. / grids_h->ky_max;
  grids_h->kx4_max_Inv = 1. / grids_h->kx4_max;
  grids_h->kperp4_max_Inv = 1. / pow(grids_h->kperp2_max,2);
  if(pars->debug) printf("kperp4_max_Inv = %f\n", grids_h->kperp4_max_Inv);


  zero <<< cdims->dimGrid,cdims->dimBlock >>> (grids->jump,1,grids->Ny,1);
  zero <<< cdims->dimGrid,cdims->dimBlock >>> (grids->kx_shift,1,grids->Ny,1);
  cudaMemcpy(grids_h->kx,grids->kx, sizeof(float)*grids->Nx, cudaMemcpyDeviceToHost);
  if(pars->debug) getError("after k memcpy 1");
  cudaMemcpy(grids_h->ky,grids->ky, sizeof(float)*grids->Ny_complex, cudaMemcpyDeviceToHost);
  if(pars->debug) getError("after k memcpy 2");

  /* Update Globals */
    kx_max=grids_h->kx_max;
    ky_max=grids_h->ky_max;
    kperp2_max=grids_h->kperp2_max;
    kx4_max=grids_h->kx4_max;
    ky4_max=grids_h->ky4_max;
    ky_max_Inv=grids_h->ky_max_Inv;
    kx4_max_Inv=grids_h->kx4_max_Inv;
    kperp4_max_Inv=grids_h->kperp4_max_Inv;
}

void zero_moving_averages(grids_struct * grids_h, cuda_dimensions_struct * cdims_h, outputs_struct * outs_hd, outputs_struct * outs_h, time_struct * time){
	int Nx = grids_h->Nx;
	int Ny = grids_h->Ny;
	int Nz = grids_h->Nz;
	dim3 dimGrid = cdims_h->dimGrid;
	dim3 dimBlock = cdims_h->dimBlock;
	
	for(int s=0; s<grids_h->Nspecies; s++) {
		outs_h->hflux_by_species_movav[s]= 0.;
	}
	outs_h->expectation_ky_movav= 0.;
	outs_h->expectation_kx_movav= 0.;
	time->dtSum= 0.;
	zero<<<dimGrid,dimBlock>>>(outs_hd->phi2_by_mode_movav, Nx,Ny/2+1,1);
	zero<<<dimGrid,dimBlock>>>(outs_hd->hflux_by_mode_movav, Nx, Ny/2+1, 1);
	zero<<<dimGrid,dimBlock>>>(outs_hd->phi2_zonal_by_kx_movav, Nx, 1,1);
	//zeroC<<<dimGrid,dimBlock>>>(Phi_sum);
	zero<<<dimGrid,dimBlock>>>(outs_hd->par_corr_by_ky_by_deltaz_movav, 1, Ny/2+1, Nz);
}
