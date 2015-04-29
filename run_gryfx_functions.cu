
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

void create_cufft_plans(grids_struct * grids, cuffts_struct * ffts){
  //set up plans for NLPS, ZDeriv, and ZDerivB
  //plan for ZDerivCovering done below
  int NLPSfftdims[2] = {grids->Nx, grids->Ny};
  cufftPlanMany(&ffts->NLPSplanR2C, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, grids->Nz);
  cufftPlanMany(&ffts->NLPSplanC2R, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, grids->Nz);
  cufftPlan1d(&ffts->ZDerivBplanR2C, grids->Nz, CUFFT_R2C, 1);
  cufftPlan1d(&ffts->ZDerivBplanC2R, grids->Nz, CUFFT_C2R, 1);  
  cufftPlan2d(&ffts->XYplanC2R, grids->Nx, grids->Ny, CUFFT_C2R);  //for diagnostics

  int n[1] = {grids->Nz};
  int inembed[1] = {grids->NxNycNz};
  int onembed[1] = {grids->NxNycNz};
  cufftPlanMany(&ffts->ZDerivplan,1,n,inembed,grids->NxNyc,1,
                                onembed,grids->NxNyc,1,CUFFT_C2C,grids->NxNyc);	
                     //    n rank  nembed  stride   dist

	/* Globals... to be removed eventually*/
	NLPSplanR2C = ffts->NLPSplanR2C;
	NLPSplanC2R = ffts->NLPSplanC2R;
	ZDerivBplanR2C = ffts->ZDerivBplanR2C;
	ZDerivBplanC2R = ffts->ZDerivBplanC2R;
	XYplanC2R = ffts->XYplanC2R;
	ZDerivplan = ffts->ZDerivplan;
  if(DEBUG) getError("after plan");
}
void initialize_z_covering(int iproc, grids_struct * grids_d, grids_struct * grids_h, input_parameters_struct * pars, cuffts_struct * ffts_h, cuda_streams_struct * streams, cuda_dimensions_struct * cdims, cuda_events_struct * events){
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //set up kxCover and kyCover for covering space z-transforms
  int naky = grids_h->naky;
  int ntheta0 = grids_h->ntheta0;// nshift;

  int idxRight[naky*ntheta0];
  int idxLeft[naky*ntheta0];

  int linksR[naky*ntheta0];
  int linksL[naky*ntheta0];

  int n_k[naky*ntheta0];

  //Local duplicate for convenience
  int icovering = pars->icovering;

  if(iproc==0) getNClasses(&grids_h->nClasses, idxRight, idxLeft, linksR, linksL, n_k, naky, ntheta0, pars->jtwist);
	grids_d->nClasses = grids_h->nClasses;

  
  if(DEBUG) getError("run_gryfx.cu, after nclasses");

	/* For brevity*/
	int ncls = grids_h->nClasses;
	int Nz = grids_h->Nz;

	/* Declare two local pointers for brevity */
  int *nlinks, *nchains;


	/* Declare two local pointers for brevity */
  int **kxCover_h, **kyCover_h;

  if(iproc==0)  {
    nlinks = grids_h->nLinks = (int*) malloc(sizeof(int)*ncls);
    nchains = grids_h->nChains = (int*) malloc(sizeof(int)*ncls);


    /*Globals - to be deleted*/
    nClasses = grids_h->nClasses;
    nLinks = nlinks;
    nChains = nchains;
    getNLinksChains(grids_h->nLinks, grids_h->nChains, n_k, ncls, naky, ntheta0);
    kxCover_h = grids_h->kxCover = (int**) malloc(sizeof(int*)*ncls);
    kyCover_h = grids_h->kyCover = (int**) malloc(sizeof(int*)*ncls);
    for(int c=0; c<ncls; c++) {   
      kyCover_h[c] = (int*) malloc(sizeof(int)*nlinks[c]*nchains[c]);
      kxCover_h[c] = (int*) malloc(sizeof(int)*nlinks[c]*nchains[c]);
    }  

    kFill(ncls, nchains, nlinks, kyCover_h, kxCover_h, linksL, linksR, idxRight, naky, ntheta0); 
    
    if(DEBUG) getError("run_gryfx.cu, after kFill");

  

  //these are the device arrays... cannot be global because jagged!
  //int *kxCover[nClasses];
  //int *kyCover[nClasses];
  //also set up a stream for each class.
    streams->zstreams = (cudaStream_t*) malloc(sizeof(cudaStream_t)*nClasses);
    events->end_of_zderiv =  (cudaEvent_t*) malloc(sizeof(cudaEvent_t)*nClasses);
    cdims->dimGridCovering = (dim3*) malloc(sizeof(dim3)*nClasses);
    cdims->dimBlockCovering.x = 8;
    cdims->dimBlockCovering.y = 8;
    cdims->dimBlockCovering.z = 8;

  // Update Globals
  zstreams = streams->zstreams ;
  end_of_zderiv = events->end_of_zderiv;
  dimBlockCovering = cdims->dimBlockCovering;
  dimGridCovering = cdims->dimGridCovering;
  

  grids_d->kxCover = (int**) malloc(sizeof(int*)*ncls);
  grids_d->kyCover = (int**) malloc(sizeof(int*)*ncls);

	/* Declare two local pointers for brevity */
  //cuComplex ** g_covering;
  //float ** kz_covering;

	//g_covering = grids_d->g_covering = (cuComplex **)malloc(sizeof(cuComplex*)*ncls);
	//kz_covering = grids_d->kz_covering = (cuComplex **)malloc(sizeof(cuComplex*)*ncls);
	grids_d->g_covering = (cuComplex **)malloc(sizeof(cuComplex*)*ncls);
	grids_d->kz_covering = (float **)malloc(sizeof(float*)*ncls);

  //cufftHandle plan_covering[nClasses];
	ffts_h->plan_covering = (cufftHandle *)malloc(sizeof(cufftHandle)*ncls);
  for(int c=0; c<ncls; c++) {    
    int n[1] = {nlinks[c]*Nz};
    cudaStreamCreate(&(streams->zstreams[c]));
    cufftPlanMany(&ffts_h->plan_covering[c],1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,nchains[c]);

    cdims->dimGridCovering[c].x = (Nz+cdims->dimBlockCovering.x-1)/cdims->dimBlockCovering.x;
    cdims->dimGridCovering[c].y = (grids_h->nChains[c]+cdims->dimBlockCovering.y-1)/cdims->dimBlockCovering.y;
    cdims->dimGridCovering[c].z = (grids_h->nLinks[c]*icovering+dimBlockCovering.z-1)/dimBlockCovering.z;
    if(pars->debug) kPrint(grids_h->nLinks[c], grids_h->nChains[c], grids_h->kyCover[c], grids_h->kxCover[c]); 
  
    //kPrint(nLinks[c], nChains[c], kyCover_h[c], kxCover_h[c]); 
    cudaMalloc((void**) &grids_d->g_covering[c], sizeof(cuComplex)*Nz*nlinks[c]*nchains[c]);
    cudaMalloc((void**) &grids_d->kz_covering[c], sizeof(float)*Nz*nlinks[c]);
    cudaMalloc((void**) &grids_d->kxCover[c], sizeof(int)*nlinks[c]*nchains[c]);
    cudaMalloc((void**) &grids_d->kyCover[c], sizeof(int)*nlinks[c]*nchains[c]);    
    cudaMemcpy(grids_d->kxCover[c], grids_h->kxCover[c], sizeof(int)*nlinks[c]*nchains[c], cudaMemcpyHostToDevice);
    cudaMemcpy(grids_d->kyCover[c], grids_h->kyCover[c], sizeof(int)*nlinks[c]*nchains[c], cudaMemcpyHostToDevice);    
  }    
  //printf("nLinks[0] = %d  nChains[0] = %d\n", nLinks[0],nChains[0]);
  

  if(DEBUG) getError("run_gryfx.cu, after kCover");
  } // end if iproc==0
  ///////////////////////////////////////////////////////////////////////////////////////////////////
}

void set_initial_conditions_no_restart(input_parameters_struct * pars_h, input_parameters_struct * pars_d, grids_struct * grids_h, grids_struct * grids_d, cuda_dimensions_struct * cdims, geometry_coefficents_struct * geo_d, fields_struct * fields_hd, temporary_arrays_struct * tmp){
    
  cuComplex *init_h;
  init_h = (cuComplex*) malloc(sizeof(cuComplex)*grids_h->NxNycNz);
	/*For brevity*/
	int Nx = grids_h->Nx;
	int Nyc = grids_h->Ny_complex;
	int Nz = grids_h->Nz;
	int NxNycNz = grids_h->NxNycNz;
	dim3 dimGrid = cdims->dimGrid;
	dim3 dimBlock = cdims->dimBlock;

      for(int index=0; index<NxNycNz; index++) 
      {
	init_h[index].x = 0;
	init_h[index].y = 0;
      }
      
      srand(22);
      float samp;
  	for(int j=0; j<Nx; j++) {
  	  for(int i=0; i<Nyc; i++) {
		samp = init_amp;
  
  	      float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
  	      float rb = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
  	      //printf("%e\n", ra);
  
  	      //loop over z here to get rid of randomness in z in initial condition
  	      for(int k=0; k<Nz; k++) {
  	        int index = i + (Ny/2+1)*j + (Ny/2+1)*Nx*k;
  		  init_h[index].x = samp;//*cos(1*z_h[k]);
  	          init_h[index].y = samp;//init_amp;//*cos(1*z_h[k]);
  	      }
  	      
  	      
  	        
  	      
  	  }
  	}

    if(pars_h->init == DENS) {
      if(pars_h->debug) getError("initializing density");    

      for(int s=0; s<pars_h->nspec; s++) {
        cudaMemset(fields_hd->dens[s], 0, sizeof(cuComplex)*NxNycNz);
      }
      
      

      cudaMemcpy(fields_hd->dens[ION], init_h, sizeof(cuComplex)*NxNycNz, cudaMemcpyHostToDevice);
      if(DEBUG) getError("after copy");    

      //enforce reality condition -- this is CRUCIAL when initializing in k-space
      reality <<< dimGrid,dimBlock >>> (fields_hd->dens[ION]);
      
      mask<<< dimGrid, dimBlock >>>(fields_hd->dens[ION]);  

      cudaMemset(fields_hd->phi, 0, sizeof(cuComplex)*NxNycNz);
      
    }
    
    if(pars_h->init == PHI) {
      
      cudaMemset(fields_hd->phi, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 
      
      cudaMemcpy(fields_hd->phi, init_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
    
      cudaMemset(fields_hd->dens[ION], 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);  
      mask<<<dimGrid,dimBlock>>>(fields_hd->phi);

      reality<<<dimGrid,dimBlock>>>(fields_hd->phi);
    }  
    if(init == FORCE) {
      cudaMemset(fields_hd->phi, 0, sizeof(cuComplex)*NxNycNz);
      cudaMemset(fields_hd->dens[ION], 0, sizeof(cuComplex)*NxNycNz);
    }   
   

    for(int s=0; s<nSpecies; s++) {
      zeroC <<< dimGrid, dimBlock >>> (fields_hd->dens1[s]);
      if(s!=0) zeroC <<< dimGrid,dimBlock >>> (fields_hd->dens[s]);

      zeroC <<< dimGrid, dimBlock >>> (fields_hd->upar[s]);
      
      zeroC <<< dimGrid, dimBlock >>> (fields_hd->tpar[s]);

      zeroC <<< dimGrid, dimBlock >>> (fields_hd->qpar[s]);

      zeroC <<< dimGrid, dimBlock >>> (fields_hd->tprp[s]);

      zeroC <<< dimGrid, dimBlock >>> (fields_hd->qprp[s]);
      
      zeroC <<< dimGrid, dimBlock >>> (fields_hd->upar1[s]);
      
      zeroC <<< dimGrid, dimBlock >>> (fields_hd->tpar1[s]);

      zeroC <<< dimGrid, dimBlock >>> (fields_hd->qpar1[s]);

      zeroC <<< dimGrid, dimBlock >>> (fields_hd->tprp1[s]);

      zeroC <<< dimGrid, dimBlock >>> (fields_hd->qprp1[s]);
    }
    
    zeroC <<< dimGrid,dimBlock >>> (fields_hd->phi1);
    if(DEBUG) getError("run_gryfx.cu, after zero");
     
    
//    if(pars_h->init == DENS) {
//      // Solve for initial phi
//      // assumes the initial conditions have been moved to the device
//      if(nSpecies!=1) { 
//	qneut <<< dimGrid,dimBlock >>> (fields_hd->phi, fields_hd->dens[ELECTRON], fields_hd->dens[ION], fields_hd->tprp[ION], pars_h->species[ION].rho, pars_d, grids_d, geo_d); 
//      } else if(ELECTRON == 0) { 
//	qneut <<< dimGrid,dimBlock >>> (fields_hd->phi, pars_h->tite, fields_hd->dens[ELECTRON], fields_hd->tprp[ELECTRON], pars_h->species[ELECTRON].rho, pars_d, grids_d, geo_d);
//      } else if(ION == 0) {
//	//qneut <<< dimGrid,dimBlock >>> (Phi, tau, Dens[ION], Tprp[ION], species[ION].rho, kx, ky,  gds2, gds21, gds22, bmagInv);
//	qneutAdiab_part1 <<< dimGrid,dimBlock >>> (tmp->CXYZ, fields_hd->field, tau, fields_hd->dens[ION], fields_hd->tprp[ION], pars_h->species[ION].rho, pars_d, grids_d, geo_d);
//	qneutAdiab_part2 <<< dimGrid,dimBlock >>> (fields_hd->phi, tmp->CXYZ, fields_hd->field, tau, fields_hd->dens[ION], fields_hd->tprp[ION], pars_h->species[ION].rho, pars_d, grids_d, geo_d);
//      }
//    }
 
    
      if(init == RH_equilibrium) {
         
        zeroC<<<dimGrid,dimBlock>>>(fields_hd->dens[0]);
        zeroC<<<dimGrid,dimBlock>>>(fields_hd->phi);
  
        RH_equilibrium_init<<<dimGrid,dimBlock>>>(
          fields_hd->dens[0], fields_hd->upar[0], fields_hd->tpar[0],
          fields_hd->tprp[0], fields_hd->qpar[0], fields_hd->qprp[0],
          grids_d->kx, geo_d->gds22, pars_h->qsf, pars_h->eps,
          geo_d->bmagInv, pars_d->shat, species[0]);
  
        //EGH: Why do we only do an initial qneut for RH_equilibrium?
        qneut(fields_hd->phi, fields_hd->dens, fields_hd->tprp,
          tmp->CXYZ, tmp->CXYZ, fields_hd->field, species, pars_d->species);
  
      }
      
   
      if(DEBUG) getError("after initial qneut");
}
