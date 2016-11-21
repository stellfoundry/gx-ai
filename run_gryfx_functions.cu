void calculate_additional_geo_arrays(
    int Nz,
    float * kz,
    float * tmpZ,
    input_parameters_struct * pars, 
    cuda_dimensions_struct * cdims, 
    geometry_coefficents_struct * geo_d, 
    geometry_coefficents_struct * geo_h){

  // NRM: for some reason, in certain situations CUDA is having trouble when the same
  // array is both the input and output of a kernel. This was causing problems
  // within ZDerivB, specifically in the zderivb kernel. see ZDerivB.cu for details.
  // for this reason, we need to declare and allocate a new array, dz_bmag_complex. 
  // doing it here should be ok, since this is all initialization and won't affect runtime.
  cuComplex* dz_bmag_complex;
  cudaMalloc((void**) &dz_bmag_complex, sizeof(cuComplex)*(Nz/2+1));

  bmagInit <<< cdims->dimGrid,cdims->dimBlock >>> (geo_d->bmag, geo_d->bmagInv);
  if(pars->igeo==0) 
    jacobianInit <<< cdims->dimGrid, cdims->dimBlock >>> (
        geo_d->jacobian,pars->drhodpsi,geo_d->gradpar,geo_d->bmag
        );
  else {
    //EGH I think this is wrong, should be line below
    //jacobianInit <<< cdims->dimGrid, cdims->dimBlock >>> (
    //			geo_d->jacobian,pars->drhodpsi,geo_d->gradpar,geo_d->bmag
    //			);
    //copy from geo output
    cudaMemcpy(geo_d->jacobian, geo_h->jacobian, sizeof(float)*Nz, cudaMemcpyHostToDevice);

    //calculate bgrad
    //calculate bgrad = d/dz ln(B(z)) = 1/B dB/dz
    if(pars->debug) printf("calculating bgrad\n");
    cudaMemset(geo_d->bmag_complex, 0, sizeof(cuComplex)*(Nz/2+1));
    //NB This function also sets bmag_complex
    ZDerivB(geo_d->bgrad, geo_d->bmag, geo_d->bmag_complex, dz_bmag_complex, kz);
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

  kInit  <<< cdims->dimGrid, cdims->dimBlock >>> (grids->kx, grids->ky, grids->kz, grids->kx_abs, NO_ZDERIV, pars->qsf, pars->shat);

  grids_h->kx_max = (float) ((int)((grids_h->Nx-1)/3))/pars->x0;
  grids_h->ky_max = (float) ((int)((grids_h->Ny-1)/3))/pars->y0;
  grids_h->kperp2_max = pow(grids_h->kx_max,2) + pow(grids_h->ky_max,2);
  grids_h->kx4_max = pow(grids_h->kx_max,4);
  grids_h->ky4_max = pow(grids_h->ky_max,4);
  grids_h->ky_max_Inv = 1. / grids_h->ky_max;
  grids_h->kx4_max_Inv = 1. / grids_h->kx4_max;
  grids_h->kperp4_max_Inv = 1. / pow(grids_h->kperp2_max,2);
  if(pars->debug) printf("kperp4_max_Inv = %f\n", grids_h->kperp4_max_Inv);


  zero <<< cdims->dimGrid,cdims->dimBlock >>> (grids->jump,1,grids->Ny_complex,1);
  zero <<< cdims->dimGrid,cdims->dimBlock >>> (grids->kx_shift,1,grids->Ny_complex,1);
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
          outs_h->hflux_by_species_old[s] = 0.;
          outs_h->hflux_by_species[s] = 0.;
          outs_h->pflux_by_species_movav[s]= 0.;
          outs_h->pflux_by_species_old[s] = 0.;
          outs_h->pflux_by_species[s] = 0.;
	}
	outs_h->expectation_ky_movav= 0.;
	outs_h->expectation_kx_movav= 0.;
	time->dtSum= 0.;
	zero<<<dimGrid,dimBlock>>>(outs_hd->phi2_by_mode_movav, Nx,Ny/2+1,1);
	zero<<<dimGrid,dimBlock>>>(outs_hd->hflux_by_mode_movav, Nx, Ny/2+1, 1);
	zero<<<dimGrid,dimBlock>>>(outs_hd->phi2_zonal_by_kx_movav, Nx, 1,1);
	//zeroC<<<dimGrid,dimBlock>>>(Phi_sum);
	zero<<<dimGrid,dimBlock>>>(outs_hd->par_corr_kydz_movav, 1, Ny/2+1, Nz);
  for (int i=0; i < grids_h->NxNyc; i++){
    outs_h->omega_out[i].x = 0.;
    outs_h->omega_out[i].y = 0.;
  }
}

void create_cufft_plans(grids_struct * grids, cuffts_struct * ffts){
  //set up plans for NLPS, ZDeriv, and ZDerivB
  //plan for ZDerivCovering done below
  int NLPSfftdims[2] = {grids->Nx, grids->Ny};
  int XYcfftdims[2] = {grids->Nx, grids->Ny_complex};
  cufftPlanMany(&ffts->NLPSplanR2C, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, grids->Nz);
  cufftPlanMany(&ffts->NLPSplanC2R, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, grids->Nz);
  cufftPlan1d(&ffts->ZDerivBplanR2C, grids->Nz, CUFFT_R2C, 1);
  cufftPlan1d(&ffts->ZDerivBplanC2R, grids->Nz, CUFFT_C2R, 1);  
  cufftPlan2d(&ffts->XYplanC2R, grids->Nx, grids->Ny, CUFFT_C2R);  //for diagnostics
  cufftPlan2d(&ffts->XYplanC2C, grids->Nx, grids->Ny/2+1, CUFFT_C2C);  //for diagnostics
  cufftPlanMany(&ffts->XYplanZ_C2C, 2, XYcfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, grids->Nz);

  int n[1] = {grids->Nz};
  int inembed[1] = {grids->NxNycNz};
  int onembed[1] = {grids->NxNycNz};
  // (ky, kx, z) <-> (ky, kx, kz)
  cufftPlanMany(&ffts->ZDerivplan, 1,   n, inembed, grids->NxNyc, 1,
                              //   dim, n, isize,   istride,      idist,
                                onembed, grids->NxNyc, 1,     CUFFT_C2C, grids->NxNyc);	
                              //osize,   ostride,      odist, type,      batchsize
  // isize = size of input data
  // istride = distance between two elements in a batch = distance between (ky,kx,z=1) and (ky,kx,z=2) = Nx*(Ny/2+1)
  // idist = distance between first element of consecutive batches = distance between (ky=1,kx=1,z=1) and (ky=2,kx=1,z=1) = 1

  int n1[1] = {grids->Nx};
  int inembed1[1] = {grids->NxNycNz};
  int onembed1[1] = {grids->NxNycNz};
  // (ky, kx, z) <-> (ky, x, z)
  cufftPlanMany(&ffts->XplanC2C, 1,   n1, inembed1, grids->Ny_complex, 1,
                              // dim, n,  isize,    istride,           idist,
                                onembed1, grids->Ny_complex, 1,     CUFFT_C2C, grids->Ny_complex*grids->Nz);	
                              //osize,    ostride,           odist, type,      batchsize

	/* Globals... to be removed eventually*/
	NLPSplanR2C = ffts->NLPSplanR2C;
	NLPSplanC2R = ffts->NLPSplanC2R;
	ZDerivBplanR2C = ffts->ZDerivBplanR2C;
	ZDerivBplanC2R = ffts->ZDerivBplanC2R;
	XYplanC2R = ffts->XYplanC2R;
	ZDerivplan = ffts->ZDerivplan;
  if(DEBUG) getError("after plan");
}
void initialize_z_covering(int iproc, grids_struct * grids_hd, grids_struct * grids_h, grids_struct * grids_d, 
                           input_parameters_struct * pars, cuffts_struct * ffts_h, cuda_streams_struct * streams, 
                           cuda_dimensions_struct * cdims, cuda_events_struct * events){
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
	grids_hd->nClasses = grids_h->nClasses;

  
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

  cudaMalloc((void**) &grids_hd->nLinks, sizeof(int)*ncls);
  cudaMalloc((void**) &grids_hd->nChains, sizeof(int)*ncls);

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

  grids_h->covering_scaler = (float*) malloc(sizeof(float)*ncls);
  cudaMalloc((void**) &grids_hd->covering_scaler, sizeof(float)*ncls);

  for(int c=0; c<ncls; c++) {
    grids_h->covering_scaler[c] = (float) 1./(Nz*grids_h->nLinks[c]);
  }

  cudaMemcpy(grids_hd->nLinks,grids_h->nLinks,sizeof(int)*ncls, cudaMemcpyHostToDevice);
  cudaMemcpy(grids_hd->nChains,grids_h->nChains,sizeof(int)*ncls, cudaMemcpyHostToDevice);
  cudaMemcpy(grids_hd->covering_scaler,grids_h->covering_scaler,sizeof(float)*ncls, cudaMemcpyHostToDevice);

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
  

  grids_hd->kxCover = (int**) malloc(sizeof(int*)*ncls);
  grids_hd->kyCover = (int**) malloc(sizeof(int*)*ncls);
  grids_hd->g_covering = (cuComplex **)malloc(sizeof(cuComplex*)*ncls);
  grids_hd->kz_covering = (float **)malloc(sizeof(float*)*ncls);

  if(DEBUG) getError("A");

  //these pointer arrays allocated on device, will point to same memory as grids_hd->***.
  cudaMalloc((void**) &grids_hd->kxCover_d, sizeof(int*)*ncls); 
  cudaMalloc((void**) &grids_hd->kyCover_d, sizeof(int*)*ncls); 
  cudaMalloc((void**) &grids_hd->g_covering_d, sizeof(cuComplex*)*ncls); 
  cudaMalloc((void**) &grids_hd->kz_covering_d, sizeof(float*)*ncls); 

	/* Declare two local pointers for brevity */
  //cuComplex ** g_covering;
  //float ** kz_covering;

	//g_covering = grids_d->g_covering = (cuComplex **)malloc(sizeof(cuComplex*)*ncls);
	//kz_covering = grids_d->kz_covering = (cuComplex **)malloc(sizeof(cuComplex*)*ncls);
  if(DEBUG) getError("run_gryfx.cu, before kCover");
        

  //cufftHandle plan_covering[nClasses];
	ffts_h->plan_covering = (cufftHandle *)malloc(sizeof(cufftHandle)*ncls);

  int max_x = 0;
  int max_y = 0;
  int max_z = 0;
  for(int c=0; c<ncls; c++) {    
    int n[1] = {nlinks[c]*Nz};
    cudaStreamCreate(&(streams->zstreams[c]));
    cufftPlanMany(&ffts_h->plan_covering[c],1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,nchains[c]);
    //cufftSetStream(ffts_h->plan_covering[c], streams->zstreams[c]);

    cdims->dimGridCovering[c].x = (Nz+cdims->dimBlockCovering.x-1)/cdims->dimBlockCovering.x;
    cdims->dimGridCovering[c].y = (grids_h->nChains[c]+cdims->dimBlockCovering.y-1)/cdims->dimBlockCovering.y;
    cdims->dimGridCovering[c].z = (grids_h->nLinks[c]*icovering+dimBlockCovering.z-1)/dimBlockCovering.z;
    max_x = cdims->dimGridCovering[c].x > max_x ? cdims->dimGridCovering[c].x : max_x;
    max_y = cdims->dimGridCovering[c].y > max_y ? cdims->dimGridCovering[c].y : max_y;
    max_z = cdims->dimGridCovering[c].z > max_z ? cdims->dimGridCovering[c].z : max_z;
    if(pars->debug) kPrint(grids_h->nLinks[c], grids_h->nChains[c], grids_h->kyCover[c], grids_h->kxCover[c]); 
  
    //kPrint(nLinks[c], nChains[c], kyCover_h[c], kxCover_h[c]); 
    cudaMalloc((void**) &grids_hd->g_covering[c], sizeof(cuComplex)*Nz*nlinks[c]*nchains[c]);
    cudaMalloc((void**) &grids_hd->kz_covering[c], sizeof(float)*Nz*nlinks[c]);
    cudaMalloc((void**) &grids_hd->kxCover[c], sizeof(int)*nlinks[c]*nchains[c]);
    cudaMalloc((void**) &grids_hd->kyCover[c], sizeof(int)*nlinks[c]*nchains[c]);    
    cudaMemcpy(grids_hd->kxCover[c], grids_h->kxCover[c], sizeof(int)*nlinks[c]*nchains[c], cudaMemcpyHostToDevice);
    cudaMemcpy(grids_hd->kyCover[c], grids_h->kyCover[c], sizeof(int)*nlinks[c]*nchains[c], cudaMemcpyHostToDevice);    

    kzInitCovering<<<cdims->dimGridCovering[c],cdims->dimBlockCovering>>>(grids_hd->kz_covering[c], nlinks[c],NO_ZDERIV_COVERING,icovering, qsf, shat);
  }    
  cdims->dimGridCovering_all.x = max_x;
  cdims->dimGridCovering_all.y = max_y;
  cdims->dimGridCovering_all.z = max_z;
  dimGridCovering = cdims->dimGridCovering;
  dimGridCovering_all = cdims->dimGridCovering_all;

  if(DEBUG) getError("run_gryfx.cu, after kCover");

  cudaMemcpy(grids_hd->g_covering_d, grids_hd->g_covering, sizeof(cuComplex*)*ncls, cudaMemcpyHostToDevice);
  cudaMemcpy(grids_hd->kz_covering_d, grids_hd->kz_covering, sizeof(float*)*ncls, cudaMemcpyHostToDevice);
  cudaMemcpy(grids_hd->kxCover_d, grids_hd->kxCover, sizeof(int*)*ncls, cudaMemcpyHostToDevice);
  cudaMemcpy(grids_hd->kyCover_d, grids_hd->kyCover, sizeof(int*)*ncls, cudaMemcpyHostToDevice);

  
  //printf("nLinks[0] = %d  nChains[0] = %d\n", nLinks[0],nChains[0]);
  

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
	init_h[index].x = 0.;
	init_h[index].y = 0.;
      }
      
      if(pars_h->init_single) {
        //initialize single mode
        int iky = pars_h->iky_single;
        int ikx = pars_h->ikx_single;
        float fac;
        if(pars_h->nlpm_test && iky==0) fac = .5;
        else fac = 1.;
        for(int iz=0; iz<Nz; iz++) {
          int index = iky + (Ny/2+1)*ikx + (Ny/2+1)*Nx*iz;
          init_h[index].x = init_amp*fac;
          init_h[index].y = 0.; //init_amp;
        }
      }
      else {
        printf("init_amp = %e\n", init_amp);
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
  		  init_h[index].x = samp*cos(pars_h->kpar_init*z_h[k]/Zp);
  	          init_h[index].y = samp*cos(pars_h->kpar_init*z_h[k]/Zp);
  	      }
  	      
  	      
  	        
  	      
  	  }
  	}
      }

    for(int s=0; s<nSpecies; s++) {
      zeroC <<< dimGrid, dimBlock >>> (fields_hd->dens1[s]);
      zeroC <<< dimGrid,dimBlock >>> (fields_hd->dens[s]);

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
    zeroC <<< dimGrid,dimBlock >>> (fields_hd->phi);
    zeroC <<< dimGrid,dimBlock >>> (fields_hd->apar1);
    zeroC <<< dimGrid,dimBlock >>> (fields_hd->apar);
    if(DEBUG) getError("run_gryfx.cu, after zero");

    if(pars_h->init == DENS) {
      if(pars_h->debug) getError("initializing density");    
      
      for(int s=0; s<nSpecies; s++) {
        cudaMemcpy(fields_hd->dens[s], init_h, sizeof(cuComplex)*NxNycNz, cudaMemcpyHostToDevice);

        //enforce reality condition -- this is CRUCIAL when initializing in k-space
        reality <<< dimGrid,dimBlock >>> (fields_hd->dens[s]);
      
        mask<<< dimGrid, dimBlock >>>(fields_hd->dens[s]);  
      }
    }
    
    if(pars_h->init == UPAR) {
      getError("initializing upar");    
      
      cudaMemcpy(fields_hd->upar[ION], init_h, sizeof(cuComplex)*NxNycNz, cudaMemcpyHostToDevice);
      if(DEBUG) getError("after copy");    

      //enforce reality condition -- this is CRUCIAL when initializing in k-space
      reality <<< dimGrid,dimBlock >>> (fields_hd->upar[ION]);
      
      mask<<< dimGrid, dimBlock >>>(fields_hd->upar[ION]);  

    }
    if(pars_h->init == TPAR) {
      getError("initializing tpar");    
      
      cudaMemcpy(fields_hd->tpar[ION], init_h, sizeof(cuComplex)*NxNycNz, cudaMemcpyHostToDevice);
      if(DEBUG) getError("after copy");    

      //enforce reality condition -- this is CRUCIAL when initializing in k-space
      reality <<< dimGrid,dimBlock >>> (fields_hd->tpar[ION]);
      
      mask<<< dimGrid, dimBlock >>>(fields_hd->tpar[ION]);  

    }
    if(pars_h->init == TPRP) {
      getError("initializing tperp");    
      
      cudaMemcpy(fields_hd->tprp[ION], init_h, sizeof(cuComplex)*NxNycNz, cudaMemcpyHostToDevice);
      if(DEBUG) getError("after copy");    

      //enforce reality condition -- this is CRUCIAL when initializing in k-space
      reality <<< dimGrid,dimBlock >>> (fields_hd->tprp[ION]);
      
      mask<<< dimGrid, dimBlock >>>(fields_hd->tprp[ION]);  

    }
    if(pars_h->init == ODD) {
      getError("initializing odd");    
      
      cudaMemcpy(fields_hd->upar[ION], init_h, sizeof(cuComplex)*NxNycNz, cudaMemcpyHostToDevice);
      cudaMemcpy(fields_hd->qpar[ION], init_h, sizeof(cuComplex)*NxNycNz, cudaMemcpyHostToDevice);
      //cudaMemcpy(fields_hd->qprp[ION], init_h, sizeof(cuComplex)*NxNycNz, cudaMemcpyHostToDevice);
      if(DEBUG) getError("after copy");    

      scale<<<dimGrid,dimBlock>>>(fields_hd->qpar[ION],fields_hd->qpar[ION],2.);

      //enforce reality condition -- this is CRUCIAL when initializing in k-space
      reality <<< dimGrid,dimBlock >>> (fields_hd->upar[ION]);
      reality <<< dimGrid,dimBlock >>> (fields_hd->qpar[ION]);
      //reality <<< dimGrid,dimBlock >>> (fields_hd->qprp[ION]);
      
      mask<<< dimGrid, dimBlock >>>(fields_hd->upar[ION]);  
      mask<<< dimGrid, dimBlock >>>(fields_hd->qpar[ION]);  
      //mask<<< dimGrid, dimBlock >>>(fields_hd->qprp[ION]);  

    }
    if(pars_h->init == PHI) {
      
      
      cudaMemcpy(fields_hd->phi, init_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
    
      mask<<<dimGrid,dimBlock>>>(fields_hd->phi);

      reality<<<dimGrid,dimBlock>>>(fields_hd->phi);
    }  
   

     
    
//    if(pars_h->init == DENS) {
//      // Solve for initial phi
//      // assumes the initial conditions have been moved to the device
//      if(nSpecies!=1) { 
//	qneut <<< dimGrid,dimBlock >>> (fields_hd->phi, fields_hd->dens[ELECTRON], fields_hd->dens[ION], fields_hd->tprp[ION], pars_h->species[ION].rho, pars_d, grids_d, geo_d); 
//      } else if(ELECTRON == 0) { 
//	qneut <<< dimGrid,dimBlock >>> (fields_hd->phi, pars_h->ti_ov_te, fields_hd->dens[ELECTRON], fields_hd->tprp[ELECTRON], pars_h->species[ELECTRON].rho, pars_d, grids_d, geo_d);
//      } else if(ION == 0) {
//	//qneut <<< dimGrid,dimBlock >>> (Phi, ti_ov_te, Dens[ION], Tprp[ION], species[ION].rho, kx, ky,  gds2, gds21, gds22, bmagInv);
//	qneutAdiab_part1 <<< dimGrid,dimBlock >>> (tmp->CXYZ, fields_hd->field, ti_ov_te, fields_hd->dens[ION], fields_hd->tprp[ION], pars_h->species[ION].rho, pars_d, grids_d, geo_d);
//	qneutAdiab_part2 <<< dimGrid,dimBlock >>> (fields_hd->phi, tmp->CXYZ, fields_hd->field, ti_ov_te, fields_hd->dens[ION], fields_hd->tprp[ION], pars_h->species[ION].rho, pars_d, grids_d, geo_d);
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
        qneut(fields_hd->phi, fields_hd->apar, fields_hd->dens, fields_hd->tprp, fields_hd->upar, fields_hd->qprp,
          tmp->CXYZ, tmp->CXYZ, fields_hd->field, tmp->CXYZ, fields_hd->field, species, pars_d->species, pars_h);
          //pars_h->adiabatic_electrons, pars_h->fapar, pars_h->beta, pars_h->snyder_electrons);
  
      }
      
   
      if(DEBUG) getError("after initial qneut");
}

void load_fixed_arrays_from_restart(
      int Nz,
      cuComplex * CtmpZ_h,
      input_parameters_struct * pars,
      secondary_fixed_arrays_struct * sfixed, 
      fields_struct * fields /* fields on device */
)
{
//      restartRead(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi, pflxAvg, wpfxAvg, Phi2_kxky_sum, Phi2_zonal_sum,
//      			zCorr_sum,&outs->expectation_ky_movav, &outs->expectation_kx_movav, &Phi_zf_kx1_avg,
//      			&tm->dtSum, &tm->counter,&tm->runtime,&tm->dt,&tm->totaltimer,secondary_test_restartfileName);
  			
       int iky = sfixed->iky_fixed;
       int ikx = sfixed->ikx_fixed;
       printf("fixed mode: iky=%d, ikx=%d\n", iky, ikx);

       get_fixed_mode<<<dimGrid,dimBlock>>>(sfixed->phi, fields->phi, iky, ikx);
       get_fixed_mode<<<dimGrid,dimBlock>>>(sfixed->dens, fields->dens[ION], iky, ikx);
       get_fixed_mode<<<dimGrid,dimBlock>>>(sfixed->upar, fields->upar[ION], iky, ikx);
       get_fixed_mode<<<dimGrid,dimBlock>>>(sfixed->tpar, fields->tpar[ION], iky, ikx);
       get_fixed_mode<<<dimGrid,dimBlock>>>(sfixed->tprp, fields->tprp[ION], iky, ikx);
       get_fixed_mode<<<dimGrid,dimBlock>>>(sfixed->qpar, fields->qpar[ION], iky, ikx);
       get_fixed_mode<<<dimGrid,dimBlock>>>(sfixed->qprp, fields->qprp[ION], iky, ikx);

       printf("Before set_fixed_amplitude\n");
       cudaMemcpy(CtmpZ_h, sfixed->phi, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("phi_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }

       cudaMemcpy(CtmpZ_h, sfixed->dens, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("dens_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       if(SLAB) set_fixed_amplitude<<<dimGrid,dimBlock>>>(sfixed->phi, sfixed->dens, sfixed->upar, sfixed->tpar, sfixed->tprp, sfixed->qpar, sfixed->qprp, phi_test);
       else set_fixed_amplitude_withz<<<dimGrid,dimBlock>>>(sfixed->phi, sfixed->dens, sfixed->upar, sfixed->tpar, sfixed->tprp, sfixed->qpar, sfixed->qprp, phi_test);


       printf("After set_fixed_amplitude\n");
       cudaMemcpy(CtmpZ_h, sfixed->phi, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("phi_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }

       cudaMemcpy(CtmpZ_h, sfixed->dens, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("dens_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       cudaMemcpy(CtmpZ_h, sfixed->upar, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("upar_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       cudaMemcpy(CtmpZ_h, sfixed->tpar, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("tpar_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       cudaMemcpy(CtmpZ_h, sfixed->tprp, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("tprp_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       cudaMemcpy(CtmpZ_h, sfixed->qpar, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("qpar_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       cudaMemcpy(CtmpZ_h, sfixed->qprp, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("qprp_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
}

void create_cuda_events_and_streams(cuda_events_struct * events, cuda_streams_struct * streams, int nClasses){
    cudaEventCreate(&events->start);
    cudaEventCreate(&events->stop);		
    cudaEventCreate(&events->nonlin_halfstep);
    cudaEventCreate(&events->H2D);
    cudaEventCreateWithFlags(&events->D2H, cudaEventBlockingSync);
    for(int c=0; c<nClasses; c++) {
      cudaEventCreate(&events->end_of_zderiv[c]); 
    }
    cudaEventCreate(&events->GS2start);
    cudaEventCreate(&events->GS2stop);		

    //cudaEventCreate(&end_of_zderiv);  

    cudaStreamCreate(&streams->copystream);

    //int copystream = 0;
  
    //cudaEventCreate(&start1); 			    
    //cudaEventCreate(&stop1);
    //cudaEventRecord(events->start,0);
    
    //cudaProfilerStart();
}

#ifdef GS2_zonal
void initialize_hybrid_arrays(int iproc,
  grids_struct * grids,
  hybrid_zonal_arrays_struct * hybrid_h,
  hybrid_zonal_arrays_struct * hybrid_d)
{
  int nSpecies = grids->Nspecies;
  int ntheta0 = grids->ntheta0;
  int Nz = grids->Nz;
if(iproc==0) {
    cudaMemset(hybrid_d->phi, 0., sizeof(cuComplex)*ntheta0*Nz);
    for(int s=0; s<nSpecies; s++) {
      cudaMemset(hybrid_d->dens[s], 0., sizeof(cuComplex)*ntheta0*Nz);
      cudaMemset(hybrid_d->upar[s], 0., sizeof(cuComplex)*ntheta0*Nz);
      cudaMemset(hybrid_d->tpar[s], 0., sizeof(cuComplex)*ntheta0*Nz);
      cudaMemset(hybrid_d->tprp[s], 0., sizeof(cuComplex)*ntheta0*Nz);
      cudaMemset(hybrid_d->qpar[s], 0., sizeof(cuComplex)*ntheta0*Nz);
      cudaMemset(hybrid_d->qprp[s], 0., sizeof(cuComplex)*ntheta0*Nz);
    }
}    

    memset(hybrid_h->phi, 0., sizeof(cuComplex)*ntheta0*Nz);
    memset(hybrid_h->dens_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    memset(hybrid_h->upar_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    memset(hybrid_h->tpar_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    memset(hybrid_h->tprp_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    memset(hybrid_h->qpar_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    memset(hybrid_h->qprp_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);

    //set initial condition from GS2 for ky=0 modes
    gryfx_get_gs2_moments(hybrid_h);


}

void copy_hybrid_arrays_from_host_to_device_async(
  grids_struct * grids,
  hybrid_zonal_arrays_struct * hybrid_h,
  hybrid_zonal_arrays_struct * hybrid_d,
  cuda_streams_struct * streams
)
{
#ifdef PROFILE
PUSH_RANGE("copy hyb arrays H2D",0);
#endif
  int nSpecies = grids->Nspecies;
  int ntheta0 = grids->ntheta0;
  int Nz = grids->Nz;
  cudaMemcpyAsync(hybrid_d->phi, hybrid_h->phi, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
  for(int s=0; s<nSpecies; s++) {
    cudaMemcpyAsync(hybrid_d->dens[s], hybrid_h->dens_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
    cudaMemcpyAsync(hybrid_d->upar[s], hybrid_h->upar_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
    cudaMemcpyAsync(hybrid_d->tpar[s], hybrid_h->tpar_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
    cudaMemcpyAsync(hybrid_d->tprp[s], hybrid_h->tprp_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
    cudaMemcpyAsync(hybrid_d->qpar[s], hybrid_h->qpar_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
    cudaMemcpyAsync(hybrid_d->qprp[s], hybrid_h->qprp_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
  }
#ifdef PROFILE
POP_RANGE;
#endif
}

void copy_hybrid_arrays_from_device_to_host_async(
  grids_struct * grids,
  hybrid_zonal_arrays_struct * hybrid_h,
  hybrid_zonal_arrays_struct * hybrid_d,
  cuda_streams_struct * streams
)
{
#ifdef PROFILE
PUSH_RANGE("copy hyb arrays D2H",0);
#endif
  int nSpecies = grids->Nspecies;
  int ntheta0 = grids->ntheta0;
  int Nz = grids->Nz;
  for(int s=0; s<nSpecies; s++) {
    cudaMemcpyAsync(hybrid_h->dens_h + s*ntheta0*Nz, hybrid_d->dens[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
    cudaMemcpyAsync(hybrid_h->upar_h + s*ntheta0*Nz, hybrid_d->upar[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
    cudaMemcpyAsync(hybrid_h->tpar_h + s*ntheta0*Nz, hybrid_d->tpar[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
    cudaMemcpyAsync(hybrid_h->tprp_h + s*ntheta0*Nz, hybrid_d->tprp[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
    cudaMemcpyAsync(hybrid_h->qpar_h + s*ntheta0*Nz, hybrid_d->qpar[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
    cudaMemcpyAsync(hybrid_h->qprp_h + s*ntheta0*Nz, hybrid_d->qprp[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
  }
#ifdef PROFILE
POP_RANGE;
#endif
}
#endif 

void replace_zonal_fields_with_hybrid(
  int first_call,
  cuda_dimensions_struct * cdims,
  fields_struct * fields_d,
  cuComplex * phi_d, //Needs to be passed in separately
  hybrid_zonal_arrays_struct * hybrid_d,
  cuComplex * field_h
)
{
#ifdef PROFILE
PUSH_RANGE("replace zonal fields", 3);
#endif
    char filename[2000];
    dim3 dimGrid = cdims->dimGrid;
    dim3 dimBlock = cdims->dimBlock;

    //replace ky=0 modes with results from GS2
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(phi_d, hybrid_d->phi);
    reality<<<dimGrid,dimBlock>>>(phi_d);
    mask<<<dimGrid,dimBlock>>>(phi_d);

    for(int s=0; s<nSpecies; s++) {
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(fields_d->dens[s], hybrid_d->dens[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(fields_d->upar[s], hybrid_d->upar[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(fields_d->tpar[s], hybrid_d->tpar[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(fields_d->tprp[s], hybrid_d->tprp[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(fields_d->qpar[s], hybrid_d->qpar[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(fields_d->qprp[s], hybrid_d->qprp[s]);
      reality<<<dimGrid,dimBlock>>>(fields_d->dens[s]);
      reality<<<dimGrid,dimBlock>>>(fields_d->upar[s]);
      reality<<<dimGrid,dimBlock>>>(fields_d->tpar[s]);
      reality<<<dimGrid,dimBlock>>>(fields_d->tprp[s]);
      reality<<<dimGrid,dimBlock>>>(fields_d->qpar[s]);
      reality<<<dimGrid,dimBlock>>>(fields_d->qprp[s]);
      mask<<<dimGrid,dimBlock>>>(fields_d->dens[s]);
      mask<<<dimGrid,dimBlock>>>(fields_d->upar[s]);
      mask<<<dimGrid,dimBlock>>>(fields_d->tpar[s]);
      mask<<<dimGrid,dimBlock>>>(fields_d->tprp[s]);
      mask<<<dimGrid,dimBlock>>>(fields_d->qpar[s]);
      mask<<<dimGrid,dimBlock>>>(fields_d->qprp[s]);
    }

    if (first_call==1) {
      fieldWrite(phi_d, field_h, "phi_1.field", filename); 
      getky0_nopad<<<dimGrid,dimBlock>>>(hybrid_d->phi, phi_d);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(phi_d, hybrid_d->phi);
      fieldWrite(phi_d, field_h, "phi_2.field", filename); 
    }
#ifdef PROFILE
POP_RANGE;
#endif
}

void copy_fixed_modes_into_fields(
  cuda_dimensions_struct * cdims,
  fields_struct * fields_d,
  cuComplex * phi_d, //Need  phi separate cos sometimes need dens1 but not phi1 etc
  secondary_fixed_arrays_struct * sfixed,
  input_parameters_struct * pars
    )
{
    dim3 dimGrid = cdims->dimGrid;
    dim3 dimBlock = cdims->dimBlock;

       int iky = sfixed->iky_fixed;
       int ikx = sfixed->ikx_fixed;
    
    if(pars->nlpm_test) {
      cudaMemset(phi_d, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      replace_fixed_mode<<<dimGrid,dimBlock>>>(phi_d, sfixed->phi, iky, ikx, sfixed->S);
      //  replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->dens[ION], sfixed->dens, iky, ikx, 0.);
      //  replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->upar[ION], sfixed->upar, iky, ikx, 0.);
      //  replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->tpar[ION], sfixed->tpar, iky, ikx, 0.);
      //  replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->tprp[ION], sfixed->tprp, iky, ikx, 0.);
      //  replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->qpar[ION], sfixed->qpar, iky, ikx, 0.);
      //  replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->qprp[ION], sfixed->qprp, iky, ikx, 0.);
    } else {
        replace_fixed_mode<<<dimGrid,dimBlock>>>(phi_d, sfixed->phi, iky, ikx, sfixed->S);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->dens[ION], sfixed->dens, iky, ikx, sfixed->S);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->upar[ION], sfixed->upar, iky, ikx, sfixed->S);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->tpar[ION], sfixed->tpar, iky, ikx, sfixed->S);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->tprp[ION], sfixed->tprp, iky, ikx, sfixed->S);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->qpar[ION], sfixed->qpar, iky, ikx, sfixed->S);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(fields_d->qprp[ION], sfixed->qprp, iky, ikx, sfixed->S);
    }

    reality<<<dimGrid,dimBlock>>>(phi_d);
    reality<<<dimGrid,dimBlock>>>(fields_d->dens[ION]);
    reality<<<dimGrid,dimBlock>>>(fields_d->upar[ION]);
    reality<<<dimGrid,dimBlock>>>(fields_d->tpar[ION]);
    reality<<<dimGrid,dimBlock>>>(fields_d->tprp[ION]);
    reality<<<dimGrid,dimBlock>>>(fields_d->qpar[ION]);
    reality<<<dimGrid,dimBlock>>>(fields_d->qprp[ION]);

}

void write_initial_fields(
  cuda_dimensions_struct * cdims,
  fields_struct * fields_d,
  temporary_arrays_struct * tmp_d,
  cuComplex * field_h,
  float * tmpX_h
)
{
    char filename[2000];
    fieldWrite(fields_d->dens[ION], field_h, "dens0.field", filename); 
    fieldWrite(fields_d->upar[ION], field_h, "upar0.field", filename); 
    fieldWrite(fields_d->tpar[ION], field_h, "tpar0.field", filename); 
    fieldWrite(fields_d->tprp[ION], field_h, "tprp0.field", filename); 
    fieldWrite(fields_d->qpar[ION], field_h, "qpar0.field", filename); 
    fieldWrite(fields_d->qprp[ION], field_h, "qprp0.field", filename); 
    fieldWrite(fields_d->phi, field_h, "phi0.field", filename); 
    fieldWrite(fields_d->apar, field_h, "apar0.field", filename); 

  volflux(fields_d->phi,fields_d->phi,tmp_d->CXYZ,tmp_d->XY);
  sumY_neq_0<<<cdims->dimGrid,cdims->dimBlock>>>(tmp_d->X, tmp_d->XY);
  kxWrite(tmp_d->X, tmpX_h, filename, "phi2_0.kx");
}

void update_nlpm_coefficients(
    cuda_dimensions_struct * cdims,
    input_parameters_struct * pars,
    outputs_struct * outs,
    nlpm_struct * nlpm,
    nlpm_struct * nlpm_hd,
    nlpm_struct * nlpm_d,
    cuComplex * Phi,  
    temporary_arrays_struct * tmp_d,
    time_struct * tm
)
{
    //dim3 dimGrid = cdims->dimGrid;
    //dim3 dimBlock = cdims->dimBlock;
    if( strcmp(pars->nlpm_option,"constant") == 0) nlpm->D = pars->dnlpm;
    else cudaMemcpy(&nlpm->D, &nlpm_d->D, sizeof(float), cudaMemcpyDeviceToHost);

    //volflux_zonal(Phi,Phi,tmp_d->X);  //tmp_d->X = Phi_zf**2(kx)
    //get_kx1_rms<<<1,1>>>(&nlpm_d->Phi_zf_kx1, tmp_d->X);
    //nlpm->Phi_zf_kx1_old = nlpm->Phi_zf_kx1;
    //cudaMemcpy(&nlpm->Phi_zf_kx1, &nlpm_d->Phi_zf_kx1, sizeof(float), cudaMemcpyDeviceToHost);
    //
    ////volflux_zonal(Phi,Phi,tmp_d->X);  //tmp_d->X = Phi_zf**2(kx)
    //nlpm->kx2Phi_zf_rms_old = nlpm->kx2Phi_zf_rms;
    //multKx4<<<dimGrid,dimBlock>>>(tmp_d->X2, tmp_d->X, kx); 
    //nlpm->kx2Phi_zf_rms = sumReduc(tmp_d->X2, Nx, false);
    //nlpm->kx2Phi_zf_rms = sqrt(nlpm->kx2Phi_zf_rms);
    //nlpm->nu1_max = maxReduc(nlpm_hd->nu1, Nz, false);
    //nlpm->nu22_max = maxReduc(nlpm_hd->nu22, Nz, false); 
    nlpm->D_sum = nlpm->D_sum*(1.-outs->alpha_avg) + nlpm->D*tm->dt*outs->alpha_avg;

    nlpm->D_avg = nlpm->D_sum/tm->dtSum;
    nlpm->alpha = tm->dt/tau_nlpm;
    nlpm->mu = exp(-nlpm->alpha);
    //if(tm->runtime<20) {
    //  nlpm->Phi_zf_kx1_avg = nlpm->Phi_zf_kx1; //allow a build-up time of tau_nlpm
    //  nlpm->kx2Phi_zf_rms_avg = nlpm->kx2Phi_zf_rms;
    //}
    //else { 
    //  nlpm->Phi_zf_kx1_avg = nlpm->mu*nlpm->Phi_zf_kx1_avg + (1-nlpm->mu)*nlpm->Phi_zf_kx1 + (nlpm->mu - (1-nlpm->mu)/nlpm->alpha)*(nlpm->Phi_zf_kx1 - nlpm->Phi_zf_kx1_old);
    //  nlpm->kx2Phi_zf_rms_avg = nlpm->mu*nlpm->kx2Phi_zf_rms_avg + (1-nlpm->mu)*nlpm->kx2Phi_zf_rms + (nlpm->mu - (1-nlpm->mu)/nlpm->alpha)*(nlpm->kx2Phi_zf_rms - nlpm->kx2Phi_zf_rms_old);
    //}
}

void initialize_nlpm_coefficients(
    cuda_dimensions_struct * cdims,
    nlpm_struct* nlpm_h, 
    nlpm_struct* nlpm_d,
    int Nz
)
{

  dim3 dimGrid = cdims->dimGrid;
  dim3 dimBlock = cdims->dimBlock;
  nlpm_h->Phi_zf_kx1_avg = 0.;
  nlpm_h->Phi_zf_kx1 = 0.;
  nlpm_h->Phi_zf_kx1_old = 0.;
  nlpm_h->D=0.;
  nlpm_h->D_avg=0.;
  nlpm_h->D_sum=0.;
  nlpm_h->alpha=0.;
  nlpm_h->mu=0.;
  // INITIALIZE ARRAYS AS NECESSARY
  zero<<<dimGrid,dimBlock>>>(nlpm_d->nu22, 1, 1, Nz);
  zero<<<dimGrid,dimBlock>>>(nlpm_d->nu1, 1, 1, Nz);
  zero<<<dimGrid,dimBlock>>>(nlpm_d->nu, 1, 1, Nz);
}

void initialize_run_control(run_control_struct * ctrl, grids_struct * grids){
    int Nx = grids->Nx;
    int Ny = grids->Ny;

    for(int i=0; i<Nx*(Ny/2+1); i++) {
      //stability[i].x = 0;
      //stability[i].y = 0;
      ctrl->stable[i] = 0;
      ctrl->stable[i +Nx*(Ny/2+1)] = 0;
    }
    //bool STABLE_STOP=false;  
    ctrl->stable_max = 500;
      
    //float wpfxmax=0.;
    //float wpfxmin=0.;
    ctrl->converge_count=0;
}
void initialize_averaging_parameters(outputs_struct * outs, int navg){
    outs->alpha_avg = (float) 2./(navg+1.);
    outs->mu_avg = exp(-outs->alpha_avg);
}

void initialize_phi_avg_denom(
    cuda_dimensions_struct * cdims,
    input_parameters_struct * pars_h,
    grids_struct * grids_d,
    geometry_coefficents_struct * geo_d,
    specie * species_d,
    float * tmpXZ
    )
{
    cudaMalloc((void**) &PhiAvgDenom, sizeof(float)*grids_d->Nx);
    cudaMemset(PhiAvgDenom, 0, sizeof(float)*grids_d->Nx);
    phiavgdenom<<<cdims->dimGrid,cdims->dimBlock>>>(PhiAvgDenom, tmpXZ, geo_d->jacobian, species_d, grids_d->kx, grids_d->ky, pars_h->shat, geo_d->gds2, geo_d->gds21, geo_d->gds22, geo_d->bmagInv, ti_ov_te);  
}
