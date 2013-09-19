void run_gryfx(double * qflux, FILE* outfile)//, FILE* omegafile,FILE* gammafile, FILE* energyfile, FILE* fluxfile, FILE* phikyfile, FILE* phikxfile, FILE* phifile)
{
  //host variables
  
  
  cuComplex *init_h;
  float Phi_energy;
  cuComplex *omega_h;  
  float dtBox[navg];
  cuComplex* omegaAvg_h;
  float wpfx[nSpecies];
  float wpfx_sum[nSpecies];
  float tmpX_h[Nx];
  float tmpY_h[Ny/2+1];
  float tmpXY_h[Nx*(Ny/2+1)];
  float tmpYZ_h[(Ny/2+1)*Nz];
  float tmpXY_R_h[Nx*Ny];
  cuComplex field_h[Nx*(Ny/2+1)*Nz];
  
  char filename[80];
  /*
  float ky_spectrum1_h[Ny/2+1];
  float ky_spectrum2_h[Ny/2+1];
  float ky_spectrum3_h[Ny/2+1];
  float ky0_kx_spectrum_h[Nx];
  float kxky_spectrum1_h[Nx*(Ny/2+1)];  
  float kxky_spectrum2_h[Nx*(Ny/2+1)];
  float kxky_spectrum3_h[Nx*(Ny/2+1)];
  */
  //device variables; main device arrays will be capitalized
  if(DEBUG) getError("run_gryfx.cu, before device alloc");
  cuComplex *Dens[nSpecies];  
  cuComplex *Upar[nSpecies],  *Tpar[nSpecies];
  cuComplex *Qpar[nSpecies],  *Tprp[nSpecies],  *Qprp[nSpecies];
  cuComplex *Dens1[nSpecies], *Upar1[nSpecies], *Tpar1[nSpecies]; 
  cuComplex *Qpar1[nSpecies], *Tprp1[nSpecies], *Qprp1[nSpecies];
  cuComplex *Phi; cuComplex *Phi1; 
  
  //tmps for timestep routine
  cuComplex *field, *tmp;
  float *tmpZ;  
  float *tmpX;
  float *tmpX2;
  float *tmpY;
  float *tmpY2;
  float *tmpXY;
  float *tmpXY2;
  float *tmpXY3;
  float *tmpXY4;
  float *tmpXY_R;
  float *tmpXZ;
  float *tmpYZ;
  cuComplex *CtmpX;
 
  cuComplex *omega;
  cuComplex *omegaBox[navg];
  
  cuComplex *omegaAvg;
  
  //float *Phi2_XYBox[navg];
  
  
  float dt_old;
  float avgdt;
  float totaltimer;
  float timer;
  
  //diagnostics scalars
  float flux1,flux2;
  float Phi2, kPhi2;
  float expectation_ky;
  float expectation_ky_sum;
  float expectation_kx;
  float expectation_kx_sum;
  float dtSum;
  //diagnostics arrays
  float wpfxAvg[nSpecies];
  float *Phi2_kxky_sum;
  float *wpfxnorm_kxky_sum;
  float *Phi2_zonal_sum;
  //cuComplex *Phi_sum;
  float *zCorr_sum;
  
  float *kx_shift;
  int *jump;
  
  float *nu_nlpm;
  
  omega_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)); 
  omegaAvg_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1));
  kx_h = (float*) malloc(sizeof(float)*Nx);
  ky_h = (float*) malloc(sizeof(float)*(Ny/2+1));
  
  //zero dtBox array
  for(int t=0; t<navg; t++) {  dtBox[t] = 0;  }
    
  init_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  //Phi_energy = (float*) malloc(sizeof(float));
  
  cudaEventCreate(&end_of_zderiv); 
  
  for(int s=0; s<nSpecies; s++) {
    cudaMalloc((void**) &Dens[s],  sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &Dens1[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

    cudaMalloc((void**) &Upar[s],  sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &Upar1[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

    cudaMalloc((void**) &Tpar[s],  sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &Tpar1[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

    cudaMalloc((void**) &Qpar[s],  sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &Qpar1[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

    cudaMalloc((void**) &Tprp[s],  sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &Tprp1[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);

    cudaMalloc((void**) &Qprp[s],  sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &Qprp1[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  }  
  
  
  cudaMalloc((void**) &Phi, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMalloc((void**) &Phi1, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
  cudaMalloc((void**) &Phi2_kxky_sum, sizeof(float)*Nx*(Ny/2+1)); 
  cudaMalloc((void**) &wpfxnorm_kxky_sum, sizeof(float)*Nx*(Ny/2+1));
  cudaMalloc((void**) &Phi2_zonal_sum, sizeof(float)*Nx); 
  //cudaMalloc((void**) &Phi_sum, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMalloc((void**) &zCorr_sum, sizeof(float)*(Ny/2+1)*Nz);
  
  cudaMalloc((void**) &field, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMalloc((void**) &tmp,   sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMalloc((void**) &tmpZ, sizeof(float)*Nz);
  cudaMalloc((void**) &tmpX, sizeof(float)*Nx);
  cudaMalloc((void**) &tmpX2, sizeof(float)*Nx);
  cudaMalloc((void**) &tmpY, sizeof(float)*(Ny/2+1));
  cudaMalloc((void**) &tmpY2, sizeof(float)*(Ny/2+1));
  cudaMalloc((void**) &tmpXY, sizeof(float)*Nx*(Ny/2+1));
  cudaMalloc((void**) &tmpXY2, sizeof(float)*Nx*(Ny/2+1));
  cudaMalloc((void**) &tmpXY3, sizeof(float)*Nx*(Ny/2+1));
  cudaMalloc((void**) &tmpXY4, sizeof(float)*Nx*(Ny/2+1));
  cudaMalloc((void**) &tmpXY_R, sizeof(float)*Nx*Ny);
  cudaMalloc((void**) &tmpXZ, sizeof(float)*Nx*Nz);
  cudaMalloc((void**) &tmpYZ, sizeof(float)*(Ny/2+1)*Nz);
  cudaMalloc((void**) &CtmpX, sizeof(cuComplex)*Nx);
  
  cudaMalloc((void**) &deriv_nlps, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMalloc((void**) &derivR1_nlps, sizeof(float)*Nx*Ny*Nz);
  cudaMalloc((void**) &derivR2_nlps, sizeof(float)*Nx*Ny*Nz);
  cudaMalloc((void**) &resultR_nlps, sizeof(float)*Nx*Ny*Nz);

  cudaMalloc((void**) &kx, sizeof(float)*Nx);
  cudaMalloc((void**) &ky, sizeof(float)*(Ny/2+1));
  cudaMalloc((void**) &kz, sizeof(float)*Nz);

  cudaMalloc((void**) &bmagInv, sizeof(float)*Nz); 
  cudaMalloc((void**) &bmag_complex, sizeof(cuComplex)*(Nz/2+1));
  cudaMalloc((void**) &jacobian, sizeof(float)*Nz);
  
  //from input file
  cudaMalloc((void**) &gbdrift, sizeof(float)*Nz);
  cudaMalloc((void**) &grho, sizeof(float)*Nz);
  cudaMalloc((void**) &z, sizeof(float)*Nz);
  cudaMalloc((void**) &cvdrift, sizeof(float)*Nz);
  cudaMalloc((void**) &gds2, sizeof(float)*Nz);
  cudaMalloc((void**) &bmag, sizeof(float)*Nz);
  cudaMalloc((void**) &bgrad, sizeof(float)*Nz);    //
  cudaMalloc((void**) &gds21, sizeof(float)*Nz);
  cudaMalloc((void**) &gds22, sizeof(float)*Nz);
  cudaMalloc((void**) &cvdrift0, sizeof(float)*Nz);
  cudaMalloc((void**) &gbdrift0, sizeof(float)*Nz);
  
  cudaMalloc((void**) &omega, sizeof(cuComplex)*Nx*(Ny/2+1));
  for(int t=0; t<navg; t++) {
    if(LINEAR) cudaMalloc((void**) &omegaBox[t], sizeof(cuComplex)*Nx*(Ny/2+1));
    //cudaMalloc((void**) &Phi2_XYBox[t], sizeof(float)*Nx*(Ny/2+1));
  }
  cudaMalloc((void**) &omegaAvg, sizeof(cuComplex)*Nx*(Ny/2+1));
  
  cudaMalloc((void**) &kx_shift, sizeof(float)*(Ny/2+1));
  cudaMalloc((void**) &jump, sizeof(int)*(Ny/2+1));
  
  cudaMalloc((void**) &nu_nlpm, sizeof(float)*Nz);
  
  if(DEBUG) getError("run_gryfx.cu, after device alloc");
  
  cudaMemcpy(gbdrift, gbdrift_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(grho, grho_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(z, z_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(cvdrift, cvdrift_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(gds2, gds2_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(bmag, bmag_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(bgrad, bgrad_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);    //
  cudaMemcpy(gds21, gds21_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(gds22, gds22_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(cvdrift0, cvdrift0_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(gbdrift0, gbdrift0_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  
  float* val;
  cudaMalloc((void**) &val, sizeof(float));
  
  if(DEBUG) getError("run_gryfx.cu, after memcpy");
  
  
  //set up plans for NLPS, ZDeriv, and ZDerivB
  //plan for ZDerivCovering done below
  int NLPSfftdims[2] = {Nx, Ny};
  cufftPlanMany(&NLPSplanR2C, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, Nz);
  cufftPlanMany(&NLPSplanC2R, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, Nz);
  cufftPlan1d(&ZDerivBplanR2C, Nz, CUFFT_R2C, 1);
  cufftPlan1d(&ZDerivBplanC2R, Nz, CUFFT_C2R, 1);  
  cufftPlan2d(&XYplanC2R, Nx, Ny, CUFFT_C2R);  //for diagnostics
  int n[1] = {Nz};
  int inembed[1] = {(Ny/2+1)*Nx*Nz};
  int onembed[1] = {(Ny/2+1)*Nx*(Nz)};
  cufftPlanMany(&ZDerivplan,1,n,inembed,(Ny/2+1)*Nx,1,
                                onembed,(Ny/2+1)*Nx,1,CUFFT_C2C,(Ny/2+1)*Nx);	
                     //    n rank  nembed  stride   dist
  if(DEBUG) getError("after plan");
    
  // INITIALIZE ARRAYS AS NECESSARY

  kInit  <<< dimGrid, dimBlock >>> (kx, ky, kz, NO_ZDERIV);
  float kx_max = (float) ((int)((Nx-1)/3))/X0;
  float ky_max = (float) ((int)((Ny-1)/3))/Y0;
  float kperp2_max = pow(kx_max,2) + pow(ky_max,2);
  kperp2_max_Inv = 1. / kperp2_max;
  if(DEBUG) printf("kperp2_max_Inv = %f\n", kperp2_max_Inv);
  bmagInit <<<dimGrid,dimBlock>>>(bmag,bmagInv);
  if(S_ALPHA) jacobianInit<<<dimGrid,dimBlock>>> (jacobian,drhodpsi,gradpar,bmag);
  else {
    //copy from geo output
    cudaMemcpy(jacobian, jacobian_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
    
    //calculate bgrad
    ZDerivB(bgrad, bmag, bmag_complex, kz);
  }  
  if(DEBUG) getError("before cudaMemset");  
  //cudaMemset(jump, 0, sizeof(float)*Ny);
  //cudaMemset(kx_shift,0,sizeof(float)*Ny);
  if(DEBUG) getError("after cudaMemset"); 
  //for flux calculations
  multdiv<<<dimGrid,dimBlock>>>(tmpZ, jacobian, grho,1,1,Nz,1);
  fluxDen = sumReduc(tmpZ,Nz,false);
  
  if(DEBUG) getError("run_gryfx.cu, after init"); 
 
  cudaMemcpy(kx_h,kx, sizeof(float)*Nx, cudaMemcpyDeviceToHost);
  cudaMemcpy(ky_h,ky, sizeof(float)*(Ny/2+1), cudaMemcpyDeviceToHost);
  
  
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  //set up kxCover and kyCover for covering space z-transforms
  int naky, ntheta0;// nshift;
  naky = 1 + (Ny-1)/3;
  ntheta0 = 1 + 2*(Nx-1)/3;     //MASK IN MIDDLE OF ARRAY
  //nshift = Nx - ntheta0;

  int idxRight[naky*ntheta0];
  int idxLeft[naky*ntheta0];

  int linksR[naky*ntheta0];
  int linksL[naky*ntheta0];

  int n_k[naky*ntheta0];

  getNClasses(&nClasses, idxRight, idxLeft, linksR, linksL, n_k, naky, ntheta0, jtwist);
  
  if(DEBUG) getError("run_gryfx.cu, after nclasses");

  nLinks = (int*) malloc(sizeof(int)*nClasses);
  nChains = (int*) malloc(sizeof(int)*nClasses);

  getNLinksChains(nLinks, nChains, n_k, nClasses, naky, ntheta0);

  
  //int **kxCover_h, **kyCover_h;
  //kxCover_h = (int**) malloc(sizeof(int)*nClasses);
  //kyCover_h = (int**) malloc(sizeof(int)*nClasses);
  int *kxCover_h[nClasses];
  int *kyCover_h[nClasses];
  
  for(int c=0; c<nClasses; c++) {   
    kyCover_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    kxCover_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
  }  

  kFill(nClasses, nChains, nLinks, kyCover_h, kxCover_h, linksL, linksR, idxRight, naky, ntheta0); 
  
  if(DEBUG) getError("run_gryfx.cu, after kFill");

  //these are the device arrays... cannot be global because jagged!
  int *kxCover[nClasses];
  int *kyCover[nClasses];
  cuComplex *g_covering[nClasses];
  float *kz_covering[nClasses];
  cufftHandle plan_covering[nClasses];
  //also set up a stream for each class.
  streams = (cudaStream_t*) malloc(sizeof(cudaStream_t)*nClasses);
  for(int c=0; c<nClasses; c++) {    
    int n[1] = {nLinks[c]*Nz};
    cudaStreamCreate(&(streams[c]));
    cufftPlanMany(&plan_covering[c],1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,nChains[c]);
    if(DEBUG) kPrint(nLinks[c], nChains[c], kyCover_h[c], kxCover_h[c]); 
    cudaMalloc((void**) &g_covering[c], sizeof(cuComplex)*Nz*nLinks[c]*nChains[c]);
    cudaMalloc((void**) &kz_covering[c], sizeof(float)*Nz*nLinks[c]);
    cudaMalloc((void**) &kxCover[c], sizeof(int)*nLinks[c]*nChains[c]);
    cudaMalloc((void**) &kyCover[c], sizeof(int)*nLinks[c]*nChains[c]);    
    cudaMemcpy(kxCover[c], kxCover_h[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyHostToDevice);
    cudaMemcpy(kyCover[c], kyCover_h[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyHostToDevice);    
  }    
  //printf("nLinks[0] = %d  nChains[0] = %d\n", nLinks[0],nChains[0]);
  

  if(DEBUG) getError("run_gryfx.cu, after kCover");
  ///////////////////////////////////////////////////////////////////////////////////////////////////
    

  

  
  ////////////////////////////////////////////////
  // set up some diagnostics/control flow files //
  ////////////////////////////////////////////////
  
  
  //set up stopfile
  char stopfileName[60];
  strcpy(stopfileName, out_stem);
  strcat(stopfileName, "stop");
  
  printf("stopfile = %s\n", stopfileName);
  
  //time histories (.time)
  FILE *fluxfile;
  char fluxfileName[60];
  strcpy(fluxfileName, out_stem);
  strcat(fluxfileName, "flux.time");
  
  FILE *omegafile;
  char omegafileName[60];
  strcpy(omegafileName, out_stem);
  strcat(omegafileName, "omega.time");
  
  FILE *gammafile;
  char gammafileName[60];
  strcpy(gammafileName, out_stem);
  strcat(gammafileName, "gamma.time");
  
  FILE *phifile;
  char phifileName[60];
  strcpy(phifileName, out_stem);
  strcat(phifileName, "phi.time");
  
  if(!RESTART) {
    omegafile = fopen(omegafileName, "w+");
    gammafile = fopen(gammafileName, "w+");  
    fluxfile = fopen(fluxfileName, "w+");
    if(write_omega) {
      //set up omega output file 
      omegaWriteSetup(omegafile,"omega");
      //and gamma output file
      omegaWriteSetup(gammafile,"gamma");  
    }
    phifile = fopen(phifileName, "w+");
    phiWriteSetup(phifile);
  }
  else {
    omegafile = fopen(omegafileName, "a");
    gammafile = fopen(gammafileName, "a");  
    fluxfile = fopen(fluxfileName, "a");
    phifile = fopen(phifileName, "a");
  }
  
  ////////////////////////////////////////////
  

  //////////////////////////////
  // initial conditions set here
  //////////////////////////////
  
  if(DEBUG) getError("run_gryfx.cu, before initial condition"); 
  
  float runtime;
  int counter;
  
  //float amp;
  
  if(!RESTART) {
    
    if(init == DENS) {
      for(int index=0; index<Nx*(Ny/2+1)*Nz; index++) 
      {
	init_h[index].x = 0;
	init_h[index].y = 0;
      }
      //amp = 1.e-5; //e-20;
      
      srand(22);
      float samp;

      for(int k=0; k<Nz; k++) {
	for(int j=0; j<Nx; j++) {
	  for(int i=0; i<(Ny/2+1); i++) {

	    int index = i + (Ny/2+1)*j + (Ny/2+1)*Nx*k;
	    int idxy = i + (Ny/2+1)*j;
	    //if(idxy!=0) {

	      //if(i==0) amp = 1.e-5;
	      //else amp = 1.e-5;

	      if(i==0) {
	      	samp = init_amp*1.e-8;  //initialize zonal flows at much
					 //smaller amplitude
	      }
	      else {
	      	samp = init_amp;
	      }

	      

	      float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
	      float rb = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
	      //printf("%e\n", ra);

	      init_h[index].x = ra;
	      init_h[index].y = rb;
	      
	      
	        
	      
	    //}
	  }
	}
      }
      

      for(int s=0; s<nSpecies; s++) {
        cudaMemset(Dens[s], 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      }
      
      

      cudaMemcpy(Dens[ION], init_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
      if(DEBUG) getError("after copy");    

      //enforce reality condition -- this is CRUCIAL when initializing in k-space
      realityAll<<<dimGrid,dimBlock>>>(Dens[ION]);
      
      mask<<<dimGrid,dimBlock>>>(Dens[ION]);  

      cudaMemset(Phi, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      
    }
    
    if(init == PHI) {
      for(int index=0; index<Nx*(Ny/2+1)*Nz; index++) 
      {
	init_h[index].x = 0;
	init_h[index].y = 0;
      }
      
      //initialize phi(ikx = 2, iky = 0, z = 0) with some amplitude amp
      init_amp = 1.e-2;
      init_h[0 + (Ny/2+1)*2 + Nx*(Ny/2+1)*(Nz/2)].x = (float) init_amp;
      
      getError("after phi init");
      
      cudaMemset(Phi, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 
      
      cudaMemcpy(Phi, init_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
    
      cudaMemset(Dens[ION], 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);  
    }  
   

    for(int s=0; s<nSpecies; s++) {
      zeroC <<< dimGrid, dimBlock >>> (Dens1[s]);
      if(s!=0) zeroC<<<dimGrid,dimBlock>>>(Dens[s]);

      zeroC <<< dimGrid, dimBlock >>> (Upar[s]);
      
      zeroC <<< dimGrid, dimBlock >>> (Tpar[s]);

      zeroC <<< dimGrid, dimBlock >>> (Qpar[s]);

      zeroC <<< dimGrid, dimBlock >>> (Tprp[s]);

      zeroC <<< dimGrid, dimBlock >>> (Qprp[s]);
      
      zeroC <<< dimGrid, dimBlock >>> (Upar1[s]);
      
      zeroC <<< dimGrid, dimBlock >>> (Tpar1[s]);

      zeroC <<< dimGrid, dimBlock >>> (Qpar1[s]);

      zeroC <<< dimGrid, dimBlock >>> (Tprp1[s]);

      zeroC <<< dimGrid, dimBlock >>> (Qprp1[s]);
    }
    
    zeroC<<<dimGrid,dimBlock>>>(Phi1);
    if(DEBUG) getError("run_gryfx.cu, after zero");
     
    
    if(init == DENS) {
      // Solve for initial phi
      // assumes the initial conditions have been moved to the device
      if(nSpecies!=1) { 
	qneut<<<dimGrid,dimBlock>>>(Phi, Dens[ELECTRON], Dens[ION], Tprp[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv); 
      } else if(ELECTRON == 0) { 
	qneut<<<dimGrid,dimBlock>>>(Phi, tau, Dens[ELECTRON], Tprp[ELECTRON], species[ELECTRON].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      } else if(ION == 0) {
	//qneut<<<dimGrid,dimBlock>>>(Phi, tau, Dens[ION], Tprp[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
	qneutAdiab_part1<<<dimGrid,dimBlock>>>(tmp, field, tau, Dens[ION], Tprp[ION], jacobian, species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
	qneutAdiab_part2<<<dimGrid,dimBlock>>>(Phi, tmp, field, tau, Dens[ION], Tprp[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      }
    }
 
    
    
    runtime=0;
    counter=0;
    totaltimer=0.;
    
    
    for(int s=0; s<nSpecies; s++) {
      wpfx_sum[s]= 0.;
    }
    expectation_ky_sum= 0.;
    expectation_kx_sum= 0.;
    dtSum= 0.;
    zero<<<dimGrid,dimBlock>>>(Phi2_kxky_sum,Nx,Ny/2+1,1);
    zero<<<dimGrid,dimBlock>>>(wpfxnorm_kxky_sum, Nx, Ny/2+1, 1);
    zero<<<dimGrid,dimBlock>>>(Phi2_zonal_sum, Nx, 1,1);
    //zeroC<<<dimGrid,dimBlock>>>(Phi_sum);
    zero<<<dimGrid,dimBlock>>>(zCorr_sum, 1, Ny/2+1, Nz);
    
        
    
  } 
  else {
    restartRead(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,wpfx_sum,Phi2_kxky_sum, Phi2_zonal_sum,
    			zCorr_sum,&expectation_ky_sum, &expectation_kx_sum,
    			&dtSum, &counter,&runtime,&dt,&totaltimer,restartfileName);
			
    if(zero_restart_avg) {
      printf("zeroing avg sums...\n");
      for(int s=0; s<nSpecies; s++) {
        wpfx_sum[s] = 0.;
      }
      expectation_ky_sum = 0.;
      expectation_kx_sum = 0.;
      dtSum = 0.;
      zero<<<dimGrid,dimBlock>>>(Phi2_kxky_sum, Nx, Ny/2+1, 1);
      zero<<<dimGrid,dimBlock>>>(Phi2_zonal_sum, Nx, 1, 1);
      zero<<<dimGrid,dimBlock>>>(zCorr_sum, 1, Ny/2+1, Nz);
    }
	
  
  }

  if(DEBUG) fieldWrite(Dens[ION], field_h, "dens0.field", filename); 
  
  //writedat_beginning();
    
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);					    

  cudaEventRecord(start,0);
  
  cudaProfilerStart();
  
  
  int stopcount = 0;
  int nstop = 10;
  //nSteps = 100;
  
  int Stable[Nx*(Ny/2+1)*2];
  cuComplex stability[Nx*(Ny/2+1)];
  for(int i=0; i<Nx*(Ny/2+1); i++) {
    stability[i].x = 0;
    stability[i].y = 0;
    Stable[i] = 0;
    Stable[i +Nx*(Ny/2+1)] = 0;
  }
  bool STABLE_STOP=false;  
  int stableMax = 500;
    
  float wpfxmax=0.;
  float wpfxmin=0.;
  int converge_count=0;
  
  bool startavg=false;

  float dt_start = .0001;
  float alpha;
    
  while(/*counter < 1 &&*/ 
        counter<nSteps &&
	stopcount<nstop 
	/*&& converge_count<2*navg*/
	)
  {
    
    dt_old = dt;
    if(!LINEAR) dt = courant(Phi, tmp, field, resultR_nlps, species);   
    avgdt = .5*(dt_old+dt);    
    //if(counter<50 && dt>dt_start) dt = dt_start;
    

    //EXBshear bug fixed, need to check if correct
    ExBshear(Phi,Dens,Upar,Tpar,Tprp,Qpar,Qprp,kx_shift,jump,avgdt);  
    
    
    //if(DEBUG) getError("after exb");
   
    
    /*
    getPhiVal<<<dimGrid,dimBlock>>>(val, Phi, 0, 2, Nz/2);
    phiVal[0] = 0;
    cudaMemcpy(phiVal, val, sizeof(float), cudaMemcpyDeviceToHost);
    if(runtime == 0) {
      phiVal0 = phiVal[0];
    }
    fprintf(phifile, "\t%f\t%e\n", runtime, (float) phiVal[0]/phiVal0);
    */
    
    //calculate diffusion here... for now we just set it to 1
    diffusion = 1.;
     
    //cudaProfilerStart();
    
    
    for(int s=0; s<nSpecies; s++) {
           
      timestep(Dens[s], Dens[s], Dens1[s], 
               Upar[s], Upar[s], Upar1[s], 
               Tpar[s], Tpar[s], Tpar1[s], 
               Qpar[s], Qpar[s], Qpar1[s], 
               Tprp[s], Tprp[s], Tprp1[s], 
               Qprp[s], Qprp[s], Qprp1[s], 
               Phi, kxCover,kyCover, g_covering, kz_covering, species[s], dt/2.,
	       field,field,field,field,field,field,
	       tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmpZ,plan_covering,
	       nu_nlpm, tmpX, tmpXZ, CtmpX);
	         
    }
      

    if(nSpecies!=1) { 
      qneut<<<dimGrid,dimBlock>>>(Phi1, Dens1[ELECTRON], Dens1[ION], Tprp1[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv); 
    } else if(ELECTRON == 0) {
      qneut<<<dimGrid,dimBlock>>>(Phi1, 1., Dens1[ELECTRON], Tprp1[ELECTRON], species[ELECTRON].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    } else if(ION == 0) {
      //qneut<<<dimGrid,dimBlock>>>(Phi1, 1, Dens1[ION], Tprp1[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      qneutAdiab_part1<<<dimGrid,dimBlock>>>(tmp, field, tau, Dens1[ION], Tprp1[ION], jacobian, species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      qneutAdiab_part2<<<dimGrid,dimBlock>>>(Phi1, tmp, field, tau, Dens1[ION], Tprp1[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    }
    
    mask<<<dimGrid,dimBlock>>>(Phi1);
    reality<<<dimGrid,dimBlock>>>(Phi1);
    
    if(!LINEAR && NLPM) {
      for(int s=0; s<nSpecies; s++) {
        filterNLPM(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s], 
			tmpX, tmpXZ, tmpYZ, nu_nlpm, species[s], dt/2.);
		    
      }  
    }  
    
    
    
    for(int s=0; s<nSpecies; s++) {

      timestep(Dens[s], Dens1[s], Dens[s], 
               Upar[s], Upar1[s], Upar[s], 
               Tpar[s], Tpar1[s], Tpar[s], 
               Qpar[s], Qpar1[s], Qpar[s], 
               Tprp[s], Tprp1[s], Tprp[s], 
               Qprp[s], Qprp1[s], Qprp[s], 
               Phi1, kxCover,kyCover, g_covering, kz_covering, species[s], dt,
	       field,field,field,field,field,field,
	       tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmpZ,plan_covering,
	       nu_nlpm, tmpX, tmpXZ, CtmpX);
    }

    
    if(nSpecies!=1) {
      qneut<<<dimGrid,dimBlock>>>(Phi1, Dens[ELECTRON], Dens[ION], Tprp[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv); 
    } else if(ELECTRON == 0) {
      qneut<<<dimGrid,dimBlock>>>(Phi1, 1, Dens[ELECTRON], Tprp[ELECTRON], species[ELECTRON].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
    } else if(ION == 0) {
      //qneut<<<dimGrid,dimBlock>>>(Phi1, 1, Dens[ION], Tprp[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);  
      qneutAdiab_part1<<<dimGrid,dimBlock>>>(tmp, field, tau, Dens[ION], Tprp[ION], jacobian, species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
      qneutAdiab_part2<<<dimGrid,dimBlock>>>(Phi1, tmp, field, tau, Dens[ION], Tprp[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);      
    }

    mask<<<dimGrid,dimBlock>>>(Phi1);
    reality<<<dimGrid,dimBlock>>>(Phi1);
        
    if(!LINEAR && NLPM) {
      for(int s=0; s<nSpecies; s++) {
        filterNLPM(Phi1, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s], 
			tmpX, tmpXZ, tmpYZ, nu_nlpm, species[s], dt);
		    
      }  
    } 
    
    //cudaProfilerStop();
    
    
    //DIAGNOSTICS
    
    if(LINEAR) {
      growthRate<<<dimGrid,dimBlock>>>(omega,Phi1,Phi,dt);    
      //cudaMemcpy(omega_h, omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);     
      //weighted average of omega over 'navg' timesteps
      //boxAvg(omegaAvg, omega, omegaBox, dt, dtBox, navg, counter);
      cudaMemcpy(omegaAvg_h, omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);      

      
      //print growth rates to files   
      omegaWrite(omegafile,gammafile,omegaAvg_h,runtime); 
      

      if(counter>2*navg) {
	omegaStability(omega_h, omegaAvg_h, stability,Stable,stableMax);
	STABLE_STOP = stabilityCheck(Stable,stableMax);
      }
    }
    //copy Phi for next timestep
    cudaMemcpy(Phi, Phi1, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
    mask<<<dimGrid,dimBlock>>>(Phi1);
    mask<<<dimGrid,dimBlock>>>(Phi);
    
    /*
    if(counter%nwrite==0) {
      cudaMemcpy(phi_h, Phi, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
    }
    */
        
    //calculate instantaneous heat flux
    for(int s=0; s<nSpecies; s++) {  
      fluxes(&wpfx[s],Dens[s],Tpar[s],Tprp[s],Phi,tmp,tmp,tmp,field,field,tmpZ,tmpXY,species[s],runtime);        
    }
    
       
     
    //calculate tmpXY = Phi**2(kx,ky)
    volflux(Phi1,Phi1,tmp,tmpXY);
    volflux_zonal(Phi1,Phi1,tmpX);
    
    //calculate <kx> and <ky>
    expect_k<<<dimGrid,dimBlock>>>(tmpXY2, tmpXY, ky);
    kPhi2 = sumReduc(tmpXY2, Nx*(Ny/2+1), false);
    Phi2 = sumReduc(tmpXY, Nx*(Ny/2+1), false);
    expectation_ky = (float) Phi2/kPhi2;

    expect_k<<<dimGrid,dimBlock>>>(tmpXY2, tmpXY, kx);
    kPhi2 = sumReduc(tmpXY2, Nx*(Ny/2+1), false);
    expectation_kx = (float) Phi2/kPhi2;
    
    //calculate z correlation function = tmpYZ (not normalized)
    zCorrelation<<<dimGrid,dimBlock>>>(tmpYZ, Phi1);
    
    if(counter>0) { //nSteps/4) {
      //we use an exponential moving average
      // wpfx_avg[t] = alpha*wpfx[t] + (1-alpha)*wpfx_avg[t-1]
      // now with time weighting...
      // wpfx_sum[t] = alpha*dt*wpfx[t] + (1-alpha)*wpfx_avg[t-1]
      // dtSum[t] = alpha*dt[t] + (1-alpha)*dtSum[t-1]
      // wpfx_avg[t] = wpfx_sum[t]/dtSum[t]
      alpha = (float) 2./(navg+1.);
      
      // keep a running total of dt, phi**2(kx,ky), expectation values, etc.
      for(int s=0; s<nSpecies; s++) {
        wpfx_sum[s] = wpfx[s]*dt*alpha + wpfx_sum[s]*(1.-alpha);
      }
      add_scaled<<<dimGrid,dimBlock>>>(Phi2_kxky_sum, 1.-alpha, Phi2_kxky_sum, dt*alpha, tmpXY, Nx, Ny, 1);
      add_scaled<<<dimGrid,dimBlock>>>(Phi2_zonal_sum, 1.-alpha, Phi2_zonal_sum, dt*alpha, tmpX, Nx, 1, 1);
      add_scaled<<<dimGrid,dimBlock>>>(zCorr_sum, 1.-alpha, zCorr_sum, dt*alpha, tmpYZ, 1, Ny, Nz);
      //add_scaled<<<dimGrid,dimBlock>>>(omegaAvg, 1.-alpha, omegaAvg, dt*alpha, omega, Nx, Ny, 1);
      expectation_kx_sum = expectation_kx_sum*(1.-alpha) + expectation_kx*dt*alpha;
      expectation_ky_sum = expectation_ky_sum*(1.-alpha) + expectation_ky*dt*alpha;
      dtSum = dtSum*(1.-alpha) + dt*alpha;
      
      // **_sum/dtSum gives time average of **
      for(int s=0; s<nSpecies; s++) {
        if(dtSum == 0) wpfxAvg[s] = 0;
	else wpfxAvg[s] = (float) wpfx_sum[s]/dtSum;
      }        
      // try to autostop when wpfx converges
      // look at min and max of wpfxAvg over time... if wpfxAvg stays within certain bounds for a given amount of
      // time, it is converged
      if(counter >= navg*1.2) {
        //set bounds to be +/- .05*wpfxAvg
	converge_bounds = .1*wpfxAvg[ION];
	//if counter reaches navg/3, recenter bounds
	if(counter == navg || converge_count == navg/3) {
	  wpfxmax = wpfxAvg[ION] + .5*converge_bounds;
	  wpfxmin = wpfxAvg[ION] - .5*converge_bounds;
	  converge_count++;
	}
	//if wpfxAvg goes outside the bounds, reset the bounds.
	else if(wpfxAvg[ION] > wpfxmax) {
          wpfxmax = wpfxAvg[ION] + .3*converge_bounds;
	  wpfxmin = wpfxmax - converge_bounds;
	  converge_count=0;
	}
	else if(wpfxAvg[ION] < wpfxmin) {
          wpfxmin = wpfxAvg[ION] - .3*converge_bounds;
	  wpfxmax = wpfxmin + converge_bounds;
	  converge_count=0;
	}
	//if wpfxAvg stays inside the bounds, increment the convergence counter.
	else converge_count++; 
      }    
    }
	     
    if(counter%nsave==0 && write_phi) phiR_historyWrite(Phi1,omega,tmpXY_R,tmpXY_R_h, runtime, phifile); //save time history of Phi(x,y,z=0)          
    
    
    // print wpfx to screen if not printing growth rates
    if(!write_omega && counter%nwrite==0) printf("wpfx = %f, dt = %f, converge_count = %d\n", wpfx[0],dt, converge_count);
    
    // write flux to file
    fluxWrite(fluxfile,wpfx,wpfxAvg,wpfxmax,wpfxmin,converge_count,runtime,species);
    if(counter%nwrite==0) fflush(NULL);
             
    if(counter%nwrite==0 || stopcount==nstop-1 || counter==nSteps-1) {
      printf("%f    dt=%f   %d: %s\n",runtime,dt,counter,cudaGetErrorString(cudaGetLastError()));
    }
    
    
    
    //print growth rates to screen every nwrite timesteps if write_omega
    if(write_omega) {
      if (counter%nwrite==0 || stopcount==nstop-1 || counter==nSteps-1) {
	printf("ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
	//for(int i=0; i<((Nx-1)/3+1); i++) {
	for(int i=0; i<1; i++) {
	  for(int j=0; j<((Ny-1)/3+1); j++) {
	    int index = j + (Ny/2+1)*i;
	    if(index!=0) {
	      printf("%.4f\t%.4f\t\t%.6f\t%.6f", ky_h[j], kx_h[i], omegaAvg_h[index].x, omegaAvg_h[index].y);
	      if(Stable[index] >= stableMax) printf("\tomega");
	      if(Stable[index+Nx*(Ny/2+1)] >= stableMax) printf("\tgamma");
	      printf("\n");
	    }
	  }
	  printf("\n");
	}
	for(int i=2*Nx/3+1; i<2*Nx/3+1; i++) {
	//for(int i=2*Nx/3+1; i<Nx; i++) {
          for(int j=0; j<((Ny-1)/3+1); j++) {
	    int index = j + (Ny/2+1)*i;
	    printf("%.4f\t%.4f\t\t%.6f\t%.6f", ky_h[j], kx_h[i], omegaAvg_h[index].x, omegaAvg_h[index].y);
	    if(Stable[index] >= stableMax) printf("\tomega");
	    if(Stable[index+Nx*(Ny/2+1)] >= stableMax) printf("\tgamma");
	    printf("\n");
	  }
	  printf("\n");
	}	
      }            
    }
        
    
    //writedat_each();
    
    runtime+=dt;
    counter++;       
    
    
    //checkstop
    if(FILE* checkstop = fopen(stopfileName, "r") ) {
      fclose(checkstop);
      stopcount++;
    }     
    
    
    //check for problems with run
    if(!LINEAR && (isnan(wpfx[ION]) || isinf(wpfx[ION]) || wpfx[ION] < -100 || wpfx[ION] > 1000) ) {
      printf("\n-------------\n--RUN ERROR--\n-------------\n\n");
      printf("RESTARTING FROM LAST RESTART FILE...\n");
      
      restartRead(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,wpfx_sum,Phi2_kxky_sum, Phi2_zonal_sum,
    			zCorr_sum,&expectation_ky_sum, &expectation_kx_sum,
    			&dtSum, &counter,&runtime,&dt,&totaltimer,restartfileName);
      
      printf("cfl was %f. maxdt was %f.\n", cfl, maxdt);
      cfl = .8*cfl;
      maxdt = .8*maxdt;    
      printf("cfl is now %f. maxdt is now %f.\n", cfl, maxdt);
    }

    else if(counter%nsave==0) {
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&timer,start,stop);
      totaltimer+=timer;
      cudaEventRecord(start,0);
      restartWrite(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,wpfx_sum,Phi2_kxky_sum, Phi2_zonal_sum,
      			zCorr_sum, expectation_ky_sum, expectation_kx_sum, 
      			dtSum,counter,runtime,dt,totaltimer,restartfileName);
    }
    
    
    
    if(counter%nwrite == 0) gryfx_finish_diagnostics(Dens, Upar, Tpar, Tprp, Qpar, Qprp, 
                        Phi, tmp, tmp, field, tmpZ, 
                        tmpXY, tmpXY, tmpXY, tmpXY2, tmpXY3, tmpXY4, tmpYZ, tmpYZ,
  			tmpX, tmpX2, tmpY, tmpY, tmpY, tmpY, tmpY2, tmpY2, tmpY2, 
                        kxCover, kyCover, tmpX_h, tmpY_h, tmpXY_h, tmpYZ_h, field_h, 
                        kxCover_h, kyCover_h, omegaAvg_h, qflux, &expectation_ky, &expectation_kx,
			Phi2_kxky_sum, wpfxnorm_kxky_sum, Phi2_zonal_sum, zCorr_sum, expectation_ky_sum, 
			expectation_kx_sum, dtSum,
			counter, runtime, false);
  
  } 
  
  if(DEBUG) getError("just finished timestep loop");
  
  cudaProfilerStop();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timer,start,stop);
  totaltimer+=timer;
  
  nSteps = counter;     //counter at which fields were last calculated
  endtime = runtime;    //time at which fields were last calculated
  
  for(int s=0; s<nSpecies; s++) {
    qflux[s] = wpfxAvg[s];
  }
  
  ////////////////////////////////////////////////////////////

  if(DEBUG) getError("before restartWrite");  
  
  // save for restart run
  restartWrite(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,wpfx_sum, Phi2_kxky_sum, Phi2_zonal_sum, 
  			zCorr_sum, expectation_ky_sum, expectation_kx_sum,
  			dtSum,counter,runtime,dt,totaltimer,restartfileName);
  
  if(DEBUG) getError("after restartWrite");
  
  phiR_historyWrite(Phi1,omega,tmpXY_R,tmpXY_R_h, runtime, phifile); //save time history of Phi(x,y,z=0)      
  
  gryfx_finish_diagnostics(Dens, Upar, Tpar, Tprp, Qpar, Qprp, 
                        Phi, tmp, tmp, field, tmpZ, 
                        tmpXY, tmpXY, tmpXY, tmpXY2, tmpXY3, tmpXY4, tmpYZ, tmpYZ,
  			tmpX, tmpX2, tmpY, tmpY, tmpY, tmpY, tmpY2, tmpY2, tmpY2, 
                        kxCover, kyCover, tmpX_h, tmpY_h, tmpXY_h, tmpYZ_h, field_h, 
                        kxCover_h, kyCover_h, omegaAvg_h, qflux, &expectation_ky, &expectation_kx,
			Phi2_kxky_sum, wpfxnorm_kxky_sum, Phi2_zonal_sum, zCorr_sum, expectation_ky_sum, 
			expectation_kx_sum, dtSum,
			counter, runtime, true);

  
  
  //if(write_omega) stabilityWrite(stability,Stable,stableMax);
    
  //Timing       
  printf("Total time (min): %f\n",totaltimer/60000);    //convert ms to minutes
  printf("Total steps: %d\n", counter);
  printf("Avg time/timestep (s): %f\n",totaltimer/counter/1000);   //convert ms to s
  
  fprintf(outfile,"expectation val of ky = %f\n", expectation_ky);
  fprintf(outfile,"expectation val of kx = %f\n", expectation_kx);
  fprintf(outfile,"Q_i = %f\n\n", qflux[ION]);
  fprintf(outfile,"Total time (min): %f\n",totaltimer/60000);
  fprintf(outfile,"Total steps: %d\n", counter);
  fprintf(outfile,"Avg time/timestep (s): %f\n",totaltimer/counter/1000);
    
  //cleanup  
  for(int s=0; s<nSpecies; s++) {
    cudaFree(Dens[s]), cudaFree(Dens1[s]);
    cudaFree(Upar[s]), cudaFree(Upar1[s]);
    cudaFree(Tpar[s]), cudaFree(Tpar1[s]);
    cudaFree(Tprp[s]), cudaFree(Tprp1[s]);
    cudaFree(Qpar[s]), cudaFree(Qpar1[s]);
    cudaFree(Qprp[s]), cudaFree(Qprp1[s]);
  }  
  cudaFree(Phi);
  cudaFree(Phi1);
  cudaFree(Phi2_kxky_sum);
  //cudaFree(Phi_sum);
  
  cudaFree(tmp);
  cudaFree(field);      
  cudaFree(tmpZ);
  cudaFree(tmpX);
  cudaFree(tmpY);
  cudaFree(tmpY2);
  cudaFree(tmpXY);
  cudaFree(tmpXY2);
  cudaFree(tmpYZ);
  
  cudaFree(bmag), cudaFree(bmagInv), cudaFree(bmag_complex), cudaFree(bgrad);
  cudaFree(omega); cudaFree(omegaAvg);
  for(int t=0; t<navg; t++) {
    //cudaFree(Phi2_XYBox[t]);
    if(LINEAR) cudaFree(omegaBox[t]);
  }
  cudaFree(gbdrift), cudaFree(gbdrift0);
  cudaFree(grho), cudaFree(z);
  cudaFree(cvdrift), cudaFree(cvdrift0);
  cudaFree(gds2), cudaFree(gds21), cudaFree(gds22);
  
  cudaFree(kx_shift), cudaFree(jump);
  
  cudaFree(jacobian);
    
  cudaFree(kx), cudaFree(ky), cudaFree(kz);
  
  cudaFree(deriv_nlps);
  cudaFree(derivR1_nlps);
  cudaFree(derivR2_nlps);
  cudaFree(resultR_nlps);
  
  cufftDestroy(NLPSplanR2C);
  cufftDestroy(NLPSplanC2R);
  cufftDestroy(ZDerivBplanR2C);
  cufftDestroy(ZDerivBplanC2R);
  cufftDestroy(ZDerivplan);
  for(int c=0; c<nClasses; c++) {
    cufftDestroy(plan_covering[c]);
    cudaFree(kxCover[c]);
    cudaFree(kyCover[c]);
    cudaFree(g_covering[c]); 
    cudaFree(kz_covering[c]);
  }
  
  fclose(fluxfile);
  fclose(omegafile);
  fclose(gammafile);
  fclose(phifile);
  
  //cudaProfilerStop();
  
}    
    
    
    
    
