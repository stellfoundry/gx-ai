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
  float phi0_X[Nx];
  cuComplex CtmpX_h[Nx];
  cuComplex field_h[Nx*(Ny/2+1)*Nz];
 
  float Phi2_zf;
  float Phi_zf_rms;
  float Phi_zf_rms_sum;
  float Phi_zf_rms_avg;
  
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
  float *tmpXYZ;
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
  float flux1_phase, flux2_phase, Dens_phase, Tpar_phase, Tprp_phase;
  float flux1_phase_sum, flux2_phase_sum, Dens_phase_sum, Tpar_phase_sum, Tprp_phase_sum;
 
  float Phi2, kPhi2;
  float Phi2_sum, kPhi2_sum;
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
  float *shear_rate_z;
  float *shear_rate_z_nz;
  float *shear_rate_nz;  

  omega_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)); 
  omegaAvg_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1));
  kx_h = (float*) malloc(sizeof(float)*Nx);
  ky_h = (float*) malloc(sizeof(float)*(Ny/2+1));
  kz_h = (float*) malloc(sizeof(float)*(Nz/2+1));
  
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
  cudaMalloc((void**) &tmpXYZ, sizeof(float)*Nx*(Ny/2+1)*Nz); 
 
  cudaMalloc((void**) &deriv_nlps, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMalloc((void**) &derivR1_nlps, sizeof(float)*Nx*Ny*Nz);
  cudaMalloc((void**) &derivR2_nlps, sizeof(float)*Nx*Ny*Nz);
  cudaMalloc((void**) &resultR_nlps, sizeof(float)*Nx*Ny*Nz);

  cudaMalloc((void**) &kx, sizeof(float)*Nx);
  cudaMalloc((void**) &ky, sizeof(float)*(Ny/2+1));
  cudaMalloc((void**) &kz, sizeof(float)*(Nz/2+1));

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
  cudaMalloc((void**) &shear_rate_z, sizeof(float)*Nz);  
  cudaMalloc((void**) &shear_rate_nz, sizeof(float)*Nz);  
  cudaMalloc((void**) &shear_rate_z_nz, sizeof(float)*Nz);  

  float* Dnlpm_d;
  float* Phi_zf_kx1_d;
  cudaMalloc((void**) &Dnlpm_d, sizeof(float));
  cudaMalloc((void**) &Phi_zf_kx1_d, sizeof(float));

  float Dnlpm = 0;
  float Dnlpm_avg = 0;
  float Dnlpm_sum = 0;

  float Phi_zf_kx1 = 0.;
  float Phi_zf_kx1_old = 0.;
  float Phi_zf_kx1_sum = 0.;
  float Phi_zf_kx1_avg = 0.;
  float alpha_nlpm = 0.;
  float mu_nlpm = 0.;
  float tau_nlpm = 10.;

  if(DEBUG) getError("run_gryfx.cu, after device alloc");
  
  cudaMemcpy(gbdrift, gbdrift_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(grho, grho_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(z, z_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(cvdrift, cvdrift_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(gds2, gds2_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(bmag, bmag_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  if(igeo==0) cudaMemcpy(bgrad, bgrad_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);    //
  cudaMemcpy(gds21, gds21_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(gds22, gds22_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(cvdrift0, cvdrift0_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(gbdrift0, gbdrift0_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  if(DEBUG) getError("run_gryfx.cu, after memcpy");
  
  float* val;
  cudaMalloc((void**) &val, sizeof(float));
  float* phiVal; 
  phiVal = (float*) malloc(sizeof(float));
  float phiVal0;
  
  
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
  kx_max = (float) ((int)((Nx-1)/3))/X0;
  ky_max = (float) ((int)((Ny-1)/3))/Y0;
  kperp2_max = pow(kx_max,2) + pow(ky_max,2);
  kx4_max = pow(kx_max,4);
  ky4_max = pow(ky_max,4);
  ky_max_Inv = 1. / ky_max;
  kx4_max_Inv = 1. / kx4_max;
  kperp4_max_Inv = 1. / pow(kperp2_max,2);
  if(DEBUG) printf("kperp4_max_Inv = %f\n", kperp4_max_Inv);
  bmagInit <<<dimGrid,dimBlock>>>(bmag,bmagInv);
  jacobianInit<<<dimGrid,dimBlock>>> (jacobian,drhodpsi,gradpar,bmag);
  if(igeo != 0) {
    //calculate bgrad
    if(DEBUG) printf("calculating bgrad\n");
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
    int n[1] = {nLinks[c]*Nz*icovering};
    cudaStreamCreate(&(streams[c]));
    cufftPlanMany(&plan_covering[c],1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,nChains[c]);
    if(DEBUG) kPrint(nLinks[c], nChains[c], kyCover_h[c], kxCover_h[c]); 
    cudaMalloc((void**) &g_covering[c], sizeof(cuComplex)*icovering*Nz*nLinks[c]*nChains[c]);
    cudaMalloc((void**) &kz_covering[c], sizeof(float)*icovering*Nz*nLinks[c]);
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
  char stopfileName[80];
  strcpy(stopfileName, out_stem);
  strcat(stopfileName, "stop");
  
  printf("stopfile = %s\n", stopfileName);
  
  //time histories (.time)
  FILE *fluxfile;
  char fluxfileName[80];
  strcpy(fluxfileName, out_stem);
  strcat(fluxfileName, "flux.time");
  printf("flux file is %s", fluxfileName); 

 
  FILE *omegafile;
  char omegafileName[80];
  strcpy(omegafileName, out_stem);
  strcat(omegafileName, "omega.time");
  
  FILE *gammafile;
  char gammafileName[80];
  strcpy(gammafileName, out_stem);
  strcat(gammafileName, "gamma.time");
  
  FILE *phifile;
  char phifileName[80];
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
    //phiWriteSetup(phifile);
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
    
      for(int index=0; index<Nx*(Ny/2+1)*Nz; index++) 
      {
	init_h[index].x = 0;
	init_h[index].y = 0;
      }
      //amp = 1.e-5; //e-20;
      
      srand(22);
      float samp;

      //for(int k=0; k<Nz; k++) {
	for(int j=0; j<Nx; j++) {
	  for(int i=0; i<(Ny/2+1); i++) {

	    //int index = i + (Ny/2+1)*j + (Ny/2+1)*Nx*k;
	    int idxy = i + (Ny/2+1)*j;
	    //if(idxy!=0) {

	      //if(i==0) amp = 1.e-5;
	      //else amp = 1.e-5;

	      if(i==0) {
	      	samp = init_amp;//*1.e-8;  //initialize zonal flows at much
					 //smaller amplitude
	      }
	      else {
	      	samp = init_amp;
	      }

	      

	      float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
	      float rb = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
	      //printf("%e\n", ra);

	      //loop over z here to get rid of randomness in z in initial condition
	      for(int k=0; k<Nz; k++) {
	        int index = i + (Ny/2+1)*j + (Ny/2+1)*Nx*k;
		/*if(i==0) { 
		  init_h[index].x = 0.;
	          init_h[index].y = 0.;
		
		else {*/
		  init_h[index].x = init_amp;//*cos(1*z_h[k]);
	          init_h[index].y = init_amp;//init_amp;//*cos(1*z_h[k]);
		//}
	      }
	      
	      
	        
	      
	    //}
	  }
	}
      //}
      
    if(init == DENS) {
      for(int s=0; s<nSpecies; s++) {
        cudaMemset(Dens[s], 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      }
      
      

      cudaMemcpy(Dens[ION], init_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
      if(DEBUG) getError("after copy");    

      //enforce reality condition -- this is CRUCIAL when initializing in k-space
      reality<<<dimGrid,dimBlock>>>(Dens[ION]);
      
      mask<<<dimGrid,dimBlock>>>(Dens[ION]);  

      cudaMemset(Phi, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      
    }
    
    if(init == PHI) {
      
      
      
      cudaMemset(Phi, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 
      
      cudaMemcpy(Phi, init_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
    
      cudaMemset(Dens[ION], 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);  

      mask<<<dimGrid,dimBlock>>>(Phi);

      reality<<<dimGrid,dimBlock>>>(Phi);
    } 

    if(init == FORCE) {
      cudaMemset(Phi, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
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
      } else if(ION == 0) {
	if(iphi00==1) qneutETG<<<dimGrid,dimBlock>>>(Phi, tau, Dens[ION], Tprp[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
	if(iphi00==2) {
          qneutAdiab_part1<<<dimGrid,dimBlock>>>(tmp, field, tau, Dens[ION], Tprp[ION], jacobian, species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
	  qneutAdiab_part2<<<dimGrid,dimBlock>>>(Phi, tmp, field, tau, Dens[ION], Tprp[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        }
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
    flux1_phase_sum = 0.;
    flux2_phase_sum = 0.;
    Dens_phase_sum = 0.;
    Tpar_phase_sum = 0.;
    Tprp_phase_sum = 0.;
    zero<<<dimGrid,dimBlock>>>(Phi2_kxky_sum,Nx,Ny/2+1,1);
    zero<<<dimGrid,dimBlock>>>(wpfxnorm_kxky_sum, Nx, Ny/2+1, 1);
    zero<<<dimGrid,dimBlock>>>(Phi2_zonal_sum, Nx, 1,1);
    //zeroC<<<dimGrid,dimBlock>>>(Phi_sum);
    zero<<<dimGrid,dimBlock>>>(zCorr_sum, 1, Ny/2+1, Nz);
    
        
    
  } 
  else {
    restartRead(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,wpfx_sum,Phi2_kxky_sum, Phi2_zonal_sum,
    			zCorr_sum,&expectation_ky_sum, &expectation_kx_sum, &Phi_zf_kx1_avg,
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

  fieldWrite(Dens[ION], field_h, "dens0.field", filename); 
  
  //writedat_beginning();
    
  cudaEvent_t start, stop, start1, stop1;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);		
  //cudaEventCreate(&start1); 			    
  //cudaEventCreate(&stop1);
  cudaEventRecord(start,0);
  
  cudaProfilerStart();
  
  /*
  float step_timer=0.;
  float step_timer_total=0.;
  float diagnostics_timer=0.;
  float diagnostics_timer_total=0.;
  */

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
  float alpha_short;
  float navg_nlpm;
  float dtSum_short = 0;
    
  while(/*counter < 1 &&*/ 
        counter<nSteps &&
	stopcount<nstop 
	/*&& converge_count<2*navg*/
	)
  {
    
//    cudaEventRecord(start1,0);
    dt_old = dt;
    if(!LINEAR) dt = courant(Phi, tmp, field, resultR_nlps, species);   
    avgdt = .5*(dt_old+dt);    
    //if(counter<50 && dt>dt_start) dt = dt_start;
    

    //EXBshear bug fixed, need to check if correct
    ExBshear(Phi,Dens,Upar,Tpar,Tprp,Qpar,Qprp,kx_shift,jump,avgdt);  
    
    
    //if(DEBUG) getError("after exb");
   
    if(LINEAR) { 
    //volflux_zonal<<<dimGrid,dimBlock>>>(tmpX, Phi, Phi, jacobian, 1./fluxDen); 
    //getPhiVal<<<dimGrid,dimBlock>>>(val, Phi, 0, 4, Nz/2);
    
    //getPhiVal<<<dimGrid,dimBlock>>>(val, tmpX, 4);
    
    getky0z0<<<dimGrid,dimBlock>>>(tmpX, Phi);

    //volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Phi, jacobian, 1./fluxDen);
    //get_real_X<<<dimGrid,dimBlock>>>(tmpX, CtmpX);

    //phiVal[0] = 0;
    //cudaMemcpy(phiVal, val, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmpX_h, tmpX, sizeof(float)*Nx, cudaMemcpyDeviceToHost);
    /*if(runtime == 0) {
      phiVal0 = phiVal[0];
    }*/

    if(counter==0){
      fprintf(phifile,"t");
      for(int i=0; i<Nx; i++) {
        fprintf(phifile, "\tkx=%g", kx_h[i]);
        if(init==DENS || init==PHI) phi0_X[i] = tmpX_h[i];
      }

    }
    if(init== DENS || init==PHI) fprintf(phifile, "\n\t%f\t%e", runtime, (float) tmpX_h[0]/phi0_X[0]);
    else fprintf(phifile, "\n\t%f\t%e", runtime, (float) 1-tmpX_h[0]); 
    for(int i=1; i<Nx; i++) {
      if(init== DENS || init==PHI) fprintf(phifile, "\t%e", (float) tmpX_h[i]/phi0_X[i]);
      else fprintf(phifile, "\t%e", (float) 1-tmpX_h[i]); 
    }
    }

    
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
      } else if(ION == 0) {
	if(iphi00==1) qneutETG<<<dimGrid,dimBlock>>>(Phi1, tau, Dens1[ION], Tprp1[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
	if(iphi00==2) {
          qneutAdiab_part1<<<dimGrid,dimBlock>>>(tmp, field, tau, Dens1[ION], Tprp1[ION], jacobian, species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
	  qneutAdiab_part2<<<dimGrid,dimBlock>>>(Phi1, tmp, field, tau, Dens1[ION], Tprp1[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        }
      }
/*
  if(DEBUG) {*/
  if(counter==0) fieldWrite(Dens1[ION], field_h, "dens.5.field", filename); 
/*  if(counter==0) fieldWrite(Upar1[ION], field_h, "upar.5.field", filename); 
  if(counter==0) fieldWrite(Tpar1[ION], field_h, "tpar.5.field", filename); 
  if(counter==0) fieldWrite(Tprp1[ION], field_h, "tprp.5.field", filename); 
  if(counter==0) fieldWrite(Qpar1[ION], field_h, "qpar.5.field", filename); 
  if(counter==0) fieldWrite(Qprp1[ION], field_h, "qprp.5.field", filename); 
  }
*/
    mask<<<dimGrid,dimBlock>>>(Phi1);
    reality<<<dimGrid,dimBlock>>>(Phi1);
    reality<<<dimGrid,dimBlock>>>(Dens1[ION]);
    reality<<<dimGrid,dimBlock>>>(Upar1[ION]);
    reality<<<dimGrid,dimBlock>>>(Tpar1[ION]);
    reality<<<dimGrid,dimBlock>>>(Tprp1[ION]);
    reality<<<dimGrid,dimBlock>>>(Qpar1[ION]);
    reality<<<dimGrid,dimBlock>>>(Qprp1[ION]);

    
    if(!LINEAR && NLPM) {
      for(int s=0; s<nSpecies; s++) {
        filterNLPM(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s], 
			tmpX, tmpXZ, tmpYZ, nu_nlpm, species[s], dt/2., Dnlpm_d, Phi_zf_kx1_avg);
		    
      }  
    }  
    if(HYPER) {
      if(isotropic_shear) {
        for(int s=0; s<nSpecies; s++) {
          filterHyper_iso(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s], 
			tmpXYZ, shear_rate_nz, dt/2.);
		    
        }  
      }
      else {
        for(int s=0; s<nSpecies; s++) {
          filterHyper_aniso(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s],
                        tmpXYZ, shear_rate_nz, shear_rate_z, shear_rate_z_nz, dt/2.);
        }
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
      } else if(ION == 0) {
	if(iphi00==1) qneutETG<<<dimGrid,dimBlock>>>(Phi1, tau, Dens[ION], Tprp[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
	if(iphi00==2) {
          qneutAdiab_part1<<<dimGrid,dimBlock>>>(tmp, field, tau, Dens[ION], Tprp[ION], jacobian, species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
	  qneutAdiab_part2<<<dimGrid,dimBlock>>>(Phi1, tmp, field, tau, Dens[ION], Tprp[ION], species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        }
      }
    

    mask<<<dimGrid,dimBlock>>>(Phi1);
    reality<<<dimGrid,dimBlock>>>(Phi1);
    reality<<<dimGrid,dimBlock>>>(Dens[ION]);
    reality<<<dimGrid,dimBlock>>>(Upar[ION]);
    reality<<<dimGrid,dimBlock>>>(Tpar[ION]);
    reality<<<dimGrid,dimBlock>>>(Tprp[ION]);
    reality<<<dimGrid,dimBlock>>>(Qpar[ION]);
    reality<<<dimGrid,dimBlock>>>(Qprp[ION]);
        
    if(!LINEAR && NLPM) {
      for(int s=0; s<nSpecies; s++) {
        filterNLPM(Phi1, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s], 
			tmpX, tmpXZ, tmpYZ, nu_nlpm, species[s], dt, Dnlpm_d, Phi_zf_kx1_avg);
		    
      }  
    } 
    
    if(HYPER) {
      if(isotropic_shear) {
        for(int s=0; s<nSpecies; s++) {
          filterHyper_iso(Phi1, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s], 
			tmpXYZ, shear_rate_nz, dt);
		    
        }  
      }
      else {
        for(int s=0; s<nSpecies; s++) {
          filterHyper_aniso(Phi1, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s],
                        tmpXYZ, shear_rate_nz, shear_rate_z, shear_rate_z_nz, dt);
        }
      }
    }
/*
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&step_timer,start1,stop1);
    step_timer_total += step_timer;
*/    //cudaProfilerStop();
    
    
    //DIAGNOSTICS
     
  //  cudaEventRecord(start1,0);  
    
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
    cudaMemcpy(&Dnlpm, Dnlpm_d, sizeof(float), cudaMemcpyDeviceToHost);
    if( strcmp(nlpm_option,"constant") == 0) Dnlpm = dnlpm;
    mask<<<dimGrid,dimBlock>>>(Phi1);
    mask<<<dimGrid,dimBlock>>>(Phi);
    
    /*
    if(counter%nwrite==0) {
      cudaMemcpy(phi_h, Phi, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
    }
    */
        
    //calculate instantaneous heat flux
    for(int s=0; s<nSpecies; s++) {  
      fluxes(&wpfx[s],flux1,flux2,Dens[s],Tpar[s],Tprp[s],Phi,
             tmp,tmp,tmp,field,field,tmpZ,tmpXY,species[s],runtime,
             &flux1_phase, &flux2_phase, &Dens_phase, &Tpar_phase, &Tprp_phase);        
    }
    
     
    //calculate tmpXY = Phi**2(kx,ky)
    volflux(Phi1,Phi1,tmp,tmpXY);
    volflux_zonal(Phi1,Phi1,tmpX);  //tmpX = Phi_zf**2(kx)
    get_kx1_rms<<<1,1>>>(Phi_zf_kx1_d, tmpX);
    Phi_zf_kx1_old = Phi_zf_kx1;
    cudaMemcpy(&Phi_zf_kx1, Phi_zf_kx1_d, sizeof(float), cudaMemcpyDeviceToHost);

    Phi2_zf = sumReduc(tmpX, Nx, false);
    Phi_zf_rms = sqrt(Phi2_zf);   
 
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
      //navg_nlpm = (float) 10./dt;  // average over 10s = navg_nlpm*dt 
      //alpha_short = (float) 2./(navg_nlpm+1.);     
 
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
      Phi2_sum = Phi2_sum*(1.-alpha) + Phi2*dt*alpha;
      Phi_zf_rms_sum = Phi_zf_rms_sum*(1.-alpha) + Phi_zf_rms*dt*alpha;
      flux1_phase_sum = flux1_phase_sum*(1.-alpha) + flux1_phase*dt*alpha;
      flux2_phase_sum = flux2_phase_sum*(1.-alpha) + flux2_phase*dt*alpha;
      Dens_phase_sum = Dens_phase_sum*(1.-alpha) + Dens_phase*dt*alpha;
      Tpar_phase_sum = Tpar_phase_sum*(1.-alpha) + Tpar_phase*dt*alpha;
      Tprp_phase_sum = Tprp_phase_sum*(1.-alpha) + Tprp_phase*dt*alpha;
      Dnlpm_sum = Dnlpm_sum*(1.-alpha) + Dnlpm*dt*alpha;
   //   Phi_zf_kx1_sum = Phi_zf_kx1_sum*(1.-alpha_short) + Phi_zf_kx1*dt*alpha_short;

      dtSum = dtSum*(1.-alpha) + dt*alpha;
      //dtSum_short = dtSum_short*(1.-alpha_short) + dt*alpha_short;
      
      // **_sum/dtSum gives time average of **
      for(int s=0; s<nSpecies; s++) {
        if(dtSum == 0) wpfxAvg[s] = 0;
	else wpfxAvg[s] = (float) wpfx_sum[s]/dtSum;
      }        
      Phi_zf_rms_avg = Phi_zf_rms_sum/dtSum;
      Dnlpm_avg = Dnlpm_sum/dtSum;
     // Phi_zf_kx1_avg = Phi_zf_kx1_sum/dtSum_short;

      alpha_nlpm = dt/tau_nlpm;
      mu_nlpm = exp(-alpha_nlpm);
      if(runtime<tau_nlpm) Phi_zf_kx1_avg = Phi_zf_kx1; //allow a build-up time of tau_nlpm
      else Phi_zf_kx1_avg = mu_nlpm*Phi_zf_kx1_avg + (1-mu_nlpm)*Phi_zf_kx1 + (mu_nlpm - (1-mu_nlpm)/alpha_nlpm)*(Phi_zf_kx1 - Phi_zf_kx1_old);

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
    if(!write_omega && counter%nwrite==0) printf("wpfx = %f, dt = %f, Dnlpm = %f\n", wpfx[0],dt, Dnlpm);
    
    // write flux to file
    fluxWrite(fluxfile,wpfx,wpfxAvg, Dnlpm, Dnlpm_avg, Phi_zf_kx1, Phi_zf_kx1_avg, Phi_zf_rms, Phi_zf_rms_avg, wpfxmax,wpfxmin,converge_count,runtime,species);
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
/*    
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&diagnostics_timer, start1, stop1);
    diagnostics_timer_total += diagnostics_timer;    
*/    
    //writedat_each();
    
    runtime+=dt;
    counter++;       
    
    
    //checkstop
    if(FILE* checkstop = fopen(stopfileName, "r") ) {
      fclose(checkstop);
      stopcount++;
    }     
    
    
    //check for problems with run
    if(!LINEAR && (isnan(wpfx[ION]) || isinf(wpfx[ION]) || wpfx[ION] < -100 || wpfx[ION] > 100000) ) {
      printf("\n-------------\n--RUN ERROR--\n-------------\n\n");
      
      if(counter>nsave) {	
        printf("RESTARTING FROM LAST RESTART FILE...\n");
	restartRead(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,wpfx_sum,Phi2_kxky_sum, Phi2_zonal_sum,
    			zCorr_sum,&expectation_ky_sum, &expectation_kx_sum, &Phi_zf_kx1_avg,
    			&dtSum, &counter,&runtime,&dt,&totaltimer,restartfileName);
      
        printf("cfl was %f. maxdt was %f.\n", cfl, maxdt);
        cfl = .8*cfl;
        maxdt = .8*maxdt;    
        printf("cfl is now %f. maxdt is now %f.\n", cfl, maxdt);
      }
      else{
	printf("ERROR: FLUX IS NAN AT BEGINNING OF RUN\nENDING RUN...\n\n");
        stopcount=100;
      }
    }

    else if(counter%nsave==0) {
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&timer,start,stop);
      totaltimer+=timer;
      cudaEventRecord(start,0);
      restartWrite(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,wpfx_sum,Phi2_kxky_sum, Phi2_zonal_sum,
      			zCorr_sum, expectation_ky_sum, expectation_kx_sum, Phi_zf_kx1_avg,
      			dtSum,counter,runtime,dt,totaltimer,restartfileName);
    }
    
    
    
    if(counter%nwrite == 0) gryfx_finish_diagnostics(Dens, Upar, Tpar, Tprp, Qpar, Qprp, 
                        Phi, tmp, tmp, field, tmpZ, CtmpX,
                        tmpXY, tmpXY, tmpXY, tmpXY2, tmpXY3, tmpXY4, tmpYZ, tmpYZ,
  			tmpX, tmpX2, tmpY, tmpY, tmpY, tmpY, tmpY2, tmpY2, tmpY2, 
                        kxCover, kyCover, tmpX_h, tmpY_h, tmpXY_h, tmpYZ_h, field_h, 
                        kxCover_h, kyCover_h, omegaAvg_h, qflux, &expectation_ky, &expectation_kx,
			Phi2_kxky_sum, wpfxnorm_kxky_sum, Phi2_zonal_sum, zCorr_sum, expectation_ky_sum, 
			expectation_kx_sum, dtSum,
			counter, runtime, false,
			&Phi2, &flux1_phase, &flux2_phase, &Dens_phase, &Tpar_phase, &Tprp_phase,
			Phi2_sum, flux1_phase_sum, flux2_phase_sum, Dens_phase_sum, Tpar_phase_sum, Tprp_phase_sum);
  
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
  			zCorr_sum, expectation_ky_sum, expectation_kx_sum, Phi_zf_kx1_avg,
  			dtSum,counter,runtime,dt,totaltimer,restartfileName);
  
  if(DEBUG) getError("after restartWrite");
  
  //phiR_historyWrite(Phi1,omega,tmpXY_R,tmpXY_R_h, runtime, phifile); //save time history of Phi(x,y,z=0)      
  
  gryfx_finish_diagnostics(Dens, Upar, Tpar, Tprp, Qpar, Qprp, 
                        Phi, tmp, tmp, field, tmpZ, CtmpX,
                        tmpXY, tmpXY, tmpXY, tmpXY2, tmpXY3, tmpXY4, tmpYZ, tmpYZ,
  			tmpX, tmpX2, tmpY, tmpY, tmpY, tmpY, tmpY2, tmpY2, tmpY2, 
                        kxCover, kyCover, tmpX_h, tmpY_h, tmpXY_h, tmpYZ_h, field_h, 
                        kxCover_h, kyCover_h, omegaAvg_h, qflux, &expectation_ky, &expectation_kx,
			Phi2_kxky_sum, wpfxnorm_kxky_sum, Phi2_zonal_sum, zCorr_sum, expectation_ky_sum, 
			expectation_kx_sum, dtSum,
			counter, runtime, true,
			&Phi2, &flux1_phase, &flux2_phase, &Dens_phase, &Tpar_phase, &Tprp_phase,
			Phi2_sum, flux1_phase_sum, flux2_phase_sum, Dens_phase_sum, Tpar_phase_sum, Tprp_phase_sum);


  
  
  //if(write_omega) stabilityWrite(stability,Stable,stableMax);
    
  //Timing       
  printf("Total steps: %d\n", counter);
  printf("Total time (min): %f\n",totaltimer/60000.);    //convert ms to minutes
  printf("Avg time/timestep (s): %f\n",totaltimer/counter/1000.);   //convert ms to s
  //printf("Advance steps:\t%f min\t(%f%)\n", step_timer_total/60000., 100*step_timer_total/totaltimer);
  //printf("Diagnostics:\t%f min\t(%f%)\n", diagnostics_timer_total/60000., 100*diagnostics_timer_total/totaltimer);

  
  fprintf(outfile,"expectation val of ky = %f\n", expectation_ky);
  fprintf(outfile,"expectation val of kx = %f\n", expectation_kx);
  fprintf(outfile,"Q_i = %f\n Phi_zf_rms = %f\n Phi2 = %f\n", qflux[ION],Phi_zf_rms_avg, Phi2);
  fprintf(outfile, "flux1_phase = %f \t\t flux2_phase = %f\nDens_phase = %f \t\t Tpar_phase = %f \t\t Tprp_phase = %f\n", flux1_phase, flux2_phase, Dens_phase, Tpar_phase, Tprp_phase);
  fprintf(outfile,"\nTotal time (min): %f\n",totaltimer/60000);
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
    
    
    
    
