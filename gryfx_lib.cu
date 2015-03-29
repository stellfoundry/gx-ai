#include "gryfx_lib.h"
#include "global_vars.h"
//#include "write_data.cu"
#include "device_funcs.cu"
#include "c_fortran_namelist3.c"
#include "operations_kernel.cu"
#include "diagnostics_kernel.cu"
#include "exb_kernel.cu"
#include "nlps_kernel.cu"
#include "zderiv_kernel.cu"
#include "covering_kernel.cu"
#include "reduc_kernel.cu"
#include "cudaReduc_kernel.cu"
#include "init_kernel.cu"
#include "omega_kernel.cu"
#include "phi_kernel.cu"
#include "qneut_kernel.cu"
#include "nlpm_kernel.cu"
#include "zonal_kernel.cu"
#include "getfcn.cu"
#include "read_namelist.cu"
#include "read_input.cu"
#include "definitions.cu"
#include "maxReduc.cu"
#include "sumReduc.cu"
#include "diagnostics.cu"
#include "coveringSetup.cu"
#include "exb.cu"
#include "qneut.cu"
#include "ztransform_covering.cu"
#include "zderiv.cu"
#include "zderivB.cu"
#include "zderiv_covering.cu"
#include "nlps.cu"
#include "nlpm.cu"
#include "hyper.cu"
#include "courant.cu"
#include "energy.cu"
#include "timestep_gryfx.cu"
#include "run_gryfx.cu"
#include "read_geo.cu"

#ifdef GS2_zonal
extern "C" void init_gs2(int* strlength, char* namelistFile, int * mpcom, int * Nz, float * gryfx_theta, struct gryfx_parameters_struct * gryfxpars);
//extern "C" void init_gs2(int* strlength, char* namelistFile, int * mpcom,  struct gryfx_parameters_struct * gryfxpars);
extern "C" void finish_gs2();
//extern "C" void gs2_diagnostics_mp_finish_gs2_diagnostics_(int* step);
#endif

#ifdef GS2_all
extern "C" void gs2_main_mp_run_gs2_(char* namelistFile, int * strlength);
#endif

void gryfx_get_default_parameters_(struct gryfx_parameters_struct * gryfxpars, char * namelistFile, int mpcom) {  
  

#ifdef GS2_zonal



  MPI_Comm_rank(mpcom, &iproc);
  //printf("I am proc %d\n", iproc);

  char serial_full[100];
  char serial[100];
  FILE *fp;

  
  if(iproc==0) {
    fp = popen("nvidia-smi -q | grep Serial", "r");
    while(fgets(serial_full, sizeof(serial_full)-1,fp) != NULL) {
      printf("%s\n", serial_full);
    }
    pclose(fp);
    for(int i=0; i<8; i++) {
     serial[i] = serial_full[strlen(serial_full) - (9-i)];
    }
    gpuID = atoi(serial);
    printf("SN: %d\n", gpuID);
  }

  if(iproc==0) printf("\n\n========================================\nThis is a hybrid GryfX-GS2 calculation.\n========================================\n\n");
  //iproc = *mp_mp_iproc_;

  int numdev;

  cudaGetDeviceCount(&numdev);

  cudaGetDevice(&gryfxpars->mpirank);

  //gs2_main_mp_advance_gs2_();  

  //gs2_main_mp_finish_gs2_();

  //exit(1);
  
  

#endif

#ifdef GS2_all
  printf("Running GS2 simulation from within GryfX\n\n");
  int length = strlen(namelistFile);
  printf("Namelist is %s\n", namelistFile);
  gs2_main_mp_run_gs2_(namelistFile, &length);

  iproc = *mp_mp_iproc_;

  exit(1);
#endif

    read_namelist(namelistFile); // all procs read from namelist, set global variables.

  char out_dir_path[200];
  if(SCAN) {
    //default: out_stem taken from name of namelist given in argument
    //if( strcmp(scan_type, "default") == 0) {
  
      strncpy(out_stem, namelistFile, strlen(namelistFile)-2);
      //strcat(out_stem,".\0"); 
      if(iproc==0) printf("%d: out_stem = %s\n", gpuID, out_stem);
    //}


  }
#ifdef GS2_zonal
			if(iproc==0) {
#endif

     
    printf("%d: Initializing GryfX...\tNamelist is %s\n", gpuID, namelistFile);
      //update gryfxpars struct with geometry parameters (from read_geo of defaults)
    gryfxpars->equilibrium_type = equilibrium_type;
    /*char eqfile[800];*/
    gryfxpars->irho = irho;
    gryfxpars->rhoc = rhoc;
    gryfxpars->eps = eps;
    gryfxpars->bishop = bishop;
    gryfxpars->nperiod = nperiod;
    printf("nperiod is %d\n", nperiod);
    printf("Nz is %d\n", Nz);
    gryfxpars->ntheta = Nz;
  
   /* Miller parameters*/
    gryfxpars->rgeo_local = rmaj;
    gryfxpars->rgeo_lcfs = rmaj;
    gryfxpars->akappa = akappa ;
    gryfxpars->akappri = akappri;
    gryfxpars->tri = tri;
    gryfxpars->tripri = tripri;
    gryfxpars->shift = shift;
    gryfxpars->qinp = qsf;
    gryfxpars->shat = shat;
    gryfxpars->asym = asym;
    gryfxpars->asympri = asympri;
  
    /* Other geometry parameters - Bishop/Greene & Chance*/
    gryfxpars->beta_prime_input = beta_prime_input;
    gryfxpars->s_hat_input = s_hat_input;
  
    /*Flow shear*/
    gryfxpars->g_exb = g_exb;
  
    /* Species parameters... I think allowing 20 species should be enough!*/
  
    gryfxpars->ntspec = nSpecies;
  
    for (int i=0;i<nSpecies;i++){
  	  gryfxpars->dens[i] = species[i].dens;
  	  gryfxpars->temp[i] = species[i].temp;
  	  gryfxpars->fprim[i] = species[i].fprim;
  	  gryfxpars->tprim[i] = species[i].tprim;
  	  gryfxpars->nu[i] = species[i].nu_ss;
    }
  
#ifdef GS2_zonal
			} //end of iproc if
#endif

}
  

void gryfx_get_fluxes_(struct gryfx_parameters_struct *  gryfxpars, 
			struct gryfx_outputs_struct * gryfxouts, char* namelistFile, int mpcom)
{

   mpcom_global = mpcom;
   FILE* outfile;
#ifdef GS2_zonal
			if(iproc==0) {
#endif
   equilibrium_type = gryfxpars->equilibrium_type ;
  /*char eqfile[800];*/
   irho = gryfxpars->irho ;
   rhoc = gryfxpars->rhoc ;
   eps = gryfxpars->eps;
   // NB NEED TO SET EPS IN TRINITY!!!
   //eps = rhoc/rmaj;
   bishop = gryfxpars->bishop ;
   nperiod = gryfxpars->nperiod ;
    printf("nperiod2 is %d\n", nperiod);
   Nz = gryfxpars->ntheta ;

 /* Miller parameters*/
   rmaj = gryfxpars->rgeo_local ;
   //r_geo = gryfxpars->rgeo_lcfs ;
   akappa  = gryfxpars->akappa ;
   akappri = gryfxpars->akappri ;
   tri = gryfxpars->tri ;
   tripri = gryfxpars->tripri ;
   shift = gryfxpars->shift ;
   qsf = gryfxpars->qinp ;
   shat = gryfxpars->shat ;
   asym = gryfxpars->asym ;
   asympri = gryfxpars->asympri ;

  /* Other geometry parameters - Bishop/Greene & Chance*/
   beta_prime_input = gryfxpars->beta_prime_input ;
   s_hat_input = gryfxpars->s_hat_input ;

  /*Flow shear*/
   g_exb = gryfxpars->g_exb ;

  /* Species parameters... I think allowing 20 species should be enough!*/
  int oldnSpecies = nSpecies;
   nSpecies = gryfxpars->ntspec ;

  if (nSpecies!=oldnSpecies){
	  printf("oldnSpecies=%d,  nSpecies=%d\n", oldnSpecies, nSpecies);
	  printf("Number of species set in get_fluxes must equal number of species in gryfx input file\n");
	  exit(1);
  }
	 if (DEBUG) printf("nSpecies was set to %d\n", nSpecies);
  for (int i=0;i<nSpecies;i++){
	   species[i].dens = gryfxpars->dens[i] ;
	   species[i].temp = gryfxpars->temp[i] ;
	   species[i].fprim = gryfxpars->fprim[i] ;
	   species[i].tprim = gryfxpars->tprim[i] ;
	   species[i].nu_ss = gryfxpars->nu[i] ;
  }
  
  jtwist = (int) round(2*M_PI*abs(shat)*Zp);
  if(jtwist<0) jtwist=0;
  if(jtwist!=0) *&X0 = Y0*jtwist/(2*M_PI*Zp*abs(shat));  
  //else *&X0 = Y0; 
  //else use what is set in input file 
  
    if(iproc==0) printf("%d: Initializing geometry...\n\n", gpuID);
    if (igeo == 2) {
      //if (iproc==0){
        coefficients_struct *coefficients;
        constant_coefficients_struct constant_coefficients;
        read_geo(&Nz,coefficients,&constant_coefficients,gryfxpars);
      //}
    }


  
  if ( igeo == 0 ) // this is s-alpha
  {
         
    gbdrift_h = (float*) malloc(sizeof(float)*Nz);
    grho_h = (float*) malloc(sizeof(float)*Nz);
    z_h = (float*) malloc(sizeof(float)*Nz);
    cvdrift_h = (float*) malloc(sizeof(float)*Nz);
    gds2_h = (float*) malloc(sizeof(float)*Nz);
    bmag_h = (float*) malloc(sizeof(float)*Nz);
    bgrad_h = (float*) malloc(sizeof(float)*Nz);     //
    gds21_h = (float*) malloc(sizeof(float)*Nz);
    gds22_h = (float*) malloc(sizeof(float)*Nz);
    cvdrift0_h = (float*) malloc(sizeof(float)*Nz);
    gbdrift0_h = (float*) malloc(sizeof(float)*Nz); 
    jacobian_h = (float*) malloc(sizeof(float)*Nz); 
    
    gradpar = (float) 1./(qsf*rmaj);
    
    drhodpsi = 1.; 
    
    for(int k=0; k<Nz; k++) {
      z_h[k] = 2*M_PI*Zp*(k-Nz/2)/Nz;
      bmag_h[k] = 1./(1+eps*cos(z_h[k]));
      bgrad_h[k] = gradpar*eps*sin(z_h[k])*bmag_h[k];            //bgrad = d/dz ln(B(z)) = 1/B dB/dz
      gds2_h[k] = 1. + pow((shat*z_h[k]-shift*sin(z_h[k])),2);
      gds21_h[k] = -shat*(shat*z_h[k]-shift*sin(z_h[k]));
      gds22_h[k] = pow(shat,2);
      gbdrift_h[k] = 1./(2.*rmaj)*( cos(z_h[k]) + (shat*z_h[k]-shift*sin(z_h[k]))*sin(z_h[k]) );
      cvdrift_h[k] = gbdrift_h[k];
      gbdrift0_h[k] = -1./(2.*rmaj)*shat*sin(z_h[k]);
      cvdrift0_h[k] = gbdrift0_h[k];
      grho_h[k] = 1;
      if(CONST_CURV) {
        cvdrift_h[k] = 1./(2.*rmaj);
	gbdrift_h[k] = 1./(2.*rmaj);
	cvdrift0_h[k] = 0.;
	gbdrift0_h[k] = 0.;
      }
      if(SLAB) {
        //omegad=0:
	cvdrift_h[k] = 0.;
        gbdrift_h[k] = 0.;       
        cvdrift0_h[k] = 0.;
        gbdrift0_h[k] = 0.;
        //bgrad=0:
        bgrad_h[k] = 0.;
        //bmag=const:
        bmag_h[k] = 1.;
      }
    }  
  }
  else if ( igeo == 1) // read geometry from file 
  {
    FILE* geoFile = fopen(geoFileName, "r");
    printf("Reading eik geo file %s\n", geoFileName);
    read_geo_input(geoFile);
  }
  else if ( igeo == 2 ) // calculate geometry from geo module
  {
    
    //read species parameters from namelist, will overwrite geometry parameters below
      
#ifdef GS2_zonal
    //if we're already running GS2 and calculating geometry there, we don't need to recalculate, just get what GS2 calculated
//    Nz = geometry_mp_ntheta_;
//
//    gbdrift_h = (float*) malloc(sizeof(float)*Nz);
//    grho_h = (float*) malloc(sizeof(float)*Nz);
//    z_h = (float*) malloc(sizeof(float)*Nz);
//    cvdrift_h = (float*) malloc(sizeof(float)*Nz);
//    gds2_h = (float*) malloc(sizeof(float)*Nz);
//    bmag_h = (float*) malloc(sizeof(float)*Nz);
//    bgrad_h = (float*) malloc(sizeof(float)*Nz);     //
//    gds21_h = (float*) malloc(sizeof(float)*Nz);
//    gds22_h = (float*) malloc(sizeof(float)*Nz);
//    cvdrift0_h = (float*) malloc(sizeof(float)*Nz);
//    gbdrift0_h = (float*) malloc(sizeof(float)*Nz); 
//    jacobian_h = (float*) malloc(sizeof(float)*Nz); 
//    Rplot_h = (float*) malloc(sizeof(float)*Nz); 
//    Zplot_h = (float*) malloc(sizeof(float)*Nz); 
//    aplot_h = (float*) malloc(sizeof(float)*Nz); 
    Xplot_h = (float*) malloc(sizeof(float)*Nz); 
    Yplot_h = (float*) malloc(sizeof(float)*Nz); 
    deltaFL_h = (float*) malloc(sizeof(float)*Nz); 
    
    //gradpar = geometry_mp_gradpar_[0];
    //drhodpsi = theta_grid_mp_drhodpsi_;
//    rmaj = geometry_mp_rmaj_;
//    shat = geometry_mp_shat_;
//    kxfac = geometry_mp_kxfac_;
//    qsf = geometry_mp_qsf_;
//    rhoc = geometry_mp_rhoc_;
        
    for(int k=0; k<Nz; k++) {
//    gbdrift_h[k] = geometry_mp_gbdrift_[k]/4.;
//    grho_h[k] = geometry_mp_grho_[k];
//    z_h[k] = 2*M_PI*Zp*(k-Nz/2)/Nz;
//    cvdrift_h[k] = geometry_mp_cvdrift_[k]/4.;
//    gds2_h[k] = geometry_mp_gds2_[k];
//    bmag_h[k] = geometry_mp_bmag_[k];
//    gds21_h[k] = geometry_mp_gds21_[k];
//    gds22_h[k] = geometry_mp_gds22_[k];
//    cvdrift0_h[k] = geometry_mp_cvdrift0_[k]/4.;
//    gbdrift0_h[k] = geometry_mp_gbdrift0_[k]/4.;
//    jacobian_h[k] = geometry_mp_jacob_[k];
//    Rplot_h[k] = geometry_mp_rplot_[k];
//    Zplot_h[k] = geometry_mp_zplot_[k];
//      aplot_h[k] = geometry_mp_aplot_[k];
      Xplot_h[k] = Rplot_h[k]*cos(aplot_h[k]);
      Yplot_h[k] = Rplot_h[k]*sin(aplot_h[k]);
    }

#else
//  coefficients_struct *coefficients;
//  constant_coefficients_struct constant_coefficients;
//  read_geo(&Nz,coefficients,&constant_coefficients);
#endif
    eps = rhoc/rmaj;
  } 
  

  printf("\nNx=%d  Ny=%d  Nz=%d  X0=%g  Y0=%g  Zp=%d   igeo=%d\n", Nx, Ny, Nz, X0, Y0, Zp, igeo);
  printf("tprim=%g  fprim=%g\njtwist=%d   nSpecies=%d   cfl=%f\n", species[ION].tprim, species[ION].fprim,jtwist,nSpecies,cfl);
  printf("temp=%g  dens=%g nu_ss=%g  inlpm=%d  dnlpm=%f\n", species[ION].temp, species[ION].dens,species[ION].nu_ss, inlpm, dnlpm);
  printf("shat=%g  eps=%g  qsf=%g  rmaj=%g  g_exb=%g\n", shat, eps, qsf, rmaj, g_exb);
  printf("rgeo=%g  akappa=%g  akappapri=%g  tri=%g  tripri=%g\n", r_geo, akappa, akappri, tri, tripri);
  printf("asym=%g  asympri=%g  beta_prime_input=%g  rhoc=%g\n", asym, asympri, beta_prime_input, rhoc);
  if(NLPM && nlpm_kxdep) printf("USING NEW KX DEPENDENCE IN COMPLEX DORLAND NLPM EXPRESSION\n");
  if(nlpm_nlps) printf("USING NEW NLPS-style NLPM\n");

  if(DEBUG) { 
    int ct, dev;
    int driverVersion =0, runtimeVersion=0;
    struct cudaDeviceProp prop;

    cudaGetDeviceCount(&ct);
    printf("Device Count: %d\n",ct);

    cudaGetDevice(&dev);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("Driver Version / Runtime Version: %d.%d / %d.%d\n", driverVersion/1000, driverVersion%100,runtimeVersion/1000,runtimeVersion%100);
    cudaGetDeviceProperties(&prop,dev);
    printf("Device Name: %s\n", prop.name);
    printf("Global Memory (bytes): %lu\n", (unsigned long)prop.totalGlobalMem);
    printf("Shared Memory per Block (bytes): %lu\n", (unsigned long)prop.sharedMemPerBlock);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Warp Size (threads): %d\n", prop.warpSize); 
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Size of Block Dimension (threads): %d * %d * %d\n", prop.maxThreadsDim[0], 
	   prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Size of Grid Dimension (blocks): %d * %d * %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }  
  
#ifdef GS2_zonal
					}
#endif


  MPI_Barrier(mpcom);

  if(iproc==0) printf("%d: Initializing GS2...\n\n", gpuID);

  // GS2 needs to know z_h so we have to allocate it on all 
  // procs
  MPI_Bcast(&Nz, 1, MPI_INT, 0, mpcom);
  if(iproc!=0) z_h = (float*)malloc(sizeof(float)*Nz);
  printf("z_h (1) is %f %f %f\n", z_h[0], z_h[1], z_h[2]);
  MPI_Bcast(&z_h[0], Nz, MPI_FLOAT, 0, mpcom);

  printf("z_h is %f %f %f\n", z_h[0], z_h[1], z_h[2]);

  //gs2_main_mp_init_gs2_(namelistFile, &length);
  int length = strlen(namelistFile);
  //init_gs2(&length, namelistFile, &mpcom, gryfxpars);
  init_gs2(&length, namelistFile, &mpcom, &Nz, z_h, gryfxpars);
  MPI_Barrier(mpcom);

  printf("z_h after is %f %f %f\n", z_h[0], z_h[1], z_h[2]);
  if(iproc==0) printf("%d: Finished initializing GS2.\n\n", gpuID);

  //set up restart file
  strcpy(restartfileName, out_stem);
  strcat(restartfileName, "restart.bin");
 
  if(secondary_test && !LINEAR) strcpy(restartfileName, secondary_test_restartfileName);
 
  if(RESTART) {
    // check if restart file exists
    if( FILE* restartFile = fopen(restartfileName, "r") ) {
      printf("restart file found. restarting...\n");
    }
    else{
      printf("cannot restart because cannot find restart file. changing to no restart\n");
      RESTART = false;
    }
  }			
  
  if(CHECK_FOR_RESTART) {
    printf("restart mode set to exist...\n");
    //check if restart file exists
    if(FILE* restartFile = fopen(restartfileName, "r") ) {
      fclose(restartFile);
      printf("restart file exists. restarting...\n");
      RESTART = true;
    }
    else {
      printf("restart file does not exist. starting new run...\n");
      RESTART = false;
    }
  }
  
  //make an input file of form outstem.in if doesn't already exist
  FILE* input;
  FILE* namelist;
  char inputFile[200];
  strcpy(inputFile, out_stem);
  strcat(inputFile, "in");

  if(!(input = fopen(inputFile, "r"))) {
    char ch;
    input = fopen(inputFile, "w");
    namelist = fopen(namelistFile, "r");
    while( (ch = fgetc(namelist))  != EOF)
      fputc(ch, input);
    fclose(input);
    fclose(namelist);
  }

#ifdef GS2_zonal
				if(iproc==0) {
#endif

  /* 
  FILE *ifile;
  FILE *omegafile;
  FILE *gammafile;
  FILE *energyfile;
  FILE *fluxfile;
  FILE *phikyfile;
  FILE *phikxfile;
  FILE *phifile;
  
  char* ifileName;
  char omegafileName[200];
  char gammafileName[200];
  char energyfileName[200];
  
  char phikyfileName[200];
  char phikxfileName[200];
  char phifileName[200];
    
 
  ifileName = "./inputs/eik.out"; 
  if(SCAN) {             
    sprintf(omegafileName, "./scan/outputs/omega/omega%g",dt);
    sprintf(gammafileName, "./scan/outputs/gamma/gamma%g",dt);
    sprintf(energyfileName, "./scan/outputs/energy/energy%g",dt);
    //sprintf(fluxfileName, "./scan/outputs/flux");
    sprintf(phikyfileName, "./scan/outputs/spectrum/phi_ky");
    sprintf(phikxfileName, "./scan/outputs/spectrum/phi_ky0_kx");
    sprintf(phifileName, "./scan/outputs/phi");
  }
  else {
    sprintf(omegafileName, "./outputs/omega");
    sprintf(gammafileName, "./outputs/gamma");
    sprintf(energyfileName, "./outputs/energy");
    //sprintf(fluxfileName, "./outputs/flux");
    sprintf(phikyfileName, "./outputs/phi_ky");
    sprintf(phikxfileName, "./outputs/phi_ky0_kx");
    sprintf(phifileName, "./outputs/phi");
  }
  

    
  //check or set up directory structure for outputs
  struct stat st;
  if( !(stat("fields", &st) == 0 && S_ISDIR(st.st_mode)) ) {
    mkdir("./fields", 00777);
  }
  if( !(stat("fields/phi", &st) == 0 && S_ISDIR(st.st_mode)) ) {
    mkdir("./fields/phi",00777);
  }
  if( !(stat("fields/phi_covering", &st) == 0 && S_ISDIR(st.st_mode)) ) {
    mkdir("./fields/phi_covering",00777);
  }
  if( !(stat("fields/dens", &st) == 0 && S_ISDIR(st.st_mode)) ) {
    mkdir("fields/dens",00777);
  }
  if( !(stat("fields/upar", &st) == 0 && S_ISDIR(st.st_mode)) ) {
    mkdir("fields/upar",00777);
  }
  if( !(stat("fields/tpar", &st) == 0 && S_ISDIR(st.st_mode)) ) {
    mkdir("fields/tpar",00777);
  }
  if( !(stat("fields/tprp", &st) == 0 && S_ISDIR(st.st_mode)) ) {
    mkdir("fields/tprp",00777);
  }
  if( !(stat("fields/qpar", &st) == 0 && S_ISDIR(st.st_mode)) ) {
    mkdir("fields/qpar",00777);
  }
  if( !(stat("fields/qprp", &st) == 0 && S_ISDIR(st.st_mode)) ) {
    mkdir("fields/qprp",00777);
  }
  if(SCAN) {
    if( !(stat("scan", &st) == 0 && S_ISDIR(st.st_mode)) ) {
      mkdir("./scan", 00777);
    }
    if( !(stat("scan/outputs", &st) == 0 && S_ISDIR(st.st_mode)) ) {
      mkdir("./scan/outputs", 00777);
    }
    if( !(stat("scan/outputs/omega", &st) == 0 && S_ISDIR(st.st_mode)) ) {
      mkdir("./scan/outputs/omega", 00777);
    }
    if( !(stat("scan/outputs/gamma", &st) == 0 && S_ISDIR(st.st_mode)) ) {
      mkdir("./scan/outputs/gamma", 00777);
    }
    if( !(stat("scan/outputs/energy", &st) == 0 && S_ISDIR(st.st_mode)) ) {
      mkdir("./scan/outputs/energy", 00777);
    }
    if( !(stat("scan/outputs/spectrum", &st) == 0 && S_ISDIR(st.st_mode)) ) {
      mkdir("./scan/outputs/spectrum", 00777);
    }
  }
  else {
    if( !(stat("outputs", &st) == 0 && S_ISDIR(st.st_mode)) ) {
      mkdir("./outputs", 00777);
    }
  }
  
  
  
  if(DEBUG) getError("gryfx.cu, set up directories");
  
  ifile = fopen(ifileName,"r");
  if(!RESTART) {
    omegafile = fopen(omegafileName, "w+");
    gammafile = fopen(gammafileName, "w+");
    energyfile = fopen(energyfileName, "w+");  
    fluxfile = fopen(fluxfileName, "w+"); 
    phikyfile = fopen(phikyfileName, "w+"); 
    phikxfile = fopen(phikxfileName, "w+");
    phifile = fopen(phifileName, "w+");
  }
  else {
    //don't overwrite, append
    omegafile = fopen(omegafileName, "a");
    gammafile = fopen(gammafileName, "a");
    energyfile = fopen(energyfileName, "a");   
    fluxfile = fopen(fluxfileName, "a");
    phikyfile = fopen(phikyfileName, "a");
    phikxfile = fopen(phikxfileName, "a");
    phifile = fopen(phifileName, "a");
  }
  
  if(DEBUG) getError("gryfx.cu, opened files");
    
  

  
  
  
  //check directory structure for outputs
  if(omegafile == 0) {
    printf("could not open %s, check directory structure\n", omegafileName);
    exit(1);
  }
  if(gammafile == 0) {
    printf("could not open %s, check directory structure\n", gammafileName);
    exit(1);
  }
  if(energyfile == 0) {
    printf("could not open %s, check directory structure\n", energyfileName);
    exit(1);
  }  
  
  */


  ///////////////////////////////////////////////////
  // set up parallelization 
  // calculate dimBlock and dimGrid
  ///////////////////////////////////////////////////

  int dev;
  struct cudaDeviceProp prop;

  cudaGetDevice(&dev);

  cudaGetDeviceProperties(&prop,dev);

  zBlockThreads = prop.maxThreadsDim[2];

  *&zThreads = zBlockThreads*prop.maxGridSize[2];

  //printf("\nzThreads = %d\n", zThreads);

  totalThreads = prop.maxThreadsPerBlock;     


  if(Nz>zBlockThreads) dimBlock.z = zBlockThreads;
  else dimBlock.z = Nz;
  float otherThreads = totalThreads/dimBlock.z;
  int xy = floorf(otherThreads);
  if( (xy%2) != 0 ) xy = xy - 1; // make sure xy is even and less than totalThreads/dimBlock.z
  //find middle factors of xy
  int fx, fy;
  for(int f1 = 1; f1<xy; ++f1) {
    float f2 = (float) xy/f1;
    if(f2 == floorf(f2)) {
      fy = f1; fx = f2;
    }
    if(f2<=f1) break;
  }
  dimBlock.x = fx; 
  dimBlock.y = fy;
    
/*
  if(Nz>zThreads) {
    dimBlock.x = (int) sqrt(totalThreads/zBlockThreads);
    dimBlock.y = (int) sqrt(totalThreads/zBlockThreads);
    dimBlock.z = zBlockThreads;
  }  
 
  //for dirac
  if(prop.maxGridSize[2] != 1) {
    dimBlock.x = 8;
    dimBlock.y = 8;
    dimBlock.z = 8;
  }
  */
  dimGrid.x = (Nx+dimBlock.x-1)/dimBlock.x;
  dimGrid.y = (Ny+dimBlock.y-1)/dimBlock.y;
  if(prop.maxGridSize[2] == 1) dimGrid.z = 1;    
  else dimGrid.z = (Nz+dimBlock.z-1)/dimBlock.z;

  
  //if (DEBUG) 
  printf("dimGrid = (%d, %d, %d)     dimBlock = (%d, %d, %d)\n", dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z);
  
  definitions();
  
  cudaMemcpyToSymbol(nx, &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ny, &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nz, &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nspecies, &nSpecies, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(X0_d, &X0, sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Y0_d, &Y0, sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Zp_d, &Zp, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(zthreads, &zThreads, sizeof(int),0,cudaMemcpyHostToDevice);
 // cudaMemcpyToSymbol(zblockthreads, &zBlockThreads, sizeof(int),0,cudaMemcpyHostToDevice);

  if(DEBUG) getError("gryfx.cu, before run");
  
  
  char outfileName[200];
  strcpy(outfileName, out_stem);
  strcat(outfileName, "out_gryfx");
  outfile = fopen(outfileName, "w+");

#ifdef GS2_zonal
			} //end of iproc if
#endif

  run_gryfx(gryfxouts->pflux, gryfxouts->qflux, outfile);//, omegafile,gammafile,energyfile,fluxfile,phikyfile,phikxfile, phifile);


#ifdef GS2_zonal
  int last_step = 2*nSteps;
  //gs2_diagnostics_mp_finish_gs2_diagnostics_(&last_step);
#endif

#ifdef GS2_zonal
			if(iproc==0) {  
#endif
    printf("\nNx=%d  Ny=%d  Nz=%d  X0=%g  Y0=%g  Zp=%d\n", Nx, Ny, Nz, X0, Y0,Zp);
    printf("tprim=%g  fprim=%g\njtwist=%d   nSpecies=%d   cfl=%f\n", species[ION].tprim, species[ION].fprim,jtwist,nSpecies,cfl);
    printf("shat=%g  eps=%g  qsf=%g  rmaj=%g  g_exb=%g\n", shat, eps, qsf, rmaj, g_exb);
    if(LINEAR) printf("[Linear]\t");
    else printf("[Nonlinear]\t");
    if(NO_ZDERIV) printf("[No zderiv]\t");
    if(NO_ZDERIV_COVERING) printf("[No zderiv_covering]\t");
    if(SLAB) printf("[Slab limit]\t");
    if(varenna) printf("[varenna: ivarenna=%d]\t", ivarenna);
    if(CONST_CURV) printf("[constant curvature]\t");
    if(RESTART) printf("[restart]\t");
    if(NLPM && nlpm_zonal_kx1_only) {
       printf("[Nonlinear Phase Mixing: inlpm=%d, dnlpm=%f, Phi2_zf(kx=1) only]\t", inlpm, dnlpm);
    }
    if(SMAGORINSKY) printf("[Smagorinsky Diffusion]\t");
    if(HYPER && isotropic_shear) printf("[HyperViscocity: D_hyper=%f, isotropic_shear]\t", D_hyper);
    if(HYPER && !isotropic_shear) printf("[HyperViscocity: D_hyper=%f, anisotropic_shear]\t", D_hyper);
    if(no_landau_damping) printf("[No landau damping]\t");
    if(turn_off_gradients_test) printf("[Gradients turned off halfway through the run]\t");
    
    printf("\n\n");
    
    
    fprintf(outfile,"\nNx=%d  Ny=%d  Nz=%d  X0=%g  Y0=%g  Zp=%d\n", Nx, Ny, Nz, X0, Y0,Zp);
    fprintf(outfile,"tprim=%g  fprim=%g\njtwist=%d   nSpecies=%d   cfl=%f\n", species[ION].tprim, species[ION].fprim,jtwist,nSpecies,cfl);
    fprintf(outfile,"shat=%g  eps=%g  qsf=%g  rmaj=%g  g_exb=%g\n", shat, eps, qsf, rmaj, g_exb);
    if(LINEAR) fprintf(outfile,"[Linear]\t");
    else fprintf(outfile,"[Nonlinear]\t");
    if(NO_ZDERIV) fprintf(outfile,"[No zderiv]\t");
    if(NO_ZDERIV_COVERING) fprintf(outfile,"[No zderiv_covering]\t");
    if(SLAB) fprintf(outfile,"[Slab limit]\t");
    if(varenna) fprintf(outfile,"[varenna: ivarenna=%d]\t", ivarenna);
    if(CONST_CURV) fprintf(outfile,"[constant curvature]\t");
    if(RESTART) fprintf(outfile,"[restart]\t");
    if(NLPM) fprintf(outfile,"[Nonlinear Phase Mixing: inlpm=%d, dnlpm=%f]\t", inlpm, dnlpm);
    if(SMAGORINSKY) fprintf(outfile,"[Smagorinsky Diffusion]\t");
    if(HYPER && isotropic_shear) fprintf(outfile, "[HyperViscocity: D_hyper=%f, isotropic_shear]\t", D_hyper);
    if(HYPER && !isotropic_shear) fprintf(outfile, "[HyperViscocity: D_hyper=%f, anisotropic_shear]\t", D_hyper);
    
    fprintf(outfile, "\n\n");
    
    fclose(outfile);
    
    
    /*
    fclose(energyfile);
    fclose(omegafile);
    fclose(gammafile);
    fclose(fluxfile);
    fclose(phikyfile);
    fclose(phikxfile);
    fclose(phifile);
    */

#ifdef GS2_zonal
			} //end of iproc if
#endif

#ifdef GS2_zonal
  //MPI_Barrier(MPI_COMM_WORLD);
  finish_gs2();
#endif


}  	

void gryfx_main(int argc, char* argv[], int mpcom) {
	struct gryfx_parameters_struct gryfxpars;
	struct gryfx_outputs_struct gryfxouts;
//	printf("argc = %d\nargv[0] = %s   argv[1] = %s\n", argc, argv[0],argv[1]); 
  	char* namelistFile;
	if(argc == 2) {
	  namelistFile = argv[1];
	  //printf("namelist = %s\n", namelistFile);
	}
	else {
	  namelistFile = "inputs/cyclone_miller_ke.in";
	}
	gryfx_get_default_parameters_(&gryfxpars, namelistFile, mpcom);
	gryfx_get_fluxes_(&gryfxpars, &gryfxouts, namelistFile, mpcom);
}
