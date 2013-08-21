#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "sys/stat.h"
#include "cufft.h"
#include "cuda_profiler_api.h"
#include "libgen.h"
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
#include "nlpm_kernel.cu"
#include "getfcn.cu"
#include "read_namelist.cu"
#include "read_input.cu"
#include "definitions.cu"
#include "maxReduc.cu"
#include "sumReduc.cu"
#include "diagnostics.cu"
#include "coveringSetup.cu"
#include "exb.cu"
#include "ztransform_covering.cu"
#include "zderiv.cu"
#include "zderivB.cu"
#include "zderiv_covering.cu"
#include "nlps.cu"
#include "nlpm.cu"
#include "courant.cu"
#include "energy.cu"
#include "timestep_gryfx.cu"
#include "run_gryfx.cu"
//#include "read_geo.cu"
#include "gryfx_lib.h"


void gryfx_get_default_parameters_(struct gryfx_parameters_struct * gryfxpars, char * namelistFile){  
  
  read_namelist(namelistFile);
   
    //update gryfxpars struct with geometry parameters (from read_geo of defaults)
  gryfxpars->equilibrium_type = equilibrium_type;
  /*char eqfile[800];*/
  gryfxpars->irho = irho;
  gryfxpars->rhoc = rhoc;
  gryfxpars->bishop = bishop;
  gryfxpars->nperiod = nperiod;
  gryfxpars->ntheta = Nz;

 /* Miller parameters*/
  gryfxpars->rmaj = rmaj;
  gryfxpars->r_geo = r_geo;
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


}
  

void gryfx_get_fluxes_(struct gryfx_parameters_struct *  gryfxpars, 
			struct gryfx_outputs_struct * gryfxouts)
{

   equilibrium_type = gryfxpars->equilibrium_type ;
  /*char eqfile[800];*/
   irho = gryfxpars->irho ;
   rhoc = gryfxpars->rhoc ;
   bishop = gryfxpars->bishop ;
   nperiod = gryfxpars->nperiod ;
   Nz = gryfxpars->ntheta ;

 /* Miller parameters*/
   rmaj = gryfxpars->rmaj ;
   r_geo = gryfxpars->r_geo ;
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
  
  if(jtwist!=0) *&X0 = Y0*jtwist/(2*M_PI*Zp*abs(shat));  
  else *&X0 = Y0;  
  
  printf("\nNx=%d  Ny=%d  Nz=%d  X0=%g  Y0=%g  Zp=%d\n", Nx, Ny, Nz, X0, Y0, Zp);
  printf("tprim=%g  fprim=%g\njtwist=%d   nSpecies=%d   cfl=%f\n", species[ION].tprim, species[ION].fprim,jtwist,nSpecies,cfl);
  printf("temp=%g  dens=%g nu_ss=%g\n", species[ION].temp, species[ION].dens,species[ION].nu_ss);
  printf("shat=%g  eps=%g  qsf=%g  rmaj=%g  g_exb=%g\n", shat, eps, qsf, rmaj, g_exb);
  printf("rgeo=%g  akappa=%g  akappapri=%g  tri=%g  tripri=%g\n", r_geo, akappa, akappri, tri, tripri);
  printf("asym=%g  asympri=%g  beta_prime_input=%g  rhoc=%g\n", asym, asympri, beta_prime_input, rhoc);

  
  if ( S_ALPHA )
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
    
    
    
    for(int k=0; k<Nz; k++) {
      z_h[k] = 2*M_PI*Zp*(k-Nz/2)/Nz;
      bmag_h[k] = 1./(1+eps*cos(z_h[k]));
      bgrad_h[k] = gradpar*eps*sin(z_h[k])*bmag_h[k];            //
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
	cvdrift0_h[k] = 0;
	gbdrift0_h[k] = 0;
      }
    }  
  }
  else 
  {
    
    //read species parameters from namelist, will overwrite geometry parameters below
      
    //coefficients_struct *coefficients;
    //constant_coefficients_struct constant_coefficients;
    //read_geo(&Nz,coefficients,&constant_coefficients);
    
  } 
  
  gradpar = (float) 1./(qsf*rmaj);

  if(DEBUG) { 
    int ct, dev;
    int driverVersion =0, runtimeVersion=0;
    struct cudaDeviceProp prop;

    cudaGetDeviceCount(&ct);
    printf("Device Count: %d\n",ct);

    cudaGetDevice(&dev);
    printf("Device ID: %d, Device Name: %s\n",dev,prop.name);
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
  
  
  char out_dir_path[80];
  if(SCAN) {
    //q scan
    if( strcmp(scan_type, "q_scan") == 0 ) {
      sprintf(out_stem, "scan/q_scan/q%g_%d/q%g_%d.", qsf, scan_number, qsf, scan_number);
      //out_stem = scan/q_scan/qX.X_X/qX.X_X.
      
      printf("out_stem = %s\n", out_stem);

      sprintf(out_dir_path, "scan/q_scan/q%g_%d", qsf, scan_number);

      // check to make sure that the directory 
      // scan/q_scan/qX.X_X exists
      struct stat st;
      if( !(stat(out_dir_path, &st) == 0 && S_ISDIR(st.st_mode)) ) {
	mkdir(out_dir_path, 00777);
      }            
    }  

    //tprim scan
    if( strcmp(scan_type, "tprim_scan") == 0 ) {
      sprintf(out_stem, "scan/tprim_scan/tprim%g/tprim%g_%d.", species[ION].tprim, species[ION].tprim, scan_number);

      sprintf(out_dir_path, "scan/tprim_scan/tprim%g", species[ION].tprim);
      
      // check to make sure that the directory 
      // scan/tprim_scan/tprimX.X/ exists
      struct stat st;
      if( !(stat(out_dir_path, &st) == 0 && S_ISDIR(st.st_mode)) ) {
	mkdir(out_dir_path, 00777);
      }
    }  

    //shat scan
    if( strcmp(scan_type, "shat_scan") == 0 ) {
      if(!LINEAR) sprintf(out_stem, "scan/shat_scan/shat%g/shat%g_%d.", shat,shat, scan_number);
      else sprintf(out_stem, "scan/shat_scan/shat%g_lin/shat%g_%d.", shat,shat, scan_number);
      
      if(!LINEAR) 
        sprintf(out_dir_path, "scan/shat_scan/shat%g", shat);
      else
        sprintf(out_dir_path, "scan/shat_scan/shat%g_lin", shat);
      

      // check to make sure that the directory 
      // scan/shat_scan/shatX.X/ exists
      struct stat st;
      if( !(stat(out_dir_path, &st) == 0 && S_ISDIR(st.st_mode)) ) {
	mkdir(out_dir_path, 00777);
      }
    }  
    
    //cyclone
    if( strcmp(scan_type, "cyclone") == 0 ) {
      sprintf(out_stem, "scan/outputs/cyclone_%d/cyclone_%d.", scan_number, scan_number);
      //out_stem = scan/outputs/cyclone_X/cyclone_X.
      
      printf("out_stem = %s\n", out_stem);

      sprintf(out_dir_path, "scan/outputs/cyclone_%d", qsf, scan_number);

      // check to make sure that the directory exists
      struct stat st;
      if( !(stat(out_dir_path, &st) == 0 && S_ISDIR(st.st_mode)) ) {
	mkdir(out_dir_path, 00777);
      }            
    }  
    
    //test
    if( strcmp(scan_type, "test") == 0 ) {
      sprintf(out_stem, "scan/outputs/test/test.");
      //out_stem = scan/outputs/test/test.
      
      printf("out_stem = %s\n", out_stem);

      sprintf(out_dir_path, "scan/outputs/test");

      // check to make sure that the directory exists
      struct stat st;
      if( !(stat(out_dir_path, &st) == 0 && S_ISDIR(st.st_mode)) ) {
	mkdir(out_dir_path, 00777);
      }            
    }  

    //general scan
    if( strcmp(scan_type, "outputs") == 0 ) {
      sprintf(out_stem, "scan/outputs/");

      // check to make sure that the directory 
      // scan/outputs/ exists
      struct stat st;
      if( !(stat(out_stem, &st) == 0 && S_ISDIR(st.st_mode)) ) {
	mkdir(out_dir_path, 00777);
      }
    }       
  }

  //set up restart file
  strcpy(restartfileName, out_stem);
  strcat(restartfileName, "restart.bin");
  
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
  char omegafileName[100];
  char gammafileName[100];
  char energyfileName[100];
  
  char phikyfileName[100];
  char phikxfileName[100];
  char phifileName[100];
    
 
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

  int zBlockThreads = prop.maxThreadsDim[2];

  *&zThreads = zBlockThreads*prop.maxGridSize[2];

  *&totalThreads = prop.maxThreadsPerBlock;     


  if(Nz>zBlockThreads) dimBlock.z = zBlockThreads;
  else dimBlock.z = Nz;
  int xy = totalThreads/dimBlock.z;
  int blockxy = (int) sqrt(xy);
  //dimBlock = threadsPerBlock, dimGrid = numBlocks
  dimBlock.x = blockxy;
  dimBlock.y = blockxy;
    

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
  
  dimGrid.x = Nx/dimBlock.x+2;
  dimGrid.y = Ny/dimBlock.y+2;
  if(prop.maxGridSize[2] == 1) dimGrid.z = 1;    
  else dimGrid.z = Nz/dimBlock.z+2;
  
  //if (DEBUG) 
  printf("%d %d %d     %d %d %d\n", dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z);
  
  definitions();
  
  cudaMemcpyToSymbol(nx, &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ny, &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nz, &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(X0_d, &X0, sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Y0_d, &Y0, sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Zp_d, &Zp, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(zthreads, &zThreads, sizeof(int),0,cudaMemcpyHostToDevice);

  if(DEBUG) getError("gryfx.cu, before run");
  
  
  FILE* outfile;
  char outfileName[60];
  strcpy(outfileName, out_stem);
  strcat(outfileName, "out");
  outfile = fopen(outfileName, "w+");

  run_gryfx(gryfxouts->qflux, outfile);//, omegafile,gammafile,energyfile,fluxfile,phikyfile,phikxfile, phifile);


  
  printf("\nNx=%d  Ny=%d  Nz=%d  X0=%g  Y0=%g  Zp=%d\n", Nx, Ny, Nz, X0, Y0,Zp);
  printf("tprim=%g  fprim=%g\njtwist=%d   nSpecies=%d   cfl=%f\n", species[ION].tprim, species[ION].fprim,jtwist,nSpecies,cfl);
  printf("shat=%g  eps=%g  qsf=%g  rmaj=%g  g_exb=%g\n", shat, eps, qsf, rmaj, g_exb);
  if(LINEAR) printf("[Linear]\t");
  else printf("[Nonlinear]\t");
  if(NO_ZDERIV) printf("[No zderiv]\t");
  if(NO_ZDERIV_COVERING) printf("[No zderiv_covering]\t");
  if(NO_OMEGAD) printf("[No omegaD]\t");
  if(varenna) printf("[varenna]\t");
  if(CONST_CURV) printf("[constant curvature]\t");
  if(RESTART) printf("[restart]\t");
  if(NLPM) printf("[Nonlinear Phase Mixing]\t");
  if(SMAGORINSKY) printf("[Smagorinsky Diffusion]\t");
  
  printf("\n\n");
  
  
  fprintf(outfile,"\nNx=%d  Ny=%d  Nz=%d  X0=%g  Y0=%g  Zp=%d\n", Nx, Ny, Nz, X0, Y0,Zp);
  fprintf(outfile,"tprim=%g  fprim=%g\njtwist=%d   nSpecies=%d   cfl=%f\n", species[ION].tprim, species[ION].fprim,jtwist,nSpecies,cfl);
  fprintf(outfile,"shat=%g  eps=%g  qsf=%g  rmaj=%g  g_exb=%g\n", shat, eps, qsf, rmaj, g_exb);
  if(LINEAR) fprintf(outfile,"[Linear]\t");
  else fprintf(outfile,"[Nonlinear]\t");
  if(NO_ZDERIV) fprintf(outfile,"[No zderiv]\t");
  if(NO_ZDERIV_COVERING) fprintf(outfile,"[No zderiv_covering]\t");
  if(NO_OMEGAD) fprintf(outfile,"[No omegaD]\t");
  if(varenna) fprintf(outfile,"[varenna]\t");
  if(CONST_CURV) fprintf(outfile,"[constant curvature]\t");
  if(RESTART) fprintf(outfile,"[restart]\t");
  if(NLPM) fprintf(outfile,"[Nonlinear Phase Mixing]\t");
  if(SMAGORINSKY) fprintf(outfile,"[Smagorinsky Diffusion]\t");
  
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
  
}  	

