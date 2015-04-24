#include <math.h>
#include "gryfx_lib.h"
#include "simpledataio_cuda.h"
#define EXTERN_SWITCH extern
#include "everything_struct.h"
#include "read_namelist.h"
#include "global_variables.h"
#include "allocations.h"
#include "write_data.h"
#include "printout.h"
#include "global_vars.h"
//#include "write_data.cu"
#include "device_funcs.cu"
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


#ifdef GS2_zonal
//extern "C" double geometry_mp_bi_;

extern "C" void init_gs2(int* strlength, char* namelistFile, int * mpcom, int * Nz, float * gryfx_theta, struct gryfx_parameters_struct * gryfxpars);
//extern "C" void init_gs2(int* strlength, char* namelistFile, int * mpcom,  struct gryfx_parameters_struct * gryfxpars);
extern "C" void finish_gs2();
//extern "C" void gs2_diagnostics_mp_finish_gs2_diagnostics_(int* step);
#endif

#ifdef GS2_all
extern "C" void gs2_main_mp_run_gs2_(char* namelistFile, int * strlength);
#endif

//Defined at the bottom of this file
void set_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * everything);
void import_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct ** everything_ptr);

void gryfx_get_default_parameters_(struct gryfx_parameters_struct * gryfxpars, char * namelistFile, int mpcom) {  
  
  everything_struct * everything;
	everything = (everything_struct *)malloc(sizeof(everything_struct));
	everything->memory_location = ON_HOST;

#ifdef GS2_zonal

  MPI_Comm_rank(mpcom, &iproc);
  //printf("I am proc %d\n", iproc);

  char serial_full[100];
  char serial[100];
  FILE *fp;

 //Set some global defaults
  
 initialize_globals();

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

#else
  iproc=0;
#endif

#ifdef GS2_all
  printf("Running GS2 simulation from within GryfX\n\n");
  int length = strlen(namelistFile);
  printf("Namelist is %s\n", namelistFile);
  gs2_main_mp_run_gs2_(namelistFile, &length);

  iproc = *mp_mp_iproc_;

  exit(1);
#endif

  //  read_namelist(namelistFile); // all procs read from namelist, set global variables.
  read_namelist(&(everything->pars), &(everything->grids), namelistFile);
	writedat_set_run_name(&(everything->info.run_name), namelistFile);

	set_grid_masks_and_unaliased_sizes(&(everything->grids));

  allocate_or_deallocate_everything(ALLOCATE, everything);

  char out_dir_path[200];
  if(SCAN) {
    //default: out_stem taken from name of namelist given in argument
      strncpy(out_stem, namelistFile, strlen(namelistFile)-2);
      if(iproc==0) printf("%d: out_stem = %s\n", gpuID, out_stem);
  }
	if(iproc==0) {
    printf("%d: Initializing GryfX...\tNamelist is %s\n", gpuID, namelistFile);
    set_gryfxpars(gryfxpars, everything);
	} //end of iproc if

}
  

void gryfx_get_fluxes_(struct gryfx_parameters_struct *  gryfxpars, 
			struct gryfx_outputs_struct * gryfxouts, char* namelistFile, int mpcom)
{

	everything_struct * everything;
   mpcom_global = mpcom;
   FILE* outfile;
  if(iproc==0) {
    import_gryfxpars(gryfxpars, &everything);
    printf("%d: Initializing geometry...\n\n", gpuID);
    set_geometry(&everything->grids, &everything->geo, gryfxpars);
    print_initial_parameter_summary(everything);
    if(DEBUG) print_cuda_properties(everything);
	}


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

  run_gryfx(everything, gryfxouts->pflux, gryfxouts->qflux, outfile);//, omegafile,gammafile,energyfile,fluxfile,phikyfile,phikxfile, phifile);


#ifdef GS2_zonal
  int last_step = 2*nSteps;
  //gs2_diagnostics_mp_finish_gs2_diagnostics_(&last_step);
#endif

#ifdef GS2_zonal
			if(iproc==0) {  
#endif
    print_final_summary(everything, outfile);
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

void set_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * everything){
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
	gryfxpars->everything_struct_address = (void *)everything;
}
void import_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct ** everything_ptr){
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
	*everything_ptr = (everything_struct *)gryfxpars->everything_struct_address;

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
}
