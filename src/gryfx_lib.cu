#include "mpi.h"
#include "standard_headers.h"
#include "global_variables.h"
#include "run_gryfx.h"
#include "read_namelist.h"
#include "write_data.h"
#include "allocations.h"
#include "gryfx_lib.h"
#include "printout.h"
#include "gs2.h"
#include "get_error.h"

#include "definitions.cu"


//Defined at the bottom of this file
void set_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * everything);
void import_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct ** everything_ptr);
void initialize_cuda_parallelization(everything_struct * everything);

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
  gryfx_run_gs2_only(namelistFile);
#endif

	if(iproc==0) printf("%d: Initializing GryfX...\tNamelist is %s\n", gpuID, namelistFile);
  //  read_namelist(namelistFile); // all procs read from namelist, set global variables.
  read_namelist(&(everything->pars), &(everything->grids), namelistFile);
	writedat_set_run_name(&(everything->info.run_name), namelistFile);
	set_grid_masks_and_unaliased_sizes(&(everything->grids));
  //allocate_or_deallocate_everything(ALLOCATE, everything);

  //char out_dir_path[200];
  //if(SCAN) {
    //default: out_stem taken from name of namelist given in argument
      strncpy(out_stem, namelistFile, strlen(namelistFile)-2);
      if(iproc==0) printf("%d: out_stem = %s\n", gpuID, out_stem);
  //}

  if (iproc==0) set_gryfxpars(gryfxpars, everything);

  // EGH: this is a nasty way to broadcast gryfxpars... we should
  // really define a custom MPI datatype. However it should continue
  // to work as long as all MPI processes are running on the same
  // architecture. 
  MPI_Bcast(&gryfxpars, sizeof(gryfxpars), MPI_BYTE, 0, mpcom);

}
  

void gryfx_get_fluxes_(struct gryfx_parameters_struct *  gryfxpars, 
			struct gryfx_outputs_struct * gryfxouts, char* namelistFile, int mpcom)
{

	 everything_struct * everything;
   mpcom_global = mpcom;
   FILE* outfile;
  
  
  if(iproc==0) {
    //Only proc0 needst to import paramters to gryfx
    import_gryfxpars(gryfxpars, &everything);
    printf("%d: Initializing geometry...\n\n", gpuID);
    set_geometry(&everything->grids, &everything->geo, gryfxpars);
    print_initial_parameter_summary(everything);
    if(DEBUG) print_cuda_properties(everything);
	}

  if(iproc==0) printf("%d: Initializing GS2...\n\n", gpuID);
  gryfx_initialize_gs2(everything, gryfxpars, namelistFile, mpcom);
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

  // EGH to Noah... can we get rid of this?
  // do you ever use the old input file format any more?
  if(!(input = fopen(inputFile, "r"))) {
    char ch;
    input = fopen(inputFile, "w");
    namelist = fopen(namelistFile, "r");
    while( (ch = fgetc(namelist))  != EOF)
      fputc(ch, input);
    fclose(input);
    fclose(namelist);
  }

	if(iproc==0) {

    initialize_cuda_parallelization(everything); 
    definitions();
    char outfileName[200];
    strcpy(outfileName, out_stem);
    strcat(outfileName, "out_gryfx");
    outfile = fopen(outfileName, "w+");

	} //end of iproc if

  /////////////////////////
  // This is the main call
  ////////////////////////
  run_gryfx(everything, gryfxouts->pflux, gryfxouts->qflux, outfile);

	if(iproc==0) {  
    print_final_summary(everything, outfile);
    fclose(outfile);
  } //end of iproc if

#ifdef GS2_zonal
  gryfx_finish_gs2();
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

void initialize_cuda_parallelization(everything_struct * everything){

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
    
  dimGrid.x = (Nx+dimBlock.x-1)/dimBlock.x;
  dimGrid.y = (Ny+dimBlock.y-1)/dimBlock.y;
  if(prop.maxGridSize[2] == 1) dimGrid.z = 1;    
  else dimGrid.z = (Nz+dimBlock.z-1)/dimBlock.z;

  
  //if (DEBUG) 
  printf("dimGrid = (%d, %d, %d)     dimBlock = (%d, %d, %d)\n", dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z);
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
