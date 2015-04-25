#include "mpi.h"
#include "standard_headers.h"
#include "run_gryfx.h"
#include "read_namelist.h"
#include "global_variables.h"
#include "write_data.h"
#include "allocations.h"
#include "gryfx_lib.h"
#include "printout.h"
#include "gs2.h"
#include "get_error.h"

#include "definitions.cu"


//Defined at the bottom of this file
void set_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * ev);
void import_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * everything_ptr);
void initialize_cuda_parallelization(everything_struct * ev);
void setup_restart(everything_struct * ev);

void gryfx_get_default_parameters_(struct gryfx_parameters_struct * gryfxpars, char * namelistFile, int mpcom) {  
  
  everything_struct * ev;
	ev = (everything_struct *)malloc(sizeof(everything_struct));
	ev->memory_location = ON_HOST;
  int iproc;

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
    ev->info.gpuID = atoi(serial);
    printf("SN: %d\n", ev->info.gpuID);
  }

  if(iproc==0) printf("\n\n========================================\nThis is a hybrid GryfX-GS2 calculation.\n========================================\n\n");
  //iproc = *mp_mp_iproc_;

  int numdev;

  cudaGetDeviceCount(&numdev);

  cudaGetDevice(&gryfxpars->mpirank);

#else
  iproc=0;
#endif

  ev->mpi.iproc = iproc;
  ev->mpi.mpcom = mpcom;

#ifdef GS2_all
  gryfx_run_gs2_only(namelistFile);
#endif

	if(iproc==0) printf("%d: Initializing GryfX...\tNamelist is %s\n", ev->info.gpuID, namelistFile);
  //  read_namelist(namelistFile); // all procs read from namelist, set global variables.
  read_namelist(&(ev->pars), &(ev->grids), namelistFile);
	writedat_set_run_name(&(ev->info.run_name), namelistFile);
	set_grid_masks_and_unaliased_sizes(&(ev->grids));
  //allocate_or_deallocate_everything(ALLOCATE, ev);

  //char out_dir_path[200];
  //if(SCAN) {
    //default: out_stem taken from name of namelist given in argument
      //strncpy(out_stem, namelistFile, strlen(namelistFile)-2);
      if(iproc==0) printf("%d: out_stem = %s\n", ev->info.gpuID, ev->info.run_name);
  //}

  if (iproc==0) set_gryfxpars(gryfxpars, ev);

  // EGH: this is a nasty way to broadcast gryfxpars... we should
  // really define a custom MPI datatype. However it should continue
  // to work as long as all MPI processes are running on the same
  // architecture. 
  MPI_Bcast(&*gryfxpars, sizeof(gryfxpars), MPI_BYTE, 0, mpcom);
  // This has to be set after the broadcast
	gryfxpars->everything_struct_address = (void *)ev;

}
  

void gryfx_get_fluxes_(struct gryfx_parameters_struct *  gryfxpars, 
			struct gryfx_outputs_struct * gryfxouts, char* namelistFile, int mpcom)
{

	 everything_struct * ev;
   FILE* outfile;

   int iproc;
   // iproc doesn't necessarily have to be the same as it was in 
   // gryfx_get_default_parameters_
   MPI_Comm_rank(mpcom, &iproc);
	 ev = (everything_struct *)gryfxpars->everything_struct_address;
   ev->mpi.iproc = iproc;
   ev->mpi.mpcom = mpcom;
  
  
  if(iproc==0) {
    //Only proc0 needst to import paramters to gryfx
    import_gryfxpars(gryfxpars, ev);
    printf("%d: Initializing geometry...\n\n", ev->info.gpuID);
    set_geometry(&ev->grids, &ev->geo, gryfxpars);
    print_initial_parameter_summary(ev);
    if(DEBUG) print_cuda_properties(ev);
	}

  if(iproc==0) printf("%d: Initializing GS2...\n\n", ev->info.gpuID);
  gryfx_initialize_gs2(ev, gryfxpars, namelistFile, mpcom);
  if(iproc==0) printf("%d: Finished initializing GS2.\n\n", ev->info.gpuID);
 
  // Check if we should and can restart and set the file name
  setup_restart(ev);


  
  //make an input file of form outstem.in if doesn't already exist
  FILE* input;
  FILE* namelist;
  char inputFile[200];
  strcpy(inputFile, ev->info.run_name);
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

    initialize_cuda_parallelization(ev); 
    definitions(ev);
    char outfileName[200];
    strcpy(outfileName, ev->info.run_name);
    strcat(outfileName, "out_gryfx");
    outfile = fopen(outfileName, "w+");

	} //end of iproc if

  /////////////////////////
  // This is the main call
  ////////////////////////
  run_gryfx(ev, gryfxouts->pflux, gryfxouts->qflux, outfile);

	if(iproc==0) {  
    print_final_summary(ev, outfile);
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

void initialize_cuda_parallelization(everything_struct * ev){

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

void set_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * ev){
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
}
void import_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * everything_ptr){
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
}

void setup_restart(everything_struct * ev){
  //set up restart file
  input_parameters_struct * pars = &ev->pars;
  char * restartfileName;

  //Work out how long the restart file name can be
  int maxlen, temp;
  maxlen = strlen(ev->info.run_name) + 12;
  temp = strlen(pars->secondary_test_restartfileName) + 1;
  maxlen = maxlen > temp ? maxlen : temp;

  maxlen = maxlen+200;

  //Allocate restartfileName
  restartfileName = (char*)malloc(sizeof(char) * maxlen);
  ev->info.restart_file_name = restartfileName;

  
  strcpy(restartfileName, ev->info.run_name);
  strcat(restartfileName, "restart.bin");
 
  if(pars->secondary_test && !pars->linear) 
    strcpy(restartfileName, pars->secondary_test_restartfileName);
 
  if(pars->restart) {
    // check if restart file exists
    if( FILE* restartFile = fopen(restartfileName, "r") ) {
      printf("restart file found. restarting...\n");
    }
    else{
      printf("cannot restart because cannot find restart file. changing to no restart\n");
      // EGH Perhaps we should abort at this point?
      pars->restart = false;
    }
  }			
  
  if(pars->check_for_restart) {
    printf("restart mode set to exist...\n");
    //check if restart file exists
    if(FILE* restartFile = fopen(restartfileName, "r") ) {
      fclose(restartFile);
      printf("restart file exists. restarting...\n");
      pars->restart = true;
    }
    else {
      printf("restart file does not exist. starting new run...\n");
      pars->restart = false;
    }
  }
}
