#define NO_GLOBALS true
#include "mpi.h"
#include "standard_headers.h"
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
void set_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * ev);
void import_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * everything_ptr);
void initialize_cuda_parallelization(everything_struct * ev);

void gryfx_get_default_parameters_(struct gryfx_parameters_struct * gryfxpars, char * namelistFile, int mpcom) {  
  
  everything_struct * ev;
	ev = (everything_struct *)malloc(sizeof(everything_struct));
	ev->memory_location = ON_HOST;
  int iproc;

//#ifdef GS2_zonal

  
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
#ifdef GS2_zonal
  if(iproc==0) printf("\n\n========================================\nThis is a hybrid GryfX-GS2 calculation.\n========================================\n\n");
#endif
  //iproc = *mp_mp_iproc_;

  int numdev;

  cudaGetDeviceCount(&numdev);

  cudaGetDevice(&gryfxpars->mpirank);

//#else
//  iproc=0;
//#endif

  ev->mpi.iproc = iproc;
  ev->mpi.mpcom = mpcom;

#ifdef GS2_all
  gryfx_run_gs2_only(namelistFile);
#endif

	if(iproc==0) printf("%d: Initializing GryfX...\tNamelist is %s\n", ev->info.gpuID, namelistFile);
  //  read_namelist(namelistFile); // all procs read from namelist, set global variables.


  // parameters are read from namelist and put into input_parameters struct ev->pars
  read_namelist(&(ev->pars), &(ev->grids), namelistFile);

	//writedat_set_run_name(&(ev->info.run_name), namelistFile);
	set_grid_masks_and_unaliased_sizes(&(ev->grids));
  //allocate_or_deallocate_everything(ALLOCATE, ev);

  //char out_dir_path[200];
  //if(SCAN) {
    //default: out_stem taken from name of namelist given in argument
      //strncpy(out_stem, namelistFile, strlen(namelistFile)-2);
      if(iproc==0) printf("%d: out_stem = \n", ev->info.gpuID);//, ev->info.run_name);
  //}

  // copy elements of input_parameters struct ev->pars into gryfx_parameters_struct gryfxpars
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
    //Only proc0 needs to import paramters to gryfx
    // copy elements of gryfx_parameters_struct gryfxpars into input_parameters struct ev->pars.
    // this is done because gryfxpars may have been changed externally (i.e. by Trinity) 
    // between calls to gryfx_get_default_parameters and gryfx_get_fluxes.
    // ev->pars then needs to be updated since ev->pars is what is used in run_gryfx.
    import_gryfxpars(gryfxpars, ev);
    printf("%d: Initializing geometry...\n\n", ev->info.gpuID);
    set_geometry(&ev->pars, &ev->grids, &ev->geo, gryfxpars);
    //Note the printout module still uses globals which may no 
    //longer be in sync with ev at this point. Need to remove globals
    //from printout.cu
    if(ev->pars.debug) print_cuda_properties(ev);
	}

#ifdef GS2_zonal
  if(iproc==0) printf("%d: Initializing GS2...\n\n", ev->info.gpuID);
  gryfx_initialize_gs2(&ev->grids, gryfxpars, namelistFile, mpcom);
  if(iproc==0) printf("%d: Finished initializing GS2.\n\n", ev->info.gpuID);
#endif
 
  // Copy the name of the namelist file to ev->info.run_name
  // Check if we should and can restart and set the file name
  setup_info(namelistFile, &ev->pars, &ev->info);


  
  //make an input file of form outstem.in if doesn't already exist
  FILE* input;
  FILE* namelist;
  char inputFile[200];
  strcpy(inputFile, ev->info.run_name);
  strcat(inputFile, ".in");

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
    strcat(outfileName, ".out_gryfx");
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
  free(ev);
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

  //Local duplicates for convenience
  cuda_dimensions_struct * cdims = &ev->cdims;
  int totalThreads, zBlockThreads;
  int Nx = ev->grids.Nx;
  int Ny = ev->grids.Ny;
  int Nz = ev->grids.Nz;
	dim3 dimBlock;
  dim3 dimGrid;


  cudaGetDevice(&dev);

  cudaGetDeviceProperties(&prop,dev);

  zBlockThreads = cdims->zBlockThreads = prop.maxThreadsDim[2];

  cdims->zThreads = cdims->zBlockThreads*prop.maxGridSize[2];

  //printf("\nzThreads = %d\n", zThreads);

  totalThreads = cdims->totalThreads = prop.maxThreadsPerBlock;     


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

  cdims->dimGrid = dimGrid;
  cdims->dimBlock = dimBlock;

  
  //if (DEBUG) 
  printf("dimGrid = (%d, %d, %d)     dimBlock = (%d, %d, %d)\n", dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z);
}

// this function copies elements of input_parameters struct ev->pars into gryfx_parameters_struct gryfxpars
void set_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * ev){
    gryfxpars->equilibrium_type = ev->pars.equilibrium_type;
    /*char eqfile[800];*/
    input_parameters_struct * pars = &ev->pars;

    gryfxpars->irho = pars->irho;
    gryfxpars->rhoc = pars->rhoc;
    gryfxpars->eps = pars->eps;
    gryfxpars->bishop = pars->bishop;
    gryfxpars->nperiod = pars->nperiod;
    printf("nperiod is %d\n", pars->nperiod);
    printf("Nz is %d\n", ev->grids.Nz);
    gryfxpars->ntheta = ev->grids.Nz;
  
   /* Miller parameters*/
    gryfxpars->rgeo_local = pars->rmaj;
    gryfxpars->rgeo_lcfs = pars->rmaj;
    gryfxpars->akappa = pars->akappa;
    gryfxpars->akappri = pars->akappri;
    gryfxpars->tri = pars->tri;
    gryfxpars->tripri = pars->tripri;
    gryfxpars->shift = pars->shift;
    gryfxpars->qinp = pars->qsf;
    gryfxpars->shat = pars->shat;
    // EGH These appear to be redundant
    //gryfxpars->asym = pars->asym;
    //gryfxpars->asympri = pars->asympri;
  
    /* Other geometry parameters - Bishop/Greene & Chance*/
    gryfxpars->beta_prime_input = pars->beta_prime_input;
    gryfxpars->s_hat_input = pars->s_hat_input;
  
    /*Flow shear*/
    gryfxpars->g_exb = pars->g_exb;
  
    /* Species parameters... I think allowing 20 species should be enough!*/
  
    gryfxpars->ntspec = pars->nspec;
  
    for (int i=0;i<pars->nspec;i++){
  	  gryfxpars->dens[i] = pars->species[i].dens;
  	  gryfxpars->temp[i] = pars->species[i].temp;
  	  gryfxpars->fprim[i] = pars->species[i].fprim;
  	  gryfxpars->tprim[i] = pars->species[i].tprim;
  	  gryfxpars->nu[i] = pars->species[i].nu_ss;
    }
}


// this function copies elements of gryfx_parameters_struct gryfxpars into input_parameters struct ev->pars.
// this is done because gryfxpars may have been changed externally (i.e. by Trinity) 
// between calls to gryfx_get_default_parameters and gryfx_get_fluxes.
// ev->pars then needs to be updated since ev->pars is what is used in run_gryfx
void import_gryfxpars(struct gryfx_parameters_struct * gryfxpars, everything_struct * ev){
   input_parameters_struct * pars = &ev->pars;
   pars->equilibrium_type = gryfxpars->equilibrium_type ;
  /*char eqfile[800];*/
   pars->irho = gryfxpars->irho ;
   pars->rhoc = gryfxpars->rhoc ;
   pars->eps = gryfxpars->eps;
   // NB NEED TO SET EPS IN TRINITY!!!
   //eps = rhoc/rmaj;
   pars->bishop = gryfxpars->bishop ;
   pars->nperiod = gryfxpars->nperiod ;
    printf("nperiod2 is %d\n", pars->nperiod);
   ev->grids.Nz = gryfxpars->ntheta ;

 /* Miller parameters*/
   pars->rmaj = gryfxpars->rgeo_local ;
   //r_geo = gryfxpars->rgeo_lcfs ;
   pars->akappa  = gryfxpars->akappa ;
   pars->akappri = gryfxpars->akappri ;
   pars->tri = gryfxpars->tri ;
   pars->tripri = gryfxpars->tripri ;
   pars->shift = gryfxpars->shift ;
   pars->qsf = gryfxpars->qinp ;
   pars->shat = gryfxpars->shat ;
    // EGH These appear to be redundant
   //asym = gryfxpars->asym ;
   //asympri = gryfxpars->asympri ;

  /* Other geometry parameters - Bishop/Greene & Chance*/
   pars->beta_prime_input = gryfxpars->beta_prime_input ;
   pars->s_hat_input = gryfxpars->s_hat_input ;

  /*Flow shear*/
   pars->g_exb = gryfxpars->g_exb ;

  /* Species parameters... I think allowing 20 species should be enough!*/
  int oldnSpecies = pars->nspec;
   pars->nspec = gryfxpars->ntspec ;

  if (pars->nspec!=oldnSpecies){
	  printf("oldnSpecies=%d,  nSpecies=%d\n", oldnSpecies, pars->nspec);
	  printf("Number of species set in get_fluxes must equal number of species in gryfx input file\n");
	  exit(1);
  }
	 if (pars->debug) printf("nSpecies was set to %d\n", pars->nspec);
  for (int i=0;i<pars->nspec;i++){
	   pars->species[i].dens = gryfxpars->dens[i] ;
	   pars->species[i].temp = gryfxpars->temp[i] ;
	   pars->species[i].fprim = gryfxpars->fprim[i] ;
	   pars->species[i].tprim = gryfxpars->tprim[i] ;
	   pars->species[i].nu_ss = gryfxpars->nu[i] ;
  }

  //jtwist should never be < 0. If we set jtwist < 0 in the input file,
  // this triggers the use of jtwist_square... i.e. jtwist is 
  // set to what it needs to make the box square at the outboard midplane
  if (pars->jtwist < 0) {
    int jtwist_square, jtwist;
    // determine value of jtwist needed to make X0~Y0
    jtwist_square = (int) round(2*M_PI*abs(pars->shat)*pars->Zp);
    // as currently implemented, there is no way to manually set jtwist from input file
    // there could be some switch here where we choose whether to use
    // jtwist_in or jtwist_square
    jtwist = jtwist_square;
    //else use what is set in input file 
    pars->jtwist = jtwist;
  }
  if(pars->jtwist!=0 && abs(pars->shat)>1.e-6) pars->x0 = pars->y0*pars->jtwist/(2*M_PI*pars->Zp*abs(pars->shat));  
  //if(abs(pars->shat)<1.e-6) pars->x0 = pars->y0;
  
}

