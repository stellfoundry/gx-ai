#include "gryfx_lib.h"
#include "mpi.h"
#include "cufft.h"
#include "inputs.h"
#include "geometry.h"

//Defined at the bottom of this file
void set_externalpars(struct external_parameters_struct * externalpars, input_parameters_struct * pars);
//void import_externalpars(struct external_parameters_struct * externalpars, everything_struct * everything_ptr);
//void initialize_cuda_parallelization(everything_struct * ev);

void gryfx_get_default_parameters_(struct external_parameters_struct * externalpars, char * namelistFile, int mpcom) {  
  
  int iproc;

  printf("Communicator is %d\n", mpcom);
  
  MPI_Comm_rank(mpcom, &iproc);


#ifdef GS2_zonal
  if(iproc==0) printf("\n\n========================================\nThis is a hybrid GryfX-GS2 calculation.\n========================================\n\n");
#endif

  int numdev;

  cudaGetDeviceCount(&numdev);

  cudaGetDevice(&externalpars->mpirank);

//#else
//  iproc=0;
//#endif

  if(iproc==0) printf("Initializing GryfX...\tNamelist is %s\n", namelistFile);

  // read input parameters from namelist
  Inputs *inputs = new Inputs;
  inputs->read_namelist(namelistFile);
// To be moved...
//	set_grid_masks_and_unaliased_sizes(&(ev->grids));

  // copy elements of input_parameters_struct into external_parameters_struct externalpars
  if (iproc==0) inputs->set_externalpars(externalpars);

  int nprocs;

  MPI_Comm_size(mpcom, &nprocs);

  char serial_full[100];
  char serial[100];
  FILE *fp;

  fp = popen("nvidia-smi -q | grep Serial", "r");
  while(fgets(serial_full, sizeof(serial_full)-1,fp) != NULL) {
    printf("%s\n", serial_full);
  }
  pclose(fp);
  for(int i=0; i<8; i++) {
   serial[i] = serial_full[strlen(serial_full) - (9-i)];
  }
  serial[8] = NULL;
  externalpars->job_id = atoi(serial);
  printf("SN: %d\n", externalpars->job_id);

  printf("About to broadcast externalpars %d %d\n", nprocs, iproc);

  // EGH: this is a nasty way to broadcast externalpars... we should
  // really define a custom MPI datatype. However it should continue
  // to work as long as all MPI processes are running on the same
  // architecture. 
  int ret;
  ret = MPI_Bcast(&*externalpars, sizeof(external_parameters_struct), MPI_BYTE, 0, mpcom);
  printf("Broadcasted externalpars (%d) %d %d\n", ret, nprocs, iproc);
  // This has to be set after the broadcast
  externalpars->inputs_address = (void *)inputs; 
  printf("Finished gryfx_get_default_parameters_\n");

}
  
void gryfx_get_fluxes_(struct external_parameters_struct *  externalpars, 
			struct gryfx_outputs_struct * gryfxouts, char* namelistFile, int mpcom)
{

	 //everything_struct * ev;
   FILE* outfile;

   int iproc;
   // iproc doesn't necessarily have to be the same as it was in 
   // gryfx_get_default_parameters_
   MPI_Comm_rank(mpcom, &iproc);

   Inputs* inputs = (Inputs *)externalpars->inputs_address;
   //input_parameters_struct* pars = inputs->getpars();

   
/*
   ev->mpi.iproc = iproc;
   ev->mpi.mpcom = mpcom;

   //This only important for Trinity.
   ev->info.job_id = externalpars->job_id;
  
  
  // Copy the name of the namelist file to ev->info.run_name
  // Check if we should and can restart and set the file name
  setup_info(namelistFile, &ev->pars, &ev->info);
*/

  int gpuID = externalpars->job_id;

  Geometry* geo;

  if(iproc==0) {
    //Only proc0 needs to import paramters to gryfx
    // copy elements of external_parameters_struct externalpars into inputs
    // this is done because externalpars may have been changed externally (i.e. by Trinity) 
    // between calls to gryfx_get_default_parameters and gryfx_get_fluxes.
    // inputs then needs to be updated since inputs is what is used in run_gryfx.

    inputs->import_externalpars(externalpars);

    printf("%d: Initializing geometry...\n\n", gpuID);

    input_parameters_struct* pars = inputs->getpars();
    int igeo = pars->igeo;

    if(igeo==0) {
      geo = new S_alpha_geo(pars->Nz);
    } else if(igeo==1) {
      geo = new Eik_geo();
    } else if(igeo==2) {
      geo = new Gs2_geo();
    }

    //init_gs2_file_utils(&len, namelistFile);
    //set_geometry(&ev->pars, &ev->grids, &ev->geo, externalpars);
    //finish_gs2_file_utils();
    ////Note the printout module still uses globals which may no 
    ////longer be in sync with ev at this point. Need to remove globals
    ////from printout.cu
    //if(ev->pars.debug) print_cuda_properties(ev);
  }

#ifdef GS2_zonal
  if(iproc==0) printf("%d: Initializing GS2...\n\n", ev->info.gpuID);
  gryfx_initialize_gs2(&ev->grids, externalpars, namelistFile, mpcom);
  if(iproc==0) printf("%d: Finished initializing GS2.\n\n", ev->info.gpuID);
#endif
 

/*

  
  //make an input file of form outstem.in if doesn't already exist
  FILE* input;
  FILE* namelist;
  char inputFile[2000];
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
    char outfileName[2000];
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
*/

}  	

void gryfx_main(int argc, char* argv[], int mpcom) {
  struct external_parameters_struct externalpars;
  struct gryfx_outputs_struct gryfxouts;
  char* namelistFile;
  if(argc == 2) {
    namelistFile = argv[1];
    //printf("namelist = %s\n", namelistFile);
  }
  else {
    fprintf(stderr, "The correct usage is:\n./gryfx <inputfile>\n");
    exit(1);
  }
  gryfx_get_default_parameters_(&externalpars, namelistFile, mpcom);
  gryfx_get_fluxes_(&externalpars, &gryfxouts, namelistFile, mpcom);

}

/*
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
*/

