#include "gryfx_lib.h"
#include "mpi.h"
#include "cufft.h"
#include "parameters.h"
#include "run_gryfx.h"
#include "geometry.h"
#include <assert.h>


__global__ void test(float* z) {
  printf("Device: %f\n", z[0]);
}

void gryfx_get_default_parameters_(struct external_parameters_struct * externalpars, char * namelistfileName, int mpcom) {  
  
  int iproc;

  printf("Communicator is %d\n", mpcom);
  
  MPI_Comm_rank(mpcom, &iproc);

  int numdev;

  cudaGetDeviceCount(&numdev);

  cudaGetDevice(&externalpars->mpirank);

  if(iproc==0) printf("Initializing GryfX...\tNamelist is %s\n", namelistfileName);

  // read input parameters from namelist
  Parameters *pars = new Parameters;
  pars->read_namelist(namelistfileName);

  // copy elements of input_parameters_struct into external_parameters_struct externalpars
  if (iproc==0) pars->set_externalpars(externalpars);

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
  externalpars->pars_address = (void *)pars; 
  printf("Finished gryfx_get_default_parameters_\n");

}
  
void gryfx_get_fluxes_(struct external_parameters_struct *  externalpars, 
			struct gryfx_outputs_struct * gryfxouts, char* namelistfileName, int mpcom)
{
   int iproc;
   // iproc doesn't necessarily have to be the same as it was in 
   // gryfx_get_default_parameters_
   MPI_Comm_rank(mpcom, &iproc);

   Parameters* pars = (Parameters *)externalpars->pars_address;

   pars->iproc = iproc;
   
/*
   ev->mpi.iproc = iproc;
   ev->mpi.mpcom = mpcom;

   //This only important for Trinity.
   ev->info.job_id = externalpars->job_id;
  
  
  // Copy the name of the namelist file to ev->info.run_name
  // Check if we should and can restart and set the file name
  setup_info(namelistfileName, &ev->pars, &ev->info);
*/

//  int gpuID = externalpars->job_id;

  // Only proc0 needs to import paramters to gryfx
  // copy elements of external_parameters_struct externalpars into pars
  // this is done because externalpars may have been changed externally (i.e. by Trinity) 
  // between calls to gryfx_get_default_parameters and gryfx_get_fluxes.
  // pars then needs to be updated since pars is what is used in run_gryfx.
  if(iproc==0) {
    pars->import_externalpars(externalpars);
  }

  /////////////////////////
  // This is the main call
  ////////////////////////
  run_gryfx(pars, gryfxouts->pflux, gryfxouts->qflux);

  delete pars;

}  	

void gryfx_main(int argc, char* argv[], int mpcom) {
  struct external_parameters_struct externalpars;
  struct gryfx_outputs_struct gryfxouts;
  char* namelistfileName;
  if(argc == 2) {
    namelistfileName = argv[1];
    //printf("namelist = %s\n", namelistfileName);
  }
  else {
    fprintf(stderr, "The correct usage is:\n./gryfx <inputfile>\n");
    exit(1);
  }
  gryfx_get_default_parameters_(&externalpars, namelistfileName, mpcom);
  gryfx_get_fluxes_(&externalpars, &gryfxouts, namelistfileName, mpcom);

}
