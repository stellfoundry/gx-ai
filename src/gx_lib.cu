#include "gx_lib.h"
#include "mpi.h"
#include "cufft.h"
#include "parameters.h"
#include "run_gx.h"
#include "geometry.h"
#include "get_error.h"
#include <assert.h>


__global__ void test(float* z) {
  printf("Device: %f\n", z[0]);
}

void gx_get_default_parameters_(struct external_parameters_struct * externalpars, char * namelistfileName, MPI_Comm mpcom) {  
  
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
  printf("Finished gx_get_default_parameters_\n");

}
  
void gx_get_fluxes_(struct external_parameters_struct *  externalpars, 
			struct gx_outputs_struct * gxouts, char* namelistfileName, MPI_Comm mpcom)
{
   int iproc;
   // iproc doesn't necessarily have to be the same as it was in 
   // gx_get_default_parameters_
   MPI_Comm_rank(mpcom, &iproc);

   Parameters* pars = (Parameters *)externalpars->pars_address;

   pars->iproc = iproc;
   
//  int gpuID = externalpars->job_id;

  // Only proc0 needs to import paramters to gx
  // copy elements of external_parameters_struct externalpars into pars
  // this is done because externalpars may have been changed externally (i.e. by Trinity) 
  // between calls to gx_get_default_parameters and gx_get_fluxes.
  // pars then needs to be updated since pars is what is used in run_gx.
  if(iproc==0) {
    pars->import_externalpars(externalpars);
  }

  Geometry* geo;  // geometry coefficient arrays
  Grids* grids;   // grids (e.g. kx, ky, z)
  Diagnostics* diagnostics;

  printf("Initializing grids...\n");
  grids = new Grids(pars);
  checkCuda(cudaGetLastError());
  printf("Grid dimensions: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n", grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);

  if(iproc==0) {
    int igeo = pars->igeo;
    printf("Initializing geometry...\n");
    if(igeo==0) {
      geo = new S_alpha_geo(pars, grids);
    }
    else if(igeo==1) {
      geo = new File_geo(pars, grids);
    } 
    else if(igeo==2) {
      printf("igeo = 2 not yet implemented!\n");
      exit(1);
      //geo = new Eik_geo();
    } 
    else if(igeo==3) {
      printf("igeo = 3 not yet implemented!\n");
      exit(1);
      //geo = new Gs2_geo();
    }
    checkCuda(cudaGetLastError());

    printf("Initializing diagnostics...\n");
    diagnostics = new Diagnostics(pars, grids, geo);
    checkCuda(cudaGetLastError());
  }

  cudaDeviceSynchronize();

  /////////////////////////
  // This is the main call
  ////////////////////////
  run_gx(pars, grids, geo, diagnostics);

  memcpy(gxouts->qflux, diagnostics->qflux, sizeof(double)*grids->Nspecies);
  memcpy(gxouts->pflux, diagnostics->pflux, sizeof(double)*grids->Nspecies);

  delete pars;
  delete grids;
  delete geo;
  delete diagnostics;
}  	

void gx_main(int argc, char* argv[], MPI_Comm mpcom) {
  struct external_parameters_struct externalpars;
  struct gx_outputs_struct gxouts;
  char* namelistfileName;
  if(argc == 2) {
    namelistfileName = argv[1];
    //printf("namelist = %s\n", namelistfileName);
  }
  else {
    fprintf(stderr, "The correct usage is:\n./gx <inputfile>\n");
    exit(1);
  }
  gx_get_default_parameters_(&externalpars, namelistfileName, mpcom);
  gx_get_fluxes_(&externalpars, &gxouts, namelistfileName, mpcom);

}
