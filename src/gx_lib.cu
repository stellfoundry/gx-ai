#include "gx_lib.h"
#include "parameters.h"
#include "run_gx.h"
#include "geometry.h"
#include "get_error.h"
#include <assert.h>

void gx_get_default_parameters_(struct external_parameters_struct * externalpars,
				char *run_name, MPI_Comm mpcom, int devid) {  

  int iproc;

  //  printf("Communicator is %d\n", mpcom);
  
  MPI_Comm_rank(mpcom, &iproc);

  int numdev;

  cudaGetDeviceCount(&numdev);
  //
  // If there are multiple devices available, this needs work  (bug)
  cudaSetDevice(devid);

  cudaGetDevice(&externalpars->mpirank); // assuming there is one GPU per CPU this
                                         // sets externalpars->mpirank to be the devid for that GPU
  if(iproc==0 && false) printf("Initializing gx ...\t runname is %s\n", run_name);

  // read input parameters from namelist
  Parameters *pars = new Parameters;
  //  pars->read_namelist(run_name);
  pars->get_nml_vars(run_name);
  //
  // With the present setup, this call to get_nml_vars will be 
  // run on every host (for all values of iproc). 
  // Each one will attempt to open an output file called run_name.nc
  // This is fine for gx as a standalone code with one host and one dev
  // but it is not correct for one host and two devs.
  // If there were multiple calls to gx as a library, from Trinity,
  // then as long as one gx is running on one host with one device, this is fine.
  // Well, actually, it might be that Trinity maintains a family with a single
  // run_name and then this will be problematic. Depends on Trinity usage.

  
  // copy elements of input_parameters_struct into external_parameters_struct externalpars
  if (iproc==0) pars->set_externalpars(externalpars);

  int nprocs;

  MPI_Comm_size(mpcom, &nprocs);
  
  char serial_full[100];
  char serial[100];
  //  FILE *fp;

  //  fp = popen("nvidia-smi -q | grep Serial", "r");
  //  while(fgets(serial_full, sizeof(serial_full)-1,fp) != NULL) {
  //    printf("%s\n", serial_full);
  //  }
  //  pclose(fp);
  
  for(int i=0; i<8; i++) {
    serial[i] = serial_full[strlen(serial_full) - (9-i)];
  }
  serial[8] = NULL;
  externalpars->job_id = atoi(serial);
  if (false) printf("SN: %d\n", externalpars->job_id);
  
  if (false) printf("About to broadcast externalpars %d %d\n", nprocs, iproc);

  // BD: I think this is designed for a case with multiple cores that 
  // are running an instance of (linear) gs2 (with sources from gx nonlinearities)
  // on multiple CPUs, and a single copy of gx for this namespace?
  // And then Trinity is managing the calls to these gx+gs2 instances?
  
  
  //  
  // EGH: this is a nasty way to broadcast externalpars... we should
  // really define a custom MPI datatype. However it should continue
  // to work as long as all MPI processes are running on the same
  // architecture. 
  int ret;
  ret = MPI_Bcast(&*externalpars, sizeof(external_parameters_struct), MPI_BYTE, 0, mpcom);
  if (false) printf("Broadcasted externalpars (%d) %d %d\n", ret, nprocs, iproc);
  // This has to be set after the broadcast
  externalpars->pars_address = (void *)pars; 
  if (false) printf("Finished gx_get_default_parameters_\n");

}
  
void gx_get_fluxes_(struct external_parameters_struct *  externalpars, 
		    struct gx_outputs_struct * gxouts, MPI_Comm mpcom)
{
  int iproc;
  // iproc doesn't necessarily have to be the same as it was in 
  // gx_get_default_parameters_
  MPI_Comm_rank(mpcom, &iproc);
  
  Parameters* pars = (Parameters *)externalpars->pars_address;
  
  pars->iproc = iproc;
  
  //  int gpuID = externalpars->job_id;
  
  // Only proc0 needs to import parameters to gx
  // copy elements of external_parameters_struct externalpars into pars
  // this is done because externalpars may have been changed externally (i.e. by Trinity) 
  // between calls to gx_get_default_parameters and gx_get_fluxes.
  // pars then needs to be updated since pars is what is used in run_gx.
  if(iproc==0) { pars->import_externalpars(externalpars); }
  
  Geometry    * geo         = NULL;  // geometry coefficient arrays
  Grids       * grids       = NULL;  // grids (e.g. kx, ky, z)
  Diagnostics * diagnostics = NULL;
  //  HermiteTransform* herm;

  DEBUGPRINT("Initializing grids...\n");
  grids = new Grids(pars);
  CUDA_DEBUG("Initializing grids: %s \n");
  
  DEBUGPRINT("Grid dimensions: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n",
	     grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);

  if(iproc==0) {
    int igeo = pars->igeo;
    DEBUGPRINT("Initializing geometry...\n");
    if(igeo==0) {
      geo = new S_alpha_geo(pars, grids);
      CUDA_DEBUG("Initializing geometry s_alpha: %s \n");
    }
    else if(igeo==1) {
      geo = new File_geo(pars, grids);
      printf("************************* \n \n \n");
      printf("Warning: assumed grho = 1 \n \n \n");
      printf("************************* \n");
      CUDA_DEBUG("Initializing geometry from file: %s \n");
    } 
    else if(igeo==2) {
      DEBUGPRINT("igeo = 2 not yet implemented!\n");
      exit(1);
      //geo = new Eik_geo();
    } 
    else if(igeo==3) {
      DEBUGPRINT("igeo = 3 not yet implemented!\n");
      exit(1);
      //geo = new Gs2_geo();
    }

    DEBUGPRINT("Initializing diagnostics...\n");
    diagnostics = new Diagnostics(pars, grids, geo);
    CUDA_DEBUG("Initializing diagnostics: %s \n");    

    DEBUGPRINT("Initializing Hermite transforms...\n");
    //    herm = new HermiteTransform(grids, 1); // batch size could ultimately be nspec
    CUDA_DEBUG("Initializing Hermite transforms: %s \n");    
  }

  cudaDeviceSynchronize();

  run_gx(pars, grids, geo, diagnostics);

  delete pars;
  delete grids;
  delete geo;
  delete diagnostics;
}  	

void gx_main(int argc, char* argv[], MPI_Comm mpcom) {
  struct external_parameters_struct externalpars;
  struct gx_outputs_struct gxouts;

  int devid = 0; // This should be determined (optionally) on the command line
  
  char *run_name;
  if ( argc < 1) {
    fprintf(stderr, "The correct usage is:\n gx <runname>\n");
    exit(1);
  } else {    
    run_name = argv[1];
  }

  gx_get_default_parameters_(&externalpars, run_name, mpcom, devid);
  gx_get_fluxes_(&externalpars, &gxouts, mpcom);
}

