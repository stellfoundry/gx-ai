#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "gx_lib.h"
#include "version.h"

int main(int argc, char* argv[])
{

  MPI_Init(&argc, &argv);
  MPI_Comm mpcom = MPI_COMM_WORLD;
  int iproc;
  MPI_Comm_rank(mpcom, &iproc);
  
  int devid = 0; // This should be determined (optionally) on the command line
  checkCuda(cudaSetDevice(devid));
  cudaDeviceSynchronize();
 
  char *run_name;
  if ( argc < 1) {
    fprintf(stderr, "The correct usage is:\n gx <runname>\n");
    exit(1);
  } else {    
    run_name = argv[1];
    printf("Running %s \n",run_name);
  }
  
  printf("Version: %s \t Compiled: %s \n", build_git_sha, build_git_time);

  Parameters * pars         = nullptr;
  pars = new Parameters();
  pars->iproc = iproc;
  pars->get_nml_vars(run_name);
  
  Geometry    * geo         = nullptr;
  Grids       * grids       = nullptr;
  Diagnostics * diagnostics = nullptr;
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

    //    DEBUGPRINT("Initializing Hermite transforms...\n");
    //    herm = new HermiteTransform(grids, 1); // batch size could ultimately be nspec
    //    CUDA_DEBUG("Initializing Hermite transforms: %s \n");    
  }

  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  
  run_gx(pars, grids, geo, diagnostics);

  delete pars;
  delete grids;
  delete geo;
  delete diagnostics;

  MPI_Finalize();
  cudaDeviceReset();
}
