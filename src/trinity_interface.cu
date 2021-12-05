#include "run_gx.h"
#include "helper_cuda.h"
#include "trinity_interface.h"

void gx_get_fluxes_(trin_parameters_struct* tpars, trin_fluxes_struct* tfluxes, char* run_name, int mpcom_f)
{
  Parameters* pars = nullptr;
  MPI_Comm mpcom = MPI_Comm_f2c(mpcom_f);
  int iproc;
  MPI_Comm_rank(mpcom, &iproc);
  pars = new Parameters(mpcom);
  // get default values from namelist
  pars->get_nml_vars(run_name);

  // overwrite parameters based on values from trinity
  pars->set_from_trinity(tpars);

  // initialize grids, geometry, and diagnostics
  Grids       * grids       = nullptr;
  
  DEBUGPRINT("Initializing grids...\n");
  grids = new Grids(pars);
  CUDA_DEBUG("Initializing grids: %s \n");

  DEBUGPRINT("Grid dimensions: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n",
	     grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // run gx calculation using (updated) parameters
  run_gx(pars, grids);

  // copy time-averaged fluxes to trinity
  //diagnostics->copy_fluxes_to_trinity(tfluxes);

  delete pars;
  delete grids;
}
