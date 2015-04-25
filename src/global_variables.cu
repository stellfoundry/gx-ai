#define EXTERN_SWITCH 
#include "cufft.h"
#include "simpledataio_cuda.h"
#include "everything_struct.h"
#include "global_variables.h"

bool globals_initialized = false;
void initialize_globals(){

  if (!globals_initialized) {
 //   irho = 2;
  globals_initialized = true;
}
}

void set_globals_after_gryfx_lib(everything_struct * ev){
    // Defaults that used to be set in global_vars.h
    maxdt = 0.02;  // Surely this is unnecessary?
    cfl_flag = true;
    converge_bounds = 20;
    converge_stop = 10000;
    // Stuff set in definitions.cu
    D_par = ev->damps.D_par;
    D_prp = ev->damps.D_prp;
    Beta_par = ev->damps.Beta_par;
    Ny_unmasked = ev->grids.Ny_unmasked;
    Nx_unmasked = ev->grids.Nx_unmasked;
    nu = ev->damps.nu;
    mu = ev->damps.mu;
    // Set in gryfx_lib.cu
    iproc = ev->mpi.iproc;
    mpcom_global = ev->mpi.mpcom;
    gpuID = ev->info.gpuID;
    out_stem = ev->info.run_name;
    restartfileName = ev->info.restart_file_name;
    RESTART = ev->pars.restart;
    zBlockThreads = ev->cdims.zBlockThreads;
    zThreads = ev->cdims.zThreads;
    totalThreads = ev->cdims.totalThreads;
    dimBlock = ev->cdims.dimBlock;
    dimGrid = ev->cdims.dimGrid;
}
