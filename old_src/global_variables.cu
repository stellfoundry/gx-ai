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
    cflx = ev->time.cflx;
    cfly = ev->time.cfly;
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

    Nx = ev->grids.Nx;
    Ny = ev->grids.Ny;
    Nz = ev->grids.Nz;

   input_parameters_struct * pars = &ev->pars;
   irho = pars->irho ;
   rhoc = pars->rhoc ;
   eps = pars->eps;
   bishop = pars->bishop ;
   nperiod = pars->nperiod ;

 /* Miller parameters*/
   rmaj = pars->rmaj ;
   r_geo = pars->r_geo ;
   akappa  = pars->akappa ;
   akappri = pars->akappri ;
   tri = pars->tri ;
   tripri = pars->tripri ;
   shift = pars->shift ;
   qsf = pars->qsf;
   shat = pars->shat ;
    // EGH These appear to be redundant

  /* Other geometry parameters - Bishop/Greene & Chance*/
   beta_prime_input = pars->beta_prime_input ;
   s_hat_input = pars->s_hat_input ;

  /*Flow shear*/
   g_exb = pars->g_exb ;

  jtwist = pars->jtwist;
  X0 = pars->x0;

  //set in geometry.cu
  drhodpsi = pars->drhodpsi;
  kxfac = pars->kxfac;
  gradpar = ev->geo.gradpar;
  bi = ev->geo.bi;
  aminor = ev->geo.aminor;

}
