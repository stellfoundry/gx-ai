#include "mpi.h"
#include "geometry.h"
#include "parameters.h"
#include "grids.h"
#include "fields.h"
#include "moments.h"
#include "solver.h"
#include "forcing.h"
#include "timestepper.h"
#include "linear.h"
#include "diagnostics.h"
#include "cuda_constants.h"
#include "get_error.h"

#ifdef GS2_zonal
extern "C" void broadcast_integer(int* a);
#endif

void getDeviceMemoryUsage() {
  cudaDeviceSynchronize();
  // show memory usage of GPU
  size_t free_byte ;
  size_t total_byte ;
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  if ( cudaSuccess != cuda_status ){
      printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
      exit(1);
  }
  // for some reason, total_byte returned by above call is not correct. 
  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, 0) );
  double free_db = (double) free_byte;
  double total_db = (double) prop.totalGlobalMem;
  double used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f MB (%f %%), free = %f MB (%f %%), total = %f MB\n",
      used_db/1024.0/1024.0, used_db/total_db*100., free_db/1024.0/1024.0, free_db/total_db*100., total_db/1024.0/1024.0);
}

void run_gx(Parameters *pars, Grids* grids, Geometry* geo, Diagnostics* diagnostics)
{
  int iproc = pars->iproc;  

  Fields *fields;
  MomentsG *momsG;
  Solver *solver;
  Linear* linear;
  Forcing *forcing;
  Timestepper *stepper; 

  if(iproc == 0) {
    geo->initializeOperatorArrays(pars, grids);
    checkCuda(cudaGetLastError());

    printf("Initializing fields...\n");
    fields = new Fields(grids);
    checkCuda(cudaGetLastError());

    printf("Initializing moments...\n");
    momsG = new MomentsG(grids);
    checkCuda(cudaGetLastError());
    printf("Setting initial conditions...\n");
    momsG->initialConditions(pars, geo);
    checkCuda(cudaGetLastError());

    printf("Initializing field solver...\n");
    solver = new Solver(pars, grids, geo);
    checkCuda(cudaGetLastError());

    // initialize fields using field solve
    printf("Solving for initial fields...\n");
    solver->fieldSolve(momsG, fields);
    checkCuda(cudaGetLastError());

    printf("Initializing equations...\n");
    linear = new Linear(pars, grids, geo);
    checkCuda(cudaGetLastError());
   
    if (pars->forcing_init == true) {
      printf("Initializing forcing...\n");
      if (strcmp(pars->forcing_type, "Z") == 0) {
        forcing = new ZForcing(pars, grids, geo);
      } 
    }
    else {
      forcing = NULL;
    }
    checkCuda(cudaGetLastError());

    printf("Initializing timestepper...\n");
    if(pars->scheme == RK4) {
      stepper = new RungeKutta4(linear, solver, grids, forcing, pars->dt);
    } else {
      stepper = new RungeKutta2(linear, solver, grids, forcing, pars->dt);
    }
    checkCuda(cudaGetLastError());

    printf("After initialization:\n");
    getDeviceMemoryUsage();
  
    diagnostics->writeMomOrField(momsG->dens_ptr[0], "dens0");
    diagnostics->writeMomOrField(fields->phi, "phi0");

    // MFM
    if (pars->igeo == 1) {
      diagnostics->writeGridFile("geofile");
    }
  }

  // TIMESTEP LOOP
  int counter = 0;
  double time = 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timer = 0;
  cudaEventRecord(start,0);

  bool checkstop = false;
  printf("Running %d timesteps.......\n", pars->nstep);
  printf("dt = %f\n", stepper->get_dt());
  while(counter<pars->nstep) {
    if(iproc==0) {
      stepper->advance(&time, momsG, fields);
      checkstop = diagnostics->loop_diagnostics(momsG, fields, stepper->get_dt(), counter, time);
      if(checkstop) break;
      if(counter%(pars->nwrite*100)==0) diagnostics->final_diagnostics(momsG, fields);
    }
    counter++;
  }

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timer,start,stop);

  printf("Step %d\n", counter);
  printf("Finished timestep loop\n");
  checkCuda(cudaGetLastError());

  printf("Total runtime = %f s (%f s / timestep)\n", timer/1000., timer/1000./counter);

  diagnostics->final_diagnostics(momsG, fields);

  printf("Cleaning up...\n");

  delete fields;
  delete momsG;
  delete solver;
  delete linear;
  delete forcing;
  delete stepper;

}    




