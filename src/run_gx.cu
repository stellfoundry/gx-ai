#include "run_gx.h"

void getDeviceMemoryUsage() {
  cudaDeviceSynchronize();
  // show memory usage of GPU
  size_t free_byte;  size_t total_byte;  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte) ;

  if ( cudaSuccess != cuda_status ){
      printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
      exit(1);
  }
  // for some reason, total_byte returned by above call is not correct. 

  int dev;
  cudaDeviceProp prop;
  checkCuda( cudaGetDevice(&dev) );
  checkCuda( cudaGetDeviceProperties(&prop, dev) );
  double free_db = (double) free_byte;
  double total_db = (double) prop.totalGlobalMem;
  double used_db = total_db - free_db ;
  printf("GPU memory usage: used = %f MB (%f %%), free = %f MB (%f %%), total = %f MB\n",
	 used_db /1024.0/1024.0, used_db/total_db*100.,
	 free_db /1024.0/1024.0, free_db/total_db*100.,
	 total_db/1024.0/1024.0);
}

void run_gx(Parameters *pars, Grids* grids, Geometry* geo, Diagnostics* diagnostics)
{
  double time = 0;

  Fields    * fields    = NULL;
  MomentsG  * G         ;
  Solver    * solver    = NULL;
  Linear    * linear    = NULL;
  Nonlinear * nonlinear = NULL;
  Forcing   * forcing   = NULL;
  
  G         = new MomentsG (pars, grids);
  solver    = new Solver   (pars, grids, geo);    
  
  /////////////////////////////////
  //                             //
  // Initialize gyrokinetic eqs  // 
  //                             //
  /////////////////////////////////
  if (pars->gx) {

    linear = new Linear(pars, grids, geo);    
    fields = new Fields(pars, grids);
    G      -> initialConditions(geo, &time);
    solver -> fieldSolve(G, fields);    
    
    if (!pars->linear) nonlinear = new Nonlinear(pars, grids, geo);
    
    if (pars->forcing_init) {
      if (pars->forcing_type == "Kz")        forcing = new KzForcing(pars);
      if (pars->forcing_type == "KzImpulse") forcing = new KzForcingImpulse(pars);
      if (pars->forcing_type == "general")   forcing = new genForcing(pars);
    }
  }

  //////////////////////////////
  //                          //
  // Kuramoto-Sivashinsky eq  // 
  //                          //
  //////////////////////////////  
  if (pars->ks) {
    linear    = new Linear(pars, grids);    

    G -> initialConditions(&time);
    G -> qvar(grids->Naky);
    if (!pars->linear) nonlinear = new Nonlinear(pars, grids);
  }    

  Timestepper * timestep;
  if(pars->scheme_opt == K10)   timestep = new Ketcheson10 (linear, nonlinear, solver, pars, grids, forcing, pars->dt);
  if(pars->scheme_opt == RK4)   timestep = new RungeKutta4 (linear, nonlinear, solver, pars, grids, forcing, pars->dt);
  if(pars->scheme_opt == RK3)   timestep = new RungeKutta3 (linear, nonlinear, solver, pars, grids, forcing, pars->dt);
  if(pars->scheme_opt == RK2)   timestep = new RungeKutta2 (linear, nonlinear, solver, pars, grids, forcing, pars->dt);
  if(pars->scheme_opt == SSPX2) timestep = new SSPx2       (linear, nonlinear, solver, pars, grids, forcing, pars->dt);
  if(pars->scheme_opt == SSPX3) timestep = new SSPx3       (linear, nonlinear, solver, pars, grids, forcing, pars->dt);
  
  getDeviceMemoryUsage();
  
  if (pars->write_moms) diagnostics -> write_init(G, fields);
	 
  // TIMESTEP LOOP
  int counter = 0;           float timer = 0;          cudaEvent_t start, stop;    bool checkstop = false;
  cudaEventCreate(&start);   cudaEventCreate(&stop);   cudaEventRecord(start,0);

  diagnostics -> loop(G, fields, timestep->get_dt(), counter, time);
  
  while(counter<pars->nstep) {
    counter++;
    timestep -> advance(&time, G, fields);
    checkstop = diagnostics -> loop(G, fields, timestep->get_dt(), counter, time);
    if (checkstop) break;
    if (counter % pars->nreal == 0)  {
      G -> reality(grids->Nl * grids->Nm * grids->Nspecies); 
      solver -> fieldSolve(G, fields);
    }
  }

  if (pars->save_for_restart) G->restart_write(&time);
  
  cudaEventRecord(stop,0);    cudaEventSynchronize(stop);    cudaEventElapsedTime(&timer,start,stop);
  printf("Total runtime = %f s (%f s / timestep)\n", timer/1000., timer/1000./counter);

  diagnostics->finish(G, fields);

  if (G)         delete G;
  if (solver)    delete solver;
  if (linear)    delete linear;
  if (nonlinear) delete nonlinear;
  if (timestep)  delete timestep;

  if (fields)    delete fields;
  if (forcing)   delete forcing;     
}    




