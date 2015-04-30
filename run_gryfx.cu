#include <math.h>
#include "simpledataio_cuda.h"
#include "cufft.h"
#include "sys/stat.h"
#include "cuda_profiler_api.h"
#include "libgen.h"
#include "mpi.h"
#include "nvToolsExt.h"


// EGH: Until we do something more fancy with cudaGetSymbolAddress the following have
// to have file scope... ie they have to be in the same aggregate source file as the 
// kernels.

__constant__ int nx,ny,nz, zthreads, nspecies;
__constant__ float X0_d,Y0_d;
__constant__ int Zp_d;

#define EXTERN_SWITCH extern
#include "everything_struct.h"
#include "global_variables.h"
#include "allocations.h"
#include "write_data.h"
#include "gs2.h"
#include "get_error.h"
#include "device_funcs.cu"
#include "operations_kernel.cu"
#include "diagnostics_kernel.cu"
#include "exb_kernel.cu"
#include "nlps_kernel.cu"
#include "zderiv_kernel.cu"
#include "covering_kernel.cu"
#include "reduc_kernel.cu"
#include "cudaReduc_kernel.cu"
#include "init_kernel.cu"
#include "omega_kernel.cu"
#include "phi_kernel.cu"
#include "qneut_kernel.cu"
#include "nlpm_kernel.cu"
#include "zonal_kernel.cu"
#include "getfcn.cu"
#include "maxReduc.cu"
#include "sumReduc.cu"
#include "diagnostics.cu"
#include "coveringSetup.cu"
#include "exb.cu"
#include "qneut.cu"
#include "ztransform_covering.cu"
#include "zderiv.cu"
#include "zderivB.cu"
#include "zderiv_covering.cu"
#include "nlps.cu"
#include "nlpm.cu"
#include "hyper.cu"
#include "courant.cu"
#include "energy.cu"
#include "timestep_gryfx.cu"


#ifdef GS2_zonal
//extern "C" void fields_implicit_mp_advance_implicit_(int* gs2_counter, cuComplex* dens_ky0_h, cuComplex* upar_ky0_h, cuComplex* tpar_ky0_h, cuComplex* tprp_ky0_h, cuComplex* qpar_ky0_h, cuComplex* qprp_ky0_h);
//extern "C" void fields_mp_advance_(int* gs2_counter);
//extern "C" void gs2_main_mp_advance_gs2_(int* gs2_counter, cuComplex* dens_ky0_h, cuComplex* upar_ky0_h, cuComplex* tpar_ky0_h, cuComplex* tprp_ky0_h, cuComplex* qpar_ky0_h, cuComplex* qprp_ky0_h, cuComplex* phi_ky0_h, int* first_half_flag);
extern "C" void advance_gs2(int* gs2_counter, cuComplex* dens_ky0_h, cuComplex* upar_ky0_h, cuComplex* tpar_ky0_h, cuComplex* tprp_ky0_h, cuComplex* qpar_ky0_h, cuComplex* qprp_ky0_h, cuComplex* phi_ky0_h, int* first_half_flag);
//extern "C" void gs2_time_mp_update_time_();
//extern "C" void gs2_reinit_mp_check_time_step_(int* reset, int* exit);
//extern "C" void gs2_reinit_mp_reset_time_step_(int* gs2_counter, int* exit);
extern "C" void getmoms_gryfx(cuComplex* dens, cuComplex* upar, cuComplex* tpar, cuComplex* tprp, cuComplex* qpar, cuComplex* qprp, cuComplex* phi);
//extern "C" void gs2_diagnostics_mp_loop_diagnostics_(int* gs2_counter, int* exit);
extern "C" void broadcast_integer(int* a);
//extern "C" void broadcast_integer(int* a);
//extern "C" void species_mp_reinit_species_(int* nSpecies, double* dens, double* temp, double* fprim, double* tprim, double* nu);
#endif



#include "run_gryfx_functions.cu"

void set_cuda_constants(){
  cudaMemcpyToSymbol(nx, &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ny, &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nz, &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nspecies, &nSpecies, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(X0_d, &X0, sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Y0_d, &Y0, sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Zp_d, &Zp, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(zthreads, &zThreads, sizeof(int),0,cudaMemcpyHostToDevice);
 // cudaMemcpyToSymbol(zblockthreads, &zBlockThreads, sizeof(int),0,cudaMemcpyHostToDevice);

  if(DEBUG) getError("run_gryfx.cu, copied constants");
}

void run_gryfx(everything_struct * ev_h, double * pflux, double * qflux, FILE* outfile)//, FILE* omegafile,FILE* gammafile, FILE* energyfile, FILE* fluxfile, FILE* phikyfile, FILE* phikxfile, FILE* phifile)
{

    set_globals_after_gryfx_lib(ev_h);
    if (iproc==0) set_cuda_constants();

    
    //int naky, ntheta0;// nshift;
    //naky = 1 + (Ny-1)/3;
    //ntheta0 = 1 + 2*((Nx-1)/3);     //MASK IN MIDDLE OF ARRAY
    int ntheta0 = ev_h->grids.ntheta0;
    
    float* Dnlpm_d;
    float* Phi_zf_kx1_d;
    float Dnlpm = 0;
    float Dnlpm_avg = 0;
    float Dnlpm_sum = 0;
    //float* phiVal; 
    //float phiVal0;
  
    float Phi_zf_kx1 = 0.;
    float Phi_zf_kx1_old = 0.;
    //float Phi_zf_kx1_sum = 0.;
    float Phi_zf_kx1_avg = 0.;
    float alpha_nlpm = 0.;
    float mu_nlpm = 0.;
    //float tau_nlpm = 50.;
    //cuComplex *init_h;
    //float Phi_energy;
    //cuComplex *omega_h;  
    //float dtBox[navg];
    cuComplex* omegaAvg_h;
    float pflx[nSpecies];
    float wpfx_old[nSpecies];
    float pflx_old[nSpecies];
    //float wpfx_sum[nSpecies];
    float tmpX_h[Nx];
    float tmpY_h[Ny/2+1];
    float tmpXY_h[Nx*(Ny/2+1)];
    float tmpYZ_h[(Ny/2+1)*Nz];
    float tmpXY_R_h[Nx*Ny];
    //float phi0_X[Nx];
    //cuComplex CtmpX_h[Nx];
    //cuComplex field_h[Nx*(Ny/2+1)*Nz];
    //cuComplex CtmpZ_h[Nz];
   
    float Phi2_zf;
    float Phi_zf_rms;
    float kx2Phi_zf_rms;
    float kx2Phi_zf_rms_old;
    float Phi_zf_rms_sum;
    float Phi_zf_rms_avg;
    float kx2Phi_zf_rms_sum;
    float kx2Phi_zf_rms_avg;
    
    //for secondary test
    float S_fixed = 1.; // Remove eventually
    ev_h->sfixed.S = S_fixed;

    char filename[200];
  
    //int exit_flag = 0;  
  
    int gs2_counter;


    //nSteps = 100;
    
    int Stable[Nx*(Ny/2+1)*2];
    //cuComplex stability[Nx*(Ny/2+1)];
    for(int i=0; i<Nx*(Ny/2+1); i++) {
      //stability[i].x = 0;
      //stability[i].y = 0;
      Stable[i] = 0;
      Stable[i +Nx*(Ny/2+1)] = 0;
    }
    //bool STABLE_STOP=false;  
    int stableMax = 500;
      
    //float wpfxmax=0.;
    //float wpfxmin=0.;
    int converge_count=0;
    
    //bool startavg=false;
  
    //float dt_start = .02;
    float alpha_avg = (float) 2./(navg+1.);
    float mu_avg = exp(-alpha_avg);
    //float navg_nlpm;

    /*
    float ky_spectrum1_h[Ny/2+1];
    float ky_spectrum2_h[Ny/2+1];
    float ky_spectrum3_h[Ny/2+1];
    float ky0_kx_spectrum_h[Nx];
    float kxky_spectrum1_h[Nx*(Ny/2+1)];  
    float kxky_spectrum2_h[Nx*(Ny/2+1)];
    float kxky_spectrum3_h[Nx*(Ny/2+1)];
    */
    //device variables; main device arrays will be capitalized
    if(DEBUG) getError("run_gryfx.cu, before device alloc");
    
    //tmps for timestep routine
    float *tmpXYZ;
    cuComplex *CtmpX;
    cuComplex *CtmpX2;
    cuComplex *CtmpXZ;
   
    cuComplex *omegaBox[navg];
    
    cuComplex *omegaAvg;
    
    //float *Phi2_XYBox[navg];
    specie* species_d;
    
    //double dt_old;
    //double avgdt;
    //float totaltimer;
    //float timer;
    //float GS2timer;
    
    //diagnostics scalars
    float flux1,flux2;
    float flux1_phase, flux2_phase, Dens_phase, Tpar_phase, Tprp_phase;
    float flux1_phase_sum, flux2_phase_sum, Dens_phase_sum, Tpar_phase_sum, Tprp_phase_sum;
   
    float Phi2, kPhi2;
    float Phi2_sum;//, kPhi2_sum;
    float expectation_ky;
    float expectation_kx;
    //diagnostics arrays
    //cuComplex *Phi_sum;
    
    
    float *nu_nlpm;
    float *nu1_nlpm;
    float *nu22_nlpm;
    float nu1_nlpm_max;
    float nu22_nlpm_max;
    cuComplex *nu1_nlpm_complex;
    cuComplex *nu22_nlpm_complex;
    float *shear_rate_z;
    float *shear_rate_z_nz;
    float *shear_rate_nz;  
  

    printf("At the beginning of run_gryfx, gs2 time is %f\n", gs2_time_mp_code_time_/sqrt(2.0));
    
////////////  
#ifdef GS2_zonal
			if(iproc==0) {

#endif

    //omega_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)); 
    omega_out_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)); 
    omegaAvg_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1));

} //end if iproc

	/* ev_hd is on the host but the pointers point to memory on the device*/
	everything_struct * ev_hd;
	/* ev_d is on the device (and the pointers point to memory on the device)*/
	everything_struct * ev_d;
  setup_everything_structs(ev_h, &ev_hd, &ev_d);


  /* The time struct handles the 
   * run progression, e.g. runtime,
   * counter etc. Here we create a local
   * reference for brevity */
  time_struct * tm = &ev_h->time;
  //Initialize dt
  tm->dt = ev_h->pars.dt;
  /* The outputs struct contains
   * like moving averages */
  outputs_struct * outs = &ev_h->outs;
  cuda_events_struct * events = &ev_h->events;
  cuda_streams_struct * streams = &ev_h->streams;



// The file local_pointers.cu
//declares and assigns local 
// pointers to members of the everything
// structs, for example Phi... it should
// eventually be unnecessary
#include "local_pointers.cu"


   //omegaAvg_h = ev_h->outs.omega;

if (iproc==0){

  
    //zero dtBox array
    //for(int t=0; t<navg; t++) {  dtBox[t] = 0;  }
      
    //init_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    //Phi_energy = (float*) malloc(sizeof(float));
    
  //#ifdef GS2_zonal
  //#endif
    
    
    //cudaMalloc((void**) &Phi_sum, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    
    cudaMalloc((void**) &CtmpX, sizeof(cuComplex)*Nx);
    cudaMalloc((void**) &CtmpX2, sizeof(cuComplex)*Nx);
    cudaMalloc((void**) &CtmpXZ, sizeof(cuComplex)*Nx*Nz);
    cudaMalloc((void**) &tmpXYZ, sizeof(float)*Nx*(Ny/2+1)*Nz);
   
    cudaMalloc((void**) &deriv_nlps, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &derivR1_nlps, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &derivR2_nlps, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &resultR_nlps, sizeof(float)*Nx*Ny*Nz);
  
    cudaMalloc((void**) &kz_complex, sizeof(float)*(Nz/2+1));
  
    cudaMalloc((void**) &PhiAvgDenom, sizeof(float)*Nx);
    
    cudaMalloc((void**) &species_d, sizeof(specie)*nSpecies);

  /* Temporary hack */
  ev_hd->pars.species = species_d;
  
    
    for(int t=0; t<navg; t++) {
      if(LINEAR || secondary_test) cudaMalloc((void**) &omegaBox[t], sizeof(cuComplex)*Nx*(Ny/2+1));
      //cudaMalloc((void**) &Phi2_XYBox[t], sizeof(float)*Nx*(Ny/2+1));
    }
    cudaMalloc((void**) &omegaAvg, sizeof(cuComplex)*Nx*(Ny/2+1));
    
    
    cudaMalloc((void**) &nu_nlpm, sizeof(float)*Nz);
    cudaMalloc((void**) &nu1_nlpm, sizeof(float)*Nz);
    cudaMalloc((void**) &nu22_nlpm, sizeof(float)*Nz);
    cudaMalloc((void**) &nu1_nlpm_complex, sizeof(cuComplex)*Nz);
    cudaMalloc((void**) &nu22_nlpm_complex, sizeof(cuComplex)*Nz);
    cudaMalloc((void**) &shear_rate_z, sizeof(float)*Nz);  
    cudaMalloc((void**) &shear_rate_nz, sizeof(float)*Nz);  
    cudaMalloc((void**) &shear_rate_z_nz, sizeof(float)*Nz);  
  
    cudaMalloc((void**) &Dnlpm_d, sizeof(float));
    cudaMalloc((void**) &Phi_zf_kx1_d, sizeof(float));

  
  
    if(DEBUG) getError("run_gryfx.cu, after device alloc");
  
    cudaMemcpy(species_d, species, sizeof(specie)*nSpecies, cudaMemcpyHostToDevice);
    
    cudaMemcpy(z, z_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);

    copy_geo_arrays_to_device(&ev_hd->geo, &ev_h->geo, &ev_h->pars, ev_h->grids.Nz); 
    
    
    //set up plans for NLPS, ZDeriv, and ZDerivB
    //plan for ZDerivCovering done below
	  create_cufft_plans(&ev_h->grids, &ev_h->ffts);
     
    // INITIALIZE ARRAYS AS NECESSARY
    zero<<<dimGrid,dimBlock>>>(nu22_nlpm, 1, 1, Nz);
    zero<<<dimGrid,dimBlock>>>(nu1_nlpm, 1, 1, Nz);
    zero<<<dimGrid,dimBlock>>>(nu_nlpm, 1, 1, Nz);
  
	  initialize_grids(&ev_h->pars, &ev_hd->grids, &ev_h->grids, &ev_h->cdims); 

  	calculate_additional_geo_arrays(
        ev_h->grids.Nz, ev_hd->grids.kz, ev_hd->tmp.Z,
        &ev_h->pars, &ev_h->cdims, 
        &ev_hd->geo, &ev_h->geo);

  
    //PhiAvg denominator for qneut
    cudaMemset(PhiAvgDenom, 0, sizeof(float)*Nx);
    phiavgdenom<<<dimGrid,dimBlock>>>(PhiAvgDenom, tmpXZ, jacobian, species_d, kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);  
  
    if(DEBUG) getError("run_gryfx.cu, after init"); 
   
#ifdef GS2_zonal
			}
#endif
  
  
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //set up kxCover and kyCover for covering space z-transforms
    //nshift = Nx - ntheta0;
    initialize_z_covering(iproc, &ev_hd->grids, &ev_h->grids, &ev_h->pars, &ev_h->ffts, &ev_h->streams, &ev_h->cdims, &ev_h->events);  

    ///////////////////////////////////////////////////////////////////////////////////////////////////

if (iproc==0){
  
    
    ////////////////////////////////////////////////
    // set up some diagnostics/control flow files //
    ////////////////////////////////////////////////
    
    setup_files(&ev_h->files, &ev_h->pars, &ev_h->grids, ev_h->info.run_name);
    writedat_beginning(ev_h);
    
    
    
    ////////////////////////////////////////////
    
  
    //////////////////////////////
    // initial conditions set here
    //////////////////////////////
    
  
    if(DEBUG) getError("run_gryfx.cu, before initial condition"); 
    
    //if running nonlinear part of secondary test...
    if(secondary_test && !LINEAR && RESTART) { 
       restartRead(ev_h, ev_hd, &Phi_zf_rms);
       //Check restart was successful
       fieldWrite(Phi, field_h, "phi_restarted.field", filename);
       load_fixed_arrays_from_restart(
            ev_h->grids.Nz, ev_h->tmp.CZ, &ev_h->pars,
            &ev_hd->sfixed, &ev_hd->fields);
       //initialize density with noise
       RESTART = false; 
       init = DENS;

       //restart run from t=0
       tm->counter = 0;
       tm->runtime = 0.;
       tm->dtSum = 0.;

       maxdt = .1/(phi_test.x*kx_h[1]*ky_h[1]);
    } 
    
    //float amp;
    
    if(!RESTART) {
      set_initial_conditions_no_restart(
          &ev_h->pars, &ev_d->pars, &ev_h->grids, &ev_d->grids, 
          &ev_h->cdims, &ev_d->geo, &ev_hd->fields, &ev_hd->tmp);
		  zero_moving_averages(&ev_h->grids, &ev_h->cdims, &ev_hd->outs, &ev_h->outs, tm);
      
      flux1_phase_sum = 0.;
      flux2_phase_sum = 0.;
      Dens_phase_sum = 0.;
      Tpar_phase_sum = 0.;
      Tprp_phase_sum = 0.;
      //zeroC<<<dimGrid,dimBlock>>>(Phi_sum);
    } 
    else {
      restartRead(ev_h, ev_hd, &Phi_zf_rms);
      if(zero_restart_avg) {
		    zero_moving_averages(&ev_h->grids, &ev_h->cdims, &ev_hd->outs, &ev_h->outs, tm);
      }
    }
      
    create_cuda_events_and_streams(events, &ev_h->streams, ev_h->grids.nClasses);
    /*
    float step_timer=0.;
    float step_timer_total=0.;
    float diagnostics_timer=0.;
    float diagnostics_timer_total=0.;
    */ 
#ifdef GS2_zonal    
			} //end of iproc if    
    //MPI_Barrier(MPI_COMM_WORLD);
#endif 
    tm->runtime=0.;
    tm->counter=0;
    gs2_counter=1;
    tm->totaltimer=0.;
    tm->timer=0.;
      //GS2timer=0.;
    int stopcount = 0;
    int nstop = 10;
    int first_half_flag;

#ifdef GS2_zonal

    //MPI_Bcast(&ev_h->grids.Nz, 1, MPI_INT, 0, ev_h->mpi.mpcom);
    //MPI_Bcast(&ev_h->grids.ntheta0, 1, MPI_INT, 0, ev_h->mpi.mpcom);
    //MPI_Bcast(&ev_h->grids.Nspecies, 1, MPI_INT, 0, ev_h->mpi.mpcom);
      initialize_hybrid_arrays(ev_h->mpi.iproc,
       &ev_h->grids,
      &ev_h->hybrid, &ev_hd->hybrid);
      
if(iproc==0) {
    copy_hybrid_arrays_from_host_to_device_async(
        &ev_h->grids, &ev_h->hybrid, 
        &ev_hd->hybrid, streams);
    
  
    cudaEventRecord(events->H2D, streams->copystream);
    cudaStreamWaitEvent(0, events->H2D, 0);
    
    fieldWrite_nopad_h(phi_ky0_h, "phi0.field", filename, Nx, 1, Nz, ntheta0, 1);
    replace_zonal_fields_with_hybrid(
      &ev_h->cdims, &ev_hd->fields,
      &ev_hd->hybrid, ev_h->fields.field);
   
} // if iproc 
  
#endif
    
#ifdef GS2_zonal
			if(iproc==0) {  
#endif
      if(secondary_test && !LINEAR) {
        copy_fixed_modes_into_fields(
          &ev_h->cdims, &ev_hd->fields, &ev_hd->sfixed);
      }

      if(init == DENS) {
        // Solve for initial phi
        // assumes the initial conditions have been moved to the device
        qneut(Phi, Dens, Tprp, tmp, tmp, field, species, species_d);
      }


  if(DEBUG) getError("about to start timestep loop");
    fieldWrite(Dens[ION], field_h, "dens0.field", filename); 
    fieldWrite(Upar[ION], field_h, "upar0.field", filename); 
    fieldWrite(Tpar[ION], field_h, "tpar0.field", filename); 
    fieldWrite(Tprp[ION], field_h, "tprp0.field", filename); 
    fieldWrite(Qpar[ION], field_h, "qpar0.field", filename); 
    fieldWrite(Qprp[ION], field_h, "qprp0.field", filename); 
    fieldWrite(Phi, field_h, "phi0.field", filename); 

  volflux(Phi,Phi,tmp,tmpXY);
  sumY_neq_0<<<dimGrid,dimBlock>>>(tmpX, tmpXY);
  kxWrite(tmpX, tmpX_h, filename, "phi2_0.kx");

#ifdef GS2_zonal
			}
#endif 

    //uint64_t diff;
    struct timespec clockstart;//, clockend;
    //MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
   first_half_flag = 1;
   // //cudaEventRecord(GS2Istart,0);
//    clock_gettime(CLOCK_MONOTONIC, &clockstart);
//   // //getError("Before GS2start");
//    while(gs2_counter<=1) { 
//   //   //printf("gs2 step\t");
//   //   first_half_flag=1;
//    MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
//      gs2_main_mp_advance_gs2_(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h, &first_half_flag);
//      gs2_counter++;
//   //   first_half_flag=0;
//    MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
//      gs2_main_mp_advance_gs2_(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h, &first_half_flag);
//      gs2_counter++;
//    }
//    clock_gettime(CLOCK_MONOTONIC, &clockend);
//    diff = 1000000000L * (clockend.tv_sec - clockstart.tv_sec) + clockend.tv_nsec - clockstart.tv_nsec;
//    printf("elapsed time = %llu ns\n", (long long unsigned int) diff);
//   // //GS2timer = ((double) (clockend - clockstart)) / CLOCKS_PER_SEC;
//   // //cudaEventRecord(GS2stop,0);
//   // //getError("After GS2stop");
//   // //cudaEventElapsedTime(&GS2timer, GS2start, GS2stop);
//   // //getError("After GS2timer");
//    printf("proc%d: GS2 time/timestep is %e ns\n", iproc, ((double) diff)/1.);
//   // 
//    gs2_counter=1;

    clock_gettime(CLOCK_MONOTONIC, &clockstart);
#ifdef GS2_zonal
    if(iproc==0) {
#endif
      cudaEventRecord(events->start,0);
#ifdef GS2_zonal
    }
#endif

    /////////////////////
    // Begin timestep loop
    /////////////////////
   
    while(/*counter < 1 &&*/ 
        tm->counter<nSteps &&
  	stopcount<nstop 
  	/*&& converge_count<2*navg*/
  	)
     {

      //if(counter==9) cudaProfilerStart();
#ifdef GS2_zonal
//     if(turn_off_gradients_test && counter==nSteps/2) {
//       species[ION].tprim = 0.;
//       species[ION].fprim = 0.;
//       double den[1], tem[1], fp[1], tp[1], vnewk[1];
//       den[0] = (double) species[ION].dens;
//       tem[0] = (double) species[ION].temp;
//       fp[0] = (double) species[ION].fprim;
//       tp[0] = (double) species[ION].tprim;
//       vnewk[0] = (double) (species[ION].nu_ss/sqrt(2.));
//       
//       species_mp_reinit_species_(&nSpecies, den, tem, fp, tp, vnewk);
//       if(iproc==0) printf("\n\n%d: GryfX gradients have been reset to tprim = %f, fprim = %f; runtime=%f\n\n", counter, species[ION].tprim, species[ION].fprim, runtime);
//     }
//if(DEBUG && counter==0) printf("proc %d has entered the timestep loop\n", iproc);
			if(iproc==0) {
#endif
  //    cudaEventRecord(start1,0);


#ifndef GS2_zonal
      //dt_old = tm->dt;
      //if(!LINEAR && !secondary_test) tm->dt = courant(Phi, tmp, field, resultR_nlps, species);   
#endif
      //avgdt = .5*(dt_old+tm->dt);    

#ifdef GS2_zonal

      if(gs2_time_mp_code_time_/sqrt(2.) - tm->runtime > .0001) printf("\n\nRuntime mismatch! GS2 time is %f, GryfX time is %f\n\n", gs2_time_mp_code_time_/sqrt(2.), tm->runtime); 

#endif

      //if(counter<100 && tm->dt>dt_start) tm->dt = dt_start;
      
  
      //EXBshear bug fixed, need to check if correct
      //ExBshear(Phi,Dens,Upar,Tpar,Tprp,Qpar,Qprp,kx_shift,jump,avgdt);  
      
      
      //if(DEBUG) getError("after exb");
     
      //if(LINEAR) { 
      //volflux_zonal<<<dimGrid,dimBlock>>>(tmpX, Phi, Phi, jacobian, 1./fluxDen); 
      //  getPhiVal<<<dimGrid,dimBlock>>>(val, Phi1, 0, 4, Nz/2);
     //   fprintf(phifile, "\n\t%f\t%e", runtime, val);
      
      //getPhiVal<<<dimGrid,dimBlock>>>(val, tmpX, 4);
      
      //getky0z0<<<dimGrid,dimBlock>>>(tmpX, Phi);
  
      //volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Phi, jacobian, 1./fluxDen);
      //get_real_X<<<dimGrid,dimBlock>>>(tmpX, CtmpX);
  
      //phiVal[0] = 0;
      //cudaMemcpy(phiVal, val, sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(tmpX_h, tmpX, sizeof(float)*Nx, cudaMemcpyDeviceToHost);
      /*if(runtime == 0) {
        phiVal0 = phiVal[0];
      }*/
  
    //  if(counter==0){
    //    fprintf(phifile,"t");
    //    for(int i=0; i<Nx; i++) {
    //      fprintf(phifile, "\tkx=%g", kx_h[i]);
    //      if(init==DENS || init==PHI) phi0_X[i] = tmpX_h[i];
    //    }
  
    //  }
    //  if(init== DENS || init==PHI) fprintf(phifile, "\n\t%f\t%e", runtime, (float) tmpX_h[0]);///phi0_X[0]);
    //  else fprintf(phifile, "\n\t%f\t%e", runtime, (float) 1-tmpX_h[0]); 
    //  for(int i=1; i<Nx; i++) {
    //    if(init== DENS || init==PHI) fprintf(phifile, "\t%e", (float) tmpX_h[i]);///phi0_X[i]);
    //    else fprintf(phifile, "\t%e", (float) 1-tmpX_h[i]); 
    //  }
 //     }
  
      
      //calculate diffusion here... for now we just set it to 1
//      diffusion = 1.;
       
      //cudaProfilerStart();

#ifdef GS2_zonal
			} //end of iproc if
#endif

  /////////////////////////////////
  //FIRST HALF OF GRYFX TIMESTEP
  /////////////////////////////////
  
      //////nvtxRangePushA("Gryfx t->t+tm->dt/2");

      first_half_flag = 1;
#ifdef GS2_zonal
			if(iproc==0) {
#endif
      if(!LINEAR) {
        for(int s=0; s<nSpecies; s++) {
          //calculate NL(t) = NL(Moment)
          nonlinear_timestep(Dens[s], Dens[s], Dens1[s], 
                 Upar[s], Upar[s], Upar1[s], 
                 Tpar[s], Tpar[s], Tpar1[s], 
                 Qpar[s], Qpar[s], Qpar1[s], 
                 Tprp[s], Tprp[s], Tprp1[s], 
                 Qprp[s], Qprp[s], Qprp1[s], 
                 Phi, 
                 dens_ky0_d[s], upar_ky0_d[s], tpar_ky0_d[s], tprp_ky0_d[s], qpar_ky0_d[s], qprp_ky0_d[s],
                 &tm->dt, species[s],
  	       field,field,field,field,field,field,
  	       tmp,tmp, tmpX, CtmpX, first_half_flag,
	       field_h, tm->counter, kx2Phi_zf_rms, kx2Phi_zf_rms_avg);
  
          //Moment1 = Moment + (dt/2)*NL(Moment)
  


#ifdef GS2_zonal
  
          //copy NL(t)_ky=0 from D2H
  
          if(s==nSpecies-1) {  //Only after all species have been done
            cudaEventRecord(events->nonlin_halfstep, 0); //record this after all streams (ie the default stream) reach this point
            cudaStreamWaitEvent(streams->copystream, events->nonlin_halfstep,0); //wait for all streams before copying
            for(int i=0; i<nSpecies; i++) {
              cudaMemcpyAsync(dens_ky0_h + s*ntheta0*Nz, dens_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
              cudaMemcpyAsync(upar_ky0_h + s*ntheta0*Nz, upar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
              cudaMemcpyAsync(tpar_ky0_h + s*ntheta0*Nz, tpar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
              cudaMemcpyAsync(tprp_ky0_h + s*ntheta0*Nz, tprp_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
              cudaMemcpyAsync(qpar_ky0_h + s*ntheta0*Nz, qpar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
              cudaMemcpyAsync(qprp_ky0_h + s*ntheta0*Nz, qprp_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
            }
            cudaEventRecord(events->D2H, streams->copystream);
          }

          if(tm->counter==0) fieldWrite_nopad_h(dens_ky0_h, "NLdens.field", filename, Nx, 1, Nz, ntheta0, 1);
  
#endif
  
          //calculate L(t) = L(Moment)
          linear_timestep(Dens1[s], Dens[s], Dens1[s], 
                 Upar1[s], Upar[s], Upar1[s], 
                 Tpar1[s], Tpar[s], Tpar1[s], 
                 Qpar1[s], Qpar[s], Qpar1[s], 
                 Tprp1[s], Tprp[s], Tprp1[s], 
                 Qprp1[s], Qprp[s], Qprp1[s], 
                 Phi, ev_hd->grids.kxCover,ev_hd->grids.kyCover, ev_hd->grids.g_covering, ev_hd->grids.kz_covering, species[s], tm->dt/2.,
  	       field,field,field,field,field,field,
  	       tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmpZ,ev_h->ffts.plan_covering,
  	       nu_nlpm, tmpX, tmpXZ, CtmpX, CtmpX2);
   
          //Moment1 = Moment1 + (tm->dt/2)*L(Moment)

  	}         
      }
      else { //if only linear
        for(int s=0; s<nSpecies; s++) {
  
          //calculate L(t) = L(Moment)
          linear_timestep(Dens[s], Dens[s], Dens1[s], 
                 Upar[s], Upar[s], Upar1[s], 
                 Tpar[s], Tpar[s], Tpar1[s], 
                 Qpar[s], Qpar[s], Qpar1[s], 
                 Tprp[s], Tprp[s], Tprp1[s], 
                 Qprp[s], Qprp[s], Qprp1[s], 
                 Phi, ev_hd->grids.kxCover,ev_hd->grids.kyCover, ev_hd->grids.g_covering, ev_hd->grids.kz_covering, species[s], tm->dt/2.,
  	       field,field,field,field,field,field,
  	       tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmpZ,ev_h->ffts.plan_covering,
  	       nu_nlpm, tmpX, tmpXZ, CtmpX, CtmpX2);
           
          //Moment1 = Moment + (tm->dt/2)*L(Moment)

        }
      }

//if(DEBUG && counter==0) getError("after linear step"); 

      qneut(Phi1, Dens1, Tprp1, tmp, tmp, field, species, species_d);
  
      if(secondary_test && !LINEAR) {
        if(tm->runtime < .02/maxdt/M_PI) S_fixed = 1.;// sin(.01/maxdt * tm->runtime);
        else S_fixed = 1.;
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Phi1, phi_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Dens1[ION], dens_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Upar1[ION], upar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Tpar1[ION], tpar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Tprp1[ION], tprp_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Qpar1[ION], qpar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Qprp1[ION], qprp_fixed, 1, 0, S_fixed);
      }

      //f1 = f(t+tm->dt/2)
  
  /*
    if(DEBUG) {*/
    //if(counter==0) fieldWrite(Dens1[ION], field_h, "dens.5.field", filename); 
  /*  if(counter==0) fieldWrite(Upar1[ION], field_h, "upar.5.field", filename); 
    if(counter==0) fieldWrite(Tpar1[ION], field_h, "tpar.5.field", filename); 
    if(counter==0) fieldWrite(Tprp1[ION], field_h, "tprp.5.field", filename); 
    if(counter==0) fieldWrite(Qpar1[ION], field_h, "qpar.5.field", filename); 
    if(counter==0) fieldWrite(Qprp1[ION], field_h, "qprp.5.field", filename); 
    }
  */
  
  
#ifdef GS2_zonal
			} //end of iproc if
#endif

      //////nvtxRangePop();
#ifdef GS2_zonal
    
    //MPI_Barrier(MPI_COMM_WORLD); //make all procs wait


    if(!LINEAR) {
      if(iproc==0) cudaEventSynchronize(events->D2H); //have proc 0 wait for NL(t) to be copied D2H before advancing GS2 if running nonlinearly
    }
  
      //MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
    //nvtxRangePushA("GS2 t->t+tm->dt/2");
    //cudaEventRecord(GS2start,0);

    //advance GS2 t -> t + tm->dt/2  
    advance_gs2(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h, &first_half_flag);
    gs2_counter++;

    //nvtxRangePop();
    //cudaEventRecord(GS2stop,0);
    //cudaEventElapsedTime(&GS2timer, GS2start, GS2stop);
    //printf("100 calls of advance_gs2 took %f ms\n", GS2timer);

/*
    if(!LINEAR) {
      fields_implicit_mp_advance_implicit_(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h);
    } else {
      fields_mp_advance_(&gs2_counter);
    }
    if(DEBUG) printf("step %d of %d: proc %d advanced gs2 first half step\n", counter, nSteps, iproc);
    gs2_time_mp_update_time_();
    //gs2_diagnostics_mp_loop_diagnostics_(&gs2_counter, &exit_flag);
    gs2_counter++;
  
    MPI_Barrier(MPI_COMM_WORLD); 

    dist_fn_mp_getmoms_gryfx_(dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h);
*/

    //////nvtxRangePushA("copy moms(t+tm->dt/2)_ky=0 from H2D");
if(iproc==0) {  
    //copy moms(t+dt/2)_ky=0 from H2D
    cudaMemcpyAsync(phi_ky0_d, phi_ky0_h, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
    for(int s=0; s<nSpecies; s++) {
      cudaMemcpyAsync(dens_ky0_d[s], dens_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
      cudaMemcpyAsync(upar_ky0_d[s], upar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
      cudaMemcpyAsync(tpar_ky0_d[s], tpar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
      cudaMemcpyAsync(tprp_ky0_d[s], tprp_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
      cudaMemcpyAsync(qpar_ky0_d[s], qpar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
      cudaMemcpyAsync(qprp_ky0_d[s], qprp_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
    }
    
  
    cudaEventRecord(events->H2D, streams->copystream);
    cudaStreamWaitEvent(0, events->H2D, 0);
   
    //replace ky=0 modes with results from GS2
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(Phi1, phi_ky0_d);
    reality<<<dimGrid,dimBlock>>>(Phi1);
    mask<<<dimGrid,dimBlock>>>(Phi1);

    for(int s=0; s<nSpecies; s++) {
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Dens1[s], dens_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Upar1[s], upar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Tpar1[s], tpar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Tprp1[s], tprp_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Qpar1[s], qpar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Qprp1[s], qprp_ky0_d[s]);
      reality<<<dimGrid,dimBlock>>>(Dens1[s]);
      reality<<<dimGrid,dimBlock>>>(Upar1[s]);
      reality<<<dimGrid,dimBlock>>>(Tpar1[s]);
      reality<<<dimGrid,dimBlock>>>(Tprp1[s]);
      reality<<<dimGrid,dimBlock>>>(Qpar1[s]);
      reality<<<dimGrid,dimBlock>>>(Qprp1[s]);
      mask<<<dimGrid,dimBlock>>>(Dens1[s]);
      mask<<<dimGrid,dimBlock>>>(Upar1[s]);
      mask<<<dimGrid,dimBlock>>>(Tpar1[s]);
      mask<<<dimGrid,dimBlock>>>(Tprp1[s]);
      mask<<<dimGrid,dimBlock>>>(Qpar1[s]);
      mask<<<dimGrid,dimBlock>>>(Qprp1[s]);
    }
 } 
    //////nvtxRangePop();
  #endif
         

    //////nvtxRangePushA("Gryfx t->t+dt");

      first_half_flag = 0;
#ifdef GS2_zonal 
			if(iproc==0) {
#endif

      //NLPM calculated AFTER ky=0 quantities passed back from GS2!
      if(!LINEAR && NLPM && dorland_phase_complex) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPM(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s], 
        		tmpX, CtmpX, tmpXZ, CtmpXZ, tmpYZ, nu_nlpm, nu1_nlpm_complex, nu22_nlpm_complex, tmpZ, species[s], tm->dt/2., Dnlpm_d, Phi_zf_kx1_avg, kx2Phi_zf_rms);
        }	    
      }
      else if(!LINEAR && NLPM) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPM(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s], 
        		tmpX, tmpXZ, tmpYZ, nu_nlpm, nu1_nlpm, nu22_nlpm, species[s], tm->dt/2., Dnlpm_d, Phi_zf_kx1_avg, kx2Phi_zf_rms, tmp);
        }	    
      }  
      //hyper too...
      if(HYPER) {
        if(isotropic_shear) {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_iso(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s], 
  			tmpXYZ, shear_rate_nz, tm->dt/2.);
  		    
          }  
        }
        else {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_aniso(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s],
                          tmpXYZ, shear_rate_nz, shear_rate_z, shear_rate_z_nz, tm->dt/2.);
          }
        }
      }
 
  /////////////////////////////////
  //SECOND HALF OF GRYFX TIMESTEP
  /////////////////////////////////
      if(!LINEAR) {
        for(int s=0; s<nSpecies; s++) {
          //calculate NL(t+tm->dt/2) = NL(Moment1)
          nonlinear_timestep(Dens[s], Dens1[s], Dens[s], 
                 Upar[s], Upar1[s], Upar[s], 
                 Tpar[s], Tpar1[s], Tpar[s], 
                 Qpar[s], Qpar1[s], Qpar[s], 
                 Tprp[s], Tprp1[s], Tprp[s], 
                 Qprp[s], Qprp1[s], Qprp[s], 
                 Phi1, 
                 dens_ky0_d[s], upar_ky0_d[s], tpar_ky0_d[s], tprp_ky0_d[s], qpar_ky0_d[s], qprp_ky0_d[s],
                 &tm->dt, species[s],
  	       field,field,field,field,field,field,
  	       tmp,tmp, tmpX, CtmpX, first_half_flag,
	       field_h, tm->counter, kx2Phi_zf_rms, kx2Phi_zf_rms_avg);
  
          //Moment = Moment + dt * NL(Moment1)
  
  #ifdef GS2_zonal
  
          //copy NL(t+dt/2)_ky=0 from D2H
  
          if(s==nSpecies-1) {  //Only after all species have been done
            cudaEventRecord(events->nonlin_halfstep, 0); //record this after all streams (ie the default stream) reach this point
            cudaStreamWaitEvent(streams->copystream, events->nonlin_halfstep,0); //wait for all streams before copying
            for(int i=0; i<nSpecies; i++) {
              cudaMemcpyAsync(dens_ky0_h + s*ntheta0*Nz, dens_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
              cudaMemcpyAsync(upar_ky0_h + s*ntheta0*Nz, upar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
              cudaMemcpyAsync(tpar_ky0_h + s*ntheta0*Nz, tpar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
              cudaMemcpyAsync(tprp_ky0_h + s*ntheta0*Nz, tprp_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
              cudaMemcpyAsync(qpar_ky0_h + s*ntheta0*Nz, qpar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
              cudaMemcpyAsync(qprp_ky0_h + s*ntheta0*Nz, qprp_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, streams->copystream);
            }
            cudaEventRecord(events->D2H, streams->copystream);
          }
  
  #endif
  
          //calculate L(t+dt/2)=L(Moment1) 
          linear_timestep(Dens[s], Dens1[s], Dens[s], 
                 Upar[s], Upar1[s], Upar[s], 
                 Tpar[s], Tpar1[s], Tpar[s], 
                 Qpar[s], Qpar1[s], Qpar[s], 
                 Tprp[s], Tprp1[s], Tprp[s], 
                 Qprp[s], Qprp1[s], Qprp[s], 
                 Phi1, ev_hd->grids.kxCover,ev_hd->grids.kyCover, ev_hd->grids.g_covering, ev_hd->grids.kz_covering, species[s], tm->dt,
  	       field,field,field,field,field,field,
  	       tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmpZ,ev_h->ffts.plan_covering,
  	       nu_nlpm, tmpX, tmpXZ, CtmpX, CtmpX2);
  
          //Moment = Moment + dt * L(Moment1)
  	}         
      }
      else { //if only linear
        for(int s=0; s<nSpecies; s++) {
  
          linear_timestep(Dens[s], Dens1[s], Dens[s], 
                 Upar[s], Upar1[s], Upar[s], 
                 Tpar[s], Tpar1[s], Tpar[s], 
                 Qpar[s], Qpar1[s], Qpar[s], 
                 Tprp[s], Tprp1[s], Tprp[s], 
                 Qprp[s], Qprp1[s], Qprp[s], 
                 Phi1, ev_hd->grids.kxCover,ev_hd->grids.kyCover, ev_hd->grids.g_covering, ev_hd->grids.kz_covering, species[s], tm->dt,
  	       field,field,field,field,field,field,
  	       tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmpZ,ev_h->ffts.plan_covering,
  	       nu_nlpm, tmpX, tmpXZ, CtmpX, CtmpX2);
        }
      }
      

        if(!LINEAR && !secondary_test && !write_omega) qneut(Phi, Dens, Tprp, tmp, tmp, field, species, species_d); //don't need to keep Phi=Phi(t) when running nonlinearly, overwrite with Phi=Phi(t+dt)
        else qneut(Phi1, Dens, Tprp, tmp, tmp, field, species, species_d); //don't overwrite Phi=Phi(t), use Phi1=Phi(t+dt); for growth rate calculation
      
      if(secondary_test && !LINEAR) {
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Phi1, phi_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Dens[ION], dens_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Upar[ION], upar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Tpar[ION], tpar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Tprp[ION], tprp_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Qpar[ION], qpar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Qprp[ION], qprp_fixed, 1, 0, S_fixed);
      }
      //f = f(t+dt)
  
  
#ifdef GS2_zonal
			} //end of iproc if
#endif

    //////nvtxRangePop();

#ifdef GS2_zonal
    
    
    if(!LINEAR) {
      if(iproc==0) cudaEventSynchronize(events->D2H); //wait for NL(t+dt/2) to be copied D2H before advancing GS2 if running nonlinearly
    }
    //advance GS2 t+dt/2 -> t+dt
      //MPI_Barrier(MPI_COMM_WORLD);
    //nvtxRangePushA("GS2 t+dt/2->t+dt");
    advance_gs2(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h, &first_half_flag);
    gs2_counter++;
    //nvtxRangePop();
  
/*
    MPI_Barrier(MPI_COMM_WORLD);
    if(!LINEAR) {
      fields_implicit_mp_advance_implicit_(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h);
    } else {
      fields_mp_advance_(&gs2_counter);
    }
    gs2_time_mp_update_time_();
    //gs2_diagnostics_mp_loop_diagnostics_(&gs2_counter, &exit_flag);
  
    MPI_Barrier(MPI_COMM_WORLD);
    dist_fn_mp_getmoms_gryfx_(dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h);
*/

    //////nvtxRangePushA("copy moms(t+dt)_ky=0 from H2D");
if(iproc==0) {  
    //copy moms(t+dt)_ky=0 from H2D
    cudaMemcpyAsync(phi_ky0_d, phi_ky0_h, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
    for(int s=0; s<nSpecies; s++) {
      cudaMemcpyAsync(dens_ky0_d[s], dens_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
      cudaMemcpyAsync(upar_ky0_d[s], upar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
      cudaMemcpyAsync(tpar_ky0_d[s], tpar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
      cudaMemcpyAsync(tprp_ky0_d[s], tprp_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
      cudaMemcpyAsync(qpar_ky0_d[s], qpar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
      cudaMemcpyAsync(qprp_ky0_d[s], qprp_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, streams->copystream);
    }
    
  
    cudaEventRecord(events->H2D, streams->copystream);
    cudaStreamWaitEvent(0, events->H2D, 0);
   
    if(!LINEAR && !secondary_test && !write_omega) {
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Phi, phi_ky0_d);
      mask<<<dimGrid,dimBlock>>>(Phi);
      reality<<<dimGrid,dimBlock>>>(Phi);
    } else {
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Phi1, phi_ky0_d);
      mask<<<dimGrid,dimBlock>>>(Phi1);
      reality<<<dimGrid,dimBlock>>>(Phi1);
    }
    for(int s=0; s<nSpecies; s++) {
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Dens[s], dens_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Upar[s], upar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Tpar[s], tpar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Tprp[s], tprp_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Qpar[s], qpar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Qprp[s], qprp_ky0_d[s]);

      reality<<<dimGrid,dimBlock>>>(Dens[s]);
      reality<<<dimGrid,dimBlock>>>(Upar[s]);
      reality<<<dimGrid,dimBlock>>>(Tpar[s]);
      reality<<<dimGrid,dimBlock>>>(Tprp[s]);
      reality<<<dimGrid,dimBlock>>>(Qpar[s]);
      reality<<<dimGrid,dimBlock>>>(Qprp[s]);

      mask<<<dimGrid,dimBlock>>>(Dens[s]);
      mask<<<dimGrid,dimBlock>>>(Upar[s]);
      mask<<<dimGrid,dimBlock>>>(Tpar[s]);
      mask<<<dimGrid,dimBlock>>>(Tprp[s]);
      mask<<<dimGrid,dimBlock>>>(Qpar[s]);
      mask<<<dimGrid,dimBlock>>>(Qprp[s]);
    }
}  

    //////nvtxRangePop();
  
#endif
  
    //////nvtxRangePushA("Diagnostics");
    
#ifdef GS2_zonal
			if(iproc==0) {
#endif  
  
      //NLPM
      if(!LINEAR && NLPM && dorland_phase_complex) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPM(Phi, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s], 
        		tmpX, CtmpX, tmpXZ, CtmpXZ, tmpYZ, nu_nlpm, nu1_nlpm_complex, nu22_nlpm_complex, tmpZ, species[s], tm->dt, Dnlpm_d, Phi_zf_kx1_avg, kx2Phi_zf_rms);
        }	    
      }
      else if(!LINEAR && NLPM) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPM(Phi, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s], 
        		tmpX, tmpXZ, tmpYZ, nu_nlpm, nu1_nlpm, nu22_nlpm, species[s], tm->dt, Dnlpm_d, Phi_zf_kx1_avg, kx2Phi_zf_rms, tmp);
        }	    
      }  
          
      
      if(HYPER) {
        if(isotropic_shear) {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_iso(Phi, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s], 
  			tmpXYZ, shear_rate_nz, tm->dt);
  		    
          }  
        }
        else {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_aniso(Phi, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s],
                          tmpXYZ, shear_rate_nz, shear_rate_z, shear_rate_z_nz, tm->dt);
          }
        }
      }



  /*
      cudaEventRecord(stop1,0);
      cudaEventSynchronize(stop1);
      cudaEventElapsedTime(&step_timer,start1,stop1);
      step_timer_total += step_timer;
  */    //cudaProfilerStop();
      
      
      //DIAGNOSTICS
       
     // if(LINEAR) { 
     //   getPhiVal<<<dimGrid,dimBlock>>>(val, Phi1, 0, 4, Nz/2);
     //   cudaMemcpy(phiVal, val, sizeof(float), cudaMemcpyDeviceToHost);
     //   fprintf(phifile, "\n\t%f\t%e", runtime, phiVal[0]);
     // }
    //  cudaEventRecord(start1,0);  
      if(LINEAR || secondary_test || write_omega) {
        growthRate<<<dimGrid,dimBlock>>>(omega,Phi1,Phi,tm->dt);    
        //cudaMemcpy(omega_h, omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);     
        //weighted average of omega over 'navg' timesteps
        //boxAvg(omegaAvg, omega, omegaBox, dt, dtBox, navg, counter);
        //cudaMemcpy(omegaAvg_h, omegaAvg, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);      
  
        //copy Phi for next timestep
        cudaMemcpy(Phi, Phi1, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
        mask<<<dimGrid,dimBlock>>>(Phi1);
        mask<<<dimGrid,dimBlock>>>(Phi);
  
        
        //print growth rates to files   
        //omegaWrite(omegafile,gammafile,omegaAvg_h,runtime); 
        
  
        //if(counter>2*navg) {
  	//omegaStability(omega_h, omegaAvg_h, stability,Stable,stableMax);
  	//STABLE_STOP = stabilityCheck(Stable,stableMax);
        //}
      }
      
  
  
  //DIAGNOSTICS
  
      if( strcmp(nlpm_option,"constant") == 0) Dnlpm = dnlpm;
      else cudaMemcpy(&Dnlpm, Dnlpm_d, sizeof(float), cudaMemcpyDeviceToHost);
      
      /*
      if(counter%nwrite==0) {
        cudaMemcpy(phi_h, Phi, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
      }
      */
      
      for(int s=0; s<nSpecies; s++) {    
        wpfx_old[s] = wpfx[s];
        pflx_old[s] = pflx[s];
      }
  
      //calculate instantaneous heat flux
      for(int s=0; s<nSpecies; s++) {  
        fluxes(&pflx[s], &wpfx[s],flux1,flux2,Dens[s],Tpar[s],Tprp[s],Phi,
               tmp,tmp,tmp,field,field,field,tmpZ,tmpXY,species[s],tm->runtime,
               &flux1_phase, &flux2_phase, &Dens_phase, &Tpar_phase, &Tprp_phase);        
      }
       
      volflux_zonal(Phi,Phi,tmpX);  //tmpX = Phi_zf**2(kx)
      get_kx1_rms<<<1,1>>>(Phi_zf_kx1_d, tmpX);
      Phi_zf_kx1_old = Phi_zf_kx1;
      cudaMemcpy(&Phi_zf_kx1, Phi_zf_kx1_d, sizeof(float), cudaMemcpyDeviceToHost);
      
      //volflux_zonal(Phi,Phi,tmpX);  //tmpX = Phi_zf**2(kx)
      kx2Phi_zf_rms_old = kx2Phi_zf_rms;
      multKx4<<<dimGrid,dimBlock>>>(tmpX2, tmpX, kx); 
      kx2Phi_zf_rms = sumReduc(tmpX2, Nx, false);
      kx2Phi_zf_rms = sqrt(kx2Phi_zf_rms);
  
      //volflux_zonal(Phi,Phi,tmpX);  //tmpX = Phi_zf**2(kx)
      Phi2_zf = sumReduc(tmpX, Nx, false);
      Phi_zf_rms = sqrt(Phi2_zf);   

      //calculate tmpXY = Phi**2(kx,ky)
      volflux(Phi,Phi,tmp,tmpXY);
      //if(!LINEAR && write_phi2kxky_time) {
      //  cudaMemcpy(tmpXY_h, tmpXY, sizeof(float)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
      //  kxkyTimeWrite(phikxkyfile, tmpXY_h, runtime);
      //}
      sumX<<<dimGrid,dimBlock>>>(tmpY, tmpXY);
      cudaMemcpy(tmpY_h, tmpY, sizeof(float)*(Ny/2+1), cudaMemcpyDeviceToHost);
      cudaMemcpy(ev_h->outs.phi2_by_ky, tmpY, sizeof(float)*(Ny/2+1), cudaMemcpyDeviceToHost);
      sumY <<< dimGrid, dimBlock >>>(tmpX2, tmpXY);
      cudaMemcpy(ev_h->outs.phi2_by_kx, tmpX2, sizeof(float)*Nx,cudaMemcpyDeviceToHost);

      if(!LINEAR && turn_off_gradients_test) {
        kyTimeWrite(ev_h->files.phifile, tmpY_h, tm->runtime);
      }
      //calculate <kx> and <ky>
      expect_k<<<dimGrid,dimBlock>>>(tmpXY2, tmpXY, ky);
      kPhi2 = sumReduc(tmpXY2, Nx*(Ny/2+1), false);
      Phi2 = sumReduc(tmpXY, Nx*(Ny/2+1), false);

      ev_h->outs.phi2 = Phi2;

      expectation_ky = (float) Phi2/kPhi2;
  
      expect_k<<<dimGrid,dimBlock>>>(tmpXY2, tmpXY, kx);
      kPhi2 = sumReduc(tmpXY2, Nx*(Ny/2+1), false);
      expectation_kx = (float) Phi2/kPhi2;
      
      //calculate z correlation function = tmpYZ (not normalized)
      zCorrelation<<<dimGrid,dimBlock>>>(tmpYZ, Phi);
     // volflux(Phi,Phi,tmp,tmpXY);

      nu1_nlpm_max = maxReduc(nu1_nlpm, Nz, false);
      nu22_nlpm_max = maxReduc(nu22_nlpm, Nz, false); 
    
      if(tm->counter>0) { 
        //we use an exponential moving average
        // wpfx_avg[t] = alpha_avg*wpfx[t] + (1-alpha_avg)*wpfx_avg[t-1]
        // now with time weighting...
        // wpfx_sum[t] = alpha_avg*dt*wpfx[t] + (1-alpha_avg)*wpfx_avg[t-1]
        // tm->dtSum[t] = alpha_avg*tm->dt[t] + (1-alpha_avg)*tm->dtSum[t-1]
        // wpfx_avg[t] = wpfx_sum[t]/tm->dtSum[t]
   
        // keep a running total of dt, phi**2(kx,ky), expectation values, etc.
        tm->dtSum = tm->dtSum*(1.-alpha_avg) + tm->dt*alpha_avg;
        add_scaled<<<dimGrid,dimBlock>>>(Phi2_kxky_sum, 1.-alpha_avg, Phi2_kxky_sum, tm->dt*alpha_avg, tmpXY, Nx, Ny, 1);
        add_scaled<<<dimGrid,dimBlock>>>(Phi2_zonal_sum, 1.-alpha_avg, Phi2_zonal_sum, tm->dt*alpha_avg, tmpX, Nx, 1, 1);
        add_scaled<<<dimGrid,dimBlock>>>(zCorr_sum, 1.-alpha_avg, zCorr_sum, tm->dt*alpha_avg, tmpYZ, 1, Ny, Nz);
        if(LINEAR || write_omega || secondary_test) {
          if(tm->counter>0) {
            add_scaled<<<dimGrid,dimBlock>>>(omegaAvg, 1.-alpha_avg, omegaAvg, tm->dt*alpha_avg, omega, Nx, Ny, 1);
            cudaMemcpy(omegaAvg_h, omegaAvg, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);      
            //print growth rates to files   
            omegaWrite(ev_h->files.omegafile,ev_h->files.gammafile,omegaAvg_h,tm->dtSum,tm->runtime); 
          }
          else {
            cudaMemcpy(omegaAvg_h, omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);      
            //print growth rates to files   
            omegaWrite(ev_h->files.omegafile,ev_h->files.gammafile,omegaAvg_h,tm->runtime); 
          }             

        }  
        outs->expectation_kx_movav = outs->expectation_kx_movav*(1.-alpha_avg) + expectation_kx*tm->dt*alpha_avg;
        outs->expectation_ky_movav = outs->expectation_ky_movav*(1.-alpha_avg) + expectation_ky*tm->dt*alpha_avg;
        Phi2_sum = Phi2_sum*(1.-alpha_avg) + Phi2*tm->dt*alpha_avg;
        Phi_zf_rms_sum = Phi_zf_rms_sum*(1.-alpha_avg) + Phi_zf_rms*tm->dt*alpha_avg;
        kx2Phi_zf_rms_sum = kx2Phi_zf_rms_sum*(1.-alpha_avg) + kx2Phi_zf_rms*tm->dt*alpha_avg;
        flux1_phase_sum = flux1_phase_sum*(1.-alpha_avg) + flux1_phase*tm->dt*alpha_avg;
        flux2_phase_sum = flux2_phase_sum*(1.-alpha_avg) + flux2_phase*tm->dt*alpha_avg;
        Dens_phase_sum = Dens_phase_sum*(1.-alpha_avg) + Dens_phase*tm->dt*alpha_avg;
        Tpar_phase_sum = Tpar_phase_sum*(1.-alpha_avg) + Tpar_phase*tm->dt*alpha_avg;
        Tprp_phase_sum = Tprp_phase_sum*(1.-alpha_avg) + Tprp_phase*tm->dt*alpha_avg;
        Dnlpm_sum = Dnlpm_sum*(1.-alpha_avg) + Dnlpm*tm->dt*alpha_avg;
  
        
        // **_sum/dtSum gives time average of **
        Phi_zf_rms_avg = Phi_zf_rms_sum/tm->dtSum;
        //kx2Phi_zf_rms_avg = kx2Phi_zf_rms_sum/dtSum;
        Dnlpm_avg = Dnlpm_sum/tm->dtSum;
  
        for(int s=0; s<nSpecies; s++) {
          wpfxAvg[s] = mu_avg*wpfxAvg[s] + (1-mu_avg)*wpfx[s] + (mu_avg - (1-mu_avg)/alpha_avg)*(wpfx[s] - wpfx_old[s]);
          pflxAvg[s] = mu_avg*pflxAvg[s] + (1-mu_avg)*pflx[s] + (mu_avg - (1-mu_avg)/alpha_avg)*(pflx[s] - pflx_old[s]);
        }
  
        alpha_nlpm = tm->dt/tau_nlpm;
        mu_nlpm = exp(-alpha_nlpm);
        if(tm->runtime<20) {
          Phi_zf_kx1_avg = Phi_zf_kx1; //allow a build-up time of tau_nlpm
          kx2Phi_zf_rms_avg = kx2Phi_zf_rms;
        }
        else { 
          Phi_zf_kx1_avg = mu_nlpm*Phi_zf_kx1_avg + (1-mu_nlpm)*Phi_zf_kx1 + (mu_nlpm - (1-mu_nlpm)/alpha_nlpm)*(Phi_zf_kx1 - Phi_zf_kx1_old);
          kx2Phi_zf_rms_avg = mu_nlpm*kx2Phi_zf_rms_avg + (1-mu_nlpm)*kx2Phi_zf_rms + (mu_nlpm - (1-mu_nlpm)/alpha_nlpm)*(kx2Phi_zf_rms - kx2Phi_zf_rms_old);
        }
  /*
        // try to autostop when wpfx converges
        // look at min and max of wpfxAvg over time... if wpfxAvg stays within certain bounds for a given amount of
        // time, it is converged
        if(counter >= navg*1.2) {
          //set bounds to be +/- .05*wpfxAvg
  	converge_bounds = .1*wpfxAvg[ION];
  	//if counter reaches navg/3, recenter bounds
  	if(counter == navg || converge_count == navg/3) {
  	  wpfxmax = wpfxAvg[ION] + .5*converge_bounds;
  	  wpfxmin = wpfxAvg[ION] - .5*converge_bounds;
  	  converge_count++;
  	}
  	//if wpfxAvg goes outside the bounds, reset the bounds.
  	else if(wpfxAvg[ION] > wpfxmax) {
            wpfxmax = wpfxAvg[ION] + .3*converge_bounds;
  	  wpfxmin = wpfxmax - converge_bounds;
  	  converge_count=0;
  	}
  	else if(wpfxAvg[ION] < wpfxmin) {
            wpfxmin = wpfxAvg[ION] - .3*converge_bounds;
  	  wpfxmax = wpfxmin + converge_bounds;
  	  converge_count=0;
  	}
  	//if wpfxAvg stays inside the bounds, increment the convergence counter.
  	else converge_count++; 
        }    
  */
      }
  
      fluxWrite(ev_h->files.fluxfile,pflx, pflxAvg, wpfx,wpfxAvg, Dnlpm, Dnlpm_avg, Phi_zf_kx1, Phi_zf_kx1_avg, kx2Phi_zf_rms, kx2Phi_zf_rms_avg, nu1_nlpm_max,nu22_nlpm_max,converge_count,tm->runtime,species);
    
  	     
      if(tm->counter%nsave==0 && write_phi) phiR_historyWrite(Phi,omega,tmpXY_R,tmpXY_R_h, tm->runtime, ev_h->files.phifile); //save time history of Phi(x,y,z=0)          
      
      
      // print wpfx to screen if not printing growth rates
      if(!write_omega && tm->counter%nwrite==0) printf("%d: wpfx = %f, dt = %f, dt_cfl =  %f, Dnlpm = %f\n", gpuID, wpfx[0],tm->dt, dt_cfl, Dnlpm);
      
      // write flux to file
      if(tm->counter%nsave==0) fflush(NULL);
               
      
      
      
      //print growth rates to screen every nwrite timesteps if write_omega
      if(write_omega) {
        if (tm->counter%nwrite==0 || stopcount==nstop-1 || tm->counter==nSteps-1) {
  	printf("ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<((Nx-1)/3+1); i++) {
  	  for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      printf("%.4f\t%.4f\t\t%.6f\t%.6f", ky_h[j], kx_h[i], omegaAvg_h[index].x/tm->dtSum, omegaAvg_h[index].y/tm->dtSum);
        ev_h->outs.omega[index].x = omegaAvg_h[index].x/tm->dtSum;
        ev_h->outs.omega[index].y = omegaAvg_h[index].y/tm->dtSum;
  	      if(Stable[index] >= stableMax) printf("\tomega");
  	      if(Stable[index+Nx*(Ny/2+1)] >= stableMax) printf("\tgamma");
  	      printf("\n");
  	    }
  	  }
  	  printf("\n");
  	}
  	//for(int i=2*Nx/3+1; i<2*Nx/3+1; i++) {
        for(int i=2*Nx/3+1; i<Nx; i++) {
            for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	    printf("%.4f\t%.4f\t\t%.6f\t%.6f", ky_h[j], kx_h[i], omegaAvg_h[index].x/tm->dtSum, omegaAvg_h[index].y/tm->dtSum);
        ev_h->outs.omega[index].x = omegaAvg_h[index].x/tm->dtSum;
        ev_h->outs.omega[index].y = omegaAvg_h[index].y/tm->dtSum;
  	    if(Stable[index] >= stableMax) printf("\tomega");
  	    if(Stable[index+Nx*(Ny/2+1)] >= stableMax) printf("\tgamma");
  	    printf("\n");
  	  }
  	  printf("\n");
  	}	
        }            
      }
  /*    
      cudaEventRecord(stop1,0);
      cudaEventSynchronize(stop1);
      cudaEventElapsedTime(&diagnostics_timer, start1, stop1);
      diagnostics_timer_total += diagnostics_timer;    
  */    
      //writedat_each();
      
	if (tm->counter%nwrite==0){
  writedat_each(&ev_h->grids, &ev_h->outs, &ev_h->fields, &ev_h->time);
}
#ifdef GS2_zonal
			} //end of iproc if
#endif
      //////nvtxRangePop();
      //if(tm->counter==9) cudaProfilerStop();
      tm->runtime+=tm->dt;
      tm->counter++;       
      //checkstop
      if(FILE* checkstop = fopen(ev_h->files.stopfileName, "r") ) {
        fclose(checkstop);
        stopcount++;
      }     
    
#ifdef GS2_zonal
			if(iproc==0) {
      if(tm->counter%nsave==0 || stopcount==nstop-1 || tm->counter==nSteps-1) {
        printf("%d: %f    %f     dt=%f   %d: %s\n",gpuID,tm->runtime,gs2_time_mp_code_time_/sqrt(2.), tm->dt,tm->counter,cudaGetErrorString(cudaGetLastError()));
      }
#endif
      
      
      
      //check for problems with run
      if(!LINEAR && !secondary_test && (isnan(wpfx[ION]) || isinf(wpfx[ION]) || wpfx[ION] < -100 || wpfx[ION] > 100000) ) {
        printf("\n-------------\n--RUN ERROR--\n-------------\n\n");
        
        stopcount=100;
#ifdef GS2_zonal
        abort();
        broadcast_integer(&stopcount);
#endif
/*
        if(tm->counter>nsave) {	
          printf("RESTARTING FROM LAST RESTART FILE...\n");
  	restartRead(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,pflxAvg, wpfxAvg, Phi2_kxky_sum, Phi2_zonal_sum,
      			zCorr_sum,&outs->expectation_ky_movav, &expectation_kx_sum, &Phi_zf_kx1_avg,
      			&tm->dtSum, &tm->counter,&runtime,&dt,&totaltimer,restartfileName);
        
          printf("cfl was %f. maxdt was %f.\n", cfl, maxdt);
          cfl = .8*cfl;
          maxdt = .8*maxdt;    
          printf("cfl is now %f. maxdt is now %f.\n", cfl, maxdt);
        }
        else{
  	printf("ERROR: FLUX IS NAN AT BEGINNING OF RUN\nENDING RUN...\n\n");
          stopcount=100;
        } */
      }
  
      else if(tm->counter%nsave==0) {
        cudaEventRecord(events->stop,0);
        cudaEventSynchronize(events->stop);
        cudaEventElapsedTime(&tm->timer,events->start,events->stop);
        tm->totaltimer+=tm->timer;
        cudaEventRecord(events->start,0);
        restartWrite(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,pflxAvg,wpfxAvg,Phi2_kxky_sum, Phi2_zonal_sum,
        			zCorr_sum, outs->expectation_ky_movav, outs->expectation_kx_movav, Phi_zf_kx1_avg,
        			tm->dtSum,tm->counter,tm->runtime,tm->dt,tm->totaltimer,restartfileName);
      }
      
      
      
      if(tm->counter%nsave == 0) gryfx_finish_diagnostics(Dens, Upar, Tpar, Tprp, Qpar, Qprp, 
                          Phi, tmp, tmp, field, tmpZ, CtmpX,
                          tmpXY, tmpXY, tmpXY, tmpXY2, tmpXY3, tmpXY4, tmpYZ, tmpYZ,
    			tmpX, tmpX2, tmpY, tmpY, tmpY, tmpY, tmpY2, tmpY2, tmpY2, 
                          ev_hd->grids.kxCover, ev_hd->grids.kyCover, tmpX_h, tmpY_h, tmpXY_h, tmpYZ_h, field_h, 
                          ev_h->grids.kxCover, ev_h->grids.kyCover, omegaAvg_h, qflux, &expectation_ky, &expectation_kx,
  			Phi2_kxky_sum, wpfxnorm_kxky_sum, Phi2_zonal_sum, zCorr_sum, outs->expectation_ky_movav, 
  			outs->expectation_kx_movav, tm->dtSum,
  			tm->counter, tm->runtime, false,
  			&Phi2, &flux1_phase, &flux2_phase, &Dens_phase, &Tpar_phase, &Tprp_phase,
  			Phi2_sum, flux1_phase_sum, flux2_phase_sum, Dens_phase_sum, Tpar_phase_sum, Tprp_phase_sum);
  
#ifdef GS2_zonal
			} //end of iproc if
#endif


#ifdef GS2_zonal
/*  
    gs2_reinit_mp_check_time_step_(&reset, &exit_flag); //use gs2_time_mp_code_dt_cfl to check gs2_time_mp_code_dt
    if(reset) {
      gs2_reinit_mp_reset_time_step_(&gs2_counter, &exit_flag); //change gs2_time_mp_code_dt
      dt = gs2_time_mp_code_dt_ * 2. / sqrt(2.);  // pass GS2 dt to GryfX, with appropriate normalization
    }
*/  
    //dt_old = dt;
    tm->dt = gs2_time_mp_code_dt_ * 2. / sqrt(2.);
    dt = tm->dt;

    //gs2_counter++; //each proc has its own gs2_counter
//if(DEBUG) printf("proc %d thinks counter is %d, gs2_counter is %d, nsteps is %d\n", iproc, counter, gs2_counter, nSteps);
  
#endif

    } //end of timestep loop 
 
#ifdef GS2_zonal
//if(DEBUG) printf("proc %d exited the timestep loop\n", iproc);
    //clock_gettime(CLOCK_MONOTONIC, &clockend);
    //diff = 1000000000L * (clockend.tv_sec - clockstart.tv_sec) + clockend.tv_nsec - clockstart.tv_nsec;
    //printf("elapsed time = %llu ns\n", (long long unsigned int) diff);
    ////GS2timer = ((double) (clockend - clockstart)) / CLOCKS_PER_SEC;
    ////cudaEventRecord(GS2stop,0);
    ////getError("After GS2stop");
    ////cudaEventElapsedTime(&GS2timer, GS2start, GS2stop);
    ////getError("After GS2timer");
    //printf("proc%d: GS2 time/timestep is %e ns\n", iproc, ((double) diff)/nSteps);

			if(iproc==0) {
#endif
  
    //exit_flag = 1; 
   
    //gs2_diagnostics_mp_loop_diagnostics_(&gs2_counter, &exit_flag);
  
    if(DEBUG) getError("just finished timestep loop");
    
    //cudaProfilerStop();
    cudaEventRecord(events->stop,0);
    cudaEventSynchronize(events->stop);
    cudaEventElapsedTime(&tm->timer,events->start,events->stop);
    tm->totaltimer+=tm->timer;
    
    nSteps = tm->counter;     //counter at which fields were last calculated
    endtime = tm->runtime;    //time at which fields were last calculated
    
    for(int s=0; s<nSpecies; s++) {
      qflux[s] = wpfxAvg[s];
      pflux[s] = pflxAvg[s];
    }
    
    ////////////////////////////////////////////////////////////
  
    if(DEBUG) getError("before restartWrite");  
    
    // save for restart run
    restartWrite(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi, pflxAvg, wpfxAvg, Phi2_kxky_sum, Phi2_zonal_sum, 
    			zCorr_sum, outs->expectation_ky_movav, outs->expectation_kx_movav, Phi_zf_kx1_avg,
    			tm->dtSum,tm->counter,tm->runtime,tm->dt,tm->totaltimer,restartfileName);
    
    if(DEBUG) getError("after restartWrite");
    
    //phiR_historyWrite(Phi1,omega,tmpXY_R,tmpXY_R_h, runtime, ev_h->files.phifile); //save time history of Phi(x,y,z=0)      
    
    gryfx_finish_diagnostics(Dens, Upar, Tpar, Tprp, Qpar, Qprp, 
                          Phi, tmp, tmp, field, tmpZ, CtmpX,
                          tmpXY, tmpXY, tmpXY, tmpXY2, tmpXY3, tmpXY4, tmpYZ, tmpYZ,
    			tmpX, tmpX2, tmpY, tmpY, tmpY, tmpY, tmpY2, tmpY2, tmpY2, 
                          ev_hd->grids.kxCover, ev_hd->grids.kyCover, tmpX_h, tmpY_h, tmpXY_h, tmpYZ_h, field_h, 
                          ev_h->grids.kxCover, ev_h->grids.kyCover, omegaAvg_h, qflux, &expectation_ky, &expectation_kx,
  			Phi2_kxky_sum, wpfxnorm_kxky_sum, Phi2_zonal_sum, zCorr_sum, outs->expectation_ky_movav, 
  			outs->expectation_kx_movav, tm->dtSum,
  			tm->counter, tm->runtime, true,
  			&Phi2, &flux1_phase, &flux2_phase, &Dens_phase, &Tpar_phase, &Tprp_phase,
  			Phi2_sum, flux1_phase_sum, flux2_phase_sum, Dens_phase_sum, Tpar_phase_sum, Tprp_phase_sum);
  
  
    
    
    //if(write_omega) stabilityWrite(stability,Stable,stableMax);
      
    //Timing       
    printf("Total steps: %d\n", tm->counter);
    printf("Total time (min): %f\n",tm->totaltimer/60000.);    //convert ms to minutes
    printf("Avg time/timestep (s): %f\n",tm->totaltimer/tm->counter/1000.);   //convert ms to s
    //printf("Advance steps:\t%f min\t(%f%)\n", step_timer_total/60000., 100*step_timer_total/totaltimer);
    //printf("Diagnostics:\t%f min\t(%f%)\n", diagnostics_timer_total/60000., 100*diagnostics_timer_total/totaltimer);
  
    
    fprintf(outfile,"expectation val of ky = %f\n", expectation_ky);
    fprintf(outfile,"expectation val of kx = %f\n", expectation_kx);
    fprintf(outfile,"Q_i = %f\n Phi_zf_rms = %f\n Phi2 = %f\n", qflux[ION],Phi_zf_rms_avg, Phi2);
    fprintf(outfile, "flux1_phase = %f \t\t flux2_phase = %f\nDens_phase = %f \t\t Tpar_phase = %f \t\t Tprp_phase = %f\n", flux1_phase, flux2_phase, Dens_phase, Tpar_phase, Tprp_phase);
    fprintf(outfile,"\nTotal time (min): %f\n",tm->totaltimer/60000);
    fprintf(outfile,"Total steps: %d\n", tm->counter);
    fprintf(outfile,"Avg time/timestep (s): %f\n",tm->totaltimer/tm->counter/1000);
      
    //cleanup  
    for(int s=0; s<nSpecies; s++) {
//      cudaFree(Dens[s]), cudaFree(Dens1[s]);
//      cudaFree(Upar[s]), cudaFree(Upar1[s]);
//      cudaFree(Tpar[s]), cudaFree(Tpar1[s]);
//      cudaFree(Tprp[s]), cudaFree(Tprp1[s]);
//      cudaFree(Qpar[s]), cudaFree(Qpar1[s]);
//      cudaFree(Qprp[s]), cudaFree(Qprp1[s]);
    }  
    //cudaFree(Phi);
    //cudaFree(Phi1);
    //cudaFree(Phi2_kxky_sum);
    //cudaFree(Phi_sum);
    
    //cudaFree(tmp);
    //cudaFree(field);      
    //cudaFree(tmpZ);
    //cudaFree(tmpX);
    //cudaFree(tmpY);
    //cudaFree(tmpY2);
    //cudaFree(tmpXY);
    //cudaFree(tmpXY2);
    //cudaFree(tmpYZ);
    
    //cudaFree(bmag);
    cudaFree(bmagInv), cudaFree(bmag_complex), cudaFree(bgrad);
    //cudaFree(omega);
    cudaFree(omegaAvg);
    for(int t=0; t<navg; t++) {
      //cudaFree(Phi2_XYBox[t]);
      if(LINEAR) cudaFree(omegaBox[t]);
    }
    
    //cudaFree(kx_shift), cudaFree(jump);
    
      
    //cudaFree(kx), cudaFree(ky), cudaFree(kz);
    
    cudaFree(deriv_nlps);
    cudaFree(derivR1_nlps);
    cudaFree(derivR2_nlps);
    cudaFree(resultR_nlps);

    cudaFree(nu_nlpm);
    cudaFree(nu1_nlpm);
    cudaFree(nu22_nlpm);
    
    cufftDestroy(NLPSplanR2C);
    cufftDestroy(NLPSplanC2R);
    cufftDestroy(ZDerivBplanR2C);
    cufftDestroy(ZDerivBplanC2R);
    cufftDestroy(ZDerivplan);
    for(int c=0; c<nClasses; c++) {
      cufftDestroy(ev_h->ffts.plan_covering[c]);
      cudaFree(ev_hd->grids.kxCover[c]);
      cudaFree(ev_hd->grids.kyCover[c]);
      cudaFree(ev_hd->grids.g_covering[c]); 
      cudaFree(ev_hd->grids.kz_covering[c]);
    }
    
    close_files(&ev_h->files);
    //fclose(fluxfile);
    //fclose(omegafile);
    //fclose(gammafile);
    //fclose(ev_h->files.phifile);
   writedat_end(ev_h->outs);
    
    //cudaProfilerStop();
#ifdef GS2_zonal
  			} //end of iproc if  
#endif
}    
    
    
    
    
