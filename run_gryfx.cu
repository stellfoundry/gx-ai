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
#include "gryfx_run_diagnostics.cu"


#ifdef GS2_zonal
extern "C" void advance_gs2(int* gs2_counter, cuComplex* dens_ky0_h, cuComplex* upar_ky0_h, cuComplex* tpar_ky0_h, cuComplex* tprp_ky0_h, cuComplex* qpar_ky0_h, cuComplex* qprp_ky0_h, cuComplex* phi_ky0_h, int* first_half_flag);
extern "C" void getmoms_gryfx(cuComplex* dens, cuComplex* upar, cuComplex* tpar, cuComplex* tprp, cuComplex* qpar, cuComplex* qprp, cuComplex* phi);
extern "C" void broadcast_integer(int* a);
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

    char filename[200];
    //device variables; main device arrays will be capitalized
    if(DEBUG) getError("run_gryfx.cu, before device alloc");
    specie* species_d;

    printf("At the beginning of run_gryfx, gs2 time is %f\n", gs2_time_mp_code_time_/sqrt(2.0));
    
////////////  
#ifdef GS2_zonal
			if(iproc==0) {

#endif
    omega_out_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)); 

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
  nlpm_struct * nlpm = &ev_h->nlpm;
  run_control_struct * ctrl = &ev_h->ctrl;

  initialize_run_control(ctrl, &ev_h->grids);

// The file local_pointers.cu
//declares and assigns local 
// pointers to members of the everything
// structs, for example Phi... it should
// eventually be unnecessary
#include "local_pointers.cu"



if (iproc==0){
   
    cudaMalloc((void**) &deriv_nlps, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &derivR1_nlps, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &derivR2_nlps, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &resultR_nlps, sizeof(float)*Nx*Ny*Nz);
  
    cudaMalloc((void**) &kz_complex, sizeof(float)*(Nz/2+1));
  
    cudaMalloc((void**) &PhiAvgDenom, sizeof(float)*Nx);
    
    cudaMalloc((void**) &species_d, sizeof(specie)*nSpecies);

    /* Temporary hack */
    ev_hd->pars.species = species_d;
  
  
    if(DEBUG) getError("run_gryfx.cu, after device alloc");
  
    cudaMemcpy(species_d, species, sizeof(specie)*nSpecies, cudaMemcpyHostToDevice);
    
    cudaMemcpy(z, z_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);

    copy_geo_arrays_to_device(&ev_hd->geo, &ev_h->geo, &ev_h->pars, ev_h->grids.Nz); 
    
    
    //set up plans for NLPS, ZDeriv, and ZDerivB
    //plan for ZDerivCovering done below
	  create_cufft_plans(&ev_h->grids, &ev_h->ffts);
     
    initialize_nlpm_coefficients(&ev_h->cdims, nlpm, &ev_hd->nlpm, ev_h->grids.Nz);
  
	  initialize_grids(&ev_h->pars, &ev_hd->grids, &ev_h->grids, &ev_h->cdims); 

  	calculate_additional_geo_arrays(
        ev_h->grids.Nz, ev_hd->grids.kz, ev_hd->tmp.Z,
        &ev_h->pars, &ev_h->cdims, 
        &ev_hd->geo, &ev_h->geo);

  
    //PhiAvg denominator for qneut
    cudaMemset(PhiAvgDenom, 0, sizeof(float)*Nx);
    phiavgdenom<<<dimGrid,dimBlock>>>(PhiAvgDenom, tmpXZ, jacobian, species_d, kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);  
  
    if(DEBUG) getError("run_gryfx.cu, after init"); 
   
} // if iproc
  
  
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //set up kxCover and kyCover for covering space z-transforms
    //nshift = Nx - ntheta0;
    initialize_z_covering(iproc, &ev_hd->grids, &ev_h->grids, &ev_h->pars, &ev_h->ffts, &ev_h->streams, &ev_h->cdims, &ev_h->events);  

    ///////////////////////////////////////////////////////////////////////////////////////////////////

if (iproc==0){

  
    initialize_averaging_parameters(outs, ev_h->pars.navg);
    
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
    ev_hd->sfixed.S = ev_h->sfixed.S=1.0;
    
    //if running nonlinear part of secondary test...
    if(secondary_test && !LINEAR && RESTART) { 
       restartRead(ev_h, ev_hd);
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
       //tm->dt=ev_h->pars.dt;

       maxdt = .1/(phi_test.x*kx_h[1]*ky_h[1]);
    } 
    
    //float amp;
    
    if(!RESTART) {
      set_initial_conditions_no_restart(
          &ev_h->pars, &ev_d->pars, &ev_h->grids, &ev_d->grids, 
          &ev_h->cdims, &ev_d->geo, &ev_hd->fields, &ev_hd->tmp);
		  zero_moving_averages(&ev_h->grids, &ev_h->cdims, &ev_hd->outs, &ev_h->outs, tm);
      
      outs->phases.flux1_sum = 0.;
      outs->phases.flux2_sum = 0.;
      outs->phases.Dens_sum = 0.;
      outs->phases.Tpar_sum = 0.;
      outs->phases.Tprp_sum = 0.;
      //zeroC<<<dimGrid,dimBlock>>>(Phi_sum);
    } 
    else {
      restartRead(ev_h, ev_hd);
      if(zero_restart_avg) {
		    zero_moving_averages(&ev_h->grids, &ev_h->cdims, &ev_hd->outs, &ev_h->outs, tm);
      }
    }
      
    create_cuda_events_and_streams(events, &ev_h->streams, ev_h->grids.nClasses);

} //end of iproc if    
    tm->runtime=0.;
    tm->counter=0;
    tm->gs2_counter=1;
    tm->totaltimer=0.;
    tm->timer=0.;
      //GS2timer=0.;
    ctrl->stopcount = 0;
    ctrl->nstop = 10;

#ifdef GS2_zonal

    MPI_Bcast(&ev_h->grids.Nz, 1, MPI_INT, 0, ev_h->mpi.mpcom);
    MPI_Bcast(&ev_h->grids.ntheta0, 1, MPI_INT, 0, ev_h->mpi.mpcom);
    MPI_Bcast(&ev_h->grids.Nspecies, 1, MPI_INT, 0, ev_h->mpi.mpcom);
      initialize_hybrid_arrays(ev_h->mpi.iproc,
       &ev_h->grids,
      &ev_h->hybrid, &ev_hd->hybrid);
      
if(iproc==0) {
    copy_hybrid_arrays_from_host_to_device_async(
        &ev_h->grids, &ev_h->hybrid, 
        &ev_hd->hybrid, streams);
    
  
    cudaEventRecord(events->H2D, streams->copystream);
    cudaStreamWaitEvent(0, events->H2D, 0);
    
    fieldWrite_nopad_h(phi_ky0_h, "phi0.field", filename, Nx, 1, Nz, ev_h->grids.ntheta0, 1);
    replace_zonal_fields_with_hybrid(
      1,
      &ev_h->cdims, &ev_hd->fields,
      ev_hd->fields.phi,
      &ev_hd->hybrid, ev_h->fields.field);
   
} // if iproc 
  
#endif
    
#ifdef GS2_zonal
			if(iproc==0) {  
#endif
      if(secondary_test && !LINEAR) {
        copy_fixed_modes_into_fields(
          &ev_h->cdims, &ev_hd->fields, ev_hd->fields.phi, &ev_hd->sfixed);
      }

      if(init == DENS) {
        // Solve for initial phi
        // assumes the initial conditions have been moved to the device
        qneut(Phi, Dens, Tprp, tmp, tmp, field, species, species_d);
      }


  if(DEBUG) getError("about to start timestep loop");

  write_initial_fields(
    &ev_h->cdims, &ev_hd->fields,
    &ev_hd->tmp, ev_h->fields.field,
    ev_h->tmp.X);

#ifdef GS2_zonal
			}
#endif 

    //uint64_t diff;
    struct timespec clockstart;//, clockend;
    //MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
   tm->first_half_flag = 1;

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
   
    while(tm->counter<nSteps && ctrl->stopcount<ctrl->nstop)
     {

      //if(counter==9) cudaProfilerStart();
			if(iproc==0) {

#ifdef GS2_zonal

      if(gs2_time_mp_code_time_/sqrt(2.) - tm->runtime > .0001) printf("\n\nRuntime mismatch! GS2 time is %f, GryfX time is %f\n\n", gs2_time_mp_code_time_/sqrt(2.), tm->runtime); 

#endif

      
  
      //EXBshear bug fixed, need to check if correct
      //ExBshear(Phi,Dens,Upar,Tpar,Tprp,Qpar,Qprp,kx_shift,jump,avgdt);  
      
      //if(DEBUG) getError("after exb");
     

			} //end of iproc if

  /////////////////////////////////
  //FIRST HALF OF GRYFX TIMESTEP
  /////////////////////////////////
  
      //////nvtxRangePushA("Gryfx t->t+tm->dt/2");

      tm->first_half_flag = 1;
#ifdef GS2_zonal
			if(iproc==0) {
#endif
      if(!LINEAR) {
        for(int s=0; s<nSpecies; s++) {
          //calculate NL(t) = NL(Moment)
          // first_half_flag determines which half of 
          // RK2 we are doing
          nonlinear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
          //Moment1 = Moment + (dt/2)*NL(Moment)

#ifdef GS2_zonal
  
          //copy NL(t)_ky=0 from D2H
  
          if(s==nSpecies-1) {  //Only after all species have been done
            cudaEventRecord(events->nonlin_halfstep, 0); //record this after all streams (ie the default stream) reach this point
            cudaStreamWaitEvent(streams->copystream, events->nonlin_halfstep,0); //wait for all streams before copying
            copy_hybrid_arrays_from_device_to_host_async(
                &ev_h->grids, &ev_h->hybrid, &ev_hd->hybrid,
                &ev_h->streams);
            cudaEventRecord(events->D2H, streams->copystream);
          }

          if(tm->counter==0) fieldWrite_nopad_h(dens_ky0_h, "NLdens.field", filename, Nx, 1, Nz, ev_h->grids.ntheta0, 1);
  
#endif
  
          //calculate L(t) = L(Moment)
          // first_half_flag determines which half of RK2 we are doing
          // The new fields end up in dens1 etc
          //Moment1 = Moment1 + (tm->dt/2)*L(Moment)
          linear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
  	}         
      }
      else { //if only linear
        for(int s=0; s<nSpecies; s++) {
          //calculate L(t) = L(Moment)
          // first_half_flag determines which half of  RK2 we are doing
          // The new fields end up in dens1 etc
          //Moment1 = Moment + (tm->dt/2)*L(Moment)
          linear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);

        }
      }

//if(DEBUG && counter==0) getError("after linear step"); 

      qneut(Phi1, Dens1, Tprp1, tmp, tmp, field, species, species_d);
  
      if(secondary_test && !LINEAR) {
        if(tm->runtime < .02/maxdt/M_PI) ev_h->sfixed.S = 1.;// sin(.01/maxdt * tm->runtime);
        else ev_h->sfixed.S = 1.;
        ev_hd->sfixed.S = ev_h->sfixed.S;
        copy_fixed_modes_into_fields( &ev_h->cdims, &ev_hd->fields1,
               ev_hd->fields1.phi, &ev_hd->sfixed);
      }

  
			} //end of iproc if

      //////nvtxRangePop();
#ifdef GS2_zonal
    
    if(!LINEAR) {
      if(iproc==0) cudaEventSynchronize(events->D2H); //have proc 0 wait for NL(t) to be copied D2H before advancing GS2 if running nonlinearly
    }
    advance_gs2(&tm->gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h, &tm->first_half_flag);
    tm->gs2_counter++;

if(iproc==0) {  
    //copy moms(t+dt/2)_ky=0 from H2D
    copy_hybrid_arrays_from_host_to_device_async(
        &ev_h->grids, &ev_h->hybrid, 
        &ev_hd->hybrid, streams);
    
  
    cudaEventRecord(events->H2D, streams->copystream);
    cudaStreamWaitEvent(0, events->H2D, 0);

    replace_zonal_fields_with_hybrid(
      0,
      &ev_h->cdims, &ev_hd->fields1,
      ev_hd->fields1.phi,
      &ev_hd->hybrid, ev_h->fields.field);
   
 } 
    //////nvtxRangePop();
  #endif
         

    //////nvtxRangePushA("Gryfx t->t+dt");

      tm->first_half_flag = 0;
if(iproc==0) {

      //NLPM calculated AFTER ky=0 quantities passed back from GS2!
      if(!LINEAR && NLPM && dorland_phase_complex) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPMcomplex(s, 
            &ev_hd->fields1, &ev_hd->tmp, &ev_hd->nlpm, nlpm, tm->dt/2.,
            ev_h->pars.species[s], &ev_d->nlpm.D); //NB this is ev_d here
        }	    
      }
      else if(!LINEAR && NLPM) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPM(s, 
            &ev_hd->fields1, &ev_hd->tmp, &ev_hd->nlpm, nlpm, tm->dt/2.,
            ev_h->pars.species[s], &ev_d->nlpm.D); //NB this is ev_d here
        }	    
      }  
      //hyper too...
      if(HYPER) {
        if(isotropic_shear) {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_iso(s, &ev_hd->fields1, ev_hd->tmp.XYZ, ev_hd->hyper.shear_rate_nz, tm->dt/2.);
  		    
          }  
        }
        else {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_aniso(s, &ev_hd->fields1, ev_hd->tmp.XYZ, &ev_hd->hyper, tm->dt/2.);
          }
        }
      }
 
  /////////////////////////////////
  //SECOND HALF OF GRYFX TIMESTEP
  /////////////////////////////////
      if(!LINEAR) {
        for(int s=0; s<nSpecies; s++) {
          // first_half_flag determines which half of 
          // RK2 we are doing
          //calculate NL(t+tm->dt/2) = NL(Moment1)
          nonlinear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
  
          //Moment = Moment + dt * NL(Moment1)
  
  #ifdef GS2_zonal
  
          //copy NL(t+dt/2)_ky=0 from D2H
  
          if(s==nSpecies-1) {  //Only after all species have been done
            cudaEventRecord(events->nonlin_halfstep, 0); //record this after all streams (ie the default stream) reach this point
            cudaStreamWaitEvent(streams->copystream, events->nonlin_halfstep,0); //wait for all streams before copying
            copy_hybrid_arrays_from_device_to_host_async(
                &ev_h->grids, &ev_h->hybrid, &ev_hd->hybrid,
                &ev_h->streams);
            cudaEventRecord(events->D2H, streams->copystream);
          }
  
  #endif
  
          // first_half_flag determines which half of 
          // RK2 we are doing
          // The new fields end up in dens etc
          //calculate L(t+dt/2)=L(Moment1) 
          //Moment = Moment + dt * L(Moment1)
          linear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
  	}         
      }
      else { //if only linear
        for(int s=0; s<nSpecies; s++) {
  
          // first_half_flag determines which half of 
          // RK2 we are doing
          // The new fields end up in dens etc
          linear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
        }
      }
      

        if(!LINEAR && !secondary_test && !write_omega) qneut(Phi, Dens, Tprp, tmp, tmp, field, species, species_d); //don't need to keep Phi=Phi(t) when running nonlinearly, overwrite with Phi=Phi(t+dt)
        else qneut(Phi1, Dens, Tprp, tmp, tmp, field, species, species_d); //don't overwrite Phi=Phi(t), use Phi1=Phi(t+dt); for growth rate calculation
      
      if(secondary_test && !LINEAR) {
        copy_fixed_modes_into_fields(
          &ev_h->cdims, &ev_hd->fields, ev_hd->fields1.phi, &ev_hd->sfixed);
      }
      //f = f(t+dt)
  
} //end of iproc if

    //////nvtxRangePop();

#ifdef GS2_zonal
    
    
    if(!LINEAR) {
      if(iproc==0) cudaEventSynchronize(events->D2H); //wait for NL(t+dt/2) to be copied D2H before advancing GS2 if running nonlinearly
    }
    //advance GS2 t+dt/2 -> t+dt
    advance_gs2(&tm->gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h, &tm->first_half_flag);
    tm->gs2_counter++;

if(iproc==0) {  
    //copy moms(t+dt)_ky=0 from H2D
    copy_hybrid_arrays_from_host_to_device_async(
        &ev_h->grids, &ev_h->hybrid, 
        &ev_hd->hybrid, streams);
    
  
    cudaEventRecord(events->H2D, streams->copystream);
    cudaStreamWaitEvent(0, events->H2D, 0);

    cuComplex * phiptr;
   
    if(!LINEAR && !secondary_test && !write_omega) {
      phiptr = ev_hd->fields.phi;
    } else {
      phiptr = ev_hd->fields1.phi;
    }

    replace_zonal_fields_with_hybrid(
      0,
      &ev_h->cdims, &ev_hd->fields,
      phiptr,
      &ev_hd->hybrid, ev_h->fields.field);
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
          filterNLPMcomplex(s, &ev_hd->fields, &ev_hd->tmp,
            &ev_hd->nlpm, nlpm, tm->dt, ev_h->pars.species[s],
            &ev_d->nlpm.D); //NB this is ev_d here
        }	    
      }
      else if(!LINEAR && NLPM) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPM(s, &ev_hd->fields, &ev_hd->tmp,
            &ev_hd->nlpm, nlpm, tm->dt, ev_h->pars.species[s],
            &ev_d->nlpm.D); //NB this is ev_d here
        }	    
      }  
          
      
      if(HYPER) {
        if(isotropic_shear) {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_iso(s, &ev_hd->fields, ev_hd->tmp.XYZ, ev_hd->hyper.shear_rate_nz, tm->dt/2.);
  		    
          }  
        }
        else {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_aniso(s, &ev_hd->fields, ev_hd->tmp.XYZ, &ev_hd->hyper, tm->dt/2.);
          }
        }
      }

      update_nlpm_coefficients( &ev_h->cdims, &ev_h->pars, &ev_h->outs,
        &ev_h->nlpm, &ev_hd->nlpm, &ev_d->nlpm, ev_hd->fields.phi, &ev_hd->tmp, tm);
    

      //DIAGNOSTICS
      gryfx_run_diagnostics(ev_h, ev_hd);
     	if (tm->counter%nwrite==0) writedat_each(&ev_h->grids, &ev_h->outs, &ev_h->fields, &ev_h->time);

	} //end of iproc if
      tm->runtime+=tm->dt;
      tm->counter++;       
      //checkstop
      if(FILE* checkstop = fopen(ev_h->files.stopfileName, "r") ) {
        fclose(checkstop);
        ctrl->stopcount++;
      }     
    
#ifdef GS2_zonal
			if(iproc==0) {
      if(tm->counter%nsave==0 || ctrl->stopcount==ctrl->nstop-1 || tm->counter==nSteps-1) {
        printf("%d: %f    %f     dt=%f   %d: %s\n",gpuID,tm->runtime,gs2_time_mp_code_time_/sqrt(2.), tm->dt,tm->counter,cudaGetErrorString(cudaGetLastError()));
      }
#endif
      
      
      
      //check for problems with run
      if(!LINEAR && !secondary_test && (isnan(wpfx[ION]) || isinf(wpfx[ION]) || wpfx[ION] < -100 || wpfx[ION] > 100000) ) {
        printf("\n-------------\n--RUN ERROR--\n-------------\n\n");
        
        ctrl->stopcount=100;
#ifdef GS2_zonal
        abort();
        broadcast_integer(&ctrl->stopcount);
#endif
      }
  
      else if(tm->counter%nsave==0) {
        cudaEventRecord(events->stop,0);
        cudaEventSynchronize(events->stop);
        cudaEventElapsedTime(&tm->timer,events->start,events->stop);
        tm->totaltimer+=tm->timer;
        cudaEventRecord(events->start,0);
        restartWrite(ev_h, ev_hd);
      }
      
      
      
     if(tm->counter%nsave == 0) gryfx_finish_diagnostics(ev_h, ev_hd, false);
  
} //end of iproc if


#ifdef GS2_zonal
    tm->dt = gs2_time_mp_code_dt_ * 2. / sqrt(2.);
    dt = tm->dt;
#endif

} 
////////////////////////////
////end of timestep loop 
/////////////////////////////
 
if(iproc==0) {
  
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
    restartWrite(ev_h, ev_hd);
    
    if(DEBUG) getError("after restartWrite");
    
    
    gryfx_finish_diagnostics(ev_h, ev_hd, true);
      
    //Timing       
    printf("Total steps: %d\n", tm->counter);
    printf("Total time (min): %f\n",tm->totaltimer/60000.);    //convert ms to minutes
    printf("Avg time/timestep (s): %f\n",tm->totaltimer/tm->counter/1000.);   //convert ms to s
    //printf("Advance steps:\t%f min\t(%f%)\n", step_timer_total/60000., 100*step_timer_total/totaltimer);
    //printf("Diagnostics:\t%f min\t(%f%)\n", diagnostics_timer_total/60000., 100*diagnostics_timer_total/totaltimer);
  
    
    fprintf(outfile,"expectation val of ky = %f\n", outs->expectation_ky);
    fprintf(outfile,"expectation val of kx = %f\n", outs->expectation_kx);
    fprintf(outfile,"Q_i = %f\n Phi_zf_rms = %f\n Phi2 = %f\n", qflux[ION],outs->phi2_zf_rms_avg, outs->phi2);
    fprintf(outfile, "flux1_phase = %f \t\t flux2_phase = %f\nDens_phase = %f \t\t Tpar_phase = %f \t\t Tprp_phase = %f\n", outs->phases.flux1, outs->phases.flux2, outs->phases.Dens, outs->phases.Tpar, outs->phases.Tprp);
    fprintf(outfile,"\nTotal time (min): %f\n",tm->totaltimer/60000);
    fprintf(outfile,"Total steps: %d\n", tm->counter);
    fprintf(outfile,"Avg time/timestep (s): %f\n",tm->totaltimer/tm->counter/1000);
      
    //cleanup  
    
    cudaFree(deriv_nlps);
    cudaFree(derivR1_nlps);
    cudaFree(derivR2_nlps);
    cudaFree(resultR_nlps);

    
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
   writedat_end(ev_h->outs);
    
    //cudaProfilerStop();
 } //end of iproc if  
}    
    
     
    
    
