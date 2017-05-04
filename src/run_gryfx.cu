#include "mpi.h"
#include "standard_headers.h"
//#include "allocations.h"
//#include "everything_else.h"
//#include "global_variables.h"
//#include "get_error.h"
//#include "write_data.h"
//#include "gs2.h"
//#include "profile.h"
//#include "printout.h"
//#include "time.h"
#include "parameters.h"
#include "grids.h"
#include "fields.h"

#ifdef GS2_zonal
extern "C" void broadcast_integer(int* a);
#endif

void run_gryfx(Parameters *pars, double * pflux, double * qflux, FILE* outfile)
{
  // geometry coefficient arrays
  Geometry* geo;
  if(iproc == 0) {
    int igeo = pars->igeo;

    printf("Initializing geometry...\n\n");

    if(igeo==0) {
      geo = new S_alpha_geo(pars);
    } else if(igeo==1) {
      printf("igeo = 1 not yet implemented!\n");
      exit(1);
      //geo = new Eik_geo();
    } else if(igeo==2) {
      printf("igeo = 2 not yet implemented!\n");
      exit(1);
      //geo = new Gs2_geo();
    }
  }

  // grids (e.g. kx)
  Grids* grids = new Grids(pars);
  grids->initialize();

  Fields* fields = new Fields(pars);
  fields->initialize(pars);
  
  // cufft plans
  Ffts* ffts = new Ffts();


//  set_globals_after_gryfx_lib(ev_h);
//  if (iproc==0) set_cuda_constants();
//
//  char filename[2000];
//  //device variables; main device arrays will be capitalized
//  if(DEBUG) getError(&ev_h->info,"run_gryfx.cu, before device alloc");
//  specie* species_d;
//
//#ifdef GS2_zonal
//  if(iproc==0) printf("At the beginning of run_gryfx, gs2 time is %f\n", gs2_time()/sqrt(2.0));
//#endif
//
//  if (iproc==0 && ev_h->info.job_id!=-1)  //This usually means we are running inside Trinity
//    printf("Job %d is running on GPU %d\n", ev_h->info.job_id, ev_h->info.gpuID); 
//
//  /* ev_hd is on the host but the pointers point to memory on the device*/
//  everything_struct * ev_hd;
//  /* ev_d is on the device (and the pointers point to memory on the device)*/
//  everything_struct * ev_d;
//  setup_everything_structs(ev_h, &ev_hd, &ev_d);
//
//
//  /* The time struct handles the 
//   * run progression, e.g. runtime,
//   * counter etc. Here we create a local
//   * reference for brevity */
//  time_struct * tm = &ev_h->time;
//  //Initialize dt
//  tm->dt = ev_h->pars.dt;
//  /* The outputs struct contains
//   * like moving averages */
//  outputs_struct * outs = &ev_h->outs;
//  cuda_events_struct * events = &ev_h->events;
//  cuda_streams_struct * streams = &ev_h->streams;
//  nlpm_struct * nlpm = &ev_h->nlpm;
//  run_control_struct * ctrl = &ev_h->ctrl;
//
//  initialize_run_control(ctrl, &ev_h->grids);
//
//  // The file local_pointers.cu
//  //declares and assigns local 
//  // pointers to members of the everything
//  // structs, for example Phi... it should
//  // eventually be unnecessary
//#include "local_pointers.cu"
//
//
//
//  if (iproc==0){
//
//    cudaMalloc((void**) &deriv_nlps, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
//    cudaMalloc((void**) &derivR1_nlps, sizeof(float)*Nx*Ny*Nz);
//    cudaMalloc((void**) &derivR2_nlps, sizeof(float)*Nx*Ny*Nz);
//    cudaMalloc((void**) &derivR3_nlps, sizeof(float)*Nx*Ny*Nz);
//    cudaMalloc((void**) &resultR_nlps, sizeof(float)*Nx*Ny*Nz);
//
//    cudaMalloc((void**) &kz_complex, sizeof(float)*(Nz/2+1));
//
//
//    cudaMalloc((void**) &species_d, sizeof(specie)*nSpecies);
//    cudaMemcpy(species_d, species, sizeof(specie)*nSpecies, cudaMemcpyHostToDevice);
//    /* Temporary hack */
//    ev_hd->pars.species = species_d;
//
//    if(DEBUG) getError(&ev_h->info,"run_gryfx.cu, after device alloc");
//
//    copy_geo_arrays_to_device(&ev_hd->geo, &ev_h->geo, &ev_h->pars, ev_h->grids.Nz); 
//    //set up plans for NLPS, ZDeriv, and ZDerivB
//    //plan for ZDerivCovering done below
//    create_cufft_plans(&ev_h->grids, &ev_h->ffts);
//    initialize_nlpm_coefficients(&ev_h->cdims, nlpm, &ev_hd->nlpm, ev_h->grids.Nz);
//    initialize_grids(&ev_h->pars, &ev_hd->grids, &ev_h->grids, &ev_h->cdims); 
//    calculate_additional_geo_arrays( ev_h->grids.Nz, ev_hd->grids.kz, ev_hd->tmp.Z,
//        &ev_h->pars, &ev_h->cdims, &ev_hd->geo, &ev_h->geo);
//    //PhiAvg denominator for qneut
//    initialize_phi_avg_denom(&ev_h->cdims, &ev_h->pars, &ev_hd->grids, &ev_hd->geo, species_d, ev_hd->tmp.XZ);
//    if(ev_h->pars.debug) getError(&ev_h->info,"run_gryfx.cu, after init"); 
//
//  } // if iproc
//
//
//  ///////////////////////////////////////////////////////////////////////////////////////////////////
//  //set up kxCover and kyCover for covering space z-transforms
//  //nshift = Nx - ntheta0;
//  initialize_z_covering(iproc, &ev_hd->grids, &ev_h->grids, &ev_d->grids, &ev_h->pars, &ev_h->ffts, &ev_h->streams, &ev_h->cdims, &ev_h->events);  
//
//  ///////////////////////////////////////////////////////////////////////////////////////////////////
//
//  if (iproc==0){
//
//
//    initialize_averaging_parameters(outs, ev_h->pars.navg);
//
//    ////////////////////////////////////////////////
//    // set up some diagnostics/control flow files //
//    ////////////////////////////////////////////////
//
//    setup_files(&ev_h->files, &ev_h->pars, &ev_h->grids, ev_h->info.run_name);
//    if (ev_h->pars.write_netcdf){
//      writedat_beginning(ev_h);
//      //This first call to writedat_each creates the variables only.
//      writedat_each(&ev_h->grids, &ev_h->outs, &ev_h->fields, &ev_h->time);
//      ev_h->outs.gnostics.create = false;
//      ev_h->outs.gnostics.write = true;
//    }
//    ////////////////////////////////////////////
//
//
//    //////////////////////////////
//    // initial conditions set here
//    //////////////////////////////
//
//
//    if(DEBUG) getError(&ev_h->info,"run_gryfx.cu, before initial condition"); 
//
//    //if running nonlinear part of secondary test or nlpm_test
//    if( (secondary_test && !LINEAR && RESTART) || (ev_h->pars.nlpm_test && !LINEAR && RESTART) ) { 
//
//      ev_hd->sfixed.S = ev_h->sfixed.S=1.0;
//      if(secondary_test) {
//        // for secondary test, fixed mode is always (ky=ky_min, kx=0)
//        ev_h->pars.iky_fixed = 1;
//        ev_h->pars.ikx_fixed = 0;
//      }
//      ev_hd->sfixed.iky_fixed = ev_h->sfixed.iky_fixed = ev_h->pars.iky_fixed;
//      ev_hd->sfixed.ikx_fixed = ev_h->sfixed.ikx_fixed = ev_h->pars.ikx_fixed;
//
//      //restartRead(ev_h, ev_hd);
//      writedat_read_restart(ev_h, ev_hd);
//      //Check restart was successful
//      fieldWrite(ev_hd->fields.phi, field_h, "phi_restarted.field", filename);
//      //Here we copy the fields from the restart file into the 
//      // arrays in the sfixed struct, which will then be copied back
//      // at the appropriate points in the timestep loop.
//      load_fixed_arrays_from_restart( ev_h->grids.Nz, ev_h->tmp.CZ, &ev_h->pars,
//          &ev_hd->sfixed, &ev_hd->fields);
//      //initialize density with noise
//      ev_h->pars.restart = RESTART = false; 
//      //ev_h->pars.init = init = DENS;
//      strcpy(ev_h->info.restart_file_name, ev_h->info.run_name);
//      strcat(ev_h->info.restart_file_name, ".restart.bin");
//
//      //restart run from t=0
//      tm->counter = 0;
//      tm->runtime = 0.;
//      tm->dtSum = 0.;
//      //tm->dt=ev_h->pars.dt;
//
//      maxdt = ev_h->pars.dt;
//      //maxdt = .1/(phi_test.x*kx_h[1]*ky_h[1]);
//    } 
//
//    //float amp;
//
//    if(!RESTART) {
//      set_initial_conditions_no_restart(
//          &ev_h->pars, &ev_d->pars, &ev_h->grids, &ev_d->grids, 
//          &ev_h->cdims, &ev_d->geo, &ev_hd->fields, &ev_hd->tmp);
//      zero_moving_averages(&ev_h->grids, &ev_h->cdims, &ev_hd->outs, &ev_h->outs, tm);
//
//      tm->runtime=0.;
//      tm->timer=0.;
//
//      outs->phases.flux1_sum = 0.;
//      outs->phases.flux2_sum = 0.;
//      outs->phases.Dens_sum = 0.;
//      outs->phases.Tpar_sum = 0.;
//      outs->phases.Tprp_sum = 0.;
//      //zeroC<<<dimGrid,dimBlock>>>(Phi_sum);
//    } 
//    else {
//      //set_initial_conditions_no_restart(
//        //  &ev_h->pars, &ev_d->pars, &ev_h->grids, &ev_d->grids, 
//          //&ev_h->cdims, &ev_d->geo, &ev_hd->fields, &ev_hd->tmp);
//      //restartRead(ev_h, ev_hd);
//      writedat_read_restart(ev_h, ev_hd);
//      dt = ev_h->time.dt;
//      if(zero_restart_avg) {
//        zero_moving_averages(&ev_h->grids, &ev_h->cdims, &ev_hd->outs, &ev_h->outs, tm);
//      }
//    }
//
//    create_cuda_events_and_streams(events, &ev_h->streams, ev_h->grids.nClasses);
//
//  } //end of iproc if    
//  tm->counter=0;
//  tm->gs2_counter=1;
//  tm->totaltimer=0.;
//  //GS2timer=0.;
//  ctrl->stopcount = 0;
//  ctrl->nstop = 10;
//
//#ifdef GS2_zonal
//
//  MPI_Bcast(&ev_h->grids.Nz, 1, MPI_INT, 0, ev_h->mpi.mpcom);
//  MPI_Bcast(&ev_h->grids.ntheta0, 1, MPI_INT, 0, ev_h->mpi.mpcom);
//  MPI_Bcast(&ev_h->grids.Nspecies, 1, MPI_INT, 0, ev_h->mpi.mpcom);
//  initialize_hybrid_arrays(ev_h->mpi.iproc, &ev_h->grids,
//      &ev_h->hybrid, &ev_hd->hybrid);
//
//  if(iproc==0) {
//    copy_hybrid_arrays_from_host_to_device_async(
//        &ev_h->grids, &ev_h->hybrid, 
//        &ev_hd->hybrid, streams);
//
//
//    cudaEventRecord(events->H2D, streams->copystream);
//    cudaStreamWaitEvent(0, events->H2D, 0);
//
//    fieldWrite_nopad_h(phi_ky0_h, "phi0.field", filename, Nx, 1, Nz, ev_h->grids.ntheta0, 1);
//    if(!ev_h->pars.higher_order_moments) {
//      replace_zonal_fields_with_hybrid(
//        1,
//        &ev_h->cdims, &ev_hd->fields,
//        ev_hd->fields.phi,
//        &ev_hd->hybrid, ev_h->fields.field);
//    }
//
//  } // if iproc 
//
//#endif
//
// // MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
//  if(iproc==0) {  
//    if((secondary_test || ev_h->pars.nlpm_test) && !LINEAR) {
//      copy_fixed_modes_into_fields(
//          &ev_h->cdims, &ev_hd->fields, ev_hd->fields.phi, &ev_hd->sfixed, &ev_h->pars);
//    }
//
//    // Solve for initial phi
//    // assumes the initial conditions have been moved to the device
//    if(init == DENS) qneut(Phi, Apar, Dens, Tprp, Upar, Qprp, tmp, tmp, field, tmp, field, species, species_d, &ev_h->pars);
//               //ev_h->pars.adiabatic_electrons, ev_h->pars.fapar, ev_h->pars.beta, ev_h->pars.snyder_electrons);
//    if((secondary_test || ev_h->pars.nlpm_test) && !LINEAR) {
//      copy_fixed_modes_into_fields(
//          &ev_h->cdims, &ev_hd->fields, ev_hd->fields.phi, &ev_hd->sfixed, &ev_h->pars);
//    }
//    if(DEBUG) getError(&ev_h->info,"about to start timestep loop");
//
//
//    write_initial_fields(
//        &ev_h->cdims, &ev_hd->fields, &ev_hd->tmp, ev_h->fields.field, ev_h->tmp.X);
//
//  } // end if iproc
//
//  //uint64_t diff;
//  struct timespec clockstart;//, clockend;
//  //MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
//  tm->first_half_flag = 1;
//
//  clock_gettime(CLOCK_MONOTONIC, &clockstart);
////#ifdef GS2_zonal
//  if(iproc==0) {
////#endif
//    cudaEventRecord(events->start,0);
////#ifdef GS2_zonal
//  }
////#endif
//
//  if(iproc==0) print_initial_parameter_summary(ev_h);
//  /////////////////////
//  // Begin timestep loop
//  /////////////////////
//  if(iproc==0) {
//      if(ev_h->pars.nlpm_test) gryfx_run_diagnostics(ev_h, ev_hd);
//  }
//
//  //Broadcast nstep just in case it was changed by Trinity
//  MPI_Bcast(&ev_h->pars.nstep, 1, MPI_INT, 0, ev_h->mpi.mpcom);
//  nSteps = ev_h->pars.nstep;
//
//  MPI_Bcast(&tm->end_time, 1, MPI_DOUBLE, 0, ev_h->mpi.mpcom); // make sure all procs know the end time
//
//  if(iproc==0&&DEBUG) getError(&ev_h->info,"run_gryfx.cu, about to start timestep loop"); 
//
//
//  while(tm->counter<nSteps && ctrl->stopcount<ctrl->nstop)
//  {
//    double epoch_seconds= (double)time(NULL);
//    MPI_Bcast(&epoch_seconds, 1, MPI_DOUBLE, 0, ev_h->mpi.mpcom); // make sure we all agree what the time is!
//    if (tm->end_time < epoch_seconds){
//      if (iproc==0) printf("Available time exceeded; quitting timestep loop\n");
//      break; 
//    }
//    else
//    {
//      if (ev_h->pars.debug&&iproc==0) 
//        printf("Time is %e, end time is %e, difference %e, avail_cpu_time is %e and margin_cpu_time is %e\n",
//            epoch_seconds, tm->end_time, tm->end_time-epoch_seconds, ev_h->pars.avail_cpu_time, ev_h->pars.margin_cpu_time);
//    }
//
//    if(iproc==0) {
//#ifdef GS2_zonal
//      if(gs2_time()/sqrt(2.) - tm->runtime > .0001){
//        printf("\n\nRuntime mismatch! GS2 time is %f, GryfX time is %f\n\n", gs2_time()/sqrt(2.), tm->runtime); 
//        abort();
//      }
//#endif
//      //EXBshear bug fixed, need to check if correct
//      //ExBshear(Phi,Dens,Upar,Tpar,Tprp,Qpar,Qprp,kx_shift,jump,avgdt);  
//
//      //if(DEBUG) getError("after exb");
//    } //end of iproc if
//    
//    if(iproc==0&&DEBUG) getError(&ev_h->info,"Starting timestep \n", tm->counter);
//
//    /////////////////////////////////
//    //FIRST HALF OF GRYFX TIMESTEP
//    /////////////////////////////////
//
//    //////nvtxRangePushA("Gryfx t->t+tm->dt/2");
//
//    tm->first_half_flag = 1;
////#ifdef GS2_zonal
//    if(iproc==0) {
////#endif
//      if(!LINEAR) {
//        for(int s=0; s<nSpecies; s++) {
//          //calculate NL(t) = NL(Moment)
//          // first_half_flag determines which half of 
//          // RK2 we are doing
//          nonlinear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
//          //Moment1 = Moment + (dt/2)*NL(Moment)
//
//#ifdef GS2_zonal
//
//#ifdef PROFILE
//PUSH_RANGE("copy NL(t)_ky=0 from D2H",1);
//#endif
//          //copy NL(t)_ky=0 from D2H
//
//          if(s==nSpecies-1) {  //Only after all species have been done
//            cudaEventRecord(events->nonlin_halfstep, 0); //record this after all streams (ie the default stream) reach this point
//#ifdef PROFILE
//PUSH_RANGE("waiting for all streams to finish nonlin_halfstep",2);
//#endif
//            cudaStreamWaitEvent(streams->copystream, events->nonlin_halfstep,0); //wait for all streams before copying
//#ifdef PROFILE
//POP_RANGE;
//#endif
//            copy_hybrid_arrays_from_device_to_host_async(
//                &ev_h->grids, &ev_h->hybrid, &ev_hd->hybrid, &ev_h->streams);
//            cudaEventRecord(events->D2H, streams->copystream);
//          }
//
//          if(tm->counter==0) fieldWrite_nopad_h(dens_ky0_h, "NLdens.field", filename, Nx, 1, Nz, ev_h->grids.ntheta0, 1);
//
//#ifdef PROFILE
//POP_RANGE;
//#endif
//
//#endif
//
//          //calculate L(t) = L(Moment)
//          // first_half_flag determines which half of RK2 we are doing
//          // The new fields end up in dens1 etc
//          //Moment1 = Moment1 + (tm->dt/2)*L(Moment)
//        }         
//        if(iproc==0&&DEBUG) getError(&ev_h->info,"run_gryfx.cu completed nonlin 1st half"); 
//      }
//      
//      //LINEAR STEP
//      // when using snyder finite beta electron model, need to evolve electron equations before ion equations
//      if(ev_h->pars.snyder_electrons) {
//        linear_electron_timestep(nSpecies-1, tm->first_half_flag, ev_h, ev_hd, ev_d);
//        for(int s=0; s<nSpecies-1; s++) {
//          linear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
//        }
//      } else {
//        for(int s=0; s<nSpecies; s++) {
//          linear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
//        }
//      }
//
//      if(iproc==0&&DEBUG) getError(&ev_h->info,"run_gryfx.cu completed lin 1st half"); 
//
//      //if(DEBUG && counter==0) getError("after linear step"); 
//
//      qneut(Phi1, Apar1, Dens1, Tprp1, Upar1, Qprp1, tmp, tmp, field, tmp, field, species, species_d, &ev_h->pars);
//            //ev_h->pars.adiabatic_electrons, ev_h->pars.fapar, ev_h->pars.beta, ev_h->pars.snyder_electrons);
//
//    } //end of iproc if
//
//
//    //////nvtxRangePop();
//#ifdef GS2_zonal
//
//    if(!LINEAR) {
//      if(iproc==0) cudaEventSynchronize(events->D2H); //have proc 0 wait for NL(t) to be copied D2H before advancing GS2 if running nonlinearly
//    }
//    gryfx_advance_gs2(&ev_h->hybrid, tm);
//    // proc0 will not advance past this point until gryfx_advance_gs2 is complete.
//
//    if(iproc==0) {  
//#ifdef PROFILE
//PUSH_RANGE("copy moms(t+dt/2)_ky=0 H2D",5);
//#endif
//      //copy moms(t+dt/2)_ky=0 from H2D
//      copy_hybrid_arrays_from_host_to_device_async(
//          &ev_h->grids, &ev_h->hybrid, &ev_hd->hybrid, streams);
//      cudaEventRecord(events->H2D, streams->copystream);
//#ifdef PROFILE
//PUSH_RANGE("waiting for copystream to finish H2D",1);
//#endif
//      cudaStreamWaitEvent(0, events->H2D, 0);
//#ifdef PROFILE
//POP_RANGE;
//#endif
//
//    if(!ev_h->pars.higher_order_moments) {
//      replace_zonal_fields_with_hybrid( 0, &ev_h->cdims, &ev_hd->fields1,
//          ev_hd->fields1.phi, &ev_hd->hybrid, ev_h->fields.field);
//    }
//#ifdef PROFILE
//POP_RANGE;
//#endif
//    } 
//    //////nvtxRangePop();
//#endif
//
//    //////nvtxRangePushA("Gryfx t->t+dt");
//
//    tm->first_half_flag = 0;
//    if(iproc==0) {
//      if((secondary_test || ev_h->pars.nlpm_test) && !LINEAR) {
//        //ev_h->sfixed.S = 1.;
//        //ev_hd->sfixed.S = ev_h->sfixed.S;
//        copy_fixed_modes_into_fields( &ev_h->cdims, &ev_hd->fields1,
//            ev_hd->fields1.phi, &ev_hd->sfixed, &ev_h->pars);
//      }
//
//#ifdef PROFILE
//PUSH_RANGE("NLPM and hyper", 3);
//#endif
//      //NLPM calculated AFTER ky=0 quantities passed back from GS2!
//      if(!LINEAR && NLPM && dorland_phase_complex) {
//        for(int s=0; s<nSpecies; s++) {
//          filterNLPMcomplex(s, 
//              &ev_hd->fields1, &ev_hd->tmp, &ev_hd->nlpm, nlpm, tm->dt/2.,
//              ev_h->pars.species[s], &ev_d->nlpm.D); //NB this is ev_d here
//        }	    
//      }
//      else if(!LINEAR && NLPM) {
//        for(int s=0; s<nSpecies; s++) {
//          filterNLPM(s, 
//              &ev_hd->fields1, &ev_hd->tmp, &ev_hd->nlpm, nlpm, tm->dt/2.,
//              ev_h->pars.species[s], &ev_d->nlpm.D); //NB this is ev_d here
//        }	    
//      }  
//      //hyper too...
//      if(HYPER) {
//        if(isotropic_shear) {
//          for(int s=0; s<nSpecies; s++) {
//            filterHyper_iso(s, &ev_hd->fields1, ev_hd->tmp.XYZ, ev_hd->hyper.shear_rate_nz, tm->dt/2.);
//
//          }  
//        }
//        else {
//          for(int s=0; s<nSpecies; s++) {
//            filterHyper_aniso(s, &ev_hd->fields1, ev_hd->tmp.XYZ, &ev_hd->hyper, tm->dt/2.);
//          }
//        }
//      }
//#ifdef PROFILE
//POP_RANGE;
//#endif
//
//    if(iproc==0&&DEBUG) getError(&ev_h->info,"Starting 2nd half of timestep %d\n", tm->counter);
//      /////////////////////////////////
//      //SECOND HALF OF GRYFX TIMESTEP
//      /////////////////////////////////
//      if(!LINEAR) {
//        for(int s=0; s<nSpecies; s++) {
//          // first_half_flag determines which half of 
//          // RK2 we are doing
//          //calculate NL(t+tm->dt/2) = NL(Moment1)
//          nonlinear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
//
//
//          //Moment = Moment + dt * NL(Moment1)
//
//#ifdef GS2_zonal
//
//          //copy NL(t+dt/2)_ky=0 from D2H
//
//#ifdef PROFILE
//PUSH_RANGE("copy NL(t+dt/2)_ky=0 from D2H",1);
//#endif
//          if(s==nSpecies-1) {  //Only after all species have been done
//            cudaEventRecord(events->nonlin_halfstep, 0); //record this after all streams (ie the default stream) reach this point
//#ifdef PROFILE
//PUSH_RANGE("waiting for all streams to finish nonlin_halfstep",2);
//#endif
//            cudaStreamWaitEvent(streams->copystream, events->nonlin_halfstep,0); //wait for all streams before copying
//#ifdef PROFILE
//POP_RANGE;
//#endif
//            copy_hybrid_arrays_from_device_to_host_async(
//                &ev_h->grids, &ev_h->hybrid, &ev_hd->hybrid,
//                &ev_h->streams);
//            cudaEventRecord(events->D2H, streams->copystream);
//          }
//#ifdef PROFILE
//POP_RANGE;
//#endif
//
//#endif
//        }         
//        if(iproc==0&&DEBUG) getError(&ev_h->info,"run_gryfx.cu completed nonlin 2nd half"); 
//      }
//
//      //LINEAR STEP
//      // when using snyder finite beta electron model, need to evolve electron equations before ion equations
//      if(ev_h->pars.snyder_electrons) {
//        linear_electron_timestep(nSpecies-1, tm->first_half_flag, ev_h, ev_hd, ev_d);
//        for(int s=0; s<nSpecies-1; s++) {
//          linear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
//        }
//      } else {
//        for(int s=0; s<nSpecies; s++) {
//          linear_timestep(s, tm->first_half_flag, ev_h, ev_hd, ev_d);
//        }
//      }
//      if(iproc==0&&DEBUG) getError(&ev_h->info,"run_gryfx.cu completed lin 2nd half"); 
//
//
//      if(!LINEAR && !secondary_test && !ev_h->pars.nlpm_test && !write_omega) {
//        qneut(Phi, Apar, Dens, Tprp, Upar, Qprp, tmp, tmp, field, tmp, field, species, species_d, &ev_h->pars);
//        //      ev_h->pars.adiabatic_electrons, ev_h->pars.fapar, ev_h->pars.beta, ev_h->pars.snyder_electrons); 
//        //don't need to keep Phi=Phi(t) when we don't need growth rates, i.e. when running nonlinearly, so overwrite with Phi=Phi(t+dt)
//      } else { 
//        qneut(Phi1, Apar, Dens, Tprp, Upar, Qprp, tmp, tmp, field, tmp, field, species, species_d, &ev_h->pars);
//        //      ev_h->pars.adiabatic_electrons, ev_h->pars.fapar, ev_h->pars.beta, ev_h->pars.snyder_electrons); 
//        //don't overwrite Phi=Phi(t), use Phi1=Phi(t+dt); for growth rate calculation
//      }
//
//      if(iproc==0&&DEBUG) getError(&ev_h->info,"run_gryfx.cu completed quasineutrality 2nd half"); 
//      //f = f(t+dt)
//
//    } //end of iproc if
//
//    //////nvtxRangePop();
//
//#ifdef GS2_zonal
//
//    if(!LINEAR) {
//      if(iproc==0) cudaEventSynchronize(events->D2H); //wait for NL(t+dt/2) to be copied D2H before advancing GS2 if running nonlinearly
//    }
//
//    if(iproc==0&&DEBUG) getError(&ev_h->info,"run_gryfx.cu copied nonlinearity to hybrid 2nd half"); 
//
//    //advance GS2 t+dt/2 -> t+dt
//    gryfx_advance_gs2(&ev_h->hybrid, tm);
//    // proc0 will not advance past this point until gryfx_advance_gs2 is complete.
//
//    if(iproc==0) {  
//#ifdef PROFILE
//PUSH_RANGE("copy moms(t+dt/2)_ky=0 H2D",5);
//#endif
//      //copy moms(t+dt)_ky=0 from H2D
//      copy_hybrid_arrays_from_host_to_device_async(
//          &ev_h->grids, &ev_h->hybrid, 
//          &ev_hd->hybrid, streams);
//
//
//      cudaEventRecord(events->H2D, streams->copystream);
//#ifdef PROFILE
//PUSH_RANGE("waiting for copystream to finish H2D",1);
//#endif
//      cudaStreamWaitEvent(0, events->H2D, 0);
//#ifdef PROFILE
//POP_RANGE;
//#endif
//
//      cuComplex * phiptr;
//
//      if(!LINEAR && !secondary_test && !ev_h->pars.nlpm_test && !write_omega) {
//        phiptr = ev_hd->fields.phi;
//      } else {
//        phiptr = ev_hd->fields1.phi;
//      }
//
//    if(!ev_h->pars.higher_order_moments) {
//      replace_zonal_fields_with_hybrid(
//          0,
//          &ev_h->cdims, &ev_hd->fields,
//          phiptr,
//          &ev_hd->hybrid, ev_h->fields.field);
//    }
//#ifdef PROFILE
//POP_RANGE;
//#endif
//    } // if iproc  
//
//    //////nvtxRangePop();
//
//#endif
//
//    if(iproc==0&&DEBUG) getError(&ev_h->info,"Finished 2nd half of timestep\n");
//    //////nvtxRangePushA("Diagnostics");
//
////#ifdef GS2_zonal
//    if(iproc==0) {
////#endif  
//      if((secondary_test || ev_h->pars.nlpm_test) && !LINEAR) {
//        copy_fixed_modes_into_fields(
//            &ev_h->cdims, &ev_hd->fields, ev_hd->fields1.phi, &ev_hd->sfixed, &ev_h->pars);
//      }
//
//#ifdef PROFILE
//PUSH_RANGE("NLPM and hyper", 3);
//#endif
//      //NLPM
//      if(!LINEAR && NLPM && dorland_phase_complex) {
//        for(int s=0; s<nSpecies; s++) {
//          filterNLPMcomplex(s, &ev_hd->fields, &ev_hd->tmp,
//              &ev_hd->nlpm, nlpm, tm->dt, ev_h->pars.species[s],
//              &ev_d->nlpm.D); //NB this is ev_d here
//        }	    
//      }
//      else if(!LINEAR && NLPM) {
//        for(int s=0; s<nSpecies; s++) {
//          filterNLPM(s, &ev_hd->fields, &ev_hd->tmp,
//              &ev_hd->nlpm, nlpm, tm->dt, ev_h->pars.species[s],
//              &ev_d->nlpm.D); //NB this is ev_d here
//        }	    
//      }  
//
//
//      if(HYPER) {
//        if(isotropic_shear) {
//          for(int s=0; s<nSpecies; s++) {
//            filterHyper_iso(s, &ev_hd->fields, ev_hd->tmp.XYZ, ev_hd->hyper.shear_rate_nz, tm->dt/2.);
//
//          }  
//        }
//        else {
//          for(int s=0; s<nSpecies; s++) {
//            filterHyper_aniso(s, &ev_hd->fields, ev_hd->tmp.XYZ, &ev_hd->hyper, tm->dt/2.);
//          }
//        }
//      }
//
//      update_nlpm_coefficients( &ev_h->cdims, &ev_h->pars, &ev_h->outs,
//          &ev_h->nlpm, &ev_hd->nlpm, &ev_d->nlpm, ev_hd->fields.phi, &ev_hd->tmp, tm);
//
//#ifdef PROFILE
//POP_RANGE;
//#endif
//
//
////#ifdef GS2_zonal
//    } //end of iproc if
////#endif
//    tm->runtime+=tm->dt;
//    tm->counter++;       
//    //checkstop
//
//    wpfx[ION]=0.0;
//    if(iproc==0&&DEBUG) getError(&ev_h->info,"Running diagnostics\n");
//    if(iproc==0) {
//      //DIAGNOSTICS
//      gryfx_run_diagnostics(ev_h, ev_hd);
//      if(tm->counter%nsave == 0) gryfx_finish_diagnostics(ev_h, ev_hd, false);
//      if (tm->counter%nwrite==0 && tm->counter!=0 && ev_h->pars.write_netcdf) writedat_each(&ev_h->grids, &ev_h->outs, &ev_h->fields, &ev_h->time);
//    }
//
//    if (iproc==0){
//      if(FILE* checkstop = fopen(ev_h->files.stopfileName, "r") ) {
//        fclose(checkstop);
//        ctrl->stopcount++;
//      }     
//    }
//    if(iproc==0&&DEBUG) getError(&ev_h->info,"Checked for stop file\n");
//
////#ifdef GS2_zonal
//    if(iproc==0) {
//#ifdef GS2_zonal
//      if(tm->counter%nsave==0 || ctrl->stopcount==ctrl->nstop-1 || tm->counter==nSteps-1) {
//        printf("%d: %f    %f     dt=%f   %d: %s\n",gpuID,tm->runtime,gs2_time()/sqrt(2.), tm->dt,tm->counter,cudaGetErrorString(cudaGetLastError()));
//      }
//#else
//      if(tm->counter%nsave==0 || ctrl->stopcount==ctrl->nstop-1 || tm->counter==nSteps-1) {
//        printf("%d: %f     dt=%f   %d: %s\n",gpuID,tm->runtime, tm->dt,tm->counter,cudaGetErrorString(cudaGetLastError()));
//      }
//#endif
//
//
//
//      //check for problems with run
//      if(tm->counter%nwrite==0&&tm->counter>0&&!LINEAR && !secondary_test && !ev_h->pars.nlpm_test && (isnan(wpfx[ION]) || isinf(wpfx[ION]) || wpfx[ION] < -100000.0 || wpfx[ION] > 100000.0) ) {
//        printf("wpfx: %f, hflux1 %f, hflux2 %f\n", wpfx[ION], ev_h->outs.hflux_tot, ev_h->outs.hflux_by_species[0]);
//        printf("\n-------------\n--RUN ERROR--\n-------------\n\n");
//
//        //ctrl->stopcount=100;
//#ifdef GS2_zonal
//        //abort();
//        //broadcast_integer(&ctrl->stopcount);
//#endif
//      }
//
//      else if(tm->counter%nsave==0) {
//        cudaEventRecord(events->stop,0);
//        cudaEventSynchronize(events->stop);
//        cudaEventElapsedTime(&tm->timer,events->start,events->stop);
//        tm->totaltimer+=tm->timer;
//        cudaEventRecord(events->start,0);
//        restartWrite(ev_h, ev_hd);
//        writedat_write_restart(ev_h, ev_hd);
//      }
//
//
//
////#ifdef GS2_zonal
//    } //end of iproc if
////#endif
//
//#ifdef GS2_zonal
//    tm->dt = gs2_dt() * 2. / sqrt(2.);
//    dt = tm->dt;
//#endif
//
//  } 
//  ////////////////////////////
//  ////end of timestep loop 
//  /////////////////////////////
//
//  if(iproc==0) {
//
//    if(DEBUG) getError(&ev_h->info,"just finished timestep loop");
//
//    //cudaProfilerStop();
//    cudaEventRecord(events->stop,0);
//    cudaEventSynchronize(events->stop);
//    cudaEventElapsedTime(&tm->timer,events->start,events->stop);
//    tm->totaltimer+=tm->timer;
//
//    nSteps = tm->counter;     //counter at which fields were last calculated
//    endtime = tm->runtime;    //time at which fields were last calculated
//
//    for(int s=0; s<nSpecies; s++) {
//      qflux[s] = wpfxAvg[s];
//      pflux[s] = pflxAvg[s];
//    }
//
//    ////////////////////////////////////////////////////////////
//
//    if(DEBUG) getError(&ev_h->info,"before restartWrite");  
//
//    // save for restart run
//    restartWrite(ev_h, ev_hd);
//    writedat_write_restart(ev_h, ev_hd);
//
//    if(DEBUG) getError(&ev_h->info,"after restartWrite");
//
//
//    gryfx_finish_diagnostics(ev_h, ev_hd, true);
//
//    if(ev_h->pars.debug) getError(&ev_h->info,"after gryfx_finish_diagnostics");
//    //Timing       
//    printf("Total steps: %d\n", tm->counter);
//    printf("Total time (min): %f\n",tm->totaltimer/60000.);    //convert ms to minutes
//    printf("Avg time/timestep (s): %f\n",tm->totaltimer/tm->counter/1000.);   //convert ms to s
//    //printf("Advance steps:\t%f min\t(%f%)\n", step_timer_total/60000., 100*step_timer_total/totaltimer);
//    //printf("Diagnostics:\t%f min\t(%f%)\n", diagnostics_timer_total/60000., 100*diagnostics_timer_total/totaltimer);
//
//
//    fprintf(outfile,"expectation val of ky = %f\n", outs->expectation_ky);
//    fprintf(outfile,"expectation val of kx = %f\n", outs->expectation_kx);
//    fprintf(outfile,"Q_i = %f\n Phi_zf_rms = %f\n Phi2 = %f\n", qflux[ION],outs->phi2_zf_rms_avg, outs->phi2);
//    fprintf(outfile, "flux1_phase = %f \t\t flux2_phase = %f\nDens_phase = %f \t\t Tpar_phase = %f \t\t Tprp_phase = %f\n", outs->phases.flux1, outs->phases.flux2, outs->phases.Dens, outs->phases.Tpar, outs->phases.Tprp);
//    fprintf(outfile,"\nTotal time (min): %f\n",tm->totaltimer/60000);
//    fprintf(outfile,"Total steps: %d\n", tm->counter);
//    fprintf(outfile,"Avg time/timestep (s): %f\n",tm->totaltimer/tm->counter/1000);
//
//    //cleanup  
//
//    cudaFree(deriv_nlps);
//    cudaFree(derivR1_nlps);
//    cudaFree(derivR2_nlps);
//    cudaFree(derivR3_nlps);
//    cudaFree(resultR_nlps);
//
//
//    cufftDestroy(NLPSplanR2C);
//    cufftDestroy(NLPSplanC2R);
//    cufftDestroy(ZDerivBplanR2C);
//    cufftDestroy(ZDerivBplanC2R);
//    cufftDestroy(ZDerivplan);
//    for(int c=0; c<nClasses; c++) {
//      cufftDestroy(ev_h->ffts.plan_covering[c]);
//      cudaFree(ev_hd->grids.kxCover[c]);
//      cudaFree(ev_hd->grids.kyCover[c]);
//      cudaFree(ev_hd->grids.g_covering[c]); 
//      cudaFree(ev_hd->grids.g_covering[c]); 
//      cudaFree(ev_hd->grids.kz_covering[c]);
//    }
//    //cudaFree(ev_hd->grids.g_covering_d);
//    //cudaFree(ev_hd->grids.kz_covering_d);
//    //cudaFree(ev_hd->grids.nLinks_d);
//    //cudaFree(ev_hd->grids.nChains_d);
//    //cudaFree(ev_hd->grids.kxCover_d);
//    //cudaFree(ev_hd->grids.kyCover_d);
//    //cudaFree(g_covering_d);
//
//    close_files(&ev_h->files);
//    if (ev_h->pars.write_netcdf) writedat_end(&ev_h->outs);
//
//    allocate_or_deallocate_everything(DEALLOCATE, ev_hd);
//    allocate_or_deallocate_everything(DEALLOCATE, ev_h);
//    free(ev_hd);
//    cudaFree(ev_d);
//
//    //cudaProfilerStop();
//  } //end of iproc if  
}    




