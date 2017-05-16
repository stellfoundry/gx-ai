void gryfx_run_diagnostics(
  everything_struct * ev_h,
  everything_struct * ev_hd
)
{
#ifdef PROFILE
PUSH_RANGE("gryfx diagnostics", 5);
#endif
  /* Some local shortcuts */
  cuda_dimensions_struct * cdims = &ev_h->cdims;
  input_parameters_struct * pars_h = &ev_h->pars;
  fields_struct * fields_d = &ev_hd->fields;
  fields_struct * fields1_d = &ev_hd->fields1;
  outputs_struct * outs_h = &ev_h->outs;
  outputs_struct * outs_d = &ev_hd->outs;
  files_struct * files_h = &ev_h->files;
  nlpm_struct * nlpm_h = &ev_h->nlpm;
  //nlpm_struct * nlpm_d,
  time_struct * tm_h = &ev_h->time;
  temporary_arrays_struct * tmp_h = &ev_h->tmp;
  temporary_arrays_struct * tmp_d = &ev_hd->tmp;
  run_control_struct * ctrl = &ev_h->ctrl;
  cuffts_struct * ffts = &ev_h->ffts;


    dim3 dimGrid = cdims->dimGrid;
    dim3 dimBlock = cdims->dimBlock;
    cuComplex * Phi = fields_d->phi;
    cuComplex * Phi1 = fields1_d->phi;

    cuComplex ** Dens = fields_d->dens;
    cuComplex ** Upar = fields_d->upar;
    cuComplex ** Tpar = fields_d->tpar;
    cuComplex ** Tprp = fields_d->tprp;
    cuComplex ** Qpar = fields_d->qpar;
    cuComplex ** Qprp = fields_d->qprp;
    //cuComplex ** Dens1 = fields1_d->dens;
    //cuComplex ** Upar1 = fields1_d->upar;
    //cuComplex ** Tpar1 = fields1_d->tpar;
    //cuComplex ** Tprp1 = fields1_d->tprp;
    //cuComplex ** Qpar1 = fields1_d->qpar;
    //cuComplex ** Qprp1 = fields1_d->qprp;

    float * wpfx = outs_h->hflux_by_species;
    float * wpfxAvg = outs_h->hflux_by_species_movav;
    float * pflxAvg = outs_h->pflux_by_species_movav;

    float alphav = outs_h->alpha_avg;
    float muav = outs_h->mu_avg;
       
      if(pars_h->nlpm_test) { 

        int init = pars_h->init;
        int idx0y0 = pars_h->inlpm; //(Ny/2+1)*(Nx/2+1);
        if(init==DENS || init==TPRP || init==TPAR) {
          cudaMemcpy(tmp_d->CXYZ, Dens[0], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
          mask<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          reality<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);

          cufftExecC2C(ffts->XYplanZ_C2C, tmp_d->CXYZ, tmp_d->CXYZ, CUFFT_INVERSE);
          get_z0<<<dimGrid,dimBlock>>>(outs_d->omega, tmp_d->CXYZ);
          cudaMemcpy(outs_h->omega, outs_d->omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
          fprintf(files_h->phifile, "\n\t%f\t%e", tm_h->runtime, outs_h->omega[idx0y0].x);
          fprintf(files_h->phifile, "\t\t\t%e", outs_h->omega[idx0y0].y);
          
          cudaMemcpy(tmp_d->CXYZ, Tprp[0], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
          mask<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          reality<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          
          cufftExecC2C(ffts->XYplanZ_C2C, tmp_d->CXYZ, tmp_d->CXYZ, CUFFT_INVERSE);
          get_z0<<<dimGrid,dimBlock>>>(outs_d->omega, tmp_d->CXYZ);
          cudaMemcpy(outs_h->omega, outs_d->omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
          fprintf(files_h->phifile, "\t\t\t%e", outs_h->omega[idx0y0].x);
          fprintf(files_h->phifile, "\t\t\t%e", outs_h->omega[idx0y0].y);

          cudaMemcpy(tmp_d->CXYZ, Tpar[0], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
          mask<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          reality<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          
          cufftExecC2C(ffts->XYplanZ_C2C, tmp_d->CXYZ, tmp_d->CXYZ, CUFFT_INVERSE);
          get_z0<<<dimGrid,dimBlock>>>(outs_d->omega, tmp_d->CXYZ);
          cudaMemcpy(outs_h->omega, outs_d->omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
          fprintf(files_h->phifile, "\t\t\t%e", outs_h->omega[idx0y0].x);
          fprintf(files_h->phifile, "\t\t\t%e", outs_h->omega[idx0y0].y);
        }

        if(init==ODD || init==UPAR) {
          cudaMemcpy(tmp_d->CXYZ, Upar[0], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
          mask<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          reality<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          
          cufftExecC2C(ffts->XYplanZ_C2C, tmp_d->CXYZ, tmp_d->CXYZ, CUFFT_INVERSE);
          get_z0<<<dimGrid,dimBlock>>>(outs_d->omega, tmp_d->CXYZ);
          cudaMemcpy(outs_h->omega, outs_d->omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
          fprintf(files_h->phifile, "\n\t%f\t%e", tm_h->runtime, outs_h->omega[idx0y0].x);
          fprintf(files_h->phifile, "\t\t\t%e", outs_h->omega[idx0y0].y);

          cudaMemcpy(tmp_d->CXYZ, Qprp[0], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
          mask<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          reality<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          
          cufftExecC2C(ffts->XYplanZ_C2C, tmp_d->CXYZ, tmp_d->CXYZ, CUFFT_INVERSE);
          get_z0<<<dimGrid,dimBlock>>>(outs_d->omega, tmp_d->CXYZ);
          cudaMemcpy(outs_h->omega, outs_d->omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
          fprintf(files_h->phifile, "\t\t\t%e", outs_h->omega[idx0y0].x);
          fprintf(files_h->phifile, "\t\t\t%e", outs_h->omega[idx0y0].y);

          cudaMemcpy(tmp_d->CXYZ, Qpar[0], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
          mask<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          reality<<<dimGrid,dimBlock>>>(tmp_d->CXYZ);
          
          cufftExecC2C(ffts->XYplanZ_C2C, tmp_d->CXYZ, tmp_d->CXYZ, CUFFT_INVERSE);
          get_z0<<<dimGrid,dimBlock>>>(outs_d->omega, tmp_d->CXYZ);
          cudaMemcpy(outs_h->omega, outs_d->omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
          fprintf(files_h->phifile, "\t\t\t%e", outs_h->omega[idx0y0].x);
          fprintf(files_h->phifile, "\t\t\t%e", outs_h->omega[idx0y0].y);
        }
        //fprintf(files_h->omegafile,"t=%f\n", tm_h->runtime);
        //for(int j=0; j<=Nx; j++) {
        //  for(int i=0; i<=Ny; i++) {
        //    int index = i + Ny*j;
        //    if(j==Nx) index = i;
        //    if(i==Ny) index = Ny*j;
        //    if(i==Ny && j==Nx) index= 0;
        //    fprintf(files_h->omegafile,"%.6f\t%.6f\t%f\n", 2*Y0*(i-Ny/2)/Ny, 2*X0*(j-Nx/2)/Nx, outs_h->omega[index]);
        //  }
        //  fprintf(files_h->omegafile,"\n");
        //}
        //fprintf(files_h->omegafile,"\n\n");
         
        //getModeValReal<<<dimGrid,dimBlock>>>(outs_d->val, Dens[0], pars_h->iky_single, pars_h->ikx_single, Nz/2);
        //cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        //fprintf(files_h->phifile, "\n\t%f\t%e", tm_h->runtime, outs_h->val[0]);
        //getModeValImag<<<dimGrid,dimBlock>>>(outs_d->val, Dens[0], pars_h->iky_single, pars_h->ikx_single, Nz/2);
        //cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        //fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);
        //getModeValReal<<<dimGrid,dimBlock>>>(outs_d->val, Tprp[0], pars_h->iky_single, pars_h->ikx_single, Nz/2);
        //cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        //fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);
        //getModeValImag<<<dimGrid,dimBlock>>>(outs_d->val, Tprp[0], pars_h->iky_single, pars_h->ikx_single, Nz/2);
        //cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        //fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);
      }
 
      if(init == FORCE) {
        field_line_avg<<<dimGrid,dimBlock>>>(tmp_d->CXY, Phi1, jacobian, 1./fluxDen);
        getModeValReal<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\n\t%f\t%e", tm_h->runtime, outs_h->val[0]);
        getModeValImag<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", tm_h->runtime, outs_h->val[0]);

        field_line_avg<<<dimGrid,dimBlock>>>(tmp_d->CXY, Dens[0], jacobian, 1./fluxDen);
        getModeValReal<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);
        getModeValImag<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);

        field_line_avg<<<dimGrid,dimBlock>>>(tmp_d->CXY, Upar[0], jacobian, 1./fluxDen);
        getModeValReal<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);
        getModeValImag<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);

        field_line_avg<<<dimGrid,dimBlock>>>(tmp_d->CXY, Tpar[0], jacobian, 1./fluxDen);
        getModeValReal<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);
        getModeValImag<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);

        field_line_avg<<<dimGrid,dimBlock>>>(tmp_d->CXY, Tprp[0], jacobian, 1./fluxDen);
        getModeValReal<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);
        getModeValImag<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);

        field_line_avg<<<dimGrid,dimBlock>>>(tmp_d->CXY, Qpar[0], jacobian, 1./fluxDen);
        getModeValReal<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);
        getModeValImag<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);

        field_line_avg<<<dimGrid,dimBlock>>>(tmp_d->CXY, Qprp[0], jacobian, 1./fluxDen);
        getModeValReal<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);
        getModeValImag<<<dimGrid,dimBlock>>>(outs_d->val, tmp_d->CXY, pars_h->iky_single, pars_h->ikx_single, 0);
        cudaMemcpy(outs_h->val, outs_d->val, sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(files_h->phifile, "\t\t\t%e", outs_h->val[0]);
      }
    //  cudaEventRecord(start1,0);  
      if(LINEAR || secondary_test || write_omega) {

        growthRate<<<dimGrid,dimBlock>>>(outs_d->omega,Phi1,Phi,tm_h->dt);    
        //cudaMemcpy(omega_h, outs_d->omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);     
        //weighted average of outs_d->omega over 'navg' timesteps
        //boxAvg(omegaAvg, outs_d->omega, omegaBox, dt, dtBox, navg, counter);
        //cudaMemcpy(omegaAvg_h, omegaAvg, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);      
  
        //copy Phi for next timestep
        cudaMemcpy(Phi, Phi1, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
        mask<<<dimGrid,dimBlock>>>(Phi1);
        mask<<<dimGrid,dimBlock>>>(Phi);
        //print growth rates to files   
        //omegaWrite(omegafile,gammafile,omegaAvg_h,runtime); 
        
  
        //if(counter>2*navg) {
  	//omegaStability(omega_h, omegaAvg_h, stability,ctrl->stable,stableMax);
  	//STABLE_STOP = stabilityCheck(ctrl->stable,stableMax);
        //}
      }
      
  
      //Copy Phi to host for writing
      // this is incredibly slow, should never be done in timestep loop.
      if(ev_h->pars.write_phi) {
        cudaMemcpy(ev_h->fields.phi, Phi, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
        //kxkyz0TimeWrite(files_h->phifile, ev_h->fields.phi, tm_h->runtime);
      }
  
  //DIAGNOSTICS
  
      
      /*
      if(counter%nwrite==0) {
        cudaMemcpy(phi_h, Phi, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
      }
      */
      
      for(int s=0; s<nSpecies; s++) {    
        outs_h->hflux_by_species_old[s] = wpfx[s];
        outs_h->pflux_by_species_old[s] = outs_h->pflux_by_species[s];
      }
  
      outs_h->hflux_tot=0.0;
      //calculate instantaneous heat flux
#ifdef PROFILE
PUSH_RANGE("fluxes",5);
#endif
      for(int s=0; s<nSpecies; s++) {  
        fluxes(&outs_h->pflux_by_species[s], &wpfx[s],Dens[s],Tpar[s],Tprp[s],Phi,
               tmp_d->CXYZ,tmp_d->CXYZ,tmp_d->CXYZ,fields_d->field,fields_d->field,fields_d->field,tmp_d->Z,tmp_d->XY, tmp_d->XY2,species[s],tm_h->runtime,
               &outs_h->phases.flux1, &outs_h->phases.flux2, &outs_h->phases.Dens, &outs_h->phases.Tpar, &outs_h->phases.Tprp);        
        outs_h->hflux_tot=outs_h->hflux_tot+wpfx[s];
      }
#ifdef PROFILE
POP_RANGE;
#endif
      volflux<<<dimGrid,dimBlock>>>(outs_d->phi2_by_mode, Phi, Phi, jacobian, 1./fluxDen);
       
      //if(ev_h->pars.write_netcdf) {
      if(write_phi) {
        volflux_zonal(Phi,Phi,tmp_d->X);  //tmp_d->X = Phi_zf**2(kx)
        //get_kx1_rms<<<1,1>>>(&ev_d->nlpm.Phi_zf_kx1, tmp_d->X);
        //nlpm->Phi_zf_kx1_old = nlpm->Phi_zf_kx1;
        //cudaMemcpy(&nlpm->Phi_zf_kx1, &ev_d->nlpm.Phi_zf_kx1, sizeof(float), cudaMemcpyDeviceToHost);
        
        //volflux_zonal(Phi,Phi,tmp_d->X);  //tmp_d->X = Phi_zf**2(kx)
        //nlpm->kx2Phi_zf_rms_old = nlpm->kx2Phi_zf_rms;
        multKx4<<<dimGrid,dimBlock>>>(tmp_d->X2, tmp_d->X, kx); 
        //nlpm->kx2Phi_zf_rms = sumReduc(tmp_d->X2, Nx, false);
        //nlpm->kx2Phi_zf_rms = sqrt(nlpm->kx2Phi_zf_rms);
  
        //volflux_zonal(Phi,Phi,tmp_d->X);  //tmp_d->X = Phi_zf**2(kx)
        outs_h->phi2_zf = sumReduc(tmp_d->X, Nx, tmp_d->X2, tmp_d->X2);
        outs_h->phi2_zf_rms = sqrt(outs_h->phi2_zf);   

        //calculate tmp_d->XY = Phi**2(kx,ky)
        volflux(Phi,Phi,tmp_d->CXYZ,tmp_d->XY);
        //if(!LINEAR && write_phi2kxky_time) {
        //  cudaMemcpy(tmpXY_h, tmp_d->XY, sizeof(float)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
        //  kxkyTimeWrite(phikxkyfile, tmpXY_h, runtime);
        //}
        sumX<<<dimGrid,dimBlock>>>(tmp_d->Y, tmp_d->XY);
        cudaMemcpy(tmp_h->Y, tmp_d->Y, sizeof(float)*(Ny/2+1), cudaMemcpyDeviceToHost);
        cudaMemcpy(outs_h->phi2_by_ky, tmp_d->Y, sizeof(float)*(Ny/2+1), cudaMemcpyDeviceToHost);
        sumY <<< dimGrid, dimBlock >>>(tmp_d->X2, tmp_d->XY);
        cudaMemcpy(outs_h->phi2_by_kx, tmp_d->X2, sizeof(float)*Nx,cudaMemcpyDeviceToHost);

        if(!LINEAR && turn_off_gradients_test) {
          kyTimeWrite(files_h->phifile, tmp_h->Y, tm_h->runtime);
        }
        //calculate <kx> and <ky>
        expect_k<<<dimGrid,dimBlock>>>(tmp_d->XY2, tmp_d->XY, ky);
        outs_h->kphi2 = sumReduc(tmp_d->XY2, Nx*(Ny/2+1), tmp_d->XY3, tmp_d->XY3);
        outs_h->phi2 = sumReduc(tmp_d->XY, Nx*(Ny/2+1), tmp_d->XY3, tmp_d->XY3);

        //outs_h->phi2 = outs->phi2;

        outs_h->expectation_ky = (float) outs_h->phi2/outs_h->kphi2;
  
        expect_k<<<dimGrid,dimBlock>>>(tmp_d->XY2, tmp_d->XY, kx);
        outs_h->kphi2 = sumReduc(tmp_d->XY2, Nx*(Ny/2+1), tmp_d->XY3, tmp_d->XY3);
        outs_h->expectation_kx = (float) outs_h->phi2/outs_h->kphi2;
        
        //calculate z correlation function = tmp_d->YZ (not normalized)
        zCorrelation<<<dimGrid,dimBlock>>>(tmp_d->YZ, Phi);
        //volflux(Phi,Phi,tmp_d->CXYZ,tmp_d->XY);
      }
 
      if(tm_h->counter>0) { 
        //we use an exponential moving average
        // wpfx_avg[t] = alphav*wpfx[t] + (1-alphav)*wpfx_avg[t-1]
        // now with time weighting...
        // wpfx_sum[t] = alphav*dt*wpfx[t] + (1-alphav)*wpfx_avg[t-1]
        // tm_h->dtSum[t] = alphav*tm_h->dt[t] + (1-alphav)*tm_h->dtSum[t-1]
        // wpfx_avg[t] = wpfx_sum[t]/tm_h->dtSum[t]
   
        // keep a running total of dt, phi**2(kx,ky), expectation values, etc.
        tm_h->dtSum = tm_h->dtSum*(1.-alphav) + tm_h->dt*alphav;
        add_scaled<<<dimGrid,dimBlock>>>(outs_d->phi2_by_mode_movav, 1.-alphav, outs_d->phi2_by_mode_movav, tm_h->dt*alphav, outs_d->phi2_by_mode, Nx, Ny, 1);
        if(write_phi) {
        add_scaled<<<dimGrid,dimBlock>>>(outs_d->phi2_zonal_by_kx_movav, 1.-alphav, outs_d->phi2_zonal_by_kx_movav, tm_h->dt*alphav, tmp_d->X, Nx, 1, 1);
        add_scaled<<<dimGrid,dimBlock>>>(outs_d->par_corr_kydz_movav, 1.-alphav, outs_d->par_corr_kydz_movav, tm_h->dt*alphav, tmp_d->YZ, 1, Ny, Nz);
        }
        if(LINEAR || write_omega || secondary_test) {
          if(tm_h->counter>0) {
            add_scaled<<<dimGrid,dimBlock>>>(outs_d->omega_avg, 1.-alphav, outs_d->omega_avg, tm_h->dt*alphav, outs_d->omega, Nx, Ny, 1);
            cudaMemcpy(outs_h->omega_avg, outs_d->omega_avg, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);      
            //print growth rates to files   
            omegaWrite(files_h->omegafile,files_h->gammafile,outs_h->omega_avg,tm_h->dtSum,tm_h->runtime); 
          }
          else {
            tm_h->dtSum = 1.;
            cudaMemcpy(outs_h->omega_avg, outs_d->omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);      
            //print growth rates to files   
            omegaWrite(files_h->omegafile,files_h->gammafile,outs_h->omega_avg,tm_h->runtime); 
          }             

        }  
        if(write_phi) {
        outs_h->expectation_kx_movav = outs_h->expectation_kx_movav*(1.-alphav) + outs_h->expectation_kx*tm_h->dt*alphav;
        outs_h->expectation_ky_movav = outs_h->expectation_ky_movav*(1.-alphav) + outs_h->expectation_ky*tm_h->dt*alphav;
        outs_h->phi2_movav = outs_h->phi2_movav*(1.-alphav) + outs_h->phi2*tm_h->dt*alphav;
        outs_h->phi2_zf_rms_sum = outs_h->phi2_zf_rms_sum*(1.-alphav) + outs_h->phi2_zf_rms*tm_h->dt*alphav;
        //kx2Phi_zf_rms_sum = kx2Phi_zf_rms_sum*(1.-alphav) + nlpm->kx2Phi_zf_rms*tm_h->dt*alphav;
        outs_h->phases.flux1_sum = outs_h->phases.flux1_sum*(1.-alphav) + outs_h->phases.flux1*tm_h->dt*alphav;
        outs_h->phases.flux2_sum = outs_h->phases.flux2_sum*(1.-alphav) + outs_h->phases.flux2*tm_h->dt*alphav;
        outs_h->phases.Dens_sum = outs_h->phases.Dens_sum*(1.-alphav) + outs_h->phases.Dens*tm_h->dt*alphav;
        outs_h->phases.Tpar_sum = outs_h->phases.Tpar_sum*(1.-alphav) + outs_h->phases.Tpar*tm_h->dt*alphav;
        outs_h->phases.Tprp_sum = outs_h->phases.Tprp_sum*(1.-alphav) + outs_h->phases.Tprp*tm_h->dt*alphav;
        //nlpm->D_sum = nlpm->D_sum*(1.-alphav) + nlpm->D*tm_h->dt*alphav;
  
        
        // **_sum/dtSum gives time average of **
        outs_h->phi2_zf_rms_avg = outs_h->phi2_zf_rms_sum/tm_h->dtSum;
        //nlpm->kx2Phi_zf_rms_avg = kx2Phi_zf_rms_sum/dtSum;
        }
 
        for(int s=0; s<nSpecies; s++) {
          if (ev_h->time.counter == 1 && ev_h->pars.zero_restart_avg) wpfxAvg[s] = wpfx[s];  
          //wpfxAvg[s] = muav*wpfxAvg[s] + (1.-muav)*wpfx[s] + (muav - (1.-muav)/alphav)*(wpfx[s] - outs_h->hflux_by_species_old[s]);
          wpfxAvg[s] = muav*wpfxAvg[s] + (1.-muav)*wpfx[s];
          //wpfxAvg[s] = (1.0-muav)*wpfxAvg[s] + muav*wpfx[s];
          pflxAvg[s] = muav*pflxAvg[s] + (1-muav)*outs_h->pflux_by_species[s] + (muav - (1-muav)/alphav)*(outs_h->pflux_by_species[s] - outs_h->pflux_by_species_old[s]);
        }
  
  /*
        // try to autostop when wpfx converges
        // look at min and max of wpfxAvg over time... if wpfxAvg stays within certain bounds for a given amount of
        // time, it is converged
        if(counter >= navg*1.2) {
          //set bounds to be +/- .05*wpfxAvg
  	converge_bounds = .1*wpfxAvg[ION];
  	//if counter reaches navg/3, recenter bounds
  	if(counter == navg || ctrl->converge_count == navg/3) {
  	  wpfxmax = wpfxAvg[ION] + .5*converge_bounds;
  	  wpfxmin = wpfxAvg[ION] - .5*converge_bounds;
  	  ctrl->converge_count++;
  	}
  	//if wpfxAvg goes outside the bounds, reset the bounds.
  	else if(wpfxAvg[ION] > wpfxmax) {
            wpfxmax = wpfxAvg[ION] + .3*converge_bounds;
  	  wpfxmin = wpfxmax - converge_bounds;
  	  ctrl->converge_count=0;
  	}
  	else if(wpfxAvg[ION] < wpfxmin) {
            wpfxmin = wpfxAvg[ION] - .3*converge_bounds;
  	  wpfxmax = wpfxmin + converge_bounds;
  	  ctrl->converge_count=0;
  	}
  	//if wpfxAvg stays inside the bounds, increment the convergence counter.
  	else ctrl->converge_count++; 
        }    
  */
      }
  
      fluxWrite(files_h->fluxfile,outs_h->pflux_by_species, pflxAvg, wpfx,wpfxAvg, nlpm_h->D, nlpm_h->D_avg, nlpm_h->Phi_zf_kx1, nlpm_h->Phi_zf_kx1_avg, nlpm_h->kx2Phi_zf_rms, nlpm_h->kx2Phi_zf_rms_avg, nlpm_h->nu1_max,nlpm_h->nu22_max,ctrl->converge_count,tm_h->runtime,species);
    
  	     
      if(tm_h->counter==0 && pars_h->nlpm_test) phiR_historyWrite(Phi,outs_d->omega,tmp_d->XY_R,tmp_h->XY_R, tm_h->runtime, files_h->omegafile); //save time history of Phi(x,y,z=0)          
      if(tm_h->counter==0 && pars_h->nlpm_test) phiR_historyWrite(Dens[0],outs_d->omega,tmp_d->XY_R,tmp_h->XY_R, tm_h->runtime, files_h->omegafile); //save time history of Phi(x,y,z=0)          
      
      if(tm_h->counter==0 && pars_h->nlpm_test) phiRcomplex_historyWrite(Phi,outs_d->omega,outs_h->omega, tm_h->runtime, files_h->gammafile, ffts->XYplanC2C); //save time history of Phi(x,y,z=0)          
      if(tm_h->counter==0 && pars_h->nlpm_test) phiRcomplex_historyWrite(Dens[0],outs_d->omega,outs_h->omega, tm_h->runtime, files_h->gammafile, ffts->XYplanC2C); //save time history of Phi(x,y,z=0)          
      
      // print wpfx to screen if not printing growth rates
      if(!write_omega && tm_h->counter%nwrite==0) printf("%d: wpfx = %f, dt = %f, dt_cfl =  %f, Dnlpm = %f\n", gpuID, wpfx[0],tm_h->dt, dt_cfl, nlpm_h->D);
      
      // write flux to file
      if(tm_h->counter%nwrite==0) fflush(NULL);
               
      
      
      
      //print growth rates to screen every 10 timesteps if write_omega
      if(write_omega) {
        if (tm_h->counter%10==0 || ctrl->stopcount==ctrl->nstop-1 || tm_h->counter==nSteps-1) {
  	printf("ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<((Nx-1)/3+1); i++) {
  	  for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      printf("%.4f\t%.4f\t\t%.6f\t%.6f", ky_h[j], kx_h[i], outs_h->omega_avg[index].x/tm_h->dtSum, outs_h->omega_avg[index].y/tm_h->dtSum);
        //outs_h->omega[index].x = outs_h->omega_avg[index].x/tm_h->dtSum;
        //outs_h->omega[index].y = outs_h->omega_avg[index].y/tm_h->dtSum;
  	      if(ctrl->stable[index] >= ctrl->stable_max) printf("\tomega");
  	      if(ctrl->stable[index+Nx*(Ny/2+1)] >= ctrl->stable_max) printf("\tgamma");
  	      printf("\n");
  	    }
  	  }
  	  printf("\n");
  	}
  	//for(int i=2*Nx/3+1; i<2*Nx/3+1; i++) {
        for(int i=2*Nx/3+1; i<Nx; i++) {
            for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	    printf("%.4f\t%.4f\t\t%.6f\t%.6f", ky_h[j], kx_h[i], outs_h->omega_avg[index].x/tm_h->dtSum, outs_h->omega_avg[index].y/tm_h->dtSum);
        //outs_h->omega[index].x = outs_h->omega_avg[index].x/tm_h->dtSum;
        //outs_h->omega[index].y = outs_h->omega_avg[index].y/tm_h->dtSum;
  	    if(ctrl->stable[index] >= ctrl->stable_max) printf("\tomega");
  	    if(ctrl->stable[index+Nx*(Ny/2+1)] >= ctrl->stable_max) printf("\tgamma");
  	    printf("\n");
  	  }
  	  printf("\n");
  	}	
        }            
      }
#ifdef PROFILE
POP_RANGE;
#endif
}
