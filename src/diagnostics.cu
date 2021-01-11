#include "diagnostics.h"
#include "device_funcs.h"
#include "cuda_constants.h"
#include "get_error.h"
#include "gx_lib.h"
#include "netcdf.h"
#include <sys/stat.h>

Diagnostics::Diagnostics(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{  
  id = new NetCDF_ids(grids_, pars_, geo_); cudaDeviceSynchronize(); CUDA_DEBUG("NetCDF_ids: %s \n");
  fields_old = new Fields(pars_, grids_);   cudaDeviceSynchronize(); CUDA_DEBUG("Fields: %s \n");

  int nS  = grids_->Nspecies;
  int nM  = grids_->Nm;
  int nL  = grids_->Nl;
  int nY  = grids_->Nyc;
  int nYk = grids_->Naky;
  int nX  = grids_->Nx;
  int nXk = grids_->Nakx;
  int nZ  = grids_->Nz;
  int nR  = nX * nY * nZ;
  int nK  = nXk * nYk * nZ;
  int nG  = nR * grids_->Nmoms * nS;
  
  if (id->wphik.write || id->denk.write) {
    // Only appropriate for unsheared configurations ... otherwise this diagnostic has a bug
    grad_parallel = new GradParallelPeriodic(grids_); cudaDeviceSynchronize();  CUDA_DEBUG("Grad parallel periodic: %s \n");
  }
  if (pars_->diagnosing_spectra) {    
    float dum = 1.0;
    red = new Red(grids_, pars_->wspectra);       cudaDeviceSynchronize(); CUDA_DEBUG("Reductions: %s \n");
    pot = new Red(grids_, pars_->pspectra, true); cudaDeviceSynchronize(); CUDA_DEBUG("Reductions: %s \n");
    ph2 = new Red(grids_, pars_->aspectra,  dum); cudaDeviceSynchronize(); CUDA_DEBUG("Reductions: %s \n");

    volDenom = 0.;  for (int i=0; i<nZ; i++)   volDenom += geo_->jacobian_h[i]; 
    volDenomInv = 1./volDenom;   

    if (id->Wm.write) {
      cudaMalloc     (&Wm_d,        sizeof(float) * nM * nS);
      cudaMallocHost (&Wm_h,        sizeof(float) * nM * nS);
    }
    if (id->Wl.write) {
      cudaMalloc     (&Wl_d,        sizeof(float) * nL * nS); 
      cudaMallocHost (&Wl_h,        sizeof(float) * nL * nS); 
    }
    if (id->Wlm.write) {
      cudaMalloc     (&Wlm_d,       sizeof(float) * nL * nM * nS);
      cudaMallocHost (&Wlm_h,       sizeof(float) * nL * nM * nS);
    }
    if (id->Wky.write) {
      cudaMalloc     (&Wky_d,       sizeof(float) * nY * nS); 
      cudaMallocHost (&tmp_Wky_h,   sizeof(float) * nY * nS); 
      cudaMallocHost (&Wky_h,       sizeof(float) * nYk * nS);
    }
    if (id->Wkx.write) {
      cudaMalloc     (&Wkx_d,       sizeof(float) * nX * nS);
      cudaMallocHost (&tmp_Wkx_h,   sizeof(float) * nX * nS);
      cudaMallocHost (&Wkx_h,       sizeof(float) * nXk * nS);
    }
    if (id->Wkxky.write) {
      cudaMalloc     (&Wkxky_d,     sizeof(float) * nX * nY * nS);
      cudaMallocHost (&tmp_Wkxky_h, sizeof(float) * nX * nY * nS);
      cudaMallocHost (&Wkxky_h,     sizeof(float) * nXk * nYk * nS);
    }
    if (id->Wz.write) {
      cudaMalloc     (&Wz_d,        sizeof(float) * nZ * nS);
      cudaMallocHost (&Wz_h,        sizeof(float) * nZ * nS);
    }
    if (id->Ws.write) {
      cudaMalloc     (&Ws_d,        sizeof(float) * nS);
      cudaMallocHost (&Ws_h,        sizeof(float) * nS);
    }
    if (id->Pky.write) {
      cudaMalloc     (&Pky_d,       sizeof(float) * nY * nS);
      cudaMallocHost (&tmp_Pky_h,   sizeof(float) * nY * nS);
      cudaMallocHost (&Pky_h,       sizeof(float) * nYk * nS);
    }
    if (id->Pz.write) {
      cudaMalloc     (&Pz_d,        sizeof(float) * nZ * nS); 
      cudaMallocHost (&Pz_h,        sizeof(float) * nZ * nS); 
    }
    if (id->Pkx.write) {
      cudaMalloc     (&Pkx_d,       sizeof(float) * nX * nS); 
      cudaMallocHost (&tmp_Pkx_h,   sizeof(float) * nX * nS); 
      cudaMallocHost (&Pkx_h,       sizeof(float) * nXk * nS); 
    }
    if (id->Pkxky.write) {
      cudaMalloc     (&Pkxky_d,     sizeof(float) * nX * nY * nS);
      cudaMallocHost (&tmp_Pkxky_h, sizeof(float) * nX * nY * nS);
      cudaMallocHost (&Pkxky_h,     sizeof(float) * nXk * nYk * nS);
    }
    if (id->Ps.write) {
      cudaMalloc     (&Ps_d,        sizeof(float) * nS);
      cudaMallocHost (&Ps_h,        sizeof(float) * nS);
    }    
    if (id->Aky.write) {
      cudaMalloc     (&Aky_d,       sizeof(float) * nY); 
      cudaMallocHost (&tmp_Aky_h,   sizeof(float) * nY); 
      cudaMallocHost (&Aky_h,       sizeof(float) * nYk);
    }
    if (id->Akx.write) {
      cudaMalloc     (&Akx_d,       sizeof(float) * nX);
      cudaMallocHost (&tmp_Akx_h,   sizeof(float) * nX);
      cudaMallocHost (&Akx_h,       sizeof(float) * nXk);
    }
    if (id->Akxky.write) {
      cudaMalloc     (&Akxky_d,     sizeof(float) * nX * nY);
      cudaMallocHost (&tmp_Akxky_h, sizeof(float) * nX * nY);
      cudaMallocHost (&Akxky_h,     sizeof(float) * nXk * nYk);
    }
    if (id->Az.write) {
      cudaMalloc     (&Az_d,        sizeof(float) * nZ);
      cudaMallocHost (&Az_h,        sizeof(float) * nZ);
    }
    if (id->As.write) {
      cudaMalloc     (&As_d,        sizeof(float));
      cudaMallocHost (&As_h,        sizeof(float));
    }
    cudaMalloc (&G2, sizeof(float) * nG); // G2 is the only big array associated with these diagnostics.
  }
  cudaMalloc (&P2s, sizeof(float) * nR * nS);
  if (!pars_->all_kinetic) {
    if (pars_->iphi00 == 1) {
      cudaMalloc (&Phi2, sizeof(float) * nR);  
    }
  }
  
  if (id->omg.write_v_time) {
    cudaMalloc     (&omg_d,       sizeof(cuComplex) * nX * nY);
    cudaMallocHost (&tmp_omg_h,   sizeof(cuComplex) * nX * nY); 
    cudaMallocHost (&omg_h,       sizeof(cuComplex) * nXk * nYk);

    for (int i=0; i < nXk * nYk * 2; i++) omg_h[i] = 0.;
    cudaMemset (omg_d, 0., sizeof(cuComplex) * nX * nY);
  }  
  
  if (pars_->diagnosing_moments) {    
    cudaMallocHost (&amom_h,     sizeof(cuComplex) * nK * nS);
    cudaMallocHost (&tmp_amom_h, sizeof(cuComplex) * nR);
    cudaMalloc     (&amom_d,     sizeof(cuComplex) * nR);
  }
  if (pars_->diagnosing_pzt) {
    cudaMallocHost (&primary,   sizeof(float));    primary[0] = 0.;  
    cudaMallocHost (&secondary, sizeof(float));    secondary[0] = 0.;
    cudaMallocHost (&tertiary,  sizeof(float));    tertiary[0] = 0.;
    cudaMalloc     (&t_bar,     sizeof(cuComplex) * nR * nS);
  }
  if (id->flx.write_v_time) {
    cudaMallocHost(&pflux,  sizeof(float) * nS);
    cudaMallocHost(&qflux,  sizeof(float) * nS);
    cudaMalloc    (&qs_d,   sizeof(float) * nS);
    all_red = new Red(nR, nS);  cudaDeviceSynchronize();  CUDA_DEBUG("Reductions: %s \n");
    
    fluxDenom = 0.;
    for(int i=0; i<grids_->Nz; i++)   fluxDenom += geo_->jacobian_h[i]*geo_->grho_h[i];
    fluxDenomInv = 1./fluxDenom;
  }
  cudaMallocHost(&val, sizeof(float)*2);
    
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  maxThreadsPerBlock_ = prop.maxThreadsPerBlock;

  // note that delta theta is a constant in this formalism!   

  // set up stop file
  sprintf(stopfilename_, "%s.stop", pars_->run_name);

  dB_scale = 512;
  dG_scale = nR/dB_scale.x+1;
  
  //  dB_spec = dim3(32, 1, 1);
  dB_spec = dim3(16, 8, 8);
  dG_spec = dim3(nY/dB_spec.x+1, nX/dB_spec.y+1, nZ/dB_spec.z+1);  

  int nyx =  nY * nX;
  int nslm = nL * nM * nS;

  int nbx = 32;
  int ngx = (nyx-1)/nbx + 1;

  int nby = 32;
  int ngy = (grids_->Nz-1)/nby + 1;
  
  dB_all = dim3(nbx, nby, 1);
  dG_all = dim3(ngx, ngy, nslm);

}

Diagnostics::~Diagnostics()
{
  if (fields_old) delete fields_old;
  if (red)        delete red;
  if (pot)        delete pot;
  if (ph2)        delete ph2;
  if (all_red)    delete all_red;
  if (id)         delete id;

  if (G2)                 cudaFree(G2);
  if (P2s)                cudaFree(P2s);
  
  if (pflux)              cudaFreeHost(pflux);
  if (qflux)              cudaFreeHost(qflux);
  if (qs_d)               cudaFree(qs_d);
  
  if (amom_h)             cudaFreeHost(amom_h);
  if (tmp_amom_h)         cudaFreeHost(tmp_amom_h);
  if (amom_d)             cudaFree(amom_d);
  if (t_bar)              cudaFree(t_bar);
  
  if (primary)            cudaFreeHost(primary);
  if (secondary)          cudaFreeHost(secondary);
  if (tertiary)           cudaFreeHost(tertiary);

  if (val)                cudaFreeHost(val);

  if (omg_d)              cudaFree    (omg_d);
  if (tmp_omg_h)          cudaFreeHost(tmp_omg_h);
  if (omg_h)              cudaFreeHost(omg_h);

  if (Wm_d)               cudaFree    (Wm_d);
  if (Wm_h)               cudaFreeHost(Wm_h);

  if (Wl_d)               cudaFree    (Wl_d);
  if (Wl_h)               cudaFreeHost(Wl_h);

  if (Wlm_d)              cudaFree    (Wlm_d);
  if (Wlm_h)              cudaFreeHost(Wlm_h);

  if (Ws_d)               cudaFree    (Ws_d);
  if (Ws_h)               cudaFreeHost(Ws_h);

  if (Wz_d)               cudaFree    (Wz_d);
  if (Wz_h)               cudaFreeHost(Wz_h);

  if (Wky_d)              cudaFree    (Wky_d);
  if (tmp_Wky_h)          cudaFreeHost(tmp_Wky_h);
  if (Wky_h)              cudaFreeHost(Wky_h);

  if (Wkx_d)              cudaFree    (Wkx_d);
  if (tmp_Wkx_h)          cudaFreeHost(tmp_Wkx_h);
  if (Wkx_h)              cudaFreeHost(Wkx_h);

  if (Wkxky_d)            cudaFree    (Wkxky_d);
  if (Wkxky_h)            cudaFreeHost(Wkxky_h);
  if (tmp_Wkxky_h)        cudaFreeHost(tmp_Wkxky_h);

  if (Ps_d)               cudaFree    (Ps_d);
  if (Ps_h)               cudaFreeHost(Ps_h);

  if (Pz_d)               cudaFree    (Pz_d);
  if (Pz_h)               cudaFreeHost(Pz_h);

  if (Pky_d)              cudaFree    (Pky_d);
  if (tmp_Pky_h)          cudaFreeHost(tmp_Pky_h);
  if (Pky_h)              cudaFreeHost(Pky_h);

  if (Pkx_d)              cudaFree    (Pkx_d);
  if (tmp_Pkx_h)          cudaFreeHost(tmp_Pkx_h);
  if (Pkx_h)              cudaFreeHost(Pkx_h);

  if (Pkxky_d)            cudaFree    (Pkxky_d);
  if (tmp_Pkxky_h)        cudaFreeHost(tmp_Pkxky_h);
  if (Pkxky_h)            cudaFreeHost(Pkxky_h);

  if (As_d)               cudaFree    (As_d);
  if (As_h)               cudaFreeHost(As_h);

  if (Az_d)               cudaFree    (Az_d);
  if (Az_h)               cudaFreeHost(Az_h);

  if (Aky_d)              cudaFree    (Aky_d);
  if (tmp_Aky_h)          cudaFreeHost(tmp_Aky_h);
  if (Aky_h)              cudaFreeHost(Aky_h);

  if (Akx_d)              cudaFree    (Akx_d);
  if (tmp_Akx_h)          cudaFreeHost(tmp_Akx_h);
  if (Akx_h)              cudaFreeHost(Akx_h);

  if (Akxky_d)            cudaFree    (Akxky_d);
  if (tmp_Akxky_h)        cudaFreeHost(tmp_Akxky_h);
  if (Akxky_h)            cudaFreeHost(Akxky_h);
}

bool Diagnostics::loop(MomentsG* G, Fields* fields, double dt, int counter, double time) 
{
  int retval;
  bool stop = false;
  int nw;

  nw = pars_->nwrite;

  if(counter%nw == 0) {
    fflush(NULL);

    write_nc (id->file, id->time, time, false);
    id->time.increment_ts();
  }

  if(id->omg.write_v_time) {
    if (counter%nw == nw-1) fields_old->copyPhiFrom(fields);

    if (counter%nw == 0 && counter > 0) {
      freqs(fields, fields_old, dt);
      write_omg(omg_d, false);
    }
  }
  
  if (counter%nw == 0 && pars_->write_spec_v_time && pars_->diagnosing_spectra) {
    W_summand <<<dG_all, dB_all>>> (G2, G->G(), geo_->jacobian, volDenomInv, pars_->species_h);

    for (int is=0; is < grids_->Nspecies; is++) {
      Wphi_summand <<<dG_spec, dB_spec>>> (P2(is), fields->phi,
					   geo_->jacobian, volDenomInv, geo_->kperp2,
					   pars_->species_h[is].rho2);
      
      Wphi_scale <<<dG_spec, dB_spec>>> (P2(is), pars_->species_h[is].qneut);
    }
    if (!pars_->all_kinetic) {
      if (pars_->iphi00==1) {
	Wphi2_summand <<< dG_spec, dB_spec>>> (Phi2, fields->phi, geo_->jacobian, volDenomInv);
	float fac = 1./pars_->ti_ov_te;
	Wphi_scale <<<dG_spec, dB_spec>>> (Phi2, fac);	// assume hydrogenic ion species only
      }
    }
        
    if (id->Wm.write)    write_Wm (G2, false);
    if (id->Wl.write)    write_Wl (G2, false);
    if (id->Wlm.write)   write_Wlm(G2, false);

    if (id->Wz.write)    write_Wz(G2, false);
    if (id->Wky.write)   write_Wky(G2, false);
    if (id->Wkx.write)   write_Wkx(G2, false);
    if (id->Wkxky.write) write_Wkxky(G2, false);

    if (id->Pz.write)    write_Pz(P2(), false);
    if (id->Pky.write)   write_Pky(P2(), false);
    if (id->Pkx.write)   write_Pkx(P2(), false);
    if (id->Pkxky.write) write_Pkxky(P2(), false);

    if (id->Az.write)    write_Az(Phi2, false);
    if (id->Aky.write)   write_Aky(Phi2, false);
    if (id->Akx.write)   write_Akx(Phi2, false);
    if (id->Akxky.write) write_Akxky(Phi2, false);

    // Do not move these next two calls
    if (id->Ps.write)    write_Ps(P2(), false);
    if (id->Ws.write)    write_Ws(G2, false);

    if (id->As.write_v_time) write_As(Phi2, false);
    
    if (id->Wtot.write_v_time) {
      totW = 0.;
      for (int is=0; is<grids_->Nspecies; is++) totW += Ws_h[is] + Ps_h[is];

      if (!pars_->all_kinetic && (pars_->iphi00==1)) totW += As_h[0];
      write_Wtot(totW, false);
    }
  }
  
  // Rosenbluth-Hinton diagnostics
  if(counter%nw == 0 && id->rh.write) {
    get_rh(fields);
    write_nc (id->file, id->rh, val, false);
    id->rh.increment_ts();
  }

  if( counter%nw == 0 && id->Pzt.write_v_time) {
    pzt(G, fields);  // calculate each of P, Z, and T (very very rough diagnostic)
    cudaDeviceSynchronize();

    write_nc (id->file, id->Pzt, primary, false);
    id->Pzt.increment_ts();
    
    write_nc (id->file, id->pZt, secondary, false);
    id->pZt.increment_ts();
    
    write_nc (id->file, id->pzT, tertiary, false);
    id->pZt.increment_ts();
  }

  if (counter%nw == 0) {
    if (id->flx.write_v_time) {
      printf("Step %d: Time = %f \t Flux = ", counter, time);
      fluxes(G, fields, false);
    } else {
      printf("Step %d: Time = %f\n", counter, time);
    }    
    // sync the netcdf data 
    if (retval = nc_sync(id->file)) ERR(retval);
  }
  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

void Diagnostics::finish(MomentsG* G, Fields* fields) 
{

  if (pars_->diagnosing_moments)  {

    if (id->den.write)   writeMomOrField (G->dens_ptr[0], id->den);
    if (id->wphi.write)  writeMomOrField (fields->phi, id->wphi);
    
    if (id->wphik.write) {
      grad_parallel->fft_only(fields->phi, amom_d, CUFFT_FORWARD);
      writeMomOrField (amom_d, id->wphik);
    }
    
    if (id->denk.write) {
      grad_parallel->fft_only(G->dens_ptr[0], amom_d, CUFFT_FORWARD);
      writeMomOrField (amom_d, id->denk); 
    }
  }
  if (pars_->diagnosing_spectra) {

    W_summand <<<dG_all, dB_all>>> (G2, G->G(), geo_->jacobian, volDenomInv, pars_->species_h);

    for (int is=0; is < grids_->Nspecies; is++) {
      Wphi_summand <<<dG_spec, dB_spec>>> (P2(is), fields->phi,
					   geo_->jacobian, volDenomInv, geo_->kperp2,
					   pars_->species_h[is].rho2);
      Wphi_scale <<<dG_spec, dB_spec>>> (P2(is), pars_->species_h[is].qneut);
    }
    
    if (id->Wm.write   )  write_Wm  (G2, true);
    if (id->Wl.write   )  write_Wl  (G2, true);
    if (id->Wlm.write  )  write_Wlm (G2, true);
    
    if (id->Ps.write   )   write_Ps (P2(), true);
    if (id->Wz.write   )   write_Wz (G2, true);
    if (id->Wky.write  )  write_Wky (G2, true);
    if (id->Wkx.write  )  write_Wkx (G2, true);
    if (id->Wkxky.write) write_Wkxky(G2, true);
    
    if (id->Pz.write   )   write_Pz (P2(), true);
    if (id->Pky.write  )   write_Pky(P2(), true);
    if (id->Pkx.write  )   write_Pkx(P2(), true);
    if (id->Pkxky.write) write_Pkxky(P2(), true);
  }
  id->close_nc_file();  fflush(NULL);
}

void Diagnostics::write_init(MomentsG* G, Fields* fields) {
  if (id->den.write)  writeMomOrField (G->dens_ptr[0], id->den0); 
  if (id->wphi.write) writeMomOrField (fields->phi,    id->wphi0);  
}

void Diagnostics::writeMomOrField(cuComplex* m, nca lid) { 
  int retval;
  int i = grids_->NxNycNz;
  int j = i * grids_->Nmoms;
  int k = 2 * grids_->Nakx*grids_->Naky*grids_->Nz*grids_->Nmoms;
  
  for (int is=0; is<lid.ns; is++) {
    CP_TO_CPU(tmp_amom_h, &m[is*j], sizeof(cuComplex)*i);
    reduce2z(&amom_h[is*k], tmp_amom_h);
  }

  if (retval = nc_put_vara(id->file, lid.idx, lid.start, lid.count, amom_h)) ERR(retval);
}

void Diagnostics::write_nc(int ncid, nca D, const float *data, const bool endrun) {
  int retval;

  if (endrun) {
    if (retval=nc_put_vara(ncid, D.idx, D.start, D.count, data)) ERR(retval);
  } else {   
    if (retval=nc_put_vara(ncid, D.time, D.time_start, D.time_count, data)) ERR(retval);
  }
}

void Diagnostics::write_nc(int ncid, nca D, const double data, const bool endrun) {
  int retval;

  if (endrun) {
    if (retval=nc_put_vara(ncid, D.idx, D.start, D.count, &data)) ERR(retval);
  } else {   
    if (retval=nc_put_vara(ncid, D.time, D.time_start, D.time_count, &data)) ERR(retval);
  }
}

void Diagnostics::write_Wlm(float* G2, bool endrun)
{
  int i = grids_->Nmoms*grids_->Nspecies;

  red->Sum(G2, Wlm_d, WSPECTRA_lm);                CP_TO_CPU(Wlm_h, Wlm_d, sizeof(float)*i);
  write_nc(id->file, id->Wlm, Wlm_h, endrun);      id->Wlm.increment_ts();
}

void Diagnostics::write_Wl(float* G2, bool endrun)
{
  int i = grids_->Nl*grids_->Nspecies;

  red->Sum(G2, Wl_d, WSPECTRA_l);                  CP_TO_CPU(Wl_h, Wl_d, sizeof(float)*i);
  write_nc(id->file, id->Wl, Wl_h, endrun);        id->Wl.increment_ts();
}

void Diagnostics::write_Wm(float* G2, bool endrun)
{
  int i = grids_->Nm*grids_->Nspecies;

  red-> Sum(G2, Wm_d, WSPECTRA_m);                 CP_TO_CPU(Wm_h, Wm_d, sizeof(float)*i);
  write_nc(id->file, id->Wm,  Wm_h, endrun);       id->Wm.increment_ts();
}

void Diagnostics::write_Ps(float* P2, bool endrun)
{
  pot->pSum(P2, Ps_d, PSPECTRA_species);             CP_TO_CPU(Ps_h, Ps_d, sizeof(float)*grids_->Nspecies);
  write_nc(id->file, id->Ps, Ps_h, endrun);          id->Ps.increment_ts();
}

void Diagnostics::write_Pky(float* P2, bool endrun)
{
  int i = grids_->Nyc*grids_->Nspecies;
  
  pot->pSum(P2, Pky_d, PSPECTRA_ky);               CP_TO_CPU(tmp_Pky_h, Pky_d, sizeof(float)*i);

  for (int is = 0; is < grids_->Nspecies; is++) {
    for (int ik = 0; ik < grids_->Naky; ik++) {
      Pky_h[ik + is*grids_->Naky] = tmp_Pky_h[ik + is*grids_->Nyc];
    }
  }
  write_nc(id->file, id->Pky, Pky_h, endrun);      id->Pky.increment_ts();  
}

void Diagnostics::write_Pz(float* P2, bool endrun)
{
  int i = grids_->Nz*grids_->Nspecies;

  pot->pSum(P2, Pz_d, PSPECTRA_z);                 CP_TO_CPU(Pz_h, Pz_d, sizeof(float)*i);
  write_nc(id->file, id->Pz, Pz_h, endrun);        id->Pz.increment_ts();
}

void Diagnostics::write_Pkx(float* P2, bool endrun)
{
  int i = grids_->Nx*grids_->Nspecies;             int NK = (grids_->Nx-1)/3+1;

  pot->pSum(G2, Pkx_d, PSPECTRA_kx);               CP_TO_CPU(tmp_Pkx_h, Pkx_d, sizeof(float)*i);

  for (int is = 0; is < grids_->Nspecies; is++) {
    for (int it = 0; it < NK; it++) {
      Pkx_h[it + is*grids_->Nakx] = tmp_Pkx_h[it + is*grids_->Nx];
    }

    for (int it = 2*grids_->Nx/3+1; it<grids_->Nx; it++) {
      int ith = it - 2*grids_->Nx/3 + (grids_->Nx-1)/3;
      Pkx_h[ith + is*grids_->Nakx] = tmp_Pkx_h[it + is*grids_->Nx];
    }
  }  
  write_nc(id->file, id->Pkx, Pkx_h, endrun);      id->Pkx.increment_ts();  
}

void Diagnostics::write_Pkxky(float* P2, bool endrun)
{
  int i = grids_->Nyc*grids_->Nx*grids_->Nspecies; int NK = (grids_->Nx-1)/3+1;
  pot->pSum(P2, Pkxky_d, PSPECTRA_kxky);
  CP_TO_CPU(tmp_Pkxky_h, Pkxky_d, sizeof(float)*i);
  for (int is = 0; is < grids_->Nspecies; is++) {
    for (int it = 0; it < NK; it++) {
      for (int ik = 0; ik < grids_->Naky; ik++) {
	int Q = ik + it*grids_->Naky + is*grids_->Naky*grids_->Nakx;
	int R = ik + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
	Pkxky_h[Q] = tmp_Pkxky_h[R];
      }
    }
    for (int it = 2*grids_->Nx/3+1; it<grids_->Nx; it++) {
      for (int ik = 0; ik < grids_->Naky; ik++) {     
	int ith = it - 2*grids_->Nx/3 + (grids_->Nx-1)/3;
	int Q = ik + ith*grids_->Naky + is*grids_->Naky*grids_->Nakx;
	int R = ik + it *grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
	Pkxky_h[Q] = tmp_Pkxky_h[R];
      }
    }
  }
  write_nc(id->file, id->Pkxky, Pkxky_h, endrun);  id->Pkxky.increment_ts();  
}

void Diagnostics::write_Wky(float* G2, bool endrun)
{
  int i = grids_->Nyc*grids_->Nspecies;

  red->Sum(G2, Wky_d, WSPECTRA_ky);                CP_TO_CPU(tmp_Wky_h, Wky_d, sizeof(float)*i);

  for (int is = 0; is < grids_->Nspecies; is++) {
    for (int ik = 0; ik < grids_->Naky; ik++) {
      Wky_h[ik + is*grids_->Naky] = tmp_Wky_h[ik + is*grids_->Nyc];
    }
  }
  write_nc(id->file, id->Wky, Wky_h, endrun);      id->Wky.increment_ts();
}

void Diagnostics::write_Wkx(float* G2, bool endrun)
{
  int i = grids_->Nx*grids_->Nspecies;             int NK = (grids_->Nx-1)/3+1;

  red->Sum(G2, Wkx_d, WSPECTRA_kx);                CP_TO_CPU(tmp_Wkx_h, Wkx_d, sizeof(float)*i);

  for (int is = 0; is < grids_->Nspecies; is++) {
    for (int it = 0; it < NK; it++) {
      Wkx_h[it + is*grids_->Nakx] = tmp_Wkx_h[it + is*grids_->Nx];
    }
    for (int it = 2*grids_->Nx/3+1; it<grids_->Nx; it++) {
      int ith = it - 2*grids_->Nx/3 + (grids_->Nx-1)/3;
      Wkx_h[ith + is*grids_->Nakx] = tmp_Wkx_h[it + is*grids_->Nx];
    }
  }
  write_nc(id->file, id->Wkx, Wkx_h, endrun);      id->Wkx.increment_ts();  
}

void Diagnostics::write_Wkxky(float* G2, bool endrun)
{
  int i = grids_->Nyc*grids_->Nx*grids_->Nspecies; int NK = (grids_->Nx-1)/3+1;
  
  red->Sum(G2, Wkxky_d, WSPECTRA_kxky);            CP_TO_CPU(tmp_Wkxky_h, Wkxky_d, sizeof(float)*i);

  for (int is = 0; is < grids_->Nspecies; is++) {
    for (int it = 0; it < NK; it++) {
      for (int ik = 0; ik < grids_->Naky; ik++) {
	int Q = ik + it*grids_->Naky + is*grids_->Naky*grids_->Nakx;
	int R = ik + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
	Wkxky_h[Q] = tmp_Wkxky_h[R];
      }
    }
    for (int it = 2*grids_->Nx/3+1; it<grids_->Nx; it++) {
      for (int ik = 0; ik < grids_->Naky; ik++) {     
	int ith = it - 2*grids_->Nx/3 + (grids_->Nx-1)/3;
	int Q = ik + ith*grids_->Naky + is*grids_->Naky*grids_->Nakx;
	int R = ik + it *grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
	Wkxky_h[Q] = tmp_Wkxky_h[R];
      }
    }
  }
  write_nc(id->file, id->Wkxky, Wkxky_h, endrun);  id->Wkxky.increment_ts();  
}

void Diagnostics::write_Wz(float* G2, bool endrun)
{
  int i = grids_->Nz*grids_->Nspecies;
  
  red->Sum(G2, Wz_d, WSPECTRA_z);                  CP_TO_CPU(Wz_h, Wz_d, sizeof(float)*i);
  write_nc(id->file, id->Wz, Wz_h, endrun);        id->Wz.increment_ts();
}

void Diagnostics::write_Wtot(float Wh, bool endrun)
{
  write_nc(id->file, id->Wtot, &Wh, endrun);
  id->Wtot.increment_ts();
}
void Diagnostics::write_Ws(float* G2, bool endrun)
{
  red->Sum(G2, Ws_d, WSPECTRA_species);            CP_TO_CPU(Ws_h, Ws_d, sizeof(float)*grids_->Nspecies);
  write_nc(id->file, id->Ws, Ws_h, endrun);        id->Ws.increment_ts();
}

void Diagnostics::write_omg(cuComplex *W, bool endrun)
{
  CP_TO_CPU (tmp_omg_h, W, sizeof(cuComplex)*grids_->NxNyc);
  print_growth_rates_to_screen(tmp_omg_h);

  reduce2k(omg_h, tmp_omg_h);
  write_nc(id->file, id->omg, omg_h, endrun);
  id->omg.increment_ts();
}

void Diagnostics::write_Q (float* Q, bool endrun)
{
  all_red->sSum(Q, qs_d);                           CP_TO_CPU (qflux, qs_d, sizeof(float)*grids_->Nspecies);
  write_nc(id->file, id->flx, qflux, endrun);       id->flx.increment_ts();
  for (int is=0; is<grids_->Nspecies; is++) printf ("%e \t ",qflux[is]);
  printf("\n");
}

void Diagnostics::write_As(float *P2, bool endrun)
{
  printf("write_As \n");
  ph2 ->iSum(Phi2, As_d, ASPECTRA_species);         CP_TO_CPU (As_h, As_d, sizeof(float));
  write_nc(id->file, id->As, As_h, endrun);         id->As.increment_ts();
}

void Diagnostics::write_Aky(float* P2, bool endrun)
{
  int i = grids_->Naky;
  
  printf("write_Aky \n");
  pot->iSum(P2, Aky_d, ASPECTRA_ky);               CP_TO_CPU(Aky_h, Aky_d, sizeof(float)*i);
  write_nc(id->file, id->Aky, Aky_h, endrun);      id->Aky.increment_ts();  
}

void Diagnostics::write_Az(float* P2, bool endrun)
{
  int i = grids_->Nz;

  printf("write_Az \n");
  pot->iSum(P2, Az_d, ASPECTRA_z);                 CP_TO_CPU(Az_h, Az_d, sizeof(float)*i);
  write_nc(id->file, id->Az, Az_h, endrun);        id->Az.increment_ts();
}

void Diagnostics::write_Akx(float* P2, bool endrun)
{
  int i = grids_->Nx;                              int NK = (grids_->Nx-1)/3+1;

  printf("write_Akx \n");
  pot->iSum(G2, Akx_d, ASPECTRA_kx);               CP_TO_CPU(tmp_Akx_h, Akx_d, sizeof(float)*i);

  for (int it = 0; it < NK; it++) Akx_h[it] = tmp_Akx_h[it];;

  for (int it = 2*grids_->Nx/3+1; it<grids_->Nx; it++) {
    int ith = it - 2*grids_->Nx/3 + (grids_->Nx-1)/3;
    Akx_h[ith] = tmp_Akx_h[it];
  }
  write_nc(id->file, id->Akx, Akx_h, endrun);      id->Akx.increment_ts();  
}

void Diagnostics::write_Akxky(float* P2, bool endrun)
{
  int i = grids_->Nyc*grids_->Nx; int NK = (grids_->Nx-1)/3+1;
  printf("write_Akxky \n");
  pot->iSum(P2, Akxky_d, ASPECTRA_kxky);
  CP_TO_CPU(tmp_Akxky_h, Akxky_d, sizeof(float)*i);

  for (int it = 0; it < NK; it++) {
    for (int ik = 0; ik < grids_->Naky; ik++) {
      int Q = ik + it*grids_->Naky;
      int R = ik + it*grids_->Nyc ;
      Akxky_h[Q] = tmp_Akxky_h[R];
    }
  }
  for (int it = 2*grids_->Nx/3+1; it<grids_->Nx; it++) {
    for (int ik = 0; ik < grids_->Naky; ik++) {     
      int ith = it - 2*grids_->Nx/3 + (grids_->Nx-1)/3;
      int Q = ik + ith*grids_->Naky;
      int R = ik + it *grids_->Nyc;
      Akxky_h[Q] = tmp_Akxky_h[R];
    }
  }
  write_nc(id->file, id->Akxky, Akxky_h, endrun);  id->Akxky.increment_ts();  
}

  // For each kx, z, l and m, sum the moments of G**2 + Phi**2 (1-Gamma_0) with weights:
  //
  // The full sum is 
  //
  // sum_s sum_l,m sum_ky sum_kx sum_z J(z) == SUM
  //
  // and there is a denominator for the z-integration, to normalize out the volume,
  // SUM => SUM / sum J(z)
  // 
  // W = SUM n_s tau_s G**2 / 2 + SUM n_s Z_s**2 / (2 tau_s) (1 - Gamma_0) Phi**2
  //
  // Since we store only half of the spatial Fourier coefficients for all but the
  // ky = 0 modes
  //
  // (for which we have the full set of kx modes, kx > 0 and kx < 0, and everything is zero for kx = ky = 0)  
  //
  // we multiply all the ky != 0 modes by a factor of two. This is easiest to see in device_functions/vol_summand.
  //
  // To make later manipulations easiest, all partial summations will be carried such that a simple sum of
  // whatever is leftover will give W.
  //

void Diagnostics::get_rh(Fields* f)
{
    ikx_local = 1; iky_local = 0; iz_local=grids_->Nz/2; // correct values for usual RH tests

    int idx = iky_local + grids_->Nyc*ikx_local + grids_->NxNyc*iz_local;
  
    CP_TO_CPU(&valphi, &f->phi[idx], sizeof(cuComplex));
    val[0] = valphi.x;
    val[1] = valphi.y;
}

void Diagnostics::pzt(MomentsG* G, Fields* f)
{
  int threads=256;
  int blocks=(grids_->NxNycNz+threads-1)/threads;
  
  primary[0]=0.; secondary[0]=0.; tertiary[0]=0.;
  
  Tbar <<<blocks, threads>>> (t_bar, G->G(), f->phi, geo_->kperp2);
  get_pzt <<<blocks, threads>>> (&primary[0], &secondary[0], &tertiary[0], f->phi, t_bar);
}

void Diagnostics::freqs (Fields* f, Fields* f_old, double dt)
{
  growthRates <<< grids_->NxNyc/maxThreadsPerBlock_+1, maxThreadsPerBlock_>>>
    (f->phi, f_old->phi, dt, omg_d);
}
void Diagnostics::fluxes(MomentsG* G, Fields* f, bool endrun)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    heat_flux_summand <<<dG_spec, dB_spec>>> (P2(is), f->phi, G->G(0,0,is), grids_->ky,
    					      geo_->jacobian, fluxDenomInv, geo_->kperp2, pars_->species_h[is].rho2);
  }
  write_Q(P2(), endrun); 
}

bool Diagnostics::checkstop() 
{
  struct stat buffer;   
  bool stop = (stat (stopfilename_, &buffer) == 0);
  if (stop) remove(stopfilename_);
  return stop;
}

// condense a (ky,kx) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
void Diagnostics::reduce2k(float *fk, cuComplex* f) {
  
  int Nx = grids_->Nx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  for(int i=0; i<((Nx-1)/3+1); i++) {
    for(int j=0; j<Naky; j++) {
      int index     = j + Nyc *i; 
      int index_out = j + Naky*i; 
      fk[2*index_out]   = f[index].x;
      fk[2*index_out+1] = f[index].y;
    }
  }
  
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<Naky; j++) {
      int index = j + Nyc *i;
      int index_out = j + Naky*( i - 2*Nx/3 + (Nx-1)/3 );
      fk[2*index_out]   = f[index].x;
      fk[2*index_out+1] = f[index].y;
    }
  }	  
}

// condense a (ky,kx,z) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
void Diagnostics::reduce2z(float *fz, cuComplex* f) {

  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;

  for (int k=0; k<Nz; k++) {
    for (int i=0; i<(Nx-1)/3+1; i++) {
      for (int j=0; j<Naky; j++) {
	int index     = j + Nyc *i + Nyc*Nx*k; 
	int index_out = j + Naky*i + Naky*Nakx*k; 
	fz[2*index_out]   = f[index].x;
	fz[2*index_out+1] = f[index].y;
      }
    }
  }
  
  for (int k=0; k<Nz; k++) {
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<Naky; j++) {
	int index     = j + Nyc *i + Nyc*Nx*k;
	int index_out = j + Naky*( i - 2*Nx/3 + (Nx-1)/3 ) + Naky*Nakx*k;
	fz[2*index_out]   = f[index].x;
	fz[2*index_out+1] = f[index].y;
      }
    }	
  }	
}

void Diagnostics::print_growth_rates_to_screen(cuComplex* w)
{
  int Nx = grids_->Nx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  printf("ky\tkx\t\tomega\t\tgamma\n");

  for(int j=0; j<Naky; j++) {
    for(int i=2*Nx/3+1; i<Nx; i++) {
      int index = j + Nyc*i;
      printf("%.4f\t%.4f\t\t%.6f\t%.6f",
	     grids_->ky_h[j], grids_->kx_outh[i], w[index].x, w[index].y);
      printf("\n");
    }
    for(int i=0; i<((Nx-1)/3+1); i++) {
      int index = j + Nyc*i;
      if(index!=0) {
	printf("%.4f\t%.4f\t\t%.6f\t%.6f",
	       grids_->ky_h[j], grids_->kx_outh[i], w[index].x, w[index].y);
	printf("\n");
      } else {
	printf("%.4f\t%.4f\n", grids_->ky_h[j], grids_->kx_outh[i]);
      }
    }
    if (Nx>1) printf("\n");
  }
}


