#include "netcdf.h"
#include "ncdf.h"

NetCDF_ids::NetCDF_ids(Grids* grids, Parameters* pars, Geometry* geo) :
  grids_(grids), pars_(pars), geo_(geo)
{

  char strb[263];
  strcpy(strb, pars_->run_name); 
  strcat(strb, ".nc");

  int retval, idum;
  
  //  if (retval = nc_open(strb, NC_NETCDF4|NC_WRITE, &file)) ERR(retval);
  file = pars_->ncid;
  if (retval = nc_redef(file));
  
  int ri;
  // Get handles for the dimensions
  if (retval = nc_inq_dimid(file, "ri", &ri))  ERR(retval);
  
  if (retval = nc_def_dim(file, "ky",        grids_->Naky,  &ky_dim))    ERR(retval);
  if (retval = nc_def_dim(file, "kx",        grids_->Nakx,  &kx_dim))    ERR(retval);
  if (retval = nc_def_dim(file, "z",         grids_->Nz,    &nz))        ERR(retval);  

  if (retval = nc_inq_dimid(file, "m",       &m_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "l",       &l_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "s",       &s_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "time",    &time_dim)) ERR(retval);

  //  if (retval = nc_def_var(file, "time",       NC_DOUBLE, 1, &time_dim, &time))    ERR(retval);
  if (retval = nc_def_var(file, "periodic",       NC_INT, 0, 0, &periodic))        ERR(retval);
  if (retval = nc_def_var(file, "local_limit",    NC_INT, 0, 0, &local_limit))     ERR(retval);

  v_ky[0] = ky_dim;
  if (retval = nc_def_var(file, "ky",       NC_FLOAT, 1, v_ky, &ky))              ERR(retval);

  v_kx[0] = kx_dim;
  if (retval = nc_def_var(file, "kx",       NC_FLOAT, 1, v_kx, &kx))              ERR(retval);

  geo_v_theta[0] = nz;
  if (retval = nc_def_var(file, "theta",    NC_FLOAT, 1, geo_v_theta, &theta))    ERR(retval);
  if (retval = nc_def_var(file, "bmag",     NC_FLOAT, 1, geo_v_theta, &bmag))     ERR(retval);
  if (retval = nc_def_var(file, "bgrad",    NC_FLOAT, 1, geo_v_theta, &bgrad))    ERR(retval);
  if (retval = nc_def_var(file, "gbdrift",  NC_FLOAT, 1, geo_v_theta, &gbdrift))  ERR(retval);
  if (retval = nc_def_var(file, "gbdrift0", NC_FLOAT, 1, geo_v_theta, &gbdrift0)) ERR(retval);
  if (retval = nc_def_var(file, "cvdrift",  NC_FLOAT, 1, geo_v_theta, &cvdrift))  ERR(retval);
  if (retval = nc_def_var(file, "cvdrift0", NC_FLOAT, 1, geo_v_theta, &cvdrift0)) ERR(retval);
  if (retval = nc_def_var(file, "gds2",     NC_FLOAT, 1, geo_v_theta, &gds2))     ERR(retval);
  if (retval = nc_def_var(file, "gds21",    NC_FLOAT, 1, geo_v_theta, &gds21))    ERR(retval);
  if (retval = nc_def_var(file, "gds22",    NC_FLOAT, 1, geo_v_theta, &gds22))    ERR(retval);
  if (retval = nc_def_var(file, "grho",     NC_FLOAT, 1, geo_v_theta, &grho))     ERR(retval);
  if (retval = nc_def_var(file, "jacobian", NC_FLOAT, 1, geo_v_theta, &jacobian)) ERR(retval);
  
  //  time.dims[0] = 
  time.time_start[0] = 0;
  time.time_count[0] = 1;
  time.time_dims[0] = time_dim;
  if (retval = nc_def_var(file, "time",     NC_DOUBLE, 1, time.time_dims, &time.time))    ERR(retval);

  //  time_start[0] = 0;
  //  time_count[0] = 1;
  
  ////////////////////////////
  //                        //
  //       DENSITY          //
  //                        //
  ////////////////////////////

  den.write = pars_->write_moms;

  if (den.write) {
    den.dims[0] = s_dim;
    den.dims[1] = nz;
    den.dims[2] = kx_dim;
    den.dims[3] = ky_dim;
    den.dims[4] = ri;

    if (retval = nc_def_var(file, "density",  NC_FLOAT, 5, den.dims, &den.idx )) ERR(retval);
    
    den.start[0] = 0;
    den.start[1] = 0;
    den.start[2] = 0;
    den.start[3] = 0; 
    den.start[4] = 0; 
    
    den.count[0] = grids_->Nspecies;
    den.count[1] = grids_->Nz;
    den.count[2] = grids_->Nakx;
    den.count[3] = grids_->Naky;
    den.count[4] = 2;

    den.ns = grids_->Nspecies;
  }
  
  ////////////////////////////
  //                        //
  //       DENSITY(t=0)     //
  //                        //
  ////////////////////////////

  den0.write = pars_->write_moms;

  if (den0.write) {
    den0.dims[0] = s_dim;
    den0.dims[1] = nz;
    den0.dims[2] = kx_dim;
    den0.dims[3] = ky_dim;
    den0.dims[4] = ri;

    if (retval = nc_def_var(file, "density0",  NC_FLOAT, 5, den0.dims, &den0.idx )) ERR(retval);
    
    den0.start[0] = 0;
    den0.start[1] = 0;
    den0.start[2] = 0;
    den0.start[3] = 0; 
    den0.start[4] = 0; 
    
    den0.count[0] = grids_->Nspecies;
    den0.count[1] = grids_->Nz;
    den0.count[2] = grids_->Nakx;
    den0.count[3] = grids_->Naky;
    den0.count[4] = 2;

    den0.ns = grids_->Nspecies;
  }
  
  ////////////////////////////
  //                        //
  //       Phi              //
  //                        //
  ////////////////////////////

  wphi.write = pars_->write_phi;

  if (wphi.write) {
    wphi.dims[0] = nz;
    wphi.dims[1] = kx_dim;
    wphi.dims[2] = ky_dim;
    wphi.dims[3] = ri;
    
    if (retval = nc_def_var(file, "phi",      NC_FLOAT, 4, wphi.dims, &wphi.idx ))      ERR(retval);

    wphi.start[0] = 0;
    wphi.start[1] = 0;
    wphi.start[2] = 0;
    wphi.start[3] = 0; 
    
    wphi.count[0] = grids_->Nz;
    wphi.count[1] = grids_->Nakx;
    wphi.count[2] = grids_->Naky;
    wphi.count[3] = 2;

    wphi.ns = 1; 
  }

  ////////////////////////////
  //                        //
  //       Phi(t=0)         //
  //                        //
  ////////////////////////////

  wphi0.write = pars_->write_phi;

  if (wphi0.write) {
    wphi0.dims[0] = nz;
    wphi0.dims[1] = kx_dim;
    wphi0.dims[2] = ky_dim;
    wphi0.dims[3] = ri;
    
    if (retval = nc_def_var(file, "phi0",      NC_FLOAT, 4, wphi0.dims, &wphi0.idx ))      ERR(retval);

    wphi0.start[0] = 0;
    wphi0.start[1] = 0;
    wphi0.start[2] = 0;
    wphi0.start[3] = 0; 
    
    wphi0.count[0] = grids_->Nz;
    wphi0.count[1] = grids_->Nakx;
    wphi0.count[2] = grids_->Naky;
    wphi0.count[3] = 2;

    wphi0.ns = 1;
  }

  ////////////////////////////
  //                        //
  //   DENSITY(kpar)        //
  //                        //
  ////////////////////////////

  denk.write = (pars_->write_phi_kpar and pars_->write_moms);

  if (denk.write) {
    denk.dims[0] = s_dim;
    denk.dims[1] = nz;
    denk.dims[2] = kx_dim;
    denk.dims[3] = ky_dim;
    denk.dims[4] = ri;

    if (retval = nc_def_var(file, "density_kpar", NC_FLOAT, 5, denk.dims, &denk.idx)) ERR(retval);    

    denk.start[0] = 0;
    denk.start[1] = 0;
    denk.start[2] = 0;
    denk.start[3] = 0; 
    denk.start[4] = 0; 
    
    denk.count[0] = grids_->Nspecies;
    denk.count[1] = grids_->Nz;
    denk.count[2] = grids_->Nakx;
    denk.count[3] = grids_->Naky;
    denk.count[4] = 2;

    denk.ns = 1;
  }

  ////////////////////////////
  //                        //
  //   Phi(kpar)            //
  //                        //
  ////////////////////////////

  wphik.write = pars_->write_phi_kpar;

  if (wphik.write) {
    wphik.dims[0] = nz;
    wphik.dims[1] = kx_dim;
    wphik.dims[2] = ky_dim;
    wphik.dims[3] = ri;
    
    if (retval = nc_def_var(file, "phi_kpar",    NC_FLOAT, 4, wphik.dims, &wphik.idx))     ERR(retval);

    wphik.start[0] = 0;
    wphik.start[1] = 0;
    wphik.start[2] = 0;
    wphik.start[3] = 0; 
    
    wphik.count[0] = grids_->Nz;
    wphik.count[1] = grids_->Nakx;
    wphik.count[2] = grids_->Naky;
    wphik.count[3] = 2;

    wphi.ns = 1;
  }

  ////////////////////////////
  //                        //
  //   Frequencies          //
  //                        //
  ////////////////////////////

  omg.write_v_time = pars_->write_omega;
  
  if (omg.write_v_time) {
    omg.time_dims[0] = time_dim; 
    omg.time_dims[1] = kx_dim;
    omg.time_dims[2] = ky_dim; 
    omg.time_dims[3] = ri;
    
    if (retval = nc_def_var(file, "omega_v_time", NC_FLOAT, 4, omg.time_dims, &omg.time)) ERR(retval);

    omg.time_start[0] = 1;
    omg.time_start[1] = 0;
    omg.time_start[2] = 0;
    omg.time_start[3] = 0;
    
    omg.time_count[0] = 1;
    omg.time_count[1] = grids_->Nakx;
    omg.time_count[2] = grids_->Naky;
    omg.time_count[3] = 2;
  }

  ////////////////////////////
  //                        //
  // Rosenbluth-Hinton      //
  //                        //
  ////////////////////////////

  rh.write = pars_->write_rh;

  if (rh.write) {
    rh.time_dims[0] = time_dim;
    rh.time_dims[1] = ri;
    
    if (retval = nc_def_var(file, "phi_rh", NC_FLOAT, 2, rh.time_dims, &rh.time)) ERR(retval);

    rh.time_start[0] = 0;
    rh.time_start[1] = 0;

    rh.time_count[0] = 1;
    rh.time_count[1] = 2;
  }

  ////////////////////////////
  //                        //
  //     PZT estimates      //
  //                        //
  ////////////////////////////

  Pzt.write_v_time = pars_->write_pzt;
  
  if (Pzt.write_v_time) {

    Pzt.time_dims[0] = time_dim;
    pZt.time_dims[0] = time_dim;
    pzT.time_dims[0] = time_dim;

    if (retval = nc_def_var(file, "prim", NC_FLOAT, 1, Pzt.time_dims, &Pzt.idx)) ERR(retval);
    if (retval = nc_def_var(file, "sec",  NC_FLOAT, 1, pZt.time_dims, &pZt.idx)) ERR(retval);
    if (retval = nc_def_var(file, "tert", NC_FLOAT, 1, pzT.time_dims, &pzT.idx)) ERR(retval);

    Pzt.time_start[0] = 0;
    Pzt.time_count[0] = 1; 

    pZt.time_start[0] = 0;
    pZt.time_count[0] = 1; 

    pzT.time_start[0] = 0;
    pzT.time_count[0] = 1; 
  }
  
  flx.write_v_time = pars_->write_fluxes;
  
  if (flx.write_v_time) {
    flx.time_dims[0] = time_dim;
    flx.time_dims[1] = s_dim;
    
    if (retval = nc_def_var(file, "qflux", NC_FLOAT, 2, flx.time_dims, &flx.time)) ERR(retval);
    flx.time_start[0] = 0;
    flx.time_start[1] = 0;

    flx.time_count[0] = 1;
    flx.time_count[1] = grids_->Nspecies;
  }

  ////////////////////////////
  //                        //
  //   Free energy          //
  //                        //
  ////////////////////////////

  //
  // To do:
  //
  // Get the n Z**2 / T factors into the Wphi calculation
  // Include the W from the adiabatic species (so for ETG, Phi**2 )
  // Get the species sum of the components
  //

  Wtot.write_v_time = pars_->write_free_energy;

  if (Wtot.write_v_time) {
    Wtot.time_dims[0] = time_dim;

    if (retval = nc_def_var(file, "W", NC_FLOAT, 1, Wtot.time_dims, &Wtot.time)) ERR(retval);

    Wtot.time_start[0] = 0;
    Wtot.time_count[0] = 1;   
  }

  ////////////////////////////
  //                        //
  // (1-G0)phi**2 (species)  //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_species] > 0) Ps.write = true;
  Ps.write_v_time = pars_->write_spec_v_time;

  if (Ps.write) {
    Ps.dims[0] = s_dim;
    
    if (retval = nc_def_var(file, "Ps", NC_FLOAT, 1, Ps.dims, &Ps.idx))  ERR(retval);
    Ps.start[0] = 0;

    Ps.count[0] = grids_->Nspecies;

    if (Ps.write_v_time) {
      Ps.time_dims[0] = time_dim;
      Ps.time_dims[1] = s_dim;
      
      if (retval = nc_def_var(file, "Pst", NC_FLOAT, 2, Ps.time_dims, &Ps.time))  ERR(retval);
      Ps.time_start[0] = 0;
      Ps.time_start[1] = 0;
      
      Ps.time_count[0] = 1;
      Ps.time_count[1] = grids_->Nspecies;
    }
  }
  
  ////////////////////////////
  //                        //
  //   P (ky, species)      //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_ky] > 0) Pky.write = true;
  Pky.write_v_time = pars_->write_spec_v_time;

  if (Pky.write) {
    Pky.dims[0] = s_dim;
    Pky.dims[1] = ky_dim;
    
    if (retval = nc_def_var(file, "Pkys", NC_FLOAT, 2, Pky.dims, &Pky.idx))  ERR(retval);
    Pky.start[0] = 0;
    Pky.start[1] = 0;

    Pky.count[0] = grids_->Nspecies;
    Pky.count[1] = grids_->Naky;

    if (Pky.write_v_time) {
      Pky.time_dims[0] = time_dim;
      Pky.time_dims[1] = s_dim;
      Pky.time_dims[2] = ky_dim;
      
      if (retval = nc_def_var(file, "Pkyst", NC_FLOAT, 3, Pky.time_dims, &Pky.time))  ERR(retval);
      Pky.time_start[0] = 0;
      Pky.time_start[1] = 0;
      Pky.time_start[2] = 0;
      
      Pky.time_count[0] = 1;
      Pky.time_count[1] = grids_->Nspecies;
      Pky.time_count[2] = grids_->Naky;
      
    }
  }
  
  ////////////////////////////
  //                        //
  //   P (kx, species)      //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_kx] > 0) Pkx.write = true;
  Pkx.write_v_time = pars_->write_spec_v_time;

  if (Pkx.write) {
    Pkx.dims[0] = s_dim;
    Pkx.dims[1] = kx_dim;
    
    if (retval = nc_def_var(file, "Pkxs", NC_FLOAT, 2, Pkx.dims, &Pkx.idx))  ERR(retval);
    Pkx.start[0] = 0;
    Pkx.start[1] = 0;

    Pkx.count[0] = grids_->Nspecies;
    Pkx.count[1] = grids_->Nakx;

    if (Pkx.write_v_time) {
      Pkx.time_dims[0] = time_dim;
      Pkx.time_dims[1] = s_dim;
      Pkx.time_dims[2] = kx_dim;
      
      if (retval = nc_def_var(file, "Pkxst", NC_FLOAT, 3, Pkx.time_dims, &Pkx.time))  ERR(retval);
      Pkx.time_start[0] = 0;
      Pkx.time_start[1] = 0;
      Pkx.time_start[2] = 0;
      
      Pkx.time_count[0] = 1;
      Pkx.time_count[1] = grids_->Nspecies;
      Pkx.time_count[2] = grids_->Nakx;      
    }
  }
  
  ////////////////////////////
  //                        //
  //   P (z, species)      //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_z] > 0) Pz.write = true;
  Pz.write_v_time = pars_->write_spec_v_time;

  if (Pz.write) {
    Pz.dims[0] = s_dim;
    Pz.dims[1] = nz;
    
    if (retval = nc_def_var(file, "Pzs", NC_FLOAT, 2, Pz.dims, &Pz.idx))  ERR(retval);
    Pz.start[0] = 0;
    Pz.start[1] = 0;

    Pz.count[0] = grids_->Nspecies;
    Pz.count[1] = grids_->Nz;

    if (Pz.write_v_time) {
      Pz.time_dims[0] = time_dim;
      Pz.time_dims[1] = s_dim;
      Pz.time_dims[2] = nz;
      
      if (retval = nc_def_var(file, "Pzst", NC_FLOAT, 3, Pz.time_dims, &Pz.time))  ERR(retval);
      Pz.time_start[0] = 0;
      Pz.time_start[1] = 0;
      Pz.time_start[2] = 0;
      
      Pz.time_count[0] = 1;
      Pz.time_count[1] = grids_->Nspecies;
      Pz.time_count[2] = grids_->Nz;
    }
  }
  
  ////////////////////////////
  //                        //
  //   P (kx,ky,  species)  //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_kxky] > 0) Pkxky.write = true;
  Pkxky.write_v_time = pars_->write_spec_v_time;

  if (Pkxky.write) {
    Pkxky.dims[0] = s_dim;
    Pkxky.dims[1] = kx_dim;
    Pkxky.dims[2] = ky_dim;
    
    if (retval = nc_def_var(file, "Pkxkys", NC_FLOAT, 3, Pkxky.dims, &Pkxky.idx))  ERR(retval);
    Pkxky.start[0] = 0;
    Pkxky.start[1] = 0;
    Pkxky.start[2] = 0;

    Pkxky.count[0] = grids_->Nspecies;
    Pkxky.count[1] = grids_->Nakx;
    Pkxky.count[2] = grids_->Naky;

    if (Pkxky.write_v_time) {
      Pkxky.time_dims[0] = time_dim;
      Pkxky.time_dims[1] = s_dim;
      Pkxky.time_dims[2] = kx_dim;
      Pkxky.time_dims[3] = ky_dim;
      
      if (retval = nc_def_var(file, "Pkxkyst", NC_FLOAT, 4, Pkxky.time_dims, &Pkxky.time))  ERR(retval);
      Pkxky.time_start[0] = 0;
      Pkxky.time_start[1] = 0;
      Pkxky.time_start[2] = 0;
      Pkxky.time_start[3] = 0;
      
      Pkxky.time_count[0] = 1;
      Pkxky.time_count[1] = grids_->Nspecies;
      Pkxky.time_count[2] = grids_->Nakx;
      Pkxky.time_count[3] = grids_->Naky;
    }
  }   

  ////////////////////////////
  //                        //
  //   W (species)          //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_species] > 0) Ws.write = true;
  Ws.write_v_time = pars_->write_spec_v_time;

  if (Ws.write) {
    Ws.dims[0] = s_dim;
    
    if (retval = nc_def_var(file, "Ws", NC_FLOAT, 1, Ws.dims, &Ws.idx))  ERR(retval);
    Ws.start[0] = 0;

    Ws.count[0] = grids_->Nspecies;

    if (Ws.write_v_time) {
      Ws.time_dims[0] = time_dim;
      Ws.time_dims[1] = s_dim;
      
      if (retval = nc_def_var(file, "Wst", NC_FLOAT, 2, Ws.time_dims, &Ws.time))  ERR(retval);
      Ws.time_start[0] = 0;
      Ws.time_start[1] = 0;
      
      Ws.time_count[0] = 1;
      Ws.time_count[1] = grids_->Nspecies;
      
    }
  }
  
  ////////////////////////////
  //                        //
  //   W (ky, species)      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_ky] > 0) Wky.write = true;
  Wky.write_v_time = pars_->write_spec_v_time;

  if (Wky.write) {
    Wky.dims[0] = s_dim;
    Wky.dims[1] = ky_dim;
    
    if (retval = nc_def_var(file, "Wkys", NC_FLOAT, 2, Wky.dims, &Wky.idx))  ERR(retval);
    Wky.start[0] = 0;
    Wky.start[1] = 0;

    Wky.count[0] = grids_->Nspecies;
    Wky.count[1] = grids_->Naky;

    if (Wky.write_v_time) {
      Wky.time_dims[0] = time_dim;
      Wky.time_dims[1] = s_dim;
      Wky.time_dims[2] = ky_dim;
      
      if (retval = nc_def_var(file, "Wkyst", NC_FLOAT, 3, Wky.time_dims, &Wky.time))  ERR(retval);
      Wky.time_start[0] = 0;
      Wky.time_start[1] = 0;
      Wky.time_start[2] = 0;
      
      Wky.time_count[0] = 1;
      Wky.time_count[1] = grids_->Nspecies;
      Wky.time_count[2] = grids_->Naky;
      
    }
  }   
  
  ////////////////////////////
  //                        //
  //   W (kx, species)      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_kx] > 0) Wkx.write = true;
  Wkx.write_v_time = pars_->write_spec_v_time;

  if (Wkx.write) {
    Wkx.dims[0] = s_dim;
    Wkx.dims[1] = kx_dim;
    
    if (retval = nc_def_var(file, "Wkxs", NC_FLOAT, 2, Wkx.dims, &Wkx.idx))  ERR(retval);
    Wkx.start[0] = 0;
    Wkx.start[1] = 0;

    Wkx.count[0] = grids_->Nspecies;
    Wkx.count[1] = grids_->Nakx;

    if (Wkx.write_v_time) {
      Wkx.time_dims[0] = time_dim;
      Wkx.time_dims[1] = s_dim;
      Wkx.time_dims[2] = kx_dim;
      
      if (retval = nc_def_var(file, "Wkxst", NC_FLOAT, 3, Wkx.time_dims, &Wkx.time))  ERR(retval);
      Wkx.time_start[0] = 0;
      Wkx.time_start[1] = 0;
      Wkx.time_start[2] = 0;
      
      Wkx.time_count[0] = 1;
      Wkx.time_count[1] = grids_->Nspecies;
      Wkx.time_count[2] = grids_->Nakx;
      
    }
  }
  
  ////////////////////////////
  //                        //
  //   W (kx,ky,  species)  //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_kxky] > 0) Wkxky.write = true;
  Wkxky.write_v_time = pars_->write_spec_v_time;

  if (Wkxky.write) {
    Wkxky.dims[0] = s_dim;
    Wkxky.dims[1] = kx_dim;
    Wkxky.dims[2] = ky_dim;
    
    if (retval = nc_def_var(file, "Wkxkys", NC_FLOAT, 3, Wkxky.dims, &Wkxky.idx))  ERR(retval);
    Wkxky.start[0] = 0;
    Wkxky.start[1] = 0;
    Wkxky.start[2] = 0;

    Wkxky.count[0] = grids_->Nspecies;
    Wkxky.count[1] = grids_->Nakx;
    Wkxky.count[2] = grids_->Naky;

    if (Wkxky.write_v_time) {
      Wkxky.time_dims[0] = time_dim;
      Wkxky.time_dims[1] = s_dim;
      Wkxky.time_dims[2] = kx_dim;
      Wkxky.time_dims[3] = ky_dim;
      
      if (retval = nc_def_var(file, "Wkxkyst", NC_FLOAT, 4, Wkxky.time_dims, &Wkxky.time))  ERR(retval);
      Wkxky.time_start[0] = 0;
      Wkxky.time_start[1] = 0;
      Wkxky.time_start[2] = 0;
      Wkxky.time_start[3] = 0;
      
      Wkxky.time_count[0] = 1;
      Wkxky.time_count[1] = grids_->Nspecies;
      Wkxky.time_count[2] = grids_->Nakx;
      Wkxky.time_count[3] = grids_->Naky;      
    }
  }   

  ////////////////////////////
  //                        //
  //   W (z, species)       //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_z] > 0) Wz.write = true;
  Wz.write_v_time = pars_->write_spec_v_time;

  if (Wz.write) {
    Wz.dims[0] = s_dim;
    Wz.dims[1] = nz;
    
    if (retval = nc_def_var(file, "Wzs", NC_FLOAT, 2, Wz.dims, &Wz.idx))  ERR(retval);
    Wz.start[0] = 0;
    Wz.start[1] = 0;

    Wz.count[0] = grids_->Nspecies;
    Wz.count[1] = grids_->Nz;

    if (Wz.write_v_time) {
      Wz.time_dims[0] = time_dim;
      Wz.time_dims[1] = s_dim;
      Wz.time_dims[2] = nz;
      
      if (retval = nc_def_var(file, "Wzst", NC_FLOAT, 3, Wz.time_dims, &Wz.time))  ERR(retval);
      Wz.time_start[0] = 0;
      Wz.time_start[1] = 0;
      Wz.time_start[2] = 0;
      
      Wz.time_count[0] = 1;
      Wz.time_count[1] = grids_->Nspecies;
      Wz.time_count[2] = grids_->Nz;
      
    }
  }   
  
  ////////////////////////////
  //                        //
  // Lag-Herm spectrum      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_lm] > 0) Wlm.write = true;
  Wlm.write_v_time = pars_->write_spec_v_time;
  
  if (Wlm.write) {
    Wlm.dims[0] = s_dim;
    Wlm.dims[1] = m_dim;
    Wlm.dims[2] = l_dim;
    
    if (retval = nc_def_var(file, "Wlms", NC_FLOAT, 3, Wlm.dims, &Wlm.idx))  ERR(retval);
    Wlm.start[0] = 0;
    Wlm.start[1] = 0;
    Wlm.start[2] = 0;

    Wlm.count[0] = grids_->Nspecies;
    Wlm.count[1] = grids_->Nm;
    Wlm.count[2] = grids_->Nl;

    if (Wlm.write_v_time) {
      Wlm.time_dims[0] = time_dim;
      Wlm.time_dims[1] = s_dim;
      Wlm.time_dims[2] = m_dim;
      Wlm.time_dims[3] = l_dim;
      
      if (retval = nc_def_var(file, "Wlmst", NC_FLOAT, 4, Wlm.time_dims, &Wlm.time))  ERR(retval);
      Wlm.time_start[0] = 0;
      Wlm.time_start[1] = 0;
      Wlm.time_start[2] = 0;
      Wlm.time_start[3] = 0;
      
      Wlm.time_count[0] = 1;
      Wlm.time_count[1] = grids_->Nspecies;
      Wlm.time_count[2] = grids_->Nm;
      Wlm.time_count[3] = grids_->Nl;      
    }
  }
    
  ////////////////////////////
  //                        //
  // Laguerre spectrum      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_l] > 0) Wl.write = true;
  Wl.write_v_time = pars_->write_spec_v_time;
  
  if (Wl.write) {
    Wl.dims[0] = s_dim;
    Wl.dims[1] = l_dim;
    
    if (retval = nc_def_var(file, "Wls", NC_FLOAT, 2, Wl.dims, &Wl.idx))  ERR(retval);
    Wl.start[0] = 0;
    Wl.start[1] = 0;

    Wl.count[0] = grids_->Nspecies;    
    Wl.count[1] = grids_->Nl;    

    if (Wl.write_v_time) {
      Wl.time_dims[0] = time_dim;
      Wl.time_dims[1] = s_dim;
      Wl.time_dims[2] = l_dim;

      if (retval = nc_def_var(file, "Wlst", NC_FLOAT, 3, Wl.time_dims, &Wl.time))  ERR(retval);
      Wl.time_start[0] = 0;
      Wl.time_start[1] = 0;
      Wl.time_start[2] = 0;

      Wl.time_count[0] = 1;
      Wl.time_count[1] = grids_->Nspecies;
      Wl.time_count[2] = grids_->Nl;
    }
  }
  
  ////////////////////////////
  //                        //
  //  Hermite spectrum      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_m] > 0) Wm.write = true;
  Wm.write_v_time = pars_->write_spec_v_time;
  
  if (Wm.write) {
    Wm.dims[0] = s_dim;
    Wm.dims[1] = m_dim;

    if (retval = nc_def_var(file, "Wms", NC_FLOAT, 2, Wm.dims, &Wm.idx))  ERR(retval);
    Wm.start[0] = 0; 
    Wm.start[1] = 0; 

    Wm.count[0] = grids_->Nspecies;    
    Wm.count[1] = grids_->Nm;    

    if (Wm.write_v_time) {
      Wm.time_dims[0] = time_dim;
      Wm.time_dims[1] = s_dim;
      Wm.time_dims[2] = m_dim;
      
      if (retval = nc_def_var(file, "Wmst", NC_FLOAT, 3, Wm.time_dims, &Wm.time))  ERR(retval);
      Wm.time_start[0] = 0;
      Wm.time_start[1] = 0;
      Wm.time_start[2] = 0;

      Wm.time_count[0] = 1;    
      Wm.time_count[1] = grids_->Nspecies;    
      Wm.time_count[2] = grids_->Nm;          
    }
  }

  bool linked = (not pars_->local_limit && not pars_->boundary_option_periodic);

  if (linked) {
    zkxky[0] = nz;
    zkxky[1] = kx_dim; 
    zkxky[2] = ky_dim;
    
    if (retval = nc_def_var(file, "theta_x",  NC_FLOAT, 3, zkxky, &theta_x))  ERR(retval);
  }

  DEBUGPRINT("ncdf:  ending definition mode for NetCDF \n");
  
  if (retval = nc_enddef(file)) ERR(retval);
  
  ///////////////////////////////////
  //                               //
  //        ky                     //
  //                               //
  ///////////////////////////////////
  ky_start[0] = 0;
  ky_count[0] = grids_->Naky;

  if (retval = nc_put_vara(file, ky, ky_start, ky_count, grids_->ky_h))         ERR(retval);

  ///////////////////////////////////
  //                               //
  //        kx                     //
  //                               //
  ///////////////////////////////////
  kx_start[0] = 0;
  kx_count[0] = grids_->Nakx;

  if (retval = nc_put_vara(file, kx, kx_start, kx_count, grids_->kx_h))         ERR(retval);

  ///////////////////////////////////
  //                               //
  //  geometric information        //
  //                               //
  ///////////////////////////////////
  geo_start[0] = 0;
  geo_count[0] = grids_->Nz;
  
  if (retval = nc_put_vara(file, theta,    geo_start, geo_count, geo_->z_h))         ERR(retval);

  if (linked) {
    
    int Nx = grids_->Nx;
    int Ny = grids_->Ny;
    int Nz = grids_->Nz;
    
    int Naky = grids_->Naky;
    
    zkxky_count[0] = Nz;
    zkxky_count[1] = 1;
    zkxky_count[2] = 1;
    
    size_t size = sizeof(float)*Nz;
    cudaMallocHost((void**) &theta_extended, size);
    float th0;
    for (int i=0; i<(Nx-1)/3+1; i++) {
      for (int j=0; j<(Ny-1)/3+1; j++) {
	if (j==0) {th0 = 0.;} else {th0 = grids_->kx_h[i]/(grids_->ky_h[j]*pars_->shat);}
	for (int k=0; k<Nz; k++) {
	  theta_extended[k] = geo_->z_h[k] - th0;
	}
	zkxky_start[0] = 0;
	zkxky_start[1] = i;
	zkxky_start[2] = j;
	if (retval = nc_put_vara(file, theta_x, zkxky_start, zkxky_count, theta_extended)) ERR(retval);
      }
    }
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<Naky; j++) {
	if (j==0) {th0 = 0.;} else {th0 = grids_->kx_h[i]/(grids_->ky_h[j]*pars_->shat);}
	for (int k=0; k<Nz; k++) {
	  theta_extended[k] = geo_->z_h[k] - th0;
	}
	zkxky_start[0] = 0;
	zkxky_start[1] = i-2*Nx/3+(Nx-1)/3;
	zkxky_start[2] = j;
	if (retval = nc_put_vara(file, theta_x, zkxky_start, zkxky_count, theta_extended)) ERR(retval);
      }
    }
    if (theta_extended) cudaFreeHost(theta_extended);
  }

  if (retval = nc_put_vara(file, theta,    geo_start, geo_count, geo_->z_h))         ERR(retval);
  if (retval = nc_put_vara(file, bmag,     geo_start, geo_count, geo_->bmag_h))      ERR(retval);
  if (retval = nc_put_vara(file, bgrad,    geo_start, geo_count, geo_->bgrad_h))     ERR(retval);
  if (retval = nc_put_vara(file, gbdrift,  geo_start, geo_count, geo_->gbdrift_h))   ERR(retval);
  if (retval = nc_put_vara(file, gbdrift0, geo_start, geo_count, geo_->gbdrift0_h))  ERR(retval);
  if (retval = nc_put_vara(file, cvdrift,  geo_start, geo_count, geo_->cvdrift_h))   ERR(retval);
  if (retval = nc_put_vara(file, cvdrift0, geo_start, geo_count, geo_->cvdrift0_h))  ERR(retval);
  if (retval = nc_put_vara(file, gds2,     geo_start, geo_count, geo_->gds2_h))      ERR(retval);
  if (retval = nc_put_vara(file, gds21,    geo_start, geo_count, geo_->gds21_h))     ERR(retval);  
  if (retval = nc_put_vara(file, gds22,    geo_start, geo_count, geo_->gds22_h))     ERR(retval);
  if (retval = nc_put_vara(file, grho,     geo_start, geo_count, geo_->grho_h))      ERR(retval);
  if (retval = nc_put_vara(file, jacobian, geo_start, geo_count, geo_->jacobian_h))  ERR(retval);

  idum = pars_->boundary_option_periodic ? 1 : 0;
  if (retval = nc_put_var(file, periodic,      &idum))                   ERR(retval);

  idum = pars_->local_limit ? 1 : 0;
  if (retval = nc_put_var(file, local_limit,   &pars_->local_limit))     ERR(retval);
}

NetCDF_ids::~NetCDF_ids() {
}

void NetCDF_ids::close_nc_file() {
  int retval;
  if (retval = nc_close(file)) ERR(retval);
}


