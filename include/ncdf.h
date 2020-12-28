#pragma once
#include "grids.h"
#include "parameters.h"
#include "geometry.h"
#include <ncarr.h>

class NetCDF_ids {

 public: 
  NetCDF_ids(Grids* grids, Parameters* pars, Geometry* geo);
  ~NetCDF_ids();

  void close_nc_file();

  nca rh, omg, den, wphi, denk, wphik, den0, wphi0, flx; 
  nca Wm, Wl, Wlm;
  nca Pzt, pZt, pzT, Wtot;
  nca Ps, Pky, Pkx, Pkxky, Pz;
  nca Ws, Wky, Wkx, Wkxky, Wz;
  nca time;
  
  int nx, ny, nz, kx_dim, ky_dim, kx, ky;
  int m_dim, l_dim, s_dim;
  int theta, theta_x, bmag, bgrad, gbdrift, gbdrift0, periodic;
  int cvdrift, cvdrift0, gds2, gds21, gds22, grho, jacobian;
  int nstep, dt, restart, time_dim, nspec, char16_dim;
  int cfl, init, init_amp, init_single, iky_single, ikx_single, kpar_init;
  int nu_hyper_l, nu_hyper_m, p_hyper_l, p_hyper_m, scheme_opt;
  int closure_model_opt, file, gpu, forcing_index; 
  int iphi00, local_limit, linear, forcing, forcing_type, forcing_amp;
  int hypercollisions, snyder_electrons, nlpm, no_zonal_nlpm;
  int no_nl_flr, no_nl_cross_terms, no_nl_dens_cross_term;  
  int phi_ext, nwrite, navg, nsave, debug, nreal;
  int density, upar, phi, apar, density0, phi0, qflux;
  int write_apar, collisions;
  int source_opt;
  int write_spec_v_time, density_kpar, phi_kpar;
  int eqfix, ikx_fixed, iky_fixed, prim, sec, tert;
  int write_W, W_phi, W_phi_t;

  //  int write_omega, omega_t, gamma_t;
  
  //  char closure_model[32], scheme[32], source[32];
  
  int v_kx[1];           // dims for a real scalar as a function of kx 
  int v_ky[1];           // dims for a real scalar as a function of ky 
  int geo_v_theta[1];    // dims for a real scalar as a function of theta
  int zkxky[3];          // dims for a real quantity vs kx, ky, z
  size_t zkxky_start[3], zkxky_count[3];
  //  size_t time_start[1], time_count[1];

  //  size_t pzt_start[1], pzt_count[1];
  
  // BD eventually needs [grids_->Nspecies] instead of [1]
  //  size_t flux_start[1], flux_count[1];

  size_t W_phi_start[1], W_phi_count[1];
  size_t W_phi_t_start[2], W_phi_t_count[2];
  
  //  size_t om_start[3],  om_count[3];
  //  size_t omt_start[4], omt_count[4];

  //  size_t mom_start[4], mom_count[4];
  size_t geo_start[1], geo_count[1];
  size_t ky_start[1], ky_count[1];
  size_t kx_start[1], kx_count[1];
  
  float *theta_extended;

 private:   
  const Parameters* pars_;
  const Grids* grids_;
  const Geometry* geo_;
};
