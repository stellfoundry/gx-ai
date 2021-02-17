#pragma once
#include "grids.h"
#include "parameters.h"
#include "geometry.h"
#include "reductions.h"
#include <ncarr.h>

class NetCDF_ids {

 public: 
  NetCDF_ids(Grids* grids, Parameters* pars, Geometry* geo);
  ~NetCDF_ids();

  void close_nc_file();

  void reduce2k    (float * fk, cuComplex * f);

  void write_Wm    (float * G2, bool endrun = false);
  void write_Wl    (float * G2, bool endrun = false);
  void write_Wlm   (float * G2, bool endrun = false);

  void write_Ws    (float * G2, bool endrun = false);
  void write_Wz    (float * G2, bool endrun = false);
  void write_Wky   (float * G2, bool endrun = false);
  void write_Wkx   (float * G2, bool endrun = false);
  void write_Wkxky (float * G2, bool endrun = false);
  
  void write_Ps    (float * P2, bool endrun = false);
  void write_Pz    (float * P2, bool endrun = false);
  void write_Pky   (float * P2, bool endrun = false);
  void write_Pkx   (float * P2, bool endrun = false);  
  void write_Pkxky (float * P2, bool endrun = false);
  
  void write_As    (float * P2, bool endrun = false);
  void write_Az    (float * P2, bool endrun = false);
  void write_Aky   (float * P2, bool endrun = false);
  void write_Akx   (float * P2, bool endrun = false);
  void write_Akxky (float * P2, bool endrun = false);

  void write_Q     (float * Q,   bool endrun = false);
  void write_omg  (cuComplex *W, bool endrun = false);

  void write_nc(int ncid, nca D, const float *data, bool endrun = false);
  void write_nc(int ncid, nca D, const double data, bool endrun = false);

  void write_Wtot();
  
  nca rh, omg, den, wphi, denk, wphik, den0, wphi0, qs; 
  nca Wm, Wl, Wlm;
  nca Pzt, pZt, pzT, Wtot;
  nca Ps, Pky, Pkx, Pkxky, Pz;
  nca Ws, Wky, Wkx, Wkxky, Wz;
  nca As, Aky, Akx, Akxky, Az;
  nca time;
  
  int nx, ny, nz, nkz, kx_dim, ky_dim, kx, ky, kz;
  int m_dim, l_dim, s_dim;
  int theta, theta_x, bmag, bgrad, gbdrift, gbdrift0, periodic;
  int cvdrift, cvdrift0, gds2, gds21, gds22, grho, jacobian;
  int nstep, dt, restart, time_dim, nspec, char16_dim;
  int cfl, init, init_amp, init_single, iky_single, ikx_single, kpar_init;
  int nu_hyper_l, nu_hyper_m, p_hyper_l, p_hyper_m, scheme_opt;
  int closure_model_opt, file, gpu, forcing_index; 
  int Boltzmann_opt, local_limit, linear, forcing, forcing_type, forcing_amp;
  int hypercollisions, snyder_electrons;
  //  int nlpm, no_zonal_nlpm;
  //  int no_nl_flr, no_nl_cross_terms, no_nl_dens_cross_term;  
  int phi_ext, nwrite, navg, nsave, debug, nreal;
  int density, upar, phi, apar, density0, phi0, qflux;
  int write_apar, collisions;
  int source_opt;
  int density_kpar, phi_kpar;
  int eqfix, ikx_fixed, iky_fixed, prim, sec, tert;

  int v_kz[1];           // dims for a scalar as a function of kz 
  int v_kx[1];           // dims for a scalar as a function of kx 
  int v_ky[1];           // dims for a scalar as a function of ky 
  int geo_v_theta[1];    // dims for a scalar as a function of theta
  int zkxky[3];          // dims for a real quantity vs kx, ky, z
  size_t zkxky_start[3], zkxky_count[3];

  size_t geo_start[1], geo_count[1];
  size_t ky_start[1], ky_count[1];
  size_t kx_start[1], kx_count[1];
  size_t kz_start[1], kz_count[1];
  
  float * theta_extended ;

 private:   

  Parameters * pars_   ;
  Grids      * grids_  ;
  Geometry   * geo_    ;
  Red        * red     ;
  Red        * pot     ;
  Red        * ph2     ;
  Red        * all_red ;

  float *Wm_d, *Wm_h, *Wl_d, *Wl_h, *Wlm_d, *Wlm_h;
  float *Wz_d, *Wz_h, *Ws_d, *Ws_h;
  float *Pz_d, *Pz_h, *Ps_d, *Ps_h;
  float *Az_d, *Az_h, *As_d, *As_h;
  float *Wky_d, *Wky_h, *tmp_Wky_h;
  float *Wkx_d, *Wkx_h, *tmp_Wkx_h;
  float *Pky_d, *Pky_h, *tmp_Pky_h;
  float *Pkx_d, *Pkx_h, *tmp_Pkx_h;
  float *Aky_d, *Aky_h, *tmp_Aky_h;
  float *Akx_d, *Akx_h, *tmp_Akx_h;

  float *Wkxky_d, *Wkxky_h, *tmp_Wkxky_h;
  float *Pkxky_d, *Pkxky_h, *tmp_Pkxky_h;
  float *Akxky_d, *Akxky_h, *tmp_Akxky_h;

  float *primary, *secondary, *tertiary;
  
  float *qs_d, *qs_h;
   
  float     * omg_h     ;
  cuComplex * tmp_omg_h ;
  cuComplex * t_bar     ;
  
  float totW; 
};
