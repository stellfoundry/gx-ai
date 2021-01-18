#pragma once

#include <species.h>
#include <gx_lib.h>
#include <cufft.h>
#include <string>
#include <vector>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define PHI 0                                                                            
#define DENS 1
#define FORCE 2 
#define RH_equilibrium 3
#define TPRP 4
#define UPAR 5
#define TPAR 6
#define QPAR 7
#define QPRP 8
#define PPRP 10
#define PPAR 11
#define ODD 9
#define SSPX2 1
#define SSPX3 2
#define RK2 3
#define RK4 5
#define K10 6
#define BEER42 1
#define SMITHPERP 2
#define SMITHPAR 3
#define PHIEXT 1

#define WSPECTRA_species 0
#define WSPECTRA_kx 1
#define WSPECTRA_ky 2
#define WSPECTRA_z  3
#define WSPECTRA_l  4
#define WSPECTRA_m  5
#define WSPECTRA_lm 6
#define WSPECTRA_kperp 7
#define WSPECTRA_kxky 8

#define PSPECTRA_species 0
#define PSPECTRA_kx 1
#define PSPECTRA_ky 2
#define PSPECTRA_kperp 3
#define PSPECTRA_kxky 4
#define PSPECTRA_z 5

#define ASPECTRA_species 0
#define ASPECTRA_kx 1
#define ASPECTRA_ky 2
#define ASPECTRA_kperp 3
#define ASPECTRA_kxky 4
#define ASPECTRA_z 5

#define BOLTZMANN_IONS 1
#define BOLTZMANN_ELECTRONS 2

class Parameters {

 public:
  Parameters(void);
  ~Parameters(void);
  
  const int nw_spectra = 9;
  const int np_spectra = 6;
  const int na_spectra = 6;
  void get_nml_vars(char* file);
  int set_externalpars(external_parameters_struct* externalpars);
  int import_externalpars(external_parameters_struct* externalpars);
  void init_species(specie* species);

  int ncid;
  int p_hyper_l, p_hyper_m, irho, nwrite, navg, nsave, igeo, nreal;
  int nz_in, nperiod, Zp, bishop, scan_number, iproc, icovering;
  int nx_in, ny_in, jtwist, nm_in, nl_in, nstep, nspec_in, nspec;
  int closure_model_opt, forcing_index, smith_par_q, smith_perp_q;
  int equilibrium_type, source_option, inlpm, p_hyper, iphi00;
  int dorland_phase_ifac, ivarenna, iflr, i_share, stirf;
  int init, iky_single, ikx_single, iky_fixed, ikx_fixed;
  int Boltzmann_opt; 
  //  int lh_ikx, lh_iky;
  int zonal_dens_switch, q0_dens_switch, scheme_opt;
  // formerly part of time struct
  int trinity_timestep, trinity_iteration, trinity_conv_count, end_time;   
  
  float rhoc, eps, shat, qsf, rmaj, r_geo, shift, akappa, akappri;
  float tri, tripri, drhodpsi, epsl, kxfac, cfl, phi_ext, scale, tau_fac;
  float ti_ov_te, beta, g_exb, s_hat_input, beta_prime_input, init_amp;
  float x0, y0, dt, fphi, fapar, fbpar, kpar_init, shaping_ps;
  float forcing_amp, me_ov_mi, nu_ei, nu_hyper, D_hyper;
  float dnlpm, dnlpm_dens, dnlpm_tprp, nu_hyper_l, nu_hyper_m;
  float low_cutoff, high_cutoff, nlpm_max, tau_nlpm;
  float ion_z, ion_mass, ion_dens, ion_fprim, ion_uprim, ion_temp, ion_tprim, ion_vnewk;
  float avail_cpu_time, margin_cpu_time;
  //  float NLdensfac, NLuparfac, NLtparfac, NLtprpfac, NLqparfac, NLqprpfac;
  float tp_t0, tp_tf, tprim0, tprimf;

  cuComplex phi_test, smith_perp_w0;

  specie *species, *species_h;

  bool adiabatic_electrons, snyder_electrons, stationary_ions, dorland_qneut;
  bool all_kinetic, ks, gx, add_Boltzmann_species;
  bool nonlinear_mode, linear, iso_shear, secondary, local_limit, hyper;
  bool no_landau_damping, turn_off_gradients_test, slab, hypercollisions;
  bool write_netcdf, write_omega, write_rh, write_phi, restart, save_for_restart;
  bool append_old, no_omegad, eqfix, write_pzt, collisions;
  bool const_curv, varenna, varenna_fsa, dorland_phase_complex;
  bool new_varenna, new_catto, nlpm, dorland_nlpm, dorland_nlpm_phase;
  bool nlpm_kxdep, nlpm_nlps, nlpm_cutoff_avg, nlpm_zonal_kx1_only, smagorinsky;
  bool debug, init_single, boundary_option_periodic, forcing_init;
  bool nlpm_test, new_nlpm, hammett_nlpm_interference, nlpm_abs_sgn, nlpm_hilbert;
  bool low_b, low_b_all, higher_order_moments, nlpm_zonal_only, nlpm_vol_avg;
  bool no_nonlin_flr, no_nonlin_cross_terms, no_nonlin_dens_cross_term;
  bool zero_order_nonlin_flr_only, no_zonal_nlpm;
  bool write_l_spectrum, write_h_spectrum, write_lh_spectrum;
  bool write_phi_kpar, write_moms, write_fluxes, diagnosing_spectra;
  bool write_free_energy, diagnosing_moments, diagnosing_pzt;
  bool ostem_rname, new_varenna_fsa, qpar0_switch, qprp0_switch;
  bool zero_restart_avg, no_zderiv_covering, no_zderiv, zderiv_loop;
  //  bool tpar_omegad_corrections, tperp_omegad_corrections, qpar_gradpar_corrections ;
  //  bool qpar_bgrad_corrections, qperp_gradpar_corrections, qperp_bgrad_corrections ;
    
  char *scan_type;
  char *equilibrium_option, *nlpm_option;
  char run_name[255];

  int specs[1]; // dims for netcdf species variable arrays
  size_t is_start[1], is_count[1]; 

  int aspecdim[1]; // dimension of control structure for spectral plots (adiabatic species)
  int pspecdim[1]; // dimension of control structure for spectral plots (1-Gamma_0) Phi**2
  int wspecdim[1]; // dimension of control structure for spectral plots G**2
  size_t aspectra_start[1], aspectra_count[1]; 
  size_t pspectra_start[1], pspectra_count[1]; 
  size_t wspectra_start[1], wspectra_count[1]; 
  
  std::string Btype;
  
  std::string restart_from_file, restart_to_file;
  //  char restart_from_file[512];
  //  char restart_to_file[512];
  
  std::string scheme, forcing_type, init_field, stir_field;
  std::string closure_model, boundary, source;
  
  // char scheme[32], forcing_type[32], init_field[32], stir_field[32];
  // char boundary[32], closure_model[32], source[32];

  std::string geofilename;
  //  char geofilename[512];

  //  int *spectra = (int*) malloc (sizeof(int)*13);
  std::vector<int> wspectra;
  std::vector<int> pspectra;
  std::vector<int> aspectra;
  
  cudaDeviceProp prop;
  int maxThreadsPerBlock;

 private:

  float getfloat (int ncid, const char varname[]); 
  int   getint   (int ncid, const char varname[]); 
  bool  getbool  (int ncid, const char varname[]); 
  void  putint   (int ncid, const char varname[], int val);
  void  putfloat (int ncid, const char varname[], float val); 
  void  putbool  (int ncid, const char varname[], bool val);
  void  putspec (int ncid, int m, specie* val);
  void  put_wspectra (int ncid, std::vector<int> s);
  void  put_pspectra (int ncid, std::vector<int> s);
  void  put_aspectra (int ncid, std::vector<int> s);
  bool initialized;
};

