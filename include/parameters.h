#pragma once

#include "species.h"
#include "gryfx_lib.h"
#include "cufft.h"

#define PHI 0                                                                            
#define DENS 1
#define FORCE 2 
#define RH_equilibrium 3
#define TPRP 4
#define UPAR 5
#define TPAR 6
#define ODD 7
#define RK2 0
#define RK4 1
#define BEER42 1

class Parameters {

  public:
    Parameters(void);
    ~Parameters(void);
    int read_namelist(char* file);

    int set_externalpars(external_parameters_struct* externalpars);
    int import_externalpars(external_parameters_struct* externalpars);

    int iproc;

    // Namelist: collisions_knobs
     char * collision_model;
  
   //
   //&hyper_knobs
   // hyper_option = 'visc_only'
   // const_amp = .false.
   // isotropic_shear = .false.
   //   float D_hypervisc;
   ///
  
    // Namelist: theta_grid_parameters
  
     int nz_in;
     int nperiod;
     int Zp;
  
     float rhoc;
     float eps;
     float shat;
     float qsf;
     float rmaj;
     float r_geo;
     float shift;
     float akappa;
     float akappri;
     float tri;
     float tripri;
     float drhodpsi;
     float epsl;
     float kxfac;
  
    // Namelist: parameters
     float ti_ov_te;
     float beta;
   //  float zeff;
  
    // Namelist: theta_grid_eik_knobs
   //  int itor;
   //  int iflux;
     int irho;
  
   //ppl_eq = F
   //gen_eq = F
   //efit_eq = F
   //local_eq = T
  
     char * eqfile;
  	 /*equal_arc = T*/
     int bishop;
     float s_hat_input;
     float beta_prime_input;
   //  float delrho;
   //  int isym;
   //writelots = F
  
    // Namelist: fields_knobs
   //field_option='implicit'
  
    // Namelist: gs2_diagnostics_knobs
   //print_flux_line = T
   //write_nl_flux = T
   //print_line = F
   //write_line = F
   //write_omega = F
   //write_final_fields = T
   //write_g = F
   //write_verr = T
     int nwrite;
     int navg;
     int nsave;
   //  float omegatinst;
   //save_for_restart = .true.
   //omegatol = -1.0e-3
  
   //&le_grids_knobs
   //   int ngauss;
   //   int negrid;
   //   float vcut;
   ///
  
    // Namelist: dist_fn_knobs
   //adiabatic_option="iphi00=2"
   //  float gridfac;
   //boundary_option="linked"
     float g_exb;
  
  
    // Namelist: kt_grids_knobs
   //grid_option='box'
  
    // Namelist: kt_grids_box_parameters
   // naky = (ny-1)/3 + 1
     int ny_in;
   // nakx = 2*(nx-1)/3 + 1
     int nx_in;
   // ky_min = 1/y0
     float y0;
     float x0;
     int jtwist;
  
    // Namelist: init_g_knobs
   //chop_side = F
   //  float phiinit;
   //restart_file = "nc/cyclone_miller_ke.nc"
   //ginit_option= "noise"
   
   // Namelist: 
      // new for HL 
      int nhermite_in;
      int nlaguerre_in;  // nlaguerre will be an array 
      int closure_model;
  
  
    // Namelist: knobs
   //   float fphi;
      float fapar;
   //  float faperp;
   //    float delt;
      float dt; //Initial and maximum timestep
      float maxdt; //Obsolete
      int nstep; //Number of timesteps
      float avail_cpu_time; //Available wall clock time in s
      float margin_cpu_time; //Start finishing up when only margin_cpu_time remains
  
    // Namelist: species_knobs
     int nspec_in;

     specie * species;
  
     
     bool adiabatic_electrons;
     bool snyder_electrons;
     bool stationary_ions;
     bool dorland_qneut;
     float me_ov_mi;
     float nu_ei;
  
  
    // Namelist: dist_fn_species_knobs_2
   //  float fexpr;
   //  float bakdif;
  
    // Namelist: theta_grid_knobs
  	int equilibrium_type;
  	char * equilibrium_option;
  	 /*equilibrium_option = "miller"*/
   //equilibrium_option='eik'
  
    // Namelist: nonlinear_terms_knobs
    // nonlinear_mode='off'
  	 bool linear;
     float cfl;
  
    // Namelist: reinit_knobs
   //  float delt_adj;
   //  float delt_minimum;
  
    // Namelist: layouts_knobs
   // layout = 'lxyes'
   // local_field_solve = F
  
  
    // Namelist: gryfx_knobs
      int inlpm;
      float dnlpm;
      float dnlpm_dens;
      float dnlpm_tprp;
      bool hyper;
      float nu_hyper;
      float D_hyper;
      bool iso_shear;
      int p_hyper;
      float init_amp;
      char * scan_type;
      bool secondary_test;
      char * secondary_test_restartfileName;
      cuComplex phi_test;
      float NLdensfac;
      float NLuparfac;
      float NLtparfac;
      float NLtprpfac;
      float NLqparfac;
      float NLqprpfac;
      int scan_number;
      bool zero_restart_avg;
      bool no_zderiv_covering;
      bool no_zderiv;
      bool zderiv_loop;
      int icovering;
      int iphi00;
      bool no_landau_damping;
      bool turn_off_gradients_test;
      bool slab;
      bool write_netcdf;
      bool write_omega;
      bool write_phi;
      bool restart;
      //If restart, use the netcdf restart file
      bool netcdf_restart;
      // If a netcdf file with the right run name already exists, open it and append to it
      bool append_old;
      bool check_for_restart;
      bool no_omegad;
      bool const_curv;
      bool varenna;
      bool varenna_fsa;
      int ivarenna;
      bool new_varenna;
      bool new_catto;
      bool nlpm;
      bool dorland_nlpm;
      bool dorland_nlpm_phase;
      bool dorland_phase_complex;
      int dorland_phase_ifac;
      char * nlpm_option;
      float low_cutoff;
      float high_cutoff;
      float dnlpm_max;
      float tau_nlpm;
      bool nlpm_kxdep;
      bool nlpm_nlps;
      bool nlpm_cutoff_avg;
      bool nlpm_zonal_kx1_only;
      bool smagorinsky;
      int init;
      float phiext;
      bool debug;
      bool init_single;
      int iky_single;
      int ikx_single;
      int iky_fixed;
      int ikx_fixed;
      float kpar_init;
      bool nlpm_test;
      bool new_nlpm; 
      bool hammett_nlpm_interference; 
      bool nlpm_abs_sgn;
      bool nlpm_hilbert;
      bool low_b;
      bool low_b_all;
      int iflr;
      bool higher_order_moments;
      bool nlpm_zonal_only;
      bool nlpm_vol_avg;
      bool no_nonlin_flr;
      bool no_nonlin_cross_terms;
      bool no_nonlin_dens_cross_term;
      bool zero_order_nonlin_flr_only;
      bool no_zonal_nlpm;
  
      int igeo;
      float shaping_ps;
      char * geofilename;
      //char * fluxfile;
      //char * stopfile;
      //char * restartfile;
      bool ostem_rname;
  
      //Namelist new_varenna_knobs
      bool new_varenna_fsa;
      int zonal_dens_switch;
      int q0_dens_switch;
  
      bool tpar_omegad_corrections ;
      bool tperp_omegad_corrections ;
      bool qpar_gradpar_corrections ;
      bool qpar_bgrad_corrections ;
      bool qperp_gradpar_corrections ;
      bool qperp_bgrad_corrections ;
      bool qpar0_switch ;
      bool qprp0_switch ;
  
      // formerly part of time struct
      int trinity_timestep;
      int trinity_iteration;
      int trinity_conv_count;
      int end_time;

      int scheme;
      char run_name[2000];
};

