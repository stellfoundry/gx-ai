#include "species.h"

typedef struct {
	//input parameters for geometry
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

   int ntheta;
   int nperiod;

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
   float tite;
 //  float beta;
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
   int ny;
 // nakx = 2*(nx-1)/3 + 1
   int nx;
 // ky_min = 1/y0
   float y0;
   float x0;
   int jtwist;

  // Namelist: init_g_knobs
 //chop_side = F
 //  float phiinit;
 //restart_file = "nc/cyclone_miller_ke.nc"
 //ginit_option= "noise"


  // Namelist: knobs
 //   float fphi;
 //   float fapar;
 //  float faperp;
 //    float delt;
    float dt;
    float maxdt;
    int nstep;

  // Namelist: species_knobs
   int nspec;

	 specie * species;


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
		bool hyper;
    float nu_hyper;
    int p_hyper;
    float init_amp;
		char * scan_type;
    int scan_number;
    bool zero_restart_avg;
    bool no_zderiv_covering;
    bool write_omega;
    bool write_phi;
    bool restart;
    bool no_omegad;
    bool const_curv;
    bool varenna;
    bool nlpm;
    bool smagorinsky;
    int init;
    bool debug;
    //char * fluxfile;
    //char * stopfile;
    //char * restartfile;
		bool ostem_rname;


} input_parameters_struct;
