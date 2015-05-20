
#define PHI 0
#define DENS 1
#define FORCE 2
#define RH_equilibrium 3

#ifndef NO_GLOBALS
//Globals will be deleted eventually
//globals defined in gs2 namelist
EXTERN_SWITCH int nwrite;
EXTERN_SWITCH int nsave;
EXTERN_SWITCH int navg;
EXTERN_SWITCH int nstop;
EXTERN_SWITCH float cfl;
EXTERN_SWITCH float g_exb;
EXTERN_SWITCH int jtwist;
EXTERN_SWITCH float tau;
//
//globals defined in eik.out
EXTERN_SWITCH float drhodpsi, rmaj, shat, kxfac, qsf, gradpar, eps, aminor, epsl;
EXTERN_SWITCH float bi;

//other Miller parameters;
EXTERN_SWITCH float r_geo, akappa, akappri, tri, tripri, shift, asym, asympri;
  
//input parameters for geometry
EXTERN_SWITCH int equilibrium_type;
EXTERN_SWITCH int bishop;
EXTERN_SWITCH int irho;
EXTERN_SWITCH int nperiod;
EXTERN_SWITCH float rhoc;

//Greene & Chance/Bishop parameters
EXTERN_SWITCH float beta_prime_input, s_hat_input;

EXTERN_SWITCH int ivarenna;
EXTERN_SWITCH bool varenna_fsa;
EXTERN_SWITCH bool new_varenna_fsa;
EXTERN_SWITCH int zonal_dens_switch;
EXTERN_SWITCH int q0_dens_switch;

EXTERN_SWITCH float dnlpm_max;
EXTERN_SWITCH char* nlpm_option;
EXTERN_SWITCH float tau_nlpm;
EXTERN_SWITCH bool nlpm_zonal_kx1_only;

EXTERN_SWITCH bool dorland_nlpm;
EXTERN_SWITCH bool dorland_nlpm_phase;
EXTERN_SWITCH bool dorland_phase_complex;
EXTERN_SWITCH int dorland_phase_ifac;
EXTERN_SWITCH bool nlpm_nlps;
EXTERN_SWITCH bool nlpm_kxdep;
EXTERN_SWITCH bool nlpm_cutoff_avg;

//Grids
EXTERN_SWITCH float X0, Y0;
EXTERN_SWITCH int Zp;
EXTERN_SWITCH int nSpecies;

EXTERN_SWITCH specie *species;

EXTERN_SWITCH double dt;

EXTERN_SWITCH float maxdt;
EXTERN_SWITCH int nSteps;

EXTERN_SWITCH bool LINEAR;  
EXTERN_SWITCH bool RESTART;
EXTERN_SWITCH bool CHECK_FOR_RESTART;
EXTERN_SWITCH bool NO_ZDERIV;
EXTERN_SWITCH bool NO_ZDERIV_COVERING;

//EGH it seems SCAN is obsolete
//EXTERN_SWITCH bool SCAN;
EXTERN_SWITCH bool NO_ZDERIV_B;
EXTERN_SWITCH bool SLAB;
EXTERN_SWITCH bool CONST_CURV;
EXTERN_SWITCH bool write_omega;
EXTERN_SWITCH bool write_phi;
EXTERN_SWITCH bool write_phase;
EXTERN_SWITCH bool S_ALPHA;
EXTERN_SWITCH bool no_landau_damping;
EXTERN_SWITCH bool turn_off_gradients_test;
              
EXTERN_SWITCH bool NLPM;
EXTERN_SWITCH bool varenna;
EXTERN_SWITCH bool new_varenna;
EXTERN_SWITCH bool new_catto;
EXTERN_SWITCH bool SMAGORINSKY;
EXTERN_SWITCH bool HYPER;
EXTERN_SWITCH bool isotropic_shear;
EXTERN_SWITCH bool zero_restart_avg;

EXTERN_SWITCH bool tpar_omegad_corrections;
EXTERN_SWITCH bool tperp_omegad_corrections;
EXTERN_SWITCH bool qpar_gradpar_corrections;
EXTERN_SWITCH bool qpar_bgrad_corrections;
EXTERN_SWITCH bool qperp_gradpar_corrections;
EXTERN_SWITCH bool qperp_bgrad_corrections;
EXTERN_SWITCH bool qpar0_switch;
EXTERN_SWITCH bool qprp0_switch;

EXTERN_SWITCH float dnlpm;
EXTERN_SWITCH int inlpm;
EXTERN_SWITCH float low_cutoff;
EXTERN_SWITCH float high_cutoff;

EXTERN_SWITCH int init;
EXTERN_SWITCH float init_amp;
EXTERN_SWITCH float phiext;

EXTERN_SWITCH int icovering;

EXTERN_SWITCH float NLdensfac, NLuparfac, NLtparfac, NLtprpfac, NLqparfac, NLqprpfac;

EXTERN_SWITCH int iphi00;
EXTERN_SWITCH int igeo;
EXTERN_SWITCH float shaping_ps;
EXTERN_SWITCH char* geoFileName;

EXTERN_SWITCH bool secondary_test;
EXTERN_SWITCH cuComplex phi_test;
EXTERN_SWITCH char* secondary_test_restartfileName;

EXTERN_SWITCH float D_hyper;
EXTERN_SWITCH int p_hyper;

EXTERN_SWITCH char* scan_type;
EXTERN_SWITCH int scan_number;

EXTERN_SWITCH bool DEBUG;
#endif

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
    int icovering;
    int iphi00;
    bool no_landau_damping;
    bool turn_off_gradients_test;
    bool slab;
    bool write_netcdf;
    bool write_omega;
    bool write_phi;
    bool restart;
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


} input_parameters_struct;
