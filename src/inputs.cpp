#include "cufft.h"
#define EXTERN_SWITCH extern
#include "species.h"
#include "inputs.h"
#include "c_fortran_namelist3.c"
#include "namelist_defaults.c"

bool get_bool_on_off(struct fnr_struct * namelist_struct, const char * namelist, const char * variable)
{
  char* on_off;
  bool res;
  //on_off = (char*) malloc(sizeof(char)*4);
  fnr_get_string(namelist_struct, namelist, variable, &on_off);
  if( strcmp(on_off,"on") == 0) {
    res = true;
  }
  else if( strcmp(on_off,"off") == 0) {
    res = false;
  }
  else
  {
    printf("Illegal value %s for %s in %s: should be 'on' or 'off'\n", on_off, variable, namelist);
    abort();
  }
  free(on_off);
  return res;
}
bool get_bool(struct fnr_struct * namelist_struct, const char * namelist, const char * variable)
{
  bool return_value;
  int result;
  fnr_get_bool(namelist_struct, namelist, variable, &result);
  if (result) return_value = true;
  else return_value = false;
  return return_value;
}

Inputs::Inputs(void) {
  pars_ = new input_parameters_struct;
}

Inputs::~Inputs(void) {
  delete pars_;
}

input_parameters_struct* Inputs::getpars(void) {
  return pars_;
}

int Inputs::read_namelist(char* filename)
{
	fnr_abort_on_error = 1;
	fnr_abort_if_missing = 0;
	fnr_abort_if_no_default = 1;
  struct fnr_struct namelist_struct = fnr_read_namelist_file(filename);  
  struct fnr_struct namelist_defaults = fnr_read_namelist_string(default_namelist_string);  
	fnr_check_namelist_against_template(&namelist_struct, &namelist_defaults);
	fnr_set_defaults(&namelist_struct, &namelist_defaults);
  fnr_get_int(&namelist_struct, "theta_grid_parameters", "nperiod", &pars_->nperiod);

  fnr_get_int(&namelist_struct, "theta_grid_parameters", "ntheta", &(pars_->Nz));

  if(pars_->Nz!=1) {
    //The GS2 grid, which includes the periodic point, runs from
    // -ntgrid:ntgrid and is thus 2*ntgrid+1 in size, where ntgrid
    // is the same as what is calculated here. This little step
    // thus ensures that the Gryfx grid corresponds to the GS2 grid
    // (without the periodic) point.
      int ntgrid = pars_->Nz/2 + (pars_->nperiod-1)*pars_->Nz;
 
      pars_->Nz = 2*ntgrid;
  }
  
  if(fnr_get_int(&namelist_struct, "theta_grid_parameters", "zp", &pars_->Zp)==FNR_USED_DEFAULT){
   pars_->Zp = 2*pars_->nperiod - 1; 
  }
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "eps", &(pars_->eps));
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "shat", &(pars_->shat));

  fnr_get_float(&namelist_struct, "theta_grid_eik_knobs", "beta_prime_input", &pars_->beta_prime_input);

  fnr_get_float(&namelist_struct, "theta_grid_eik_knobs", "s_hat_input", &pars_->s_hat_input);
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "qinp", &(pars_->qsf));

  fnr_get_float(&namelist_struct, "theta_grid_parameters", "akappa", &pars_->akappa);

  fnr_get_float(&namelist_struct, "theta_grid_parameters", "akappri", &pars_->akappri);

  fnr_get_float(&namelist_struct, "theta_grid_parameters", "tri", &pars_->tri);

  fnr_get_float(&namelist_struct, "theta_grid_parameters", "tripri", &pars_->tripri);
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "Rmaj", &(pars_->rmaj));
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "rhoc", &(pars_->rhoc));
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "shift", &(pars_->shift));
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "drhodpsi", &(pars_->drhodpsi));
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "epsl", &(pars_->epsl));
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "kxfac", &(pars_->kxfac));

  fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "nx", &(pars_->nx));
    
  fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "ny", &(pars_->ny));
  
  fnr_get_float(&namelist_struct, "kt_grids_box_parameters", "y0", &(pars_->y0));
  
  //before, jtwist_old assumed Zp=1
  //now, redefining jtwist = jtwist_old*Zp
  if(fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "jtwist", &(pars_->jtwist)) == FNR_USED_DEFAULT) {
    //set default jtwist to 2pi*shat so that X0~Y0
    pars_->jtwist = (int) round(2*M_PI*pars_->shat*pars_->Zp);
  }  

  fnr_get_float(&namelist_struct, "kt_grids_box_parameters", "x0", &(pars_->x0));
  //X0 will get overwritten if shat!=0

  // set up for HL moments "grid"
  char* opt;
  int nlaguerre, vN;
  fnr_get_string(&namelist_struct, "hl_grids_knobs", "grid_option", &opt);
  if(strcmp(opt, "box")) {
    fnr_get_int(&namelist_struct, "hl_grids_knobs", "nhermite", &(pars_->nhermite));
    fnr_get_int(&namelist_struct, "hl_grids_knobs", "nlaguerre", &nlaguerre);
    pars_->nmoms = pars_->nhermite * nlaguerre;
    pars_->nlaguerre = (int*) malloc(sizeof(int)*pars_->nhermite);
    for (int l=0; l<pars_->nhermite; l++) {
      pars_->nlaguerre[l] = nlaguerre;
    }
  } else if (strcmp(opt, "poly_order")) {
    fnr_get_int(&namelist_struct, "hl_grids_knobs", "max_poly_order", &vN);
    pars_->nhermite = vN+1;
    pars_->nmoms = ( 1 + vN/2 ) * ( 1 + vN - vN/2 );
    pars_->nlaguerre = (int*) malloc(sizeof(int)*pars_->nhermite);
    for (int l=0; l<pars_->nhermite; l++) {
      pars_->nlaguerre[l] = (vN-l)/2;
    }
  }
  
  fnr_get_float(&namelist_struct, "dist_fn_knobs", "g_exb", &(pars_->g_exb));
  
  fnr_get_float(&namelist_struct, "parameters", "tite", &(pars_->ti_ov_te));
  // ti_ov_te is overwritten for non-adiabatic electrons

  fnr_get_float(&namelist_struct, "parameters", "beta", &(pars_->beta));
  
  fnr_get_float(&namelist_struct, "nonlinear_terms_knobs", "cfl", &(pars_->cfl));
  
  pars_->zero_order_nonlin_flr_only= get_bool(&namelist_struct, "gryfx_knobs", "zero_order_nonlin_flr_only");
  if(pars_->zero_order_nonlin_flr_only) pars_->no_nonlin_flr = true;

  pars_->no_nonlin_cross_terms= get_bool(&namelist_struct, "gryfx_knobs", "no_nonlin_cross_terms");
  pars_->no_nonlin_dens_cross_term= get_bool(&namelist_struct, "gryfx_knobs", "no_nonlin_dens_cross_term");
  if(pars_->no_nonlin_cross_terms || pars_->no_nonlin_dens_cross_term) {
    pars_->new_nlpm = false;
    pars_->zero_order_nonlin_flr_only = false;
    pars_->no_nonlin_flr = false;
  }

  pars_->no_zonal_nlpm= get_bool(&namelist_struct, "gryfx_knobs", "no_zonal_nlpm");

  fnr_get_int(&namelist_struct, "gs2_diagnostics_knobs", "nwrite", &(pars_->nwrite));
  
  fnr_get_int(&namelist_struct, "gs2_diagnostics_knobs", "nsave", &(pars_->nsave));
  
  fnr_get_int(&namelist_struct, "gs2_diagnostics_knobs", "navg", &(pars_->navg));
  
  fnr_get_float(&namelist_struct, "knobs", "delt", &(pars_->dt));
  
  //fnr_get_float(&namelist_struct, "knobs", "maxdt", &(pars_->maxdt));
  //maxdt=pars_->maxdt;
  
  fnr_get_int(&namelist_struct, "knobs", "nstep", &(pars_->nstep));
  
  fnr_get_float(&namelist_struct, "knobs", "fapar", &(pars_->fapar));

  fnr_get_float(&namelist_struct, "knobs", "avail_cpu_time", &(pars_->avail_cpu_time));
  fnr_get_float(&namelist_struct, "knobs", "margin_cpu_time", &(pars_->margin_cpu_time));
  //maxdt?
  
  fnr_get_int(&namelist_struct, "species_knobs", "nspec", &(pars_->nspec));
  
  pars_->linear = !get_bool_on_off(&namelist_struct, "nonlinear_terms_knobs", "nonlinear_mode");
  

  pars_->check_for_restart = false;
  char* restart;
  fnr_get_string(&namelist_struct, "gryfx_knobs", "restart", &restart);
  if( strcmp(restart,"on") == 0) {
    pars_->restart = true;
  }
  else if( strcmp(restart,"exist") == 0) {
    //check if restart file exists
    pars_->check_for_restart = true;
  }
  else if( strcmp(restart,"off") == 0) {
    pars_->restart = false;
  }

  pars_->netcdf_restart = get_bool(&namelist_struct, "gryfx_knobs", "netcdf_restart"); 

  // If a netcdf file with the right run name already exists, open it and append to it
  pars_->append_old = get_bool(&namelist_struct, "gryfx_knobs", "append_old"); 
  
  //pars_->zero_restart_avg = zero_restart_avg = get_bool(&namelist_struct, "gryfx_knobs", "zero_restart_avg");
  pars_->zero_restart_avg = get_bool_on_off(&namelist_struct, "gryfx_knobs", "zero_restart_avg");

  pars_->no_zderiv_covering = get_bool_on_off(&namelist_struct, "gryfx_knobs", "no_zderiv_covering");

  pars_->no_zderiv = get_bool_on_off(&namelist_struct, "gryfx_knobs", "no_zderiv");

  pars_->zderiv_loop = get_bool_on_off(&namelist_struct, "gryfx_knobs", "zderiv_loop");

  pars_->no_landau_damping = get_bool_on_off(&namelist_struct, "gryfx_knobs", "no_landau_damping");

  pars_->turn_off_gradients_test = get_bool_on_off(&namelist_struct, "gryfx_knobs", "turn_off_gradients_test");

  pars_->slab = get_bool_on_off(&namelist_struct, "gryfx_knobs", "slab");

  pars_->const_curv = get_bool_on_off(&namelist_struct, "gryfx_knobs", "const_curv");

  pars_->varenna = get_bool_on_off(&namelist_struct, "gryfx_knobs", "varenna");

  pars_->new_catto = get_bool_on_off(&namelist_struct, "gryfx_knobs", "new_catto");
  
 //////////////////////////
 // new varenna knobs

  pars_->new_varenna = get_bool_on_off(&namelist_struct, "gryfx_knobs", "new_varenna");
  
  pars_->new_varenna_fsa = get_bool_on_off(&namelist_struct, "new_varenna_knobs", "new_varenna_fsa");

  /////////////////////////

  pars_->nlpm = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm");

  pars_->dorland_nlpm = get_bool_on_off(&namelist_struct, "gryfx_knobs", "dorland_nlpm");

  pars_->dorland_nlpm_phase = get_bool_on_off(&namelist_struct, "gryfx_knobs", "dorland_nlpm_phase");

  pars_->dorland_phase_complex = get_bool_on_off(&namelist_struct, "gryfx_knobs", "dorland_phase_complex");

  pars_->nlpm_kxdep = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_kxdep");

  if(pars_->nlpm_kxdep) pars_->dorland_phase_complex = true;  

  pars_->nlpm_nlps = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_nlps");

  if(pars_->nlpm_nlps) pars_->nlpm = false; //turn off normal filter-style NLPM

  pars_->nlpm_cutoff_avg = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_cutoff_avg");

  fnr_get_int(&namelist_struct, "gryfx_knobs", "dorland_phase_ifac", &pars_->dorland_phase_ifac);
  fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "nlpm_option", &pars_->nlpm_option);

  fnr_get_int(&namelist_struct, "gryfx_knobs", "inlpm", &pars_->inlpm);
  fnr_get_float(&namelist_struct, "gryfx_knobs", "dnlpm", &pars_->dnlpm);
  fnr_get_float(&namelist_struct, "gryfx_knobs", "dnlpm_dens", &pars_->dnlpm_dens);
  fnr_get_float(&namelist_struct, "gryfx_knobs", "dnlpm_tprp", &pars_->dnlpm_tprp);

  fnr_get_float(&namelist_struct, "gryfx_knobs", "dnlpm_max", &pars_->dnlpm_max);
  fnr_get_float(&namelist_struct, "gryfx_knobs", "tau_nlpm", &pars_->tau_nlpm);

  pars_->nlpm_zonal_kx1_only = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_zonal_kx1_only");

  fnr_get_int(&namelist_struct, "gryfx_knobs", "ivarenna", &pars_->ivarenna);

  pars_->varenna_fsa = get_bool_on_off(&namelist_struct, "gryfx_knobs", "varenna_fsa");

  fnr_get_int(&namelist_struct, "gryfx_knobs", "icovering", &pars_->icovering);

  fnr_get_int(&namelist_struct, "gryfx_knobs", "iphi00", &pars_->iphi00);

  pars_->smagorinsky = get_bool_on_off(&namelist_struct, "gryfx_knobs", "smagorinsky");

  pars_->hyper = get_bool_on_off(&namelist_struct, "gryfx_knobs", "hyper");
  
  fnr_get_float(&namelist_struct, "gryfx_knobs", "d_hyper", &pars_->D_hyper);
  
  fnr_get_int(&namelist_struct, "gryfx_knobs", "p_hyper", &pars_->p_hyper);

  pars_->iso_shear = get_bool_on_off(&namelist_struct, "gryfx_knobs", "iso_shear");

  pars_->debug = get_bool_on_off(&namelist_struct, "gryfx_knobs", "debug");
  
  fnr_get_int(&namelist_struct, "gryfx_knobs", "igeo", &pars_->igeo);

  char* initfield;
  int init;
  //initfield = (char*) malloc(sizeof(char)*10);
  fnr_get_string(&namelist_struct, "gryfx_knobs", "init", &initfield);
  if( strcmp(initfield,"density") == 0) {
    init = DENS;
  }
  else if( strcmp(initfield,"phi") == 0) {
    init = PHI;
  }
  else if( strcmp(initfield,"force") == 0) {
    init = FORCE;
  }
  else if( strcmp(initfield,"tperp") == 0) {
    init = TPRP;
  }
  else if( strcmp(initfield,"tpar") == 0) {
    init = TPAR;
  }
  else if( strcmp(initfield,"upar") == 0) {
    init = UPAR;
  }
  else if( strcmp(initfield,"odd") == 0) {
    init = ODD;
  }
  else if (strcmp(initfield,"RH_eq") == 0) {
    init = RH_equilibrium;
    pars_->new_varenna = true;
  } 
  pars_->init = init;
  
  fnr_get_float(&namelist_struct, "gryfx_knobs", "init_amp", &(pars_->init_amp));
  fnr_get_float(&namelist_struct, "gryfx_knobs", "phiext", &pars_->phiext);

  pars_->init_single = get_bool_on_off(&namelist_struct, "gryfx_knobs", "init_single");
  fnr_get_int(&namelist_struct, "gryfx_knobs", "iky_single", &(pars_->iky_single));
  fnr_get_int(&namelist_struct, "gryfx_knobs", "ikx_single", &(pars_->ikx_single));
  
  fnr_get_float(&namelist_struct, "gryfx_knobs", "kpar_init", &(pars_->kpar_init));
  
  pars_->write_netcdf= get_bool(&namelist_struct, "gryfx_knobs", "write_netcdf");

  pars_->write_omega = get_bool_on_off(&namelist_struct, "gryfx_knobs", "write_omega");

  pars_->write_phi = get_bool_on_off(&namelist_struct, "gryfx_knobs", "write_phi");

  fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "scan_type", &pars_->scan_type);
  
  pars_->higher_order_moments = get_bool_on_off(&namelist_struct, "gryfx_knobs", "higher_order_moments");
  
  pars_->secondary_test = get_bool_on_off(&namelist_struct, "secondary_test_knobs", "secondary_test");
  pars_->nlpm_test = get_bool_on_off(&namelist_struct, "secondary_test_knobs", "nlpm_test");
  pars_->new_nlpm = get_bool_on_off(&namelist_struct, "gryfx_knobs", "new_nlpm");
  pars_->nlpm_abs_sgn = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_abs_sgn");
  pars_->hammett_nlpm_interference = get_bool_on_off(&namelist_struct, "gryfx_knobs", "hammett_nlpm_interference");
  pars_->low_b = get_bool_on_off(&namelist_struct, "secondary_test_knobs", "low_b");
  pars_->low_b_all = get_bool_on_off(&namelist_struct, "gryfx_knobs", "low_b_all");
  if(pars_->low_b_all) pars_->low_b = true;
  pars_->nlpm_zonal_only = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_zonal_only");
  pars_->nlpm_vol_avg = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_vol_avg");

  fnr_get_int(&namelist_struct, "gryfx_knobs", "iflr", &(pars_->iflr));

  cuComplex phi_test;
  fnr_get_float(&namelist_struct, "secondary_test_knobs", "phi_test_real", &phi_test.x);
  fnr_get_float(&namelist_struct, "secondary_test_knobs", "phi_test_imag", &phi_test.y);

  fnr_get_int(&namelist_struct, "secondary_test_knobs", "iky_fixed", &(pars_->iky_fixed));
  fnr_get_int(&namelist_struct, "secondary_test_knobs", "ikx_fixed", &(pars_->ikx_fixed));

  if(pars_->secondary_test && pars_->linear && phi_test.x > phi_test.y) pars_->init_amp = phi_test.x;
  if(pars_->secondary_test && pars_->linear && phi_test.x < phi_test.y) pars_->init_amp = phi_test.y;
  pars_->phi_test = phi_test;

  if(pars_->nlpm_test) { 
    float fac;
    if(pars_->iky_fixed==0) fac = .5;
    else fac = 1.;
    if(phi_test.x > 0.) phi_test.x = pars_->x0*pars_->y0*fac; 
    else phi_test.x = 0.;
    if(phi_test.y > 0.) phi_test.y = pars_->x0*pars_->y0*fac; 
    else phi_test.y = 0.;
    //if(phi_test.x > 0. && phi_test.y > 0.) {phi_test.x = phi_test.x/sqrt(2.); phi_test.y = phi_test.y/sqrt(2.);}
  }

  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "densfac", &pars_->NLdensfac);
  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "uparfac", &pars_->NLuparfac);
  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "tparfac", &pars_->NLtparfac);
  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "tprpfac", &pars_->NLtprpfac);
  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "qparfac", &pars_->NLqparfac);
  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "qprpfac", &pars_->NLqprpfac);

  if(pars_->NLdensfac!=0) pars_->NLdensfac = 1.;
  if(pars_->NLuparfac!=0) pars_->NLuparfac = 1.;
  if(pars_->NLtparfac!=0) pars_->NLtparfac = 1.;
  if(pars_->NLtprpfac!=0) pars_->NLtprpfac = 1.;
  if(pars_->NLqparfac!=0) pars_->NLqparfac = 1.;
  if(pars_->NLqprpfac!=0) pars_->NLqprpfac = 1.;
   

  fnr_get_string_no_test(&namelist_struct, "secondary_test_knobs", "restartfile", &pars_->secondary_test_restartfileName);

  fnr_get_int(&namelist_struct, "gryfx_knobs", "scan_number", &pars_->scan_number);
    
  //char* collisions;
  fnr_get_string(&namelist_struct, "collisions_knobs", "collision_model", &(pars_->collision_model));
  //collisions=pars_->collision_model;
  
  specie* species;
  pars_->species = species = (specie*) malloc(sizeof(specie)*pars_->nspec);

  pars_->adiabatic_electrons = true;

  pars_->snyder_electrons = get_bool_on_off(&namelist_struct, "gryfx_knobs", "snyder_electrons");
  pars_->stationary_ions = get_bool_on_off(&namelist_struct, "gryfx_knobs", "stationary_ions");
   
    int ionspec = 0;   
    int ispec = 1;
    float mass;
    bool main_ion_species_found = false;
  for(int s=1; s<pars_->nspec+1; s++) {
//		printf("s= %d\n", s);


    char namelist[100];
    sprintf(namelist,"species_parameters_%d",s); 
    char* type;
    fnr_get_string(&namelist_struct, namelist, "type", &type); 
  
 
    if(strcmp(type,"ion") == 0) {
      fnr_get_float(&namelist_struct, namelist, "mass", &mass);
      if((mass == 1. && !main_ion_species_found) || pars_->nspec==1) {ionspec=0; main_ion_species_found=true;} // main ion species mass assumed to be 1. main ion species indexed 0.
      else {ionspec = ispec; ispec++;}
      species[ionspec].mass = mass;
      fnr_get_float(&namelist_struct, namelist, "z", &species[ionspec].z);
      fnr_get_float(&namelist_struct, namelist, "dens", &species[ionspec].dens);
      fnr_get_float(&namelist_struct, namelist, "temp", &species[ionspec].temp);
      fnr_get_float(&namelist_struct, namelist, "tprim", &species[ionspec].tprim); //6.9
      fnr_get_float(&namelist_struct, namelist, "fprim", &species[ionspec].fprim); //2.2
      if(fnr_get_float(&namelist_struct, namelist, "uprim", &species[ionspec].uprim)) species[ionspec].uprim=0;


     // if(strcmp(collisions,"none") == 0) species[ionspec].nu_ss = 0;
     // else {
        fnr_get_float(&namelist_struct, namelist, "vnewk", &species[ionspec].nu_ss);
     // }     

      strcpy(species[ionspec].type,"ion"); 

    }
    if(strcmp(type,"electron") == 0) {

      // kinetic electrons will always be last indexed species

      fnr_get_float(&namelist_struct, namelist, "z", &species[pars_->nspec-1].z);
      fnr_get_float(&namelist_struct, namelist, "mass", &species[pars_->nspec-1].mass);
      fnr_get_float(&namelist_struct, namelist, "dens", &species[pars_->nspec-1].dens);
      fnr_get_float(&namelist_struct, namelist, "temp", &species[pars_->nspec-1].temp);
      fnr_get_float(&namelist_struct, namelist, "tprim", &species[pars_->nspec-1].tprim);
      fnr_get_float(&namelist_struct, namelist, "fprim", &species[pars_->nspec-1].fprim);
      fnr_get_float(&namelist_struct, namelist, "uprim", &species[pars_->nspec-1].uprim);

      //if(strcmp(collisions,"none") == 0) species[pars_->nspec-1].nu_ss = 0;
      //else {

      fnr_get_float(&namelist_struct, namelist, "vnewk", &species[pars_->nspec-1].nu_ss);

      //}

      strcpy(species[pars_->nspec-1].type,"electron");

      pars_->adiabatic_electrons = false;
   
    }   


  }
		//free(type);
		fnr_free(&namelist_struct);
		fnr_free(&namelist_defaults);

  return 0;
}






