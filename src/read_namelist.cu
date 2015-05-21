#include "cufft.h"
#define EXTERN_SWITCH extern
#include "species.h"
#include "input_parameters_struct.h"
#include "grids.h"
#include "read_namelist.h"
#include "c_fortran_namelist3.c"
#include "namelist_defaults.cu"

bool get_bool_on_off(struct fnr_struct * namelist_struct, char * namelist, char * variable)
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
bool get_bool(struct fnr_struct * namelist_struct, char * namelist, char * variable)
{
  bool return_value;
  int result;
  fnr_get_bool(namelist_struct, namelist, variable, &result);
  if (result) return_value = true;
  else return_value = false;
  return return_value;
}

void read_namelist(input_parameters_struct * pars, grids_struct * grids, char* filename)
{

//	printf("fnr_read_namelist_file starting\n");
	fnr_abort_on_error = 1;
	fnr_abort_if_missing = 0;
	fnr_abort_if_no_default = 1;
  struct fnr_struct namelist_struct = fnr_read_namelist_file(filename);  
  struct fnr_struct namelist_defaults = fnr_read_namelist_string(default_namelist_string);  
	fnr_check_namelist_against_template(&namelist_struct, &namelist_defaults);
	fnr_set_defaults(&namelist_struct, &namelist_defaults);
//	printf("fnr_read_namelist_file ended\n");
    
//	fnr_get_string(&namelist_struct, "theta_grid_knobs", "equilibrium_option", &(pars->equilibrium_option));
//	if (!strcmp(pars->equilibrium_option, "s-alpha")){
//		pars->equilibrium_type = SALPHA;
//		S_ALPHA = true;
//	}
//	else if (!strcmp(pars->equilibrium_option, "miller")){
//		pars->equilibrium_type = MILLER;
//		S_ALPHA = false;
//	}
  fnr_get_int(&namelist_struct, "theta_grid_parameters", "nperiod", &pars->nperiod);
  nperiod=pars->nperiod;

  fnr_get_int(&namelist_struct, "theta_grid_parameters", "ntheta", &(grids->Nz));

  //The GS2 grid, which includes the periodic point, runs from
  // -ntgrid:ntgrid and is thus 2*ntgrid+1 in size, where ntgrid
  // is the same as what is calculated here. This little step
  // thus ensures that the Gryfx grid corresponds to the GS2 grid
  // (without the periodic) point.
  int ntgrid = grids->Nz/2 + (pars->nperiod-1)*grids->Nz;
  //grids->Nz=2*(2*nperiod-1)*grids->Nz + 1
 
	Nz = grids->Nz = 2*ntgrid;

  
  if(fnr_get_int(&namelist_struct, "theta_grid_parameters", "zp", &pars->Zp)==FNR_USED_DEFAULT){
   pars->Zp = 2*nperiod - 1; 
  }
  *&Zp = pars->Zp;
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "eps", &(pars->eps));
  eps=pars->eps;
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "shat", &(pars->shat));
  shat=pars->shat;

  fnr_get_float(&namelist_struct, "theta_grid_eik_knobs", "beta_prime_input", &pars->beta_prime_input);
  beta_prime_input = pars->beta_prime_input;

  fnr_get_float(&namelist_struct, "theta_grid_eik_knobs", "s_hat_input", &pars->s_hat_input);
  s_hat_input = pars->s_hat_input;
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "qinp", &(pars->qsf));
  qsf=pars->qsf;

  fnr_get_float(&namelist_struct, "theta_grid_parameters", "akappa", &pars->akappa);
  akappa=pars->akappa;

  fnr_get_float(&namelist_struct, "theta_grid_parameters", "akappri", &pars->akappri);
  akappri=pars->akappri;

  fnr_get_float(&namelist_struct, "theta_grid_parameters", "tri", &pars->tri);
  tri=pars->tri;

  fnr_get_float(&namelist_struct, "theta_grid_parameters", "tripri", &pars->tripri);
  tripri=pars->tripri;
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "Rmaj", &(pars->rmaj));
  rmaj=pars->rmaj;
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "rhoc", &(pars->rhoc));
  rhoc=pars->rhoc;
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "shift", &(pars->shift));
  shift=pars->shift;
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "drhodpsi", &(pars->drhodpsi));
  drhodpsi=pars->drhodpsi;
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "epsl", &(pars->epsl));
  epsl=pars->epsl;
  
  fnr_get_float(&namelist_struct, "theta_grid_parameters", "kxfac", &(pars->kxfac));
  kxfac=pars->kxfac;

  
 

  fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "nx", &(grids->Nx));
  Nx=grids->Nx;
    
  fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "ny", &(grids->Ny));
  Ny=grids->Ny;
  
  fnr_get_float(&namelist_struct, "kt_grids_box_parameters", "y0", &(pars->y0));
  Y0=pars->y0;
  
  //before, jtwist_old assumed Zp=1
  //now, redefining jtwist = jtwist_old*Zp
  if(fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "jtwist", &(pars->jtwist)) == FNR_USED_DEFAULT) {
    //set default jtwist to 2pi*shat so that X0~Y0
    pars->jtwist = (int) round(2*M_PI*shat*Zp);
  }  
  jtwist=pars->jtwist;

  fnr_get_float(&namelist_struct, "kt_grids_box_parameters", "x0", &(pars->x0));
  X0=pars->x0;
  //X0 will get overwritten if shat!=0
  
  fnr_get_float(&namelist_struct, "dist_fn_knobs", "g_exb", &(pars->g_exb));
  g_exb=pars->g_exb;
  
  fnr_get_float(&namelist_struct, "parameters", "tite", &(pars->tite));
  tau=pars->tite;

  
  fnr_get_float(&namelist_struct, "nonlinear_terms_knobs", "cfl", &(pars->cfl));
  cfl=pars->cfl;
  
  fnr_get_int(&namelist_struct, "gs2_diagnostics_knobs", "nwrite", &(pars->nwrite));
  nwrite=pars->nwrite;
  
  fnr_get_int(&namelist_struct, "gs2_diagnostics_knobs", "nsave", &(pars->nsave));
  nsave=pars->nsave;
  
  fnr_get_int(&namelist_struct, "gs2_diagnostics_knobs", "navg", &(pars->navg));
  navg=pars->navg;
  
  fnr_get_float(&namelist_struct, "knobs", "delt", &(pars->dt));
  dt=pars->dt;
  
  //fnr_get_float(&namelist_struct, "knobs", "maxdt", &(pars->maxdt));
  //maxdt=pars->maxdt;
  
  fnr_get_int(&namelist_struct, "knobs", "nstep", &(pars->nstep));
  nSteps=pars->nstep;
  
  //maxdt?
  
  fnr_get_int(&namelist_struct, "species_knobs", "nspec", &(pars->nspec));
  nSpecies=pars->nspec;
	grids->Nspecies = pars->nspec;
  
  pars->linear = LINEAR = !get_bool_on_off(&namelist_struct, "nonlinear_terms_knobs", "nonlinear_mode");
  

  pars->check_for_restart = false;
  char* restart;
  fnr_get_string(&namelist_struct, "gryfx_knobs", "restart", &restart);
  if( strcmp(restart,"on") == 0) {
    pars->restart = RESTART = true;
  }
  else if( strcmp(restart,"exist") == 0) {
    //check if restart file exists
    pars->check_for_restart = CHECK_FOR_RESTART = true;
  }
  else if( strcmp(restart,"off") == 0) {
    pars->restart = RESTART = false;
  }
  
  //pars->zero_restart_avg = zero_restart_avg = get_bool(&namelist_struct, "gryfx_knobs", "zero_restart_avg");
  pars->zero_restart_avg = zero_restart_avg = get_bool_on_off(&namelist_struct, "gryfx_knobs", "zero_restart_avg");

  pars->no_zderiv_covering = NO_ZDERIV_COVERING = get_bool_on_off(&namelist_struct, "gryfx_knobs", "no_zderiv_covering");

  pars->no_zderiv= NO_ZDERIV= get_bool_on_off(&namelist_struct, "gryfx_knobs", "no_zderiv");

  pars->no_landau_damping= no_landau_damping = get_bool_on_off(&namelist_struct, "gryfx_knobs", "no_landau_damping");

  pars->turn_off_gradients_test= turn_off_gradients_test = get_bool_on_off(&namelist_struct, "gryfx_knobs", "turn_off_gradients_test");

  pars->slab = SLAB = get_bool_on_off(&namelist_struct, "gryfx_knobs", "slab");

  pars->const_curv = CONST_CURV = get_bool_on_off(&namelist_struct, "gryfx_knobs", "const_curv");

  pars->varenna= varenna = get_bool_on_off(&namelist_struct, "gryfx_knobs", "varenna");

  pars->new_catto= new_catto = get_bool_on_off(&namelist_struct, "gryfx_knobs", "new_catto");
  
 //////////////////////////
 // new varenna knobs

  pars->new_varenna= new_varenna = get_bool_on_off(&namelist_struct, "gryfx_knobs", "new_varenna");
  
  pars->new_varenna_fsa= new_varenna_fsa = get_bool_on_off(&namelist_struct, "new_varenna_knobs", "new_varenna_fsa");


  fnr_get_int(&namelist_struct, "new_varenna_knobs", "zonal_dens_switch", &pars->zonal_dens_switch);
  zonal_dens_switch=pars->zonal_dens_switch;

  if(fnr_get_int(&namelist_struct, "new_varenna_knobs", "q0_dens_switch", &q0_dens_switch)==FNR_USED_DEFAULT) q0_dens_switch = zonal_dens_switch;
  pars->q0_dens_switch = q0_dens_switch;

  qpar_gradpar_corrections = get_bool(&namelist_struct, "new_varenna_knobs", "qpar_gradpar_corrections"); 
  qperp_gradpar_corrections = get_bool(&namelist_struct, "new_varenna_knobs", "qperp_gradpar_corrections"); 
  qpar_bgrad_corrections = get_bool(&namelist_struct, "new_varenna_knobs", "qpar_bgrad_corrections"); 
  qperp_bgrad_corrections = get_bool(&namelist_struct, "new_varenna_knobs", "qperp_bgrad_corrections"); 
  qpar0_switch = get_bool(&namelist_struct, "new_varenna_knobs", "qpar0_switch"); 
  qprp0_switch = get_bool(&namelist_struct, "new_varenna_knobs", "qprp0_switch"); 

  /////////////////////////

  pars->nlpm= NLPM = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm");

  pars->dorland_nlpm= dorland_nlpm = get_bool_on_off(&namelist_struct, "gryfx_knobs", "dorland_nlpm");

  pars->dorland_nlpm_phase= dorland_nlpm_phase = get_bool_on_off(&namelist_struct, "gryfx_knobs", "dorland_nlpm_phase");

  pars->dorland_phase_complex= dorland_phase_complex = get_bool_on_off(&namelist_struct, "gryfx_knobs", "dorland_phase_complex");

  pars->nlpm_kxdep= nlpm_kxdep = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_kxdep");

  if(nlpm_kxdep) pars->dorland_phase_complex =  dorland_phase_complex = true;  

  pars->nlpm_nlps= nlpm_nlps = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_nlps");

  if(nlpm_nlps) pars->nlpm = NLPM = false; //turn off normal filter-style NLPM

  pars->nlpm_cutoff_avg= nlpm_cutoff_avg = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_cutoff_avg");

  fnr_get_int(&namelist_struct, "gryfx_knobs", "dorland_phase_ifac", &pars->dorland_phase_ifac);
  dorland_phase_ifac=pars->dorland_phase_ifac;
  fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "nlpm_option", &pars->nlpm_option);
  nlpm_option=pars->nlpm_option;

  fnr_get_int(&namelist_struct, "gryfx_knobs", "inlpm", &pars->inlpm);
  inlpm=pars->inlpm;
  fnr_get_float(&namelist_struct, "gryfx_knobs", "dnlpm", &pars->dnlpm);
  dnlpm=pars->dnlpm;
  
  fnr_get_float(&namelist_struct, "gryfx_knobs", "low_cutoff", &pars->low_cutoff);
  low_cutoff=pars->low_cutoff;
  fnr_get_float(&namelist_struct, "gryfx_knobs", "high_cutoff", &pars->high_cutoff);
  high_cutoff=pars->high_cutoff;

  if(high_cutoff == -1) {pars->high_cutoff = high_cutoff = low_cutoff; pars->low_cutoff = low_cutoff = 0.;} //backwards compatability for when only low cutoff specified

  fnr_get_float(&namelist_struct, "gryfx_knobs", "dnlpm_max", &pars->dnlpm_max);
  dnlpm_max=pars->dnlpm_max;
  fnr_get_float(&namelist_struct, "gryfx_knobs", "tau_nlpm", &pars->tau_nlpm);
  tau_nlpm=pars->tau_nlpm;

  pars->nlpm_zonal_kx1_only= nlpm_zonal_kx1_only = get_bool_on_off(&namelist_struct, "gryfx_knobs", "nlpm_zonal_kx1_only");

  

  // normalize cutoffs by Y0
  if(!dorland_nlpm && !nlpm_nlps) {
    pars->low_cutoff = low_cutoff = (float) low_cutoff / Y0;
    pars->high_cutoff = high_cutoff = (float) high_cutoff / Y0;
  }

  fnr_get_int(&namelist_struct, "gryfx_knobs", "ivarenna", &pars->ivarenna);
  ivarenna=pars->ivarenna;
  

  pars->varenna_fsa= varenna_fsa = get_bool_on_off(&namelist_struct, "gryfx_knobs", "varenna_fsa");

  
  fnr_get_int(&namelist_struct, "gryfx_knobs", "icovering", &pars->icovering);
  icovering=pars->icovering;

  fnr_get_int(&namelist_struct, "gryfx_knobs", "iphi00", &pars->iphi00);
  iphi00=pars->iphi00;

  pars->smagorinsky= SMAGORINSKY = get_bool_on_off(&namelist_struct, "gryfx_knobs", "smagorinsky");

  pars->hyper= HYPER = get_bool_on_off(&namelist_struct, "gryfx_knobs", "hyper");
  
  fnr_get_float(&namelist_struct, "gryfx_knobs", "D_hyper", &pars->D_hyper);
  D_hyper=pars->D_hyper;
  
  fnr_get_int(&namelist_struct, "gryfx_knobs", "p_hyper", &pars->p_hyper);
  p_hyper=pars->p_hyper;

  pars->iso_shear= isotropic_shear = get_bool_on_off(&namelist_struct, "gryfx_knobs", "iso_shear");

  pars->debug= DEBUG = get_bool_on_off(&namelist_struct, "gryfx_knobs", "debug");
  
  fnr_get_int(&namelist_struct, "gryfx_knobs", "igeo", &pars->igeo);
  igeo=pars->igeo;

  fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "geofile", &geoFileName);
  pars->geofilename = geoFileName;

  fnr_get_float(&namelist_struct, "gryfx_knobs", "shaping_ps", &pars->shaping_ps);
  shaping_ps=pars->shaping_ps;
  if(igeo == 0) pars->shaping_ps =  shaping_ps = 1.6;

  /// EGH As far as I can see below does nothing... if igeo != it does nothing
  // and if igeo ==0 then geometry must be s-alpha?
//  char* s_alpha;
//  s_alpha = (char*) malloc(sizeof(char)*4);
//  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "s_alpha", &s_alpha)) s_alpha="on";
//  if( strcmp(s_alpha,"on") == 0 && igeo==0) {
//    igeo = 0;
//  }
 
  char* initfield;
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
  else if (strcmp(initfield,"RH_eq") == 0) {
    init = RH_equilibrium;
    pars->new_varenna = new_varenna = true;
  } 
  pars->init = init;
  
  fnr_get_float(&namelist_struct, "gryfx_knobs", "init_amp", &(pars->init_amp));
  init_amp=pars->init_amp;
  fnr_get_float(&namelist_struct, "gryfx_knobs", "phiext", &pars->phiext);
  phiext=pars->phiext;
  
  pars->write_netcdf= get_bool(&namelist_struct, "gryfx_knobs", "write_netcdf");

  pars->write_omega= write_omega = get_bool_on_off(&namelist_struct, "gryfx_knobs", "write_omega");

  pars->write_phi= write_phi = get_bool_on_off(&namelist_struct, "gryfx_knobs", "write_phi");

  //if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "fluxfile", &fluxfileName)) fluxfileName="./scan/outputs/flux";
  
  //if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "stopfile", &stopfileName)) stopfileName="c.stop";
  
  
  //printf("stopfileName = %s\n", stopfileName);
  
  fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "scan_type", &pars->scan_type);
  scan_type=pars->scan_type;
  
  
  pars->secondary_test= secondary_test = get_bool_on_off(&namelist_struct, "secondary_test_knobs", "secondary_test");

  fnr_get_float(&namelist_struct, "secondary_test_knobs", "phi_test_real", &phi_test.x);
  fnr_get_float(&namelist_struct, "secondary_test_knobs", "phi_test_imag", &phi_test.y);

  if(secondary_test && LINEAR && phi_test.x > phi_test.y) init_amp = phi_test.x;
  if(secondary_test && LINEAR && phi_test.x < phi_test.y) init_amp = phi_test.y;
  pars->phi_test = phi_test;
  pars->init_amp = init_amp;

  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "densfac", &pars->NLdensfac);
  NLdensfac=pars->NLdensfac;
  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "uparfac", &pars->NLuparfac);
  NLuparfac=pars->NLuparfac;
  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "tparfac", &pars->NLtparfac);
  NLtparfac=pars->NLtparfac;
  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "tprpfac", &pars->NLtprpfac);
  NLtprpfac=pars->NLtprpfac;
  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "qparfac", &pars->NLqparfac);
  NLqparfac=pars->NLqparfac;
  fnr_get_float(&namelist_struct, "gryfx_nonlinear_terms_knobs", "qprpfac", &pars->NLqprpfac);
  NLqprpfac=pars->NLqprpfac;

  if(NLdensfac!=0) pars->NLdensfac = NLdensfac = 1.;
  if(NLuparfac!=0) pars->NLuparfac = NLuparfac = 1.;
  if(NLtparfac!=0) pars->NLtparfac = NLtparfac = 1.;
  if(NLtprpfac!=0) pars->NLtprpfac = NLtprpfac = 1.;
  if(NLqparfac!=0) pars->NLqparfac = NLqparfac = 1.;
  if(NLqprpfac!=0) pars->NLqprpfac = NLqprpfac = 1.;
   

  fnr_get_string_no_test(&namelist_struct, "secondary_test_knobs", "restartfile", &pars->secondary_test_restartfileName);
  secondary_test_restartfileName=pars->secondary_test_restartfileName;

  fnr_get_int(&namelist_struct, "gryfx_knobs", "scan_number", &pars->scan_number);
  scan_number=pars->scan_number;
    
  //char* collisions;
  fnr_get_string(&namelist_struct, "collisions_knobs", "collision_model", &(pars->collision_model));
  //collisions=pars->collision_model;
  
  pars->species = species = (specie*) malloc(sizeof(specie)*pars->nspec);
   
  for(int s=1; s<nSpecies+1; s++) {
//		printf("s= %d\n", s);


    char namelist[100];
    sprintf(namelist,"species_parameters_%d",s); 
    char* type;
    fnr_get_string(&namelist_struct, namelist, "type", &type); 
    
    if(strcmp(type,"ion") == 0) {
      fnr_get_float(&namelist_struct, namelist, "z", &species[ION].z);
      fnr_get_float(&namelist_struct, namelist, "mass", &species[ION].mass);
      fnr_get_float(&namelist_struct, namelist, "dens", &species[ION].dens);
      fnr_get_float(&namelist_struct, namelist, "temp", &species[ION].temp);
      fnr_get_float(&namelist_struct, namelist, "tprim", &species[ION].tprim); //6.9
      fnr_get_float(&namelist_struct, namelist, "fprim", &species[ION].fprim); //2.2
      if(fnr_get_float(&namelist_struct, namelist, "uprim", &species[ION].uprim)) species[ION].uprim=0;


     // if(strcmp(collisions,"none") == 0) species[ION].nu_ss = 0;
     // else {
        fnr_get_float(&namelist_struct, namelist, "vnewk", &species[ION].nu_ss);
     // }     

      strcpy(species[ION].type,"ion"); 

    }
    if(strcmp(type,"electron") == 0) {

      fnr_get_float(&namelist_struct, namelist, "z", &species[ELECTRON].z);
      fnr_get_float(&namelist_struct, namelist, "mass", &species[ELECTRON].mass);
      fnr_get_float(&namelist_struct, namelist, "dens", &species[ELECTRON].dens);
      fnr_get_float(&namelist_struct, namelist, "temp", &species[ELECTRON].temp);
      fnr_get_float(&namelist_struct, namelist, "tprim", &species[ELECTRON].tprim);
      fnr_get_float(&namelist_struct, namelist, "fprim", &species[ELECTRON].fprim);
      fnr_get_float(&namelist_struct, namelist, "uprim", &species[ELECTRON].uprim);

      //if(strcmp(collisions,"none") == 0) species[ELECTRON].nu_ss = 0;
      //else {

        fnr_get_float(&namelist_struct, namelist, "vnewk", &species[ELECTRON].nu_ss);

      //}

			strcpy(species[ELECTRON].type,"electron");
    }   
		//free(type);
		fnr_free(&namelist_struct);
		fnr_free(&namelist_defaults);

		printf("Got here!!\n");

  }


}


