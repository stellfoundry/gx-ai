#include "parameters.h"
#include "c_fortran_namelist3.c"
#include "inputs/namelist_defaults.c"

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

Parameters::Parameters() {
  initialized = false;

  // some cuda parameters (not from input file)
  cudaGetDeviceProperties(&prop, 0);
  maxThreadsPerBlock = prop.maxThreadsPerBlock;
}

Parameters::~Parameters() {
  cudaDeviceSynchronize();
  if(initialized) {
    cudaFree(species);
    cudaFreeHost(species_h);
  }
}

int Parameters::read_namelist(char* filename)
{
  strncpy(run_name, filename, strlen(filename)-3);

  fnr_abort_on_error = 1;
  fnr_abort_if_missing = 0;
  fnr_abort_if_no_default = 1;
  struct fnr_struct namelist_struct = fnr_read_namelist_file(filename);  
  struct fnr_struct namelist_defaults = fnr_read_namelist_string(default_namelist_string);  
  fnr_check_namelist_against_template(&namelist_struct, &namelist_defaults);
  fnr_set_defaults(&namelist_struct, &namelist_defaults);
  fnr_get_int(&namelist_struct, "gx", "nperiod", &nperiod);

  fnr_get_int(&namelist_struct, "gx", "ntheta", &(nz_in));

  if(nz_in!=1) {
    //The GS2 grid, which includes the periodic point, runs from
    // -ntgrid:ntgrid and is thus 2*ntgrid+1 in size, where ntgrid
    // is the same as what is calculated here. This little step
    // thus ensures that the Gryfx grid corresponds to the GS2 grid
    // (without the periodic) point.
      int ntgrid = nz_in/2 + (nperiod-1)*nz_in;
 
      nz_in = 2*ntgrid;
  }
  
  //if(fnr_get_int(&namelist_struct, "gx", "zp", &Zp)==FNR_USED_DEFAULT){
   Zp = 2*nperiod - 1; 
  //}
  
  fnr_get_float(&namelist_struct, "gx", "eps", &(eps));
  
  fnr_get_float(&namelist_struct, "gx", "shat", &(shat));

  fnr_get_float(&namelist_struct, "gx", "beta_prime_input", &beta_prime_input);

  fnr_get_float(&namelist_struct, "gx", "s_hat_input", &s_hat_input);
  
  fnr_get_float(&namelist_struct, "gx", "qinp", &(qsf));

  fnr_get_float(&namelist_struct, "gx", "akappa", &akappa);

  fnr_get_float(&namelist_struct, "gx", "akappri", &akappri);

  fnr_get_float(&namelist_struct, "gx", "tri", &tri);

  fnr_get_float(&namelist_struct, "gx", "tripri", &tripri);
  
  fnr_get_float(&namelist_struct, "gx", "Rmaj", &(rmaj));
  fnr_get_float(&namelist_struct, "gx", "rhoc", &(rhoc));
  
  fnr_get_float(&namelist_struct, "gx", "shift", &(shift));
  
  fnr_get_float(&namelist_struct, "gx", "drhodpsi", &(drhodpsi));
  
  // fnr_get_float(&namelist_struct, "theta_grid_parameters", "epsl", &(epsl));
  
  fnr_get_float(&namelist_struct, "gx", "kxfac", &(kxfac)); // BD: why is this here?

  fnr_get_int(&namelist_struct, "gx", "nx", &(nx_in));
    
  fnr_get_int(&namelist_struct, "gx", "ny", &(ny_in));
  
  fnr_get_float(&namelist_struct, "gx", "y0", &(y0));
  
  //before, jtwist_old assumed Zp=1
  //now, redefining jtwist = jtwist_old*Zp
  if(fnr_get_int(&namelist_struct, "gx", "jtwist", &(jtwist)) == FNR_USED_DEFAULT) {
    //set default jtwist to 2pi*shat so that X0~Y0
    jtwist = (int) round(2*M_PI*shat*Zp);
  }  

  fnr_get_float(&namelist_struct, "gx", "x0", &(x0));
  //X0 will get overwritten if shat!=0

  // set up for LH moments "grid"
  fnr_get_int(&namelist_struct, "gx", "nhermite", &(nm_in));
  fnr_get_int(&namelist_struct, "gx", "nlaguerre", &nl_in);

  char* clos_str;
  fnr_get_string(&namelist_struct, "gx", "closure_model", &clos_str);
  if(strcmp(clos_str, "beer4+2")==0) {
    printf("\nUsing Beer 4+2 closure model. Overriding nm=4, nl=2\n\n");
    nm_in = 4;
    nl_in = 2;
    closure_model = BEER42;
  }
  else if (strcmp(clos_str, "smith_perp")==0) {
    closure_model = SMITHPERP;
    // hard code this for now
    fnr_get_int(&namelist_struct, "gx", "smith_perp_q", &smith_perp_q);
  } else if (strcmp(clos_str, "smith_par")==0) {
      closure_model = SMITHPAR;
      fnr_get_int(&namelist_struct, "gx", "smith_par_q", &smith_par_q);
  }

  // set up for forcing variables
  
  forcing_init = get_bool_on_off(&namelist_struct, "gx", "forcing_init");
  fnr_get_string(&namelist_struct, "gx", "forcing_type", &forcing_type);
  fnr_get_float(&namelist_struct, "gx", "forcing_amp", &forcing_amp);
  fnr_get_int(&namelist_struct, "gx", "forcing_index", &forcing_index);

  // fnr_get_float(&namelist_struct, "dist_fn_knobs", "g_exb", &(g_exb));

  char* boundary_opt_str;
  fnr_get_string(&namelist_struct, "gx", "boundary_option", &boundary_opt_str);
  if( strcmp(boundary_opt_str,"periodic") == 0) {
    boundary_option_periodic = true;
  } else {
    boundary_option_periodic = false;
  }
  
  fnr_get_float(&namelist_struct, "gx", "tite", &(ti_ov_te));
  // ti_ov_te is overwritten for non-adiabatic electrons

  fnr_get_float(&namelist_struct, "gx", "beta", &(beta));
  
  fnr_get_float(&namelist_struct, "gx", "cfl", &(cfl));
  
  zero_order_nonlin_flr_only= get_bool(&namelist_struct, "gx", "zero_order_nonlin_flr_only");
  if(zero_order_nonlin_flr_only) no_nonlin_flr = true;

  no_nonlin_cross_terms= get_bool(&namelist_struct, "gx", "no_nonlin_cross_terms");
  no_nonlin_dens_cross_term= get_bool(&namelist_struct, "gx", "no_nonlin_dens_cross_term");
  if(no_nonlin_cross_terms || no_nonlin_dens_cross_term) {
    new_nlpm = false;
    zero_order_nonlin_flr_only = false;
    no_nonlin_flr = false;
  }

  no_zonal_nlpm= get_bool(&namelist_struct, "gx", "no_zonal_nlpm");

  //  namelistRead(&nwrite, &nsave, &navg);

  fnr_get_int(&namelist_struct, "gx", "nwrite", &(nwrite));
  
  fnr_get_int(&namelist_struct, "gx", "nsave", &(nsave));
  
  fnr_get_int(&namelist_struct, "gx", "navg", &(navg));
  
  fnr_get_float(&namelist_struct, "gx", "delt", &(dt));
  
  //fnr_get_float(&namelist_struct, "knobs", "maxdt", &(maxdt));
  //maxdt=maxdt;
  
  fnr_get_int(&namelist_struct, "gx", "nstep", &(nstep));
  
  fnr_get_float(&namelist_struct, "gx", "fapar", &(fapar));

  fnr_get_float(&namelist_struct, "gx", "avail_cpu_time", &(avail_cpu_time));
  fnr_get_float(&namelist_struct, "gx", "margin_cpu_time", &(margin_cpu_time));
  //maxdt?
  
  linear = !get_bool_on_off(&namelist_struct, "gx", "nonlinear_mode");
  printf(linear ? "linear is true" : "linear is false");


  check_for_restart = false;
  char* restart_str;
  fnr_get_string(&namelist_struct, "gx", "restart", &restart_str);
  if( strcmp(restart_str,"on") == 0) {
    restart = true;
  }
  else if( strcmp(restart_str,"exist") == 0) {
    //check if restart file exists
    check_for_restart = true;
  }
  else if( strcmp(restart_str,"off") == 0) {
    restart = false;
  }

  // netcdf_restart = get_bool(&namelist_struct, "gx", "netcdf_restart"); 

  // If a netcdf file with the right run name already exists, open it and append to it
  // append_old = get_bool(&namelist_struct, "gx", "append_old"); 
  
  //zero_restart_avg = zero_restart_avg = get_bool(&namelist_struct, "gx", "zero_restart_avg");
  // zero_restart_avg = get_bool_on_off(&namelist_struct, "gx", "zero_restart_avg");

  // no_zderiv_covering = get_bool_on_off(&namelist_struct, "gx", "no_zderiv_covering");

  // no_zderiv = get_bool_on_off(&namelist_struct, "gx", "no_zderiv");

  // zderiv_loop = get_bool_on_off(&namelist_struct, "gx", "zderiv_loop");

  // no_landau_damping = get_bool_on_off(&namelist_struct, "gx", "no_landau_damping");

  // turn_off_gradients_test = get_bool_on_off(&namelist_struct, "gx", "turn_off_gradients_test");

  slab = get_bool_on_off(&namelist_struct, "gx", "slab");

  const_curv = get_bool_on_off(&namelist_struct, "gx", "const_curv");

  // varenna = get_bool_on_off(&namelist_struct, "gx", "varenna");

  // new_catto = get_bool_on_off(&namelist_struct, "gx", "new_catto");
  
 //////////////////////////
 // new varenna knobs

  // new_varenna = get_bool_on_off(&namelist_struct, "gx", "new_varenna");
  
  // new_varenna_fsa = get_bool_on_off(&namelist_struct, "new_varenna_knobs", "new_varenna_fsa");

  /////////////////////////

  // nlpm = get_bool_on_off(&namelist_struct, "gx", "nlpm");

  // dorland_nlpm = get_bool_on_off(&namelist_struct, "gx", "dorland_nlpm");

  // dorland_nlpm_phase = get_bool_on_off(&namelist_struct, "gx", "dorland_nlpm_phase");

  // dorland_phase_complex = get_bool_on_off(&namelist_struct, "gx", "dorland_phase_complex");

  // nlpm_kxdep = get_bool_on_off(&namelist_struct, "gx", "nlpm_kxdep");

  // if(nlpm_kxdep) dorland_phase_complex = true;  

  // nlpm_nlps = get_bool_on_off(&namelist_struct, "gx", "nlpm_nlps");

  // if(nlpm_nlps) nlpm = false; //turn off normal filter-style NLPM

  // nlpm_cutoff_avg = get_bool_on_off(&namelist_struct, "gx", "nlpm_cutoff_avg");

  // fnr_get_int(&namelist_struct, "gx", "dorland_phase_ifac", &dorland_phase_ifac);
  // fnr_get_string_no_test(&namelist_struct, "gx", "nlpm_option", &nlpm_option);

  // fnr_get_int(&namelist_struct, "gx", "inlpm", &inlpm);
  // fnr_get_float(&namelist_struct, "gx", "dnlpm", &dnlpm);
  // fnr_get_float(&namelist_struct, "gx", "dnlpm_dens", &dnlpm_dens);
  // fnr_get_float(&namelist_struct, "gx", "dnlpm_tprp", &dnlpm_tprp);

  // fnr_get_float(&namelist_struct, "gx", "dnlpm_max", &dnlpm_max);
  // fnr_get_float(&namelist_struct, "gx", "tau_nlpm", &tau_nlpm);

  // nlpm_zonal_kx1_only = get_bool_on_off(&namelist_struct, "gx", "nlpm_zonal_kx1_only");

  // fnr_get_int(&namelist_struct, "gx", "ivarenna", &ivarenna);

  // varenna_fsa = get_bool_on_off(&namelist_struct, "gx", "varenna_fsa");

  // fnr_get_int(&namelist_struct, "gx", "icovering", &icovering);

  fnr_get_int(&namelist_struct, "gx", "iphi00", &iphi00);

  // smagorinsky = get_bool_on_off(&namelist_struct, "gx", "smagorinsky");

  // hyper = get_bool_on_off(&namelist_struct, "gx", "hyper");
  
  // fnr_get_float(&namelist_struct, "gx", "d_hyper", &D_hyper);
  
  // fnr_get_int(&namelist_struct, "gx", "p_hyper", &p_hyper);

  // iso_shear = get_bool_on_off(&namelist_struct, "gx", "iso_shear");

  debug = get_bool_on_off(&namelist_struct, "gx", "debug");
  
  fnr_get_int(&namelist_struct, "gx", "igeo", &igeo);

  if(qsf<0 && nz_in==1) local_limit=true;
  else local_limit = false;

  // mfm
  fnr_get_string_no_test(&namelist_struct, "gx", "geofile", &geofilename);

  char* initfield;
  //initfield = (char*) malloc(sizeof(char)*10);
  fnr_get_string(&namelist_struct, "gx", "init", &initfield);
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
    new_varenna = true;
  } 
  
  fnr_get_float(&namelist_struct, "gx", "init_amp", &(init_amp));

  init_single = get_bool_on_off(&namelist_struct, "gx", "init_single");
  fnr_get_int(&namelist_struct, "gx", "iky_single", &(iky_single));
  fnr_get_int(&namelist_struct, "gx", "ikx_single", &(ikx_single));
  
  fnr_get_float(&namelist_struct, "gx", "kpar_init", &(kpar_init));
  
  write_netcdf= get_bool(&namelist_struct, "gx", "write_netcdf");

  write_omega = get_bool_on_off(&namelist_struct, "gx", "write_omega");

  write_phi = get_bool_on_off(&namelist_struct, "gx", "write_phi");
  
  write_hermite_energy_spectrum = get_bool_on_off(&namelist_struct, "gx", "write_hermite_energy_spectrum");

  fnr_get_int(&namelist_struct, "gx", "hermite_spectrum_avg_cutoff", &(hermite_spectrum_avg_cutoff));

  // fnr_get_string_no_test(&namelist_struct, "gx", "scan_type", &scan_type);
  
  // higher_order_moments = get_bool_on_off(&namelist_struct, "gx", "higher_order_moments");
  
  // secondary_test = get_bool_on_off(&namelist_struct, "secondary_test_knobs", "secondary_test");
  // nlpm_test = get_bool_on_off(&namelist_struct, "secondary_test_knobs", "nlpm_test");
  // new_nlpm = get_bool_on_off(&namelist_struct, "gx", "new_nlpm");
  // nlpm_abs_sgn = get_bool_on_off(&namelist_struct, "gx", "nlpm_abs_sgn");
  // hammett_nlpm_interference = get_bool_on_off(&namelist_struct, "gx", "hammett_nlpm_interference");
  // low_b = get_bool_on_off(&namelist_struct, "secondary_test_knobs", "low_b");
  // low_b_all = get_bool_on_off(&namelist_struct, "gx", "low_b_all");
  // if(low_b_all) low_b = true;
  // nlpm_zonal_only = get_bool_on_off(&namelist_struct, "gx", "nlpm_zonal_only");
  // nlpm_vol_avg = get_bool_on_off(&namelist_struct, "gx", "nlpm_vol_avg");

  // fnr_get_int(&namelist_struct, "gx", "iflr", &(iflr));

  char* scheme_str;
  fnr_get_string(&namelist_struct, "gx", "scheme", &scheme_str);
  if( strcmp(scheme_str,"rk4") == 0) {
    scheme = RK4;
  } else {
    scheme = RK2;
  }

  char* source_str;
  fnr_get_string(&namelist_struct, "gx", "source_option", &source_str);
  if( strcmp(source_str,"phiext_full")==0 ) {
    source_option = PHIEXT;
    printf("Running Rosenbluth-Hinton zonal flow calculation\n");
  }
  fnr_get_float(&namelist_struct, "gx", "phi_ext", &phiext);

  // cuComplex phi_test;
  // fnr_get_float(&namelist_struct, "secondary_test_knobs", "phi_test_real", &phi_test.x);
  // fnr_get_float(&namelist_struct, "secondary_test_knobs", "phi_test_imag", &phi_test.y);

  // fnr_get_int(&namelist_struct, "secondary_test_knobs", "iky_fixed", &(iky_fixed));
  // fnr_get_int(&namelist_struct, "secondary_test_knobs", "ikx_fixed", &(ikx_fixed));

  // if(secondary_test && linear && phi_test.x > phi_test.y) init_amp = phi_test.x;
  // if(secondary_test && linear && phi_test.x < phi_test.y) init_amp = phi_test.y;
  // phi_test = phi_test;

  /*
  if(nlpm_test) { 
    float fac;
    if(iky_fixed==0) fac = .5;
    else fac = 1.;
    if(phi_test.x > 0.) phi_test.x = x0*y0*fac; 
    else phi_test.x = 0.;
    if(phi_test.y > 0.) phi_test.y = x0*y0*fac; 
    else phi_test.y = 0.;
    //if(phi_test.x > 0. && phi_test.y > 0.) {phi_test.x = phi_test.x/sqrt(2.); phi_test.y = phi_test.y/sqrt(2.);}
  }
  */

  // fnr_get_float(&namelist_struct, "gx_nonlinear_terms_knobs", "densfac", &NLdensfac);
  // fnr_get_float(&namelist_struct, "gx_nonlinear_terms_knobs", "uparfac", &NLuparfac);
  // fnr_get_float(&namelist_struct, "gx_nonlinear_terms_knobs", "tparfac", &NLtparfac);
  // fnr_get_float(&namelist_struct, "gx_nonlinear_terms_knobs", "tprpfac", &NLtprpfac);
  // fnr_get_float(&namelist_struct, "gx_nonlinear_terms_knobs", "qparfac", &NLqparfac);
  // fnr_get_float(&namelist_struct, "gx_nonlinear_terms_knobs", "qprpfac", &NLqprpfac);

  // if(NLdensfac!=0) NLdensfac = 1.;
  // if(NLuparfac!=0) NLuparfac = 1.;
  // if(NLtparfac!=0) NLtparfac = 1.;
  // if(NLtprpfac!=0) NLtprpfac = 1.;
  // if(NLqparfac!=0) NLqparfac = 1.;
  // if(NLqprpfac!=0) NLqprpfac = 1.;
   

  // fnr_get_string_no_test(&namelist_struct, "secondary_test_knobs", "restartfile", &secondary_test_restartfileName);

  // fnr_get_int(&namelist_struct, "gx", "scan_number", &scan_number);
    
  // char* collisions;
  // fnr_get_string(&namelist_struct, "collisions_knobs", "collision_model", &(collision_model));
  // collisions=collision_model;

  hypercollisions = get_bool(&namelist_struct, "gx", "hypercollisions");
  if(hypercollisions && debug) printf("Using hypercollisions.\n");
  fnr_get_float(&namelist_struct, "gx", "nu_hyper_l", &nu_hyper_l);
  fnr_get_float(&namelist_struct, "gx", "nu_hyper_m", &nu_hyper_m);
  fnr_get_int(&namelist_struct, "gx", "p_hyper_l", &p_hyper_l);
  fnr_get_int(&namelist_struct, "gx", "p_hyper_m", &p_hyper_m);
  
  adiabatic_electrons = true;

  snyder_electrons = get_bool_on_off(&namelist_struct, "gx", "snyder_electrons");
  // stationary_ions = get_bool_on_off(&namelist_struct, "gx", "stationary_ions");
   
  fnr_get_int(&namelist_struct, "gx", "nspec", &(nspec_in));

  cudaMalloc((void**) &species, sizeof(specie)*nspec_in);
  cudaMallocHost((void**) &species_h, sizeof(specie)*nspec_in);

  int ionspec = 0;   
  int ispec = 1;
  float mass;
  bool main_ion_species_found = false;
  for(int s=1; s<nspec_in+1; s++) {
    
    char namelist[100];
    sprintf(namelist,"species_parameters_%d",s); 
    char* type;
    fnr_get_string(&namelist_struct, namelist, "type", &type); 
 
    if(strcmp(type,"ion") == 0) {
      fnr_get_float(&namelist_struct, namelist, "mass", &mass);

      if((mass == 1. && !main_ion_species_found) || nspec_in==1) {
	ionspec=0; main_ion_species_found=true; // main ion species mass assumed to be 1. main ion species indexed 0.
      } else {
	ionspec = ispec; ispec++;
      }

      species_h[ionspec].mass = mass;
      fnr_get_float(&namelist_struct, namelist, "z", &species_h[ionspec].z);
      fnr_get_float(&namelist_struct, namelist, "dens", &species_h[ionspec].dens);
      fnr_get_float(&namelist_struct, namelist, "temp", &species_h[ionspec].temp);
      fnr_get_float(&namelist_struct, namelist, "tprim", &species_h[ionspec].tprim); //6.9
      fnr_get_float(&namelist_struct, namelist, "fprim", &species_h[ionspec].fprim); //2.2
      if(fnr_get_float(&namelist_struct, namelist, "uprim", &species_h[ionspec].uprim)) species_h[ionspec].uprim=0;


     // if(strcmp(collisions,"none") == 0) species_h[ionspec].nu_ss = 0;
     // else {
        fnr_get_float(&namelist_struct, namelist, "vnewk", &species_h[ionspec].nu_ss);
     // }     

      //strcpy(species_h[ionspec].type,"ion"); 

    }
    if(strcmp(type,"electron") == 0) {

      // kinetic electrons will always be last indexed species

      fnr_get_float(&namelist_struct, namelist, "z", &species_h[nspec_in-1].z);
      fnr_get_float(&namelist_struct, namelist, "mass", &species_h[nspec_in-1].mass);
      fnr_get_float(&namelist_struct, namelist, "dens", &species_h[nspec_in-1].dens);
      fnr_get_float(&namelist_struct, namelist, "temp", &species_h[nspec_in-1].temp);
      fnr_get_float(&namelist_struct, namelist, "tprim", &species_h[nspec_in-1].tprim);
      fnr_get_float(&namelist_struct, namelist, "fprim", &species_h[nspec_in-1].fprim);
      fnr_get_float(&namelist_struct, namelist, "uprim", &species_h[nspec_in-1].uprim);

      //if(strcmp(collisions,"none") == 0) species_h[nspec_in-1].nu_ss = 0;
      //else {

      fnr_get_float(&namelist_struct, namelist, "vnewk", &species_h[nspec_in-1].nu_ss);

      //}

      //strcpy(species_h[nspec_in-1].type,"electron");

      adiabatic_electrons = false;
   
    }   


  }
  cudaDeviceSynchronize();
  init_species(species_h);
  CP_TO_GPU (species, species_h, sizeof(specie)*nspec_in);

  //free(type);
  fnr_free(&namelist_struct);
  fnr_free(&namelist_defaults);

  initialized = true;
  cudaDeviceSynchronize();
  return 0;
}

void Parameters::init_species(specie* species)
{
  for(int s=0; s<nspec_in; s++) {
    species[s].vt = sqrt(species[s].temp/species[s].mass);
    species[s].zstm = species[s].z/sqrt(species[s].temp*species[s].mass);
    species[s].tz = species[s].temp/species[s].z;
    species[s].zt = species[s].z/species[s].temp;
    species[s].rho = sqrt(species[s].temp*species[s].mass)/species[s].z;
    species[s].rho2 = species[s].rho*species[s].rho;
  }
}

// this function copies elements of parameters object into external_parameters_struct externalpars
int Parameters::set_externalpars(external_parameters_struct* externalpars) {
    externalpars->equilibrium_type = equilibrium_type;

    //Defaults if we are not using Trinity
    externalpars->trinity_timestep = -1;
    externalpars->trinity_iteration = -1;
    externalpars->trinity_conv_count = -1;

    if (restart) externalpars->restart  = 1;
    else externalpars->restart = 0;

    externalpars->nstep = nstep;
    externalpars->navg = navg;
    // We increase the margin_cpu_time to make it stricter than gs2
    externalpars->end_time = time(NULL) + avail_cpu_time - margin_cpu_time*1.2;

    externalpars->irho = irho;
    externalpars->rhoc = rhoc;
    externalpars->eps = eps;
    externalpars->bishop = bishop;
    externalpars->nperiod = nperiod;
    externalpars->ntheta = nz_in;
    if(debug) printf("nz_in is %d\n", externalpars->ntheta);
  
   /* Miller parameters*/
    externalpars->rgeo_local = rmaj;
    externalpars->rgeo_lcfs = rmaj;
    externalpars->akappa = akappa;
    externalpars->akappri = akappri;
    externalpars->tri = tri;
    externalpars->tripri = tripri;
    externalpars->shift = shift;
    externalpars->qinp = qsf;
    externalpars->shat = shat;
    // EGH These appear to be redundant
    //externalpars->asym = asym;
    //externalpars->asympri = asympri;
  
    /* Other geometry parameters - Bishop/Greene & Chance*/
    externalpars->beta_prime_input = beta_prime_input;
    externalpars->s_hat_input = s_hat_input;
  
    /*Flow shear*/
    externalpars->g_exb = g_exb;
  
    /* Species parameters... I think allowing 20 species should be enough!*/
  
    externalpars->ntspec = nspec_in;
  
    for (int i=0;i<nspec_in;i++){
  	  externalpars->dens[i] = species_h[i].dens;
  	  externalpars->temp[i] = species_h[i].temp;
  	  externalpars->fprim[i] = species_h[i].fprim;
  	  externalpars->tprim[i] = species_h[i].tprim;
  	  externalpars->nu[i] = species_h[i].nu_ss;
    }
  return 0;
}

// this function copies elements of external_parameters_struct externalpars into parameters object
int Parameters::import_externalpars(external_parameters_struct* externalpars) {
   equilibrium_type = externalpars->equilibrium_type ;
   if (externalpars->restart==1) restart  = true;
   else if (externalpars->restart==2){
     restart  = true;
     zero_restart_avg = true;
   }
   else restart = false;

   if (externalpars->nstep > nstep) {
     printf("ERROR: nstep has been increased above the default value. nstep must be less than or equal to what is in the input file");
     abort();
   }
   trinity_timestep = externalpars->trinity_timestep;
   trinity_iteration = externalpars->trinity_iteration;
   trinity_conv_count = externalpars->trinity_conv_count;
   nstep = externalpars->nstep;
   navg = externalpars->navg;
   end_time = externalpars->end_time;
  /*char eqfile[800];*/
   irho = externalpars->irho ;
   rhoc = externalpars->rhoc ;
   eps = externalpars->eps;
   // NB NEED TO SET EPS IN TRINITY!!!
   //eps = rhoc/rmaj;
   bishop = externalpars->bishop ;
   nperiod = externalpars->nperiod ;
   if(debug) printf("nperiod2 is %d\n", nperiod);
   nz_in = externalpars->ntheta ;

 /* Miller parameters*/
   rmaj = externalpars->rgeo_local ;
   r_geo = externalpars->rgeo_lcfs ;
   akappa  = externalpars->akappa ;
   akappri = externalpars->akappri ;
   tri = externalpars->tri ;
   tripri = externalpars->tripri ;
   shift = externalpars->shift ;
   qsf = externalpars->qinp ;
   shat = externalpars->shat ;
    // EGH These appear to be redundant
   //asym = externalpars->asym ;
   //asympri = externalpars->asympri ;

  /* Other geometry parameters - Bishop/Greene & Chance*/
   beta_prime_input = externalpars->beta_prime_input ;
   s_hat_input = externalpars->s_hat_input ;

  /*Flow shear*/
   g_exb = externalpars->g_exb ;

  /* Species parameters... I think allowing 20 species should be enough!*/
  int oldnSpecies = nspec_in;
   nspec_in = externalpars->ntspec ;

  if (nspec_in!=oldnSpecies){
	  printf("oldnSpecies=%d,  nSpecies=%d\n", oldnSpecies, nspec_in);
	  printf("Number of species set in get_fluxes must equal number of species in gx input file\n");
	  exit(1);
  }
  if(debug) printf("nSpecies was set to %d\n", nspec_in);
  for (int i=0;i<nspec_in;i++){
	   species_h[i].dens = externalpars->dens[i] ;
	   species_h[i].temp = externalpars->temp[i] ;
	   species_h[i].fprim = externalpars->fprim[i] ;
	   species_h[i].tprim = externalpars->tprim[i] ;
	   species_h[i].nu_ss = externalpars->nu[i] ;
  }
  CP_TO_GPU(species, species_h, sizeof(specie)*nspec_in);

  //jtwist should never be < 0. If we set jtwist < 0 in the input file,
  // this triggers the use of jtwist_square... i.e. jtwist is 
  // set to what it needs to make the box square at the outboard midplane
  if (jtwist < 0) {
    int jtwist_square, jtwist;
    // determine value of jtwist needed to make X0~Y0
    jtwist_square = (int) round(2*M_PI*abs(shat)*Zp);
    if (jtwist_square == 0) jtwist_square = 1;
    // as currently implemented, there is no way to manually set jtwist from input file
    // there could be some switch here where we choose whether to use
    // jtwist_in or jtwist_square
    jtwist = jtwist_square*2;
    //else use what is set in input file 
    jtwist = jtwist;
  }
  if(jtwist!=0 && abs(shat)>1.e-6) x0 = y0*jtwist/(2*M_PI*Zp*abs(shat));  
  //if(abs(shat)<1.e-6) x0 = y0;
  
  return 0;
}







