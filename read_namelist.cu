void read_namelist(char* filename)
{

//	printf("fnr_read_namelist_file starting\n");
  struct fnr_struct namelist_struct = fnr_read_namelist_file(filename);  
//	printf("fnr_read_namelist_file ended\n");
    
  if(fnr_get_int(&namelist_struct, "theta_grid_parameters", "ntheta", &Nz)) *&Nz = 32;
  
  if(fnr_get_float(&namelist_struct, "theta_grid_parameters", "eps", &eps)) eps = 0.2;
  
  if(fnr_get_float(&namelist_struct, "theta_grid_parameters", "shat", &shat)) shat = 0.8;
  
  if(fnr_get_float(&namelist_struct, "theta_grid_parameters", "qinp", &qsf)) qsf = 1.4;
  
  if(fnr_get_float(&namelist_struct, "theta_grid_parameters", "Rmaj", &rmaj)) rmaj = 1.0;
  
  if(fnr_get_float(&namelist_struct, "theta_grid_parameters", "shift", &shift)) shift = 0.;
  
  if(fnr_get_float(&namelist_struct, "theta_grid_parameters", "drhodpsi", &drhodpsi)) drhodpsi = 1.0;
  
  if(fnr_get_float(&namelist_struct, "theta_grid_parameters", "epsl", &epsl)) epsl = 2.0;
  
  if(fnr_get_float(&namelist_struct, "theta_grid_parameters", "kxfac", &kxfac)) kxfac = 1.0;
  
  if(fnr_get_int(&namelist_struct, "theta_grid_parameters", "nperiod", &nperiod)) nperiod=1;  
  
  if(fnr_get_int(&namelist_struct, "theta_grid_parameters", "zp", &Zp)) {
   *&Zp = 2*nperiod - 1;
  }

  if(fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "nx", &Nx)) *&Nx=16;
    
  if(fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "ny", &Ny)) *&Ny=16;
  
  if(fnr_get_float(&namelist_struct, "kt_grids_box_parameters", "y0", &Y0)) *&Y0=10;
  
  //before, jtwist_old assumed Zp=1
  //now, redefining jtwist = jtwist_old*Zp
  if(fnr_get_int(&namelist_struct, "kt_grids_box_parameters", "jtwist", &jtwist)) {
    //set default jtwist to 2pi*shat so that X0~Y0
    jtwist = (int) round(2*M_PI*shat*Zp);
  }  
 
  if(fnr_get_float(&namelist_struct, "kt_grids_box_parameters", "x0", &X0)) *&X0=10;
  //X0 will get overwritten if shat!=0
  
  if(fnr_get_float(&namelist_struct, "dist_fn_knobs", "g_exb", &g_exb)) g_exb=0;
  
  if(fnr_get_float(&namelist_struct, "parameters", "TiTe", &tau)) tau = 1.0;

  
  if(fnr_get_float(&namelist_struct, "nonlinear_terms_knobs", "cfl", &cfl)) cfl=.1;
  
  if(fnr_get_int(&namelist_struct, "gs2_diagnostics_knobs", "nwrite", &nwrite)) nwrite=10;
  
  if(fnr_get_int(&namelist_struct, "gs2_diagnostics_knobs", "nsave", &nsave)) nsave=5000;
  
  if(fnr_get_int(&namelist_struct, "gs2_diagnostics_knobs", "navg", &navg)) navg=100;
  
  if(fnr_get_float(&namelist_struct, "knobs", "dt", &dt)) dt = .02;
  
  if(fnr_get_float(&namelist_struct, "knobs", "maxdt", &maxdt)) maxdt = .02;
  
  if(fnr_get_int(&namelist_struct, "knobs", "nstep", &nSteps)) nSteps = 10000;
  
  //maxdt?
  
  if(fnr_get_int(&namelist_struct, "species_knobs", "nspec", &nSpecies)) nSpecies=1;
  
  char* nonlinear;
  nonlinear = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "nonlinear_terms_knobs", "nonlinear_mode", &nonlinear)) nonlinear="off";
  if( strcmp(nonlinear,"on") == 0) {
    LINEAR = false;
  }
  else if( strcmp(nonlinear,"off") == 0) {
    LINEAR = true;
  }
  
  char* restart;
  restart = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "restart", &restart)) restart="off";
  if( strcmp(restart,"on") == 0) {
    RESTART = true;
  }
  else if( strcmp(restart,"exist") == 0) {
    //check if restart file exists
    CHECK_FOR_RESTART = true;
  }
  else if( strcmp(restart,"off") == 0) {
    RESTART = false;
  }
  
  char* zero_avg;
  zero_avg = (char* ) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "zero_restart_avg", &zero_avg)) zero_avg="off";
  if( strcmp(zero_avg, "on") == 0 ) {
    zero_restart_avg = true;
  }
  else if( strcmp(zero_avg, "off") == 0) {
    zero_restart_avg = false;
  }
  
  char* no_zderiv_covering;
  no_zderiv_covering = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "no_zderiv_covering", &no_zderiv_covering)) no_zderiv_covering="off";
  if( strcmp(no_zderiv_covering,"on") == 0) {
    NO_ZDERIV_COVERING = true;
  }
  else if( strcmp(no_zderiv_covering,"off") == 0) {
    NO_ZDERIV_COVERING = false;
  }
  
  char* no_omegad;
  no_omegad = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "no_omegad", &no_omegad)) no_omegad="off";
  if( strcmp(no_omegad,"on") == 0) {
    NO_OMEGAD = true;
  }
  else if( strcmp(no_omegad,"off") == 0) {
    NO_OMEGAD = false;
  }
  
  char* const_curv;
  const_curv = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "const_curv", &const_curv)) const_curv="off";
  if( strcmp(const_curv,"on") == 0) {
    CONST_CURV = true;
  }
  else if( strcmp(const_curv,"off") == 0) {
    CONST_CURV = false;
  }
  
  char* varenna_flag;
  varenna_flag = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "varenna", &varenna_flag)) varenna_flag="off";
  if( strcmp(varenna_flag,"on") == 0) {
    varenna = true;
  }
  else if( strcmp(varenna_flag,"off") == 0) {
    varenna = false;
  }
  
  char* nlpm;
  nlpm = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "nlpm", &nlpm)) nlpm="off";
  if( strcmp(nlpm,"on") == 0) {
    NLPM = true;
  }
  else if( strcmp(nlpm,"off") == 0) {
    NLPM = false;
  }

  if(fnr_get_int(&namelist_struct, "gryfx_knobs", "inlpm", &inlpm)) inlpm = 2;
  if(fnr_get_float(&namelist_struct, "gryfx_knobs", "dnlpm", &dnlpm)) dnlpm = 1.;
  
  char* smagorinsky;
  smagorinsky = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "smagorinsky", &smagorinsky)) smagorinsky="off";
  if( strcmp(smagorinsky,"on") == 0) {
    SMAGORINSKY = true;
  }
  else if( strcmp(smagorinsky,"off") == 0) {
    SMAGORINSKY = false;
  }
  
  char* hyper;
  hyper = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "hyper", &hyper)) hyper="off";
  if( strcmp(hyper,"on") == 0) {
    HYPER = true;
  }
  else if( strcmp(hyper,"off") == 0) {
    HYPER = false;
  }
  
  if(fnr_get_float(&namelist_struct, "gryfx_knobs", "nu_hyper", &nu_hyper)) nu_hyper = 1.;
  
  if(fnr_get_int(&namelist_struct, "gryfx_knobs", "p_hyper", &p_hyper)) p_hyper = 2;
  
  char* debug;
  debug = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "debug", &debug)) debug="off";
  if( strcmp(debug,"on") == 0) {
    DEBUG = true;
  }
  else if( strcmp(debug,"off") == 0) {
    DEBUG = false;
  }
  
  char* s_alpha;
  s_alpha = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "s_alpha", &s_alpha)) s_alpha="on";
  if( strcmp(s_alpha,"on") == 0) {
    S_ALPHA = true;
  }
  else if( strcmp(s_alpha,"off") == 0) {
    S_ALPHA = false;
  }
  
  char* initfield;
  initfield = (char*) malloc(sizeof(char)*10);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "init", &initfield)) initfield="density";
  if( strcmp(initfield,"density") == 0) {
    init = DENS;
  }
  else if( strcmp(initfield,"phi") == 0) {
    init = PHI;
  }
  
  if(fnr_get_float(&namelist_struct, "gryfx_knobs", "init_amp", &init_amp)) init_amp = 1.e-5;
  
  char* write_omega_flag;
  write_omega_flag = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "write_omega", &write_omega_flag)) write_omega_flag="on";
  if( strcmp(write_omega_flag, "on") == 0) {
    write_omega = true;
  }
  else if( strcmp(write_omega_flag, "off") == 0) {
    write_omega = false;
  }
  
  char* write_phi_flag;
  write_phi_flag = (char*) malloc(sizeof(char)*4);
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "write_phi", &write_phi_flag)) write_phi_flag="on";
  if( strcmp(write_phi_flag, "on") == 0) {
    write_phi = true;
  }
  else if( strcmp(write_phi_flag, "off") == 0) {
    write_phi = false;
  }
  
  //if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "fluxfile", &fluxfileName)) fluxfileName="./scan/outputs/flux";
  
  //if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "stopfile", &stopfileName)) stopfileName="c.stop";
  
  //if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "restartfile", &restartfileName))  restartfileName="restart.bin";
  
  //printf("stopfileName = %s\n", stopfileName);
  
  if(fnr_get_string_no_test(&namelist_struct, "gryfx_knobs", "scan_type", &scan_type))  scan_type="outputs";
  
  if(fnr_get_int(&namelist_struct, "gryfx_knobs", "scan_number", &scan_number)) scan_number = 1;
    
  char* collisions;
  if(fnr_get_string(&namelist_struct, "collision_knobs", "collision_model", &collisions)) collisions="none";
  
  species = (specie*) malloc(sizeof(specie)*nSpecies);
    
  for(int s=1; s<nSpecies+1; s++) {
//		printf("s= %d\n", s);


    char namelist[100];
    sprintf(namelist,"species_parameters_%d",s); 
    char* type;
    if(fnr_get_string(&namelist_struct, namelist, "type", &type)) type = "ion"; 
    
    if(strcmp(type,"ion") == 0) {
      if(fnr_get_float(&namelist_struct, namelist, "z", &species[ION].z)) species[ION].z=1.;
      if(fnr_get_float(&namelist_struct, namelist, "mass", &species[ION].mass)) species[ION].mass=1.;
      if(fnr_get_float(&namelist_struct, namelist, "dens", &species[ION].dens)) species[ION].dens=1.;
      if(fnr_get_float(&namelist_struct, namelist, "temp", &species[ION].temp)) species[ION].temp=1.;
      if(fnr_get_float(&namelist_struct, namelist, "tprim", &species[ION].tprim)) species[ION].tprim=6.; //6.9
      if(fnr_get_float(&namelist_struct, namelist, "fprim", &species[ION].fprim)) species[ION].fprim=2.; //2.2
      if(fnr_get_float(&namelist_struct, namelist, "uprim", &species[ION].uprim)) species[ION].uprim=0;


      if(strcmp(collisions,"none") == 0) species[ION].nu_ss = 0;
      else {
        if(fnr_get_float(&namelist_struct, namelist, "vnewk", &species[ION].nu_ss)) species[ION].nu_ss=0;
      }     

      strcpy(species[ION].type,"ion"); 

    }
    if(strcmp(type,"electron") == 0) {

      if(fnr_get_float(&namelist_struct, namelist, "z", &species[ELECTRON].z));
      if(fnr_get_float(&namelist_struct, namelist, "mass", &species[ELECTRON].mass));
      if(fnr_get_float(&namelist_struct, namelist, "dens", &species[ELECTRON].dens));
      if(fnr_get_float(&namelist_struct, namelist, "temp", &species[ELECTRON].temp));
      if(fnr_get_float(&namelist_struct, namelist, "tprim", &species[ELECTRON].tprim));
      if(fnr_get_float(&namelist_struct, namelist, "fprim", &species[ELECTRON].fprim));
      if(fnr_get_float(&namelist_struct, namelist, "uprim", &species[ELECTRON].uprim));

      if(strcmp(collisions,"none") == 0) species[ELECTRON].nu_ss = 0;
      else {

        if(fnr_get_float(&namelist_struct, namelist, "vnewk", &species[ELECTRON].nu_ss));

      }

			strcpy(species[ELECTRON].type,"electron");
    }   
		//free(type);

  }


}


