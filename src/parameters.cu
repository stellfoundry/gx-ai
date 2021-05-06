#include "parameters.h"
#include <netcdf.h>
#include "toml.hpp"
#include <iostream>
#include "version.h"
using namespace std;

Parameters::Parameters() {
  initialized = false;

  // some cuda parameters (not from input file)
  int dev; 
  cudaGetDevice(&dev);
  if (false) printf("device id = %d \n",dev);
}

Parameters::~Parameters() {
  cudaDeviceSynchronize();
  if(initialized) {
    cudaFreeHost(species_h);
  }
}


void Parameters::get_nml_vars(char* filename)
{
  strcpy (run_name, filename);
  char nml_file[263];
  
  strcpy(nml_file, run_name);
  strcat(nml_file, ".in");

  printf(ANSI_COLOR_MAGENTA);

  const auto nml = toml::parse(nml_file);
    
  repeat = toml::find_or <bool> (nml, "repeat",  false);
  
  debug = toml::find_or <bool> (nml, "debug",    false);
  nz_in = toml::find_or <int> (nml, "ntheta",    32);
  ny_in = toml::find_or <int> (nml, "ny",        32);
  nx_in = toml::find_or <int> (nml, "nx",        4);
  nm_in = toml::find_or <int> (nml, "nhermite",  4);
  nl_in = toml::find_or <int> (nml, "nlaguerre", 2);
  nspec_in = toml::find_or <int> (nml, "nspecies", 1);

  if (nx_in<4) {printf("Warning: Behavior is not guaranteed for nx = %d \n",nx_in);}
  if (ny_in<4) {printf("Warning: Behavior is not guaranteed for ny = %d \n",ny_in);}

  dt = toml::find_or <float> (nml, "dt", 0.05);
  y0 = toml::find_or <float> (nml, "y0", 10.0);
  x0 = toml::find_or <float> (nml, "x0", 10.0);
  Zp = toml::find_or <int> (nml, "zp", 1);

  nperiod = toml::find_or <int> (nml, "nperiod", 1);
  nstep   = toml::find_or <int> (nml, "nstep", 10000);
  jtwist  = toml::find_or <int> (nml, "jtwist", -1);
  nwrite  = toml::find_or <int> (nml, "nwrite", 1000);
  navg    = toml::find_or <int> (nml, "navg", 10);
  nsave   = toml::find_or <int> (nml, "nsave", 2000000);
  i_share = toml::find_or <int> (nml, "i_share", 8);
  nreal   = toml::find_or <int> (nml, "nreal", 1);
  
  ikx_fixed = toml::find_or <int> (nml, "ikx_fixed", -1);
  iky_fixed = toml::find_or <int> (nml, "iky_fixed", -1);

  vp                = toml::find_or <bool> (nml, "vp", false);
  vp_closure        = toml::find_or <bool> (nml, "vp_closure", true);
  vp_nu             = toml::find_or <float> (nml, "vp_nu", -1.0);
  vp_nuh            = toml::find_or <float> (nml, "vp_nuh", -1.0);
  vp_alpha          = toml::find_or <int> (nml, "vp_alpha", 1);
  vp_alpha_h        = toml::find_or <int> (nml, "vp_alpha_h", 2);

  ks                = toml::find_or <bool> (nml, "ks", false);
  write_ks          = toml::find_or <bool> (nml, "write_ks", false);
  eqfix             = toml::find_or <bool> (nml, "eqfix", false);
  restart           = toml::find_or <bool> (nml, "restart", false);
  save_for_restart  = toml::find_or <bool> (nml, "save_for_restart", true);
  secondary         = toml::find_or <bool> (nml, "secondary", false);
  write_omega       = toml::find_or <bool> (nml, "write_omega", false);
  write_free_energy = toml::find_or <bool> (nml, "write_free_energy", false);
  write_fluxes      = toml::find_or <bool> (nml, "write_fluxes", false);
  write_moms        = toml::find_or <bool> (nml, "write_moms", false);
  write_rh          = toml::find_or <bool> (nml, "write_rh", false);
  write_pzt         = toml::find_or <bool> (nml, "write_pzt", false);
  write_vE          = toml::find_or <bool> (nml, "write_vE", false);
  write_kvE         = toml::find_or <bool> (nml, "write_kvE", false);
  write_kden        = toml::find_or <bool> (nml, "write_kden", false);
  write_kUpar       = toml::find_or <bool> (nml, "write_kUpar", false);
  write_kTpar       = toml::find_or <bool> (nml, "write_kTpar", false);
  write_kTperp      = toml::find_or <bool> (nml, "write_kTperp", false);
  write_kqpar       = toml::find_or <bool> (nml, "write_kqpar", false);

  write_xyvE        = toml::find_or <bool> (nml, "write_xyvE", false);
  write_xykvE       = toml::find_or <bool> (nml, "write_xykvE", false);
  write_xyTperp     = toml::find_or <bool> (nml, "write_xyTperp", false);
  write_xyTpar      = toml::find_or <bool> (nml, "write_xyTpar", false);
  write_xyUpar      = toml::find_or <bool> (nml, "write_xyUpar", false);
  write_xyden       = toml::find_or <bool> (nml, "write_xyden", false);
  write_xyqpar      = toml::find_or <bool> (nml, "write_xyqpar", false);

  write_kmom = (write_vE || write_kvE || write_kden || write_kUpar || write_kTpar || write_kTperp || write_kqpar);
  write_xymom = (write_xyvE || write_xykvE || write_xyden || write_xyUpar || write_xyTpar || write_xyTperp || write_xyqpar);
  
  write_phi         = toml::find_or <bool> (nml, "write_phi", false);
  write_phi_kpar    = toml::find_or <bool> (nml, "write_phi_kpar", false);
  write_h_spectrum  = toml::find_or <bool> (nml, "write_h_spectrum", false);
  write_l_spectrum  = toml::find_or <bool> (nml, "write_l_spectrum", false);
  write_lh_spectrum = toml::find_or <bool> (nml, "write_lh_spectrum", false);
  init_single       = toml::find_or <bool> (nml, "init_single", false);

  domain_change = toml::find_or <bool> (nml, "domain_change", false);
  z0_mult = toml::find_or <int> (nml, "z0_mult", 1);  assert( (z0_mult > 0) && "z0_mult must be an integer >= 1");
  y0_mult = toml::find_or <int> (nml, "y0_mult", 1);  assert( (y0_mult > 0) && "y0_mult must be an integer >= 1");
  x0_mult = toml::find_or <int> (nml, "x0_mult", 1);  assert( (x0_mult > 0) && "x0_mult must be an integer >= 1");
  ny_mult = toml::find_or <int> (nml, "ny_mult", 1);  assert( (ny_mult > 0) && "ny_mult must be an integer >= 1");
  nx_mult = toml::find_or <int> (nml, "nx_mult", 1);  assert( (nx_mult > 0) && "nx_mult must be an integer >= 1");
  nm_add  = toml::find_or <int> (nml, "nm_add" , 0);  
  nl_add  = toml::find_or <int> (nml, "nl_add" , 0);  
  ns_add  = toml::find_or <int> (nml, "ns_add" , 0);  assert( (ns_add >= 0) && "ns_add must be an integer >= 0");
  
  ntheta_mult = toml::find_or <int> (nml, "nz_mult", 1);
  assert( (ntheta_mult > 0) && "ntheta_mult must be an integer >= 1");

  if (!domain_change) {
    assert ((nx_mult == 1) && "When domain_change is false, nx_mult must be 1");
    assert ((ny_mult == 1) && "When domain_change is false, ny_mult must be 1");
    assert ((ntheta_mult == 1) && "When domain_change is false, ntheta_mult must be 1");
    assert ((x0_mult == 1) && "When domain_change is false, x0_mult must be 1");
    assert ((y0_mult == 1) && "When domain_change is false, y0_mult must be 1");
    assert ((z0_mult == 1) && "When domain_change is false, z0_mult must be 1");
    assert ((nl_add == 0) && "When domain_change is false, nl_add must be 0");
    assert ((nm_add == 0) && "When domain_change is false, nm_add must be 0");
  }

  if (domain_change) {
    printf( "You are changing the simulation domain with this input file. \n");
    if (x0_mult > 1) printf("Compared to the restart file, you have increased x0 by a factor of %d \n",x0_mult);
    if (y0_mult > 1) printf("Compared to the restart file, you have increased y0 by a factor of %d \n",y0_mult);
    if (z0_mult > 1) printf("Compared to the restart file, you have increased z0 by a factor of %d \n",z0_mult);
    if (nx_mult > 1) printf("Compared to the restart file, you have increased nx by a factor of %d \n",nx_mult);
    if (ny_mult > 1) printf("Compared to the restart file, you have increased ny by a factor of %d \n",ny_mult);
    if (ntheta_mult > 1) printf("Compared to the restart file, you have increased nx ntheta a factor of %d \n",ntheta_mult);
    if (nl_add > 0) printf("Compared to the restart file, you have added %d Laguerre basis elements. \n",nl_add);
    if (nl_add < 0) printf("Compared to the restart file, you have removed %d Laguerre basis elements. \n",-nl_add);
    if (nm_add > 0) printf("Compared to the restart file, you have added %d Hermite basis elements. \n",nm_add);
    if (nm_add < 0) printf("Compared to the restart file, you have removed %d Hermite basis elements. \n",-nm_add);
    if (ns_add > 0) printf("Compared to the restart file, you have added %d species. \n",ns_add);
  }    
  
  eps_ks     = toml::find_or <float> (nml, "eps_ks", 0.0);
  cfl        = toml::find_or <float> (nml, "cfl", 0.1);
  init_amp   = toml::find_or <float> (nml, "init_amp", 1.0e-5);
  D_hyper    = toml::find_or <float> (nml, "D_hyper", 0.1);
  nu_hyper_l = toml::find_or <float> (nml, "nu_hyper_l", 1.0);
  nu_hyper_m = toml::find_or <float> (nml, "nu_hyper_m", 1.0);
  nu_hyper   = toml::find_or <int> (nml, "nu_hyper", 2);
  p_hyper    = toml::find_or <int> (nml, "p_hyper", 2);
  p_hyper_l  = toml::find_or <int> (nml, "p_hyper_l", 6);
  p_hyper_m  = toml::find_or <int> (nml, "p_hyper_m", 1);  
  stages     = toml::find_or <int> (nml, "stages", 10);
  //  printf("in parameters, stages = %d \n",stages);
  
  ks_t0 = toml::find_or <float> (nml, "ks_t0", -1.0);
  ks_tf = toml::find_or <float> (nml, "ks_tf", -1.0);
  ks_eps0 = toml::find_or <float> (nml, "ks_eps0", -1.0);
  ks_epsf = toml::find_or <float> (nml, "ks_epsf", -1.0);

  scheme       = toml::find_or <string> (nml, "scheme", "sspx2");
  forcing_type = toml::find_or <string> (nml, "forcing_type", "Kz");
  init_field   = toml::find_or <string> (nml, "init_field", "density");
  stir_field   = toml::find_or <string> (nml, "stir_field", "density");
  forcing_amp = toml::find_or <float> (nml, "forcing_amp", 1.0);
  scale       = toml::find_or <float> (nml, "scale", 1.0);

  forcing_index = toml::find_or <int> (nml, "forcing_index", 1);
  forcing_init  = toml::find_or <bool> (nml, "forcing_init", false);
  no_fields     = toml::find_or <bool> (nml, "no_fields", false);
  
  phi_ext = toml::find_or <float> (nml, "phi_ext", 0.0);
  kpar_init = toml::find_or <float> (nml, "kpar_init", 0.0);

  ikx_single = toml::find_or <int> (nml, "ikx_single", 0);
  iky_single = toml::find_or <int> (nml, "iky_single", 1);

  nonlinear_mode = toml::find_or <bool> (nml, "nonlinear_mode", false);    linear = !nonlinear_mode;

  closure_model = toml::find_or <std::string> (nml, "closure_model", "beer4+2");
  boundary = toml::find_or <std::string> (nml, "boundary", "linked");
  source = toml::find_or <std::string> (nml, "source", "default");

  ///////////////////////////////////////////////////////////////////////
  //
  // New way to handle Boltzmann response. 
  // 
  // Preferred: 
  //     All kinetic? Leave out iphi00 entirely; leave out add_Boltzmann_species or set add_Boltzmann_species = false
  //
  //     Include a Boltzmann species? Set add_Boltzmann_species = true, and set Boltzmann_type 
  //            as either "electrons" (to recover iphi00=2 in the old way)
  //            or "ions"             (to recover iphi00=1 in the old way)
  //     Use "tau_fac" instead of ti_ov_te as the multiplier for the Boltzmann response
  // 
  // Backward compatibility possibilities (not preferred)
  //     
  //     Assume there is always a Boltzmann species unless electromagnetic
  //     and control the response with iphi00 and ti_ov_te
  //
  //     This mode of operation is deprecated, and should be removed in early 2021.
  //
  ///////////////////////////////////////////////////////////////////////
  
  
  add_Boltzmann_species = toml::find_or <bool> (nml, "add_Boltzmann_species", false);
  Btype = toml::find_or <std::string> (nml, "Boltzmann_type", "electrons");

  iphi00 = toml::find_or <int> (nml, "iphi00", -2);

  // For backward compatibility, check if iphi00 was specified and act accordingly
  if (iphi00 > 0) {
    if (iphi00 == 1) Btype = "Ions";
    if (iphi00 == 2) Btype = "Electrons";
    add_Boltzmann_species = true;
  }
  
  // allow some sloppiness here:
  
  if (Btype == "Electrons") Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "Electron" ) Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "Ions")      Boltzmann_opt = BOLTZMANN_IONS;
  if (Btype == "Ion" )      Boltzmann_opt = BOLTZMANN_IONS;

  if (Btype == "electrons") Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "electron" ) Boltzmann_opt = BOLTZMANN_ELECTRONS;
  if (Btype == "ions")      Boltzmann_opt = BOLTZMANN_IONS;
  if (Btype == "ion" )      Boltzmann_opt = BOLTZMANN_IONS;

  ti_ov_te = toml::find_or <float> (nml, "TiTe", 1.0);     // backward compatibility, sets default overall as tau_fac = unity
  tau_fac  = toml::find_or <float> (nml, "tau_fac", -1.0); // check for new, physically sensible value in the input file
  if (tau_fac > 0.) ti_ov_te = tau_fac;                    // new definition has priority if it was provided
  tau_fac = ti_ov_te;                                      // In the body of the code, use tau_fac instead of ti_ov_te
  
  ///////////////////////////////////////////////////////////////////////
  //                                                                   //
  // Testing that we have working options                              //
  //                                                                   //
  ///////////////////////////////////////////////////////////////////////

  all_kinetic = true;
  if (add_Boltzmann_species) all_kinetic = false;

  if (all_kinetic) {
    assert( (iphi00 <= 0)
	    && "Specifying all species are kinetic and also iphi00 > 0 is not allowed");
    assert( !add_Boltzmann_species
	    && "Specifying all species are kinetic and also add_Boltzmann_species is not allowed");
  }

  if (!all_kinetic) {
    assert( (tau_fac >= 0.)
	    && "Specifying all_kinetic == false and also tau_fac < 0. is not allowed");
    assert( ( (Boltzmann_opt==BOLTZMANN_ELECTRONS) || (Boltzmann_opt==BOLTZMANN_IONS) )
	    && "If all_kinetic == false then a legal Boltzmann_opt must be specified");
    assert( add_Boltzmann_species
	    && "If all_kinetic == false then add_Boltzmann_species should be true");
  }
      
  smith_par_q = toml::find_or <int> (nml, "smith_par_q", 3);
  smith_perp_q = toml::find_or <int> (nml, "smith_perp_q", 3);
  hyper = toml::find_or <bool> (nml, "hyper", false);
  hypercollisions = toml::find_or <bool> (nml, "hypercollisions", false);

  restart_to_file   = toml::find_or <std::string> (nml, "restart_to_file", "newsave.nc");  
  restart_from_file = toml::find_or <std::string> (nml, "restart_from_file", "oldsave.nc");  
  geofilename =  toml::find_or <std::string> (nml, "geofile", "eik.out");  

  slab = toml::find_or <bool> (nml, "slab", false);
  const_curv = toml::find_or <bool> (nml, "const_curv", false);

  igeo     = toml::find_or <int> (nml, "igeo", 0);
  drhodpsi = toml::find_or <float> (nml, "drhodpsi", 1.0);
  kxfac    = toml::find_or <float> (nml, "kxfac", 1.0);
  rmaj     = toml::find_or <float> (nml, "Rmaj", 1.0);
  shift    = toml::find_or <float> (nml, "shift", 0.0);
  eps      = toml::find_or <float> (nml, "eps", 0.167);
  rhoc     = toml::find_or <float> (nml, "rhoc", 0.167);
  qsf      = toml::find_or <float> (nml, "qinp", 1.4);
  shat     = toml::find_or <float> (nml, "shat", 0.8);
  akappa   = toml::find_or <float> (nml, "akappa", 1.0);
  akappri  = toml::find_or <float> (nml, "akappri", 0.0);
  tri      = toml::find_or <float> (nml, "tri", 0.0);
  tripri   = toml::find_or <float> (nml, "tripri", 0.0);
  beta     = toml::find_or <float> (nml, "beta", -1.0);
  
  beta_prime_input = toml::find_or <float> (nml, "beta_prime_input", 0.0);
  s_hat_input      = toml::find_or <float> (nml, "s_hat_input", 0.8);

  wspectra.resize(nw_spectra);
  pspectra.resize(np_spectra);
  aspectra.resize(na_spectra);

  wspectra.assign(nw_spectra, 0);
  pspectra.assign(np_spectra, 0);
  aspectra.assign(na_spectra, 0);

  if (nml.contains("Wspectra")) {
    const auto tomlW = toml::find (nml, "Wspectra");
    wspectra [WSPECTRA_species] = (toml::find_or <bool> (tomlW, "species",          false)) == true ? 1 : 0;
    wspectra [WSPECTRA_kx]      = (toml::find_or <bool> (tomlW, "kx",               false)) == true ? 1 : 0;
    wspectra [WSPECTRA_ky]      = (toml::find_or <bool> (tomlW, "ky",               false)) == true ? 1 : 0;
    wspectra [WSPECTRA_kz]      = (toml::find_or <bool> (tomlW, "kz",               false)) == true ? 1 : 0;
    wspectra [WSPECTRA_z]       = (toml::find_or <bool> (tomlW, "z",                false)) == true ? 1 : 0;
    wspectra [WSPECTRA_l]       = (toml::find_or <bool> (tomlW, "laguerre",         false)) == true ? 1 : 0;
    wspectra [WSPECTRA_m]       = (toml::find_or <bool> (tomlW, "hermite",          false)) == true ? 1 : 0;
    wspectra [WSPECTRA_lm]      = (toml::find_or <bool> (tomlW, "hermite_laguerre", false)) == true ? 1 : 0;
    wspectra [WSPECTRA_kperp]   = (toml::find_or <bool> (tomlW, "kperp",            false)) == true ? 1 : 0;
    wspectra [WSPECTRA_kxky]    = (toml::find_or <bool> (tomlW, "kxky",             false)) == true ? 1 : 0;
  }
  if (nml.contains("Pspectra")) {
    const auto tomlP = toml::find (nml, "Pspectra");
    pspectra [PSPECTRA_species] = (toml::find_or <bool> (tomlP, "species",          false)) == true ? 1 : 0;
    pspectra [PSPECTRA_kx]      = (toml::find_or <bool> (tomlP, "kx",               false)) == true ? 1 : 0;
    pspectra [PSPECTRA_ky]      = (toml::find_or <bool> (tomlP, "ky",               false)) == true ? 1 : 0;
    pspectra [PSPECTRA_kz]      = (toml::find_or <bool> (tomlP, "kz",               false)) == true ? 1 : 0;
    pspectra [PSPECTRA_z]       = (toml::find_or <bool> (tomlP, "z",                false)) == true ? 1 : 0;
    pspectra [PSPECTRA_kperp]   = (toml::find_or <bool> (tomlP, "kperp",            false)) == true ? 1 : 0;
    pspectra [PSPECTRA_kxky]    = (toml::find_or <bool> (tomlP, "kxky",             false)) == true ? 1 : 0;
  }
  // if we have adiabatic ions, slave the aspectra to the wspectra as appropriate
  if (!all_kinetic) {
    aspectra [ ASPECTRA_species ] = wspectra [ WSPECTRA_species ];
    aspectra [ ASPECTRA_kx      ] = wspectra [ WSPECTRA_kx      ];
    aspectra [ ASPECTRA_ky      ] = wspectra [ WSPECTRA_ky      ];
    aspectra [ ASPECTRA_kz      ] = wspectra [ WSPECTRA_kz      ];
    aspectra [ ASPECTRA_z       ] = wspectra [ WSPECTRA_z       ];
    aspectra [ ASPECTRA_kperp   ] = wspectra [ WSPECTRA_kperp   ];
    aspectra [ ASPECTRA_kxky    ] = wspectra [ WSPECTRA_kxky    ];
  }
  // for backwards compatibility
  if (write_l_spectrum)  wspectra[WSPECTRA_l] = 1;
  if (write_h_spectrum)  wspectra[WSPECTRA_m] = 1;
  if (write_lh_spectrum) wspectra[WSPECTRA_lm] = 1;
  
  // Some diagnostics are not yet available:
  wspectra[ WSPECTRA_kperp] = 0;
  pspectra[ PSPECTRA_kperp] = 0;
  aspectra[ ASPECTRA_kperp] = 0;
  
  // If Wtot is requested, turn Ws, Ps, Phi2 on:
  if (write_free_energy) {
    wspectra[WSPECTRA_species] = 1;
    pspectra[PSPECTRA_species] = 1;
    if ( add_Boltzmann_species ) aspectra[ASPECTRA_species] = 1;
  }

  gx = (!ks && !vp);
  assert (!(ks && vp));
  assert (ks || vp || gx);

  int ksize = 0;
  for (int k=0; k<pspectra.size(); k++) ksize = max(ksize, pspectra[k]);
  for (int k=0; k<wspectra.size(); k++) ksize = max(ksize, wspectra[k]);
  for (int k=0; k<aspectra.size(); k++) ksize = max(ksize, aspectra[k]);

  diagnosing_pzt = write_pzt;
  
  diagnosing_spectra = false;
  if (ksize > 0) diagnosing_spectra = true;

  diagnosing_kzspec = false;
  if ((wspectra[ WSPECTRA_kz ] == 1) || (pspectra[ PSPECTRA_kz ] == 1) || (aspectra[ ASPECTRA_kz ] == 1)) {
    diagnosing_kzspec = true;
  }
  
  diagnosing_moments = false;
  if (write_moms || write_phi || write_phi_kpar) diagnosing_moments = true;
  
  species_h = (specie *) calloc(nspec_in, sizeof(specie));
  for (int is=0; is < nspec_in; is++) {
    species_h[is].uprim = 0.;
    species_h[is].nu_ss = 0.;
    species_h[is].temp = 1.;
    species_h[is].z = toml::find <float> (nml, "species", "z", is);
    species_h[is].mass = toml::find <float> (nml, "species", "mass", is);
    species_h[is].dens = toml::find <float> (nml, "species", "dens", is);
    species_h[is].temp = toml::find <float> (nml, "species", "temp", is);
    species_h[is].tprim = toml::find <float> (nml, "species", "tprim", is);
    species_h[is].fprim = toml::find <float> (nml, "species", "fprim", is);
    species_h[is].uprim = toml::find <float> (nml, "species", "uprim", is);
    species_h[is].nu_ss = toml::find <float> (nml, "species", "vnewk", is);
    std::string stype = toml::find <string> (nml, "species", "type", is);
    species_h[is].type = stype == "ion" ? 0 : 1;
  }
  
  float numax = -1.;
  collisions = false;
  for (int i=0; i<nspec_in; i++) {numax = max(numax, species_h[i].nu_ss);}
  if (numax > 0.) collisions = true;
  
  fphi = toml::find_or <float> (nml, "fphi", 1.0);
  fapar = toml::find_or <float> (nml, "fapar", 0.0);
  fbpar = toml::find_or <float> (nml, "fbpar", 0.0);

  tp_t0 = toml::find_or <float> (nml, "t0", -1.0);
  tp_tf = toml::find_or <float> (nml, "tf", -1.0);
  tprim0 = toml::find_or <float> (nml, "tprim0", -1.0);
  tprimf = toml::find_or <float> (nml, "tprimf", -1.0);

  Reservoir = false;
  add_noise = false;
  
  if (nml.contains("Reservoir")) {
    const auto tomlRes = toml::find (nml, "Reservoir");
    Reservoir          = toml::find_or <bool>  (tomlRes, "Use_reservoir", false);
    ResQ               = toml::find_or <int>   (tomlRes, "Q", 20);
    ResK               = toml::find_or <int>   (tomlRes, "K", 3);
    ResTrainingSteps   = toml::find_or <int>   (tomlRes, "training_steps", 0);
    ResPredict_Steps   = toml::find_or <int>   (tomlRes, "prediction_steps", 200);
    ResTrainingDelta   = toml::find_or <int>   (tomlRes, "training_delta", 0);
    ResSpectralRadius  = toml::find_or <float> (tomlRes, "spectral_radius", 0.6);
    ResReg             = toml::find_or <float> (tomlRes, "regularization", 1.0e-4);
    ResSigma           = toml::find_or <float> (tomlRes, "input_sigma", 0.5);
    ResSigmaNoise      = toml::find_or <float> (tomlRes, "noise", -1.0);
    
    if (ResTrainingSteps == 0) ResTrainingSteps = nstep/nwrite;
    if (ResTrainingDelta == 0) ResTrainingDelta = nwrite;
    if (ResSigmaNoise > 0.) add_noise = true;
  }  
  // open the netcdf4 file for this run
  // store all inputs for future reference

  char strb[263];
  strcpy(strb, run_name); 
  strcat(strb, ".nc");

  int retval, idim, sdim, wdim, pdim, adim;
  if (retval = nc_create(strb, NC_CLOBBER, &ncid)) ERR(retval);

  int ri = 2;
  if (retval = nc_def_dim (ncid, "ri",      ri,            &idim)) ERR(retval);
  if (retval = nc_def_dim (ncid, "x",       nx_in,         &idim)) ERR(retval);
  if (retval = nc_def_dim (ncid, "y",       ny_in,         &idim)) ERR(retval);
  if (retval = nc_def_dim (ncid, "m",       nm_in,         &idim)) ERR(retval);
  if (retval = nc_def_dim (ncid, "l",       nl_in,         &idim)) ERR(retval);
  if (retval = nc_def_dim (ncid, "s",       nspec_in,      &sdim)) ERR(retval);
  if (retval = nc_def_dim (ncid, "time",    NC_UNLIMITED,  &idim)) ERR(retval);
  if (retval = nc_def_dim (ncid, "nw",      nw_spectra,    &wdim)) ERR(retval);
  if (retval = nc_def_dim (ncid, "np",      np_spectra,    &pdim)) ERR(retval);
  if (retval = nc_def_dim (ncid, "na",      na_spectra,    &adim)) ERR(retval);

  static char file_header[] = "GX simulation data";
  if (retval = nc_put_att_text (ncid, NC_GLOBAL, "Title", strlen(file_header), file_header)) ERR(retval);
  
  int ivar; 
  if (retval = nc_def_var (ncid, "ny",                    NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nx",                    NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "ntheta",                NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nhermite",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nlaguerre",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nspecies",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "dt",                    NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "ks",                    NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "vp",                    NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "vp_closure",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  
  specs[0] = wdim;
  if (retval = nc_def_var (ncid, "wspectra",              NC_INT,   1, specs, &ivar)) ERR(retval);

  specs[0] = pdim;
  if (retval = nc_def_var (ncid, "pspectra",              NC_INT,   1, specs, &ivar)) ERR(retval);

  specs[0] = adim;
  if (retval = nc_def_var (ncid, "aspectra",              NC_INT,   1, specs, &ivar)) ERR(retval);

  specs[0] = sdim;
  if (retval = nc_def_var (ncid, "species_type",          NC_INT,   1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "z",                     NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "m",                     NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "n0",                    NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "n0_prime",              NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "u0_prime",              NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "T0",                    NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "T0_prime",              NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nu",                    NC_FLOAT, 1, specs, &ivar)) ERR(retval);
  
  if (retval = nc_def_var (ncid, "nperiod",               NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "y0",                    NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "x0",                    NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nstep",                 NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "zp",                    NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "jtwist",                NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "i_share",               NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nreal",                 NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "stages",                NC_INT,   0, NULL, &ivar)) ERR(retval);
  
  // diagnostics
  if (retval = nc_def_var (ncid, "nwrite",                NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "navg",                  NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nsave",                 NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "debug",                 NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (ncid, "repeat",                NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "restart",               NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "save_for_restart",      NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "secondary",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "restart_from_file_dum", NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", restart_from_file.size(), restart_from_file.c_str())) ERR(retval);
  if (retval = nc_def_var (ncid, "restart_to_file_dum",   NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", restart_to_file.size(), restart_to_file.c_str())) ERR(retval);
  if (retval = nc_def_var (ncid, "eqfix",                 NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (ncid, "write_vE",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_kvE",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_kden",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_kUpar",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_kTpar",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_kqpar",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_kTperp",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_xyvE",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_xykvE",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_xyden",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_xyUpar",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_xyTpar",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_xyTperp",         NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_xyqpar",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_omega",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_fluxes",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_moms",            NC_INT,   0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (ncid, "write_free_energy",     NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_phi",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_phi_kpar",        NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_rh",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "write_pzt",             NC_INT,   0, NULL, &ivar)) ERR(retval);

  // numerical parameters
  if (retval = nc_def_var (ncid, "cfl",                   NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "init_amp",              NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "D_hyper",               NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nu_hyper",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nu_hyper_l",            NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nu_hyper_m",            NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "p_hyper",               NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "p_hyper_l",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "p_hyper_m",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "eps_ks",                NC_FLOAT, 0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (ncid, "vp_nu",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "vp_nuh",                NC_FLOAT, 0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (ncid, "vp_alpha",              NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "vp_alpha_h",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  
  // model flags
  if (retval = nc_def_var (ncid, "scheme_dum",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", scheme.size(), scheme.c_str())) ERR(retval);
  if (retval = nc_def_var (ncid, "no_fields",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "forcing_init",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "forcing_type_dum",      NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", forcing_type.size(), forcing_type.c_str())) ERR(retval);
  if (retval = nc_def_var (ncid, "forcing_amp",           NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "forcing_index",         NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "phi_ext",               NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "scale",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "init_field_dum",        NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", init_field.size(), init_field.c_str())) ERR(retval);
  if (retval = nc_def_var (ncid, "stir_field_dum",        NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", stir_field.size(), stir_field.c_str())) ERR(retval);
  if (retval = nc_def_var (ncid, "init_single",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "ikx_single",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "iky_single",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "ikx_fixed",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "iky_fixed",             NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "kpar_init",             NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "nonlinear_mode",        NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "boundary_dum",          NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", boundary.size(), boundary.c_str())) ERR(retval);

  // for boltzmann opts need attribute BD bug

  if (retval = nc_def_var (ncid, "tau_fac",               NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "add_Boltzmann_species", NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "all_kinetic",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "Boltzmann_type_dum",    NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", Btype.size(), Btype.c_str() ) ) ERR(retval);
  
  if (retval = nc_def_var (ncid, "source_dum",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", source.size(), source.c_str())) ERR(retval);
  if (retval = nc_def_var (ncid, "hyper",                 NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "hypercollisions",       NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "collisions",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "closure_model_dum",     NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", closure_model.size(), closure_model.c_str())) ERR(retval);
  if (retval = nc_def_var (ncid, "smith_par_q",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "smith_perp_q",          NC_INT,   0, NULL, &ivar)) ERR(retval);

  // geometry
  if (retval = nc_def_var (ncid, "igeo",                  NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "slab",                  NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "const_curv",            NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "geofile_dum",           NC_INT,   0, NULL, &ivar)) ERR(retval);
  if (retval = nc_put_att_text (ncid, ivar, "value", geofilename.size(), geofilename.c_str())) ERR(retval);

  if (retval = nc_def_var (ncid, "drhodpsi",              NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "kxfac",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "Rmaj",                  NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "shift",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "eps",                   NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "rhoc",                  NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "q",                     NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "shat",                  NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "kappa",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "kappa_prime",           NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "tri",                   NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "tri_prime",             NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  
  if (retval = nc_def_var (ncid, "beta",                  NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "beta_prime_input",      NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "s_hat_input",           NC_FLOAT, 0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (ncid, "fphi",                  NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "fapar",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "fbpar",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (ncid, "t0",                    NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "tf",                    NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "tprim0",                NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "tprimf",                NC_FLOAT, 0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (ncid, "ks_t0",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "ks_tf",                 NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "ks_eps0",               NC_FLOAT, 0, NULL, &ivar)) ERR(retval);
  if (retval = nc_def_var (ncid, "ks_epsf",               NC_FLOAT, 0, NULL, &ivar)) ERR(retval);

  if (retval = nc_def_var (ncid, "code_info",             NC_INT,   0, NULL, &ivar)) ERR(retval);

  std::string hash(build_git_sha);                       
  if (retval = nc_put_att_text (ncid, ivar, "Hash",      hash.size(), hash.c_str() ) ) ERR(retval);
  std::string compiled(build_git_time);                  
  if (retval = nc_put_att_text (ncid, ivar, "BuildDate", compiled.size(), compiled.c_str() ) ) ERR(retval);
  std::string builder(build_user);                       
  if (retval = nc_put_att_text (ncid, ivar, "BuildUser", builder.size(), builder.c_str() ) ) ERR(retval);
  std::string build_host(build_hostname);                
  if (retval = nc_put_att_text (ncid, ivar, "BuildHost", build_host.size(), build_host.c_str() ) ) ERR(retval);

  if (retval = nc_enddef (ncid)) ERR(retval);
  
  putbool  (ncid, "all_kinetic", all_kinetic);
  putbool  (ncid, "add_Boltzmann_species", add_Boltzmann_species);
  putbool  (ncid, "debug",     debug);
  putint   (ncid, "ntheta",    nz_in);
  putint   (ncid, "nx",        nx_in);
  putint   (ncid, "ny",        ny_in);
  putint   (ncid, "nhermite",  nm_in);
  putint   (ncid, "nlaguerre", nl_in);
  putint   (ncid, "nspecies",  nspec_in);

  put_real (ncid, "dt", dt);
  put_real (ncid, "y0", y0);
  put_real (ncid, "x0", x0);
  putint   (ncid, "zp", Zp);
  
  putint   (ncid, "nperiod", nperiod);
  putint   (ncid, "nstep",   nstep);
  putint   (ncid, "jtwist",  jtwist);
  putint   (ncid, "nwrite",  nwrite);
  putint   (ncid, "navg",    navg);
  putint   (ncid, "nsave",   nsave);
  putint   (ncid, "nreal",   nreal);
  putint   (ncid, "i_share", i_share);
  putint   (ncid, "stages",  stages);

  putint   (ncid, "ikx_fixed",  ikx_fixed);
  putint   (ncid, "iky_fixed",  iky_fixed);
  putbool  (ncid, "eqfix",      eqfix);
  putbool  (ncid, "ks",         ks);
  putbool  (ncid, "vp",         vp);
  putbool  (ncid, "vp_closure", vp_closure);
  
  putbool  (ncid, "repeat",         repeat);
  putbool  (ncid, "restart",        restart);
  putbool  (ncid, "save_for_restart", save_for_restart);
  putbool  (ncid, "secondary", secondary);
  putbool  (ncid, "write_vE", write_vE);
  putbool  (ncid, "write_kvE", write_kvE);
  putbool  (ncid, "write_kden", write_kden);
  putbool  (ncid, "write_kUpar", write_kUpar);
  putbool  (ncid, "write_kTpar", write_kTpar);
  putbool  (ncid, "write_kTperp", write_kTperp);
  putbool  (ncid, "write_kqpar", write_kqpar);
  putbool  (ncid, "write_xyvE", write_xyvE);
  putbool  (ncid, "write_xykvE", write_xykvE);
  putbool  (ncid, "write_xyden", write_xyden);
  putbool  (ncid, "write_xyUpar", write_xyUpar);
  putbool  (ncid, "write_xyTpar", write_xyTpar);
  putbool  (ncid, "write_xyTperp", write_xyTperp);
  putbool  (ncid, "write_xyqpar", write_xyqpar);
  putbool  (ncid, "write_omega", write_omega);
  putbool  (ncid, "write_free_energy", write_free_energy);
  putbool  (ncid, "write_fluxes", write_fluxes);
  putbool  (ncid, "write_moms", write_moms);
  putbool  (ncid, "write_rh", write_rh);
  putbool  (ncid, "write_pzt", write_pzt);
  putbool  (ncid, "write_phi", write_phi);
  putbool  (ncid, "write_phi_kpar", write_phi_kpar);
  putbool  (ncid, "init_single", init_single);

  put_real (ncid, "cfl", cfl);
  put_real (ncid, "init_amp", init_amp);
  put_real (ncid, "D_hyper", D_hyper);
  put_real (ncid, "nu_hyper_l", nu_hyper_l);
  put_real (ncid, "nu_hyper_m", nu_hyper_m);
  put_real (ncid, "eps_ks", eps_ks);

  putint   (ncid, "nu_hyper", nu_hyper);
  putint   (ncid, "p_hyper", p_hyper);
  putint   (ncid, "p_hyper_l", p_hyper_l);
  putint   (ncid, "p_hyper_m", p_hyper_m);

  putint   (ncid, "vp_alpha",   vp_alpha);
  putint   (ncid, "vp_alpha_h", vp_alpha_h);

  put_real (ncid, "vp_nu",      vp_nu);
  put_real (ncid, "vp_nuh",     vp_nuh);
  
  put_real (ncid, "forcing_amp", forcing_amp);
  put_real (ncid, "scale", scale);
  
  putint   (ncid, "forcing_index", forcing_index);
  putbool  (ncid, "forcing_init", forcing_init);
  putbool  (ncid, "no_fields", no_fields);
  
  put_real (ncid, "phi_ext", phi_ext);
  put_real (ncid, "kpar_init", kpar_init);
  putint   (ncid, "ikx_single", ikx_single);
  putint   (ncid, "iky_single", iky_single);

  putbool  (ncid, "nonlinear_mode", nonlinear_mode); 
  
  putint   (ncid, "smith_par_q", smith_par_q);
  putint   (ncid, "smith_perp_q", smith_perp_q);
  putbool  (ncid, "hyper", hyper);
  putbool  (ncid, "hypercollisions", hypercollisions);
  putbool  (ncid, "collisions", collisions);
  
  putbool  (ncid, "slab", slab);
  putbool  (ncid, "const_curv", const_curv);
  
  //  putbool  (ncid, "snyder_electrons", snyder_electrons);

  putint   (ncid, "igeo", igeo);
  put_real (ncid, "drhodpsi", drhodpsi);
  put_real (ncid, "kxfac", kxfac);
  put_real (ncid, "Rmaj", rmaj);
  put_real (ncid, "shift", shift);
  put_real (ncid, "eps", eps);
  put_real (ncid, "rhoc", rhoc);
  put_real (ncid, "q", qsf);
  put_real (ncid, "shat", shat);  // is this always consistent with geometry inputs? should force it. BD
  put_real (ncid, "kappa", akappa);
  put_real (ncid, "kappa_prime", akappri);
  put_real (ncid, "tri", tri);
  put_real (ncid, "tri_prime", tripri);
  put_real (ncid, "beta", beta);
  put_real (ncid, "beta_prime_input", beta_prime_input);
  put_real (ncid, "s_hat_input", s_hat_input);

  put_wspectra (ncid, wspectra); 
  put_pspectra (ncid, pspectra); 
  put_aspectra (ncid, aspectra); 
  putspec (ncid, nspec_in, species_h);
  
  put_real (ncid, "tau_fac", tau_fac);
  
  put_real (ncid, "fphi", fphi);
  put_real (ncid, "fapar", fapar);
  put_real (ncid, "fbpar", fbpar);

  put_real (ncid, "t0", tp_t0);
  put_real (ncid, "tf", tp_tf);
  put_real (ncid, "tprim0", tprim0);
  put_real (ncid, "tprimf", tprimf);
  
  put_real (ncid, "ks_t0", ks_t0);
  put_real (ncid, "ks_tf", ks_tf);
  put_real (ncid, "ks_eps0", ks_eps0);
  put_real (ncid, "ks_epsf", ks_epsf);
  
  if(nz_in != 1) {
    int ntgrid = nz_in/2 + (nperiod-1)*nz_in; 
    nz_in = 2*ntgrid; // force even
  }
  
  Zp = 2*nperiod - 1; // BD This needs updating
  
  // BD  This is messy. Prefer to go back to original method
  // before, jtwist_old assumed Zp=1
  // now, redefining jtwist = jtwist_old*Zp

  if (jtwist==0) {
    // this is an error
    printf("************************** \n");
    printf("************************** \n");
    printf("jtwist = 0 is not allowed! \n");
    printf("************************** \n");
    printf("************************** \n");
  }

  // if jtwist = -1 in the input file
  // set default jtwist to 2*pi*shat to get the x0 in the input file
  
  if (jtwist == -1) {
    if (abs(shat)>1.e-6) {
      jtwist = (int) round(2*M_PI*abs(shat)*Zp/y0*x0);  // Use Zp or 1 here?
    } else {
      // no need to do anything here. x0 is set from input file and jtwist should not be used anywhere
    }
    if (jtwist == 0) jtwist = 1;  // just to be safe
  }   

  // now set x0 to be consistent with jtwist. Two cases: ~ zero shear, and otherwise
  if (abs(shat)>1.e-6) {
    x0 = y0 * jtwist/(2*M_PI*Zp*abs(shat));  
  }

  // record the values of jtwist and x0 used in runname.nc
  putint (ncid, "jtwist", jtwist);
  put_real (ncid, "x0", x0);
     
  //  if(strcmp(closure_model, "beer4+2")==0) {
  closure_model_opt = Closure::none   ;
  if( closure_model == "beer4+2") {
    printf("\nUsing Beer 4+2 closure model. Overriding nm=4, nl=2\n\n");
    nm_in = 4;
    nl_in = 2;
    closure_model_opt = Closure::beer42;
  } else if (closure_model == "smith_perp") { closure_model_opt = Closure::smithperp;
  } else if (closure_model == "smith_par")  { closure_model_opt = Closure::smithpar; 
  }

  if( boundary == "periodic") { boundary_option_periodic = true;
  } else { boundary_option_periodic = false; }
  
  local_limit = false;
  if(qsf < 0  &&  nz_in == 1) { local_limit=true; }

  if     ( init_field == "density") { initf = inits::density; }
  else if( init_field == "upar"   ) { initf = inits::upar   ; }
  else if( init_field == "tpar"   ) { initf = inits::tpar   ; }
  else if( init_field == "tperp"  ) { initf = inits::tperp  ; }
  else if( init_field == "qpar"   ) { initf = inits::qpar   ; }
  else if( init_field == "qperp"  ) { initf = inits::qperp  ; }
  
  if     ( stir_field == "density") { stirf = stirs::density; }
  else if( stir_field == "upar"   ) { stirf = stirs::upar   ; }
  else if( stir_field == "tpar"   ) { stirf = stirs::tpar   ; }
  else if( stir_field == "tperp"  ) { stirf = stirs::tperp  ; }
  else if( stir_field == "qpar"   ) { stirf = stirs::qpar   ; }
  else if( stir_field == "qperp"  ) { stirf = stirs::qperp  ; }
  else if( stir_field == "ppar"   ) { stirf = stirs::ppar   ; }
  else if( stir_field == "pperp"  ) { stirf = stirs::pperp  ; }
  
  if (scheme == "sspx3") scheme_opt = Tmethod::sspx3;
  if (scheme == "g3")    scheme_opt = Tmethod::g3;
  if (scheme == "k10")   scheme_opt = Tmethod::k10;
  if (scheme == "k2")    scheme_opt = Tmethod::k2;
  if (scheme == "rk4")   scheme_opt = Tmethod::rk4;
  if (scheme == "sspx2") scheme_opt = Tmethod::sspx2;
  if (scheme == "rk2")   scheme_opt = Tmethod::rk2;

  if (eqfix && ((scheme_opt == Tmethod::k10) || (scheme_opt == Tmethod::g3)  || (scheme_opt == Tmethod::k2))) {
    printf("\n");
    printf("\n");
    printf(ANSI_COLOR_MAGENTA);
    printf("The eqfix option is not compatible with this time-stepping algorithm. \n");
    printf(ANSI_COLOR_GREEN);
    printf("The eqfix option is not compatible with this time-stepping algorithm. \n");
    printf(ANSI_COLOR_RED);
    printf("The eqfix option is not compatible with this time-stepping algorithm. \n");
    printf(ANSI_COLOR_BLUE);
    printf("The eqfix option is not compatible with this time-stepping algorithm. \n");
    printf(ANSI_COLOR_RESET);    
    printf("\n");
    printf("\n");
  }  
  //  printf("scheme_opt = %d \n",scheme_opt);
    
  if( source == "phiext_full") {
    source_option = PHIEXT;
    printf("Running Rosenbluth-Hinton zonal flow calculation\n");
  }

  if(hypercollisions) printf("Using hypercollisions.\n");
  if(hyper) printf("Using hyperdiffusion.\n");

  if(debug) printf("nspec_in = %i \n",nspec_in);

  nspec = nspec_in;
  init_species(species_h);
  initialized = true;
  //  cudaDeviceSynchronize();
  printf(ANSI_COLOR_RESET);    
  //  if (retval = nc_close(ncid)) ERR(retval); 
  //  exit(1);
}

void Parameters::init_species(specie* species)
{
  for(int s=0; s<nspec_in; s++) {
    species[s].vt   = sqrt(species[s].temp / species[s].mass);
    species[s].tz   = species[s].temp / species[s].z;
    species[s].zt   = species[s].z / species[s].temp;
    species[s].rho2 = species[s].temp * species[s].mass / (species[s].z * species[s].z);
    species[s].nt   = species[s].dens * species[s].temp;
    species[s].qneut= species[s].dens * species[s].z * species[s].z / species[s].temp;
    species[s].nz   = species[s].dens * species[s].z;
    species[s].as   = species[s].nz * species[s].vt;
    if (debug) {
      printf("species = %d \n",s);
      printf("mass, z, temp, dens = %f, %f, %f, %f \n",
	     species[s].mass, species[s].z, species[s].temp, species[s].dens);
      printf("vt, tz, zt = %f, %f, %f \n",
	     species[s].vt, species[s].tz, species[s].zt);
      printf("rho2, nt, qneut, nz = %f, %f, %f, %f \n \n",
	     species[s].rho2, species[s].nt, species[s].qneut, species[s].nz);
    }      
  }
}

int Parameters::getint (int ncid, const char varname[]) {
  int idum, retval, res;
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_get_var  (ncid, idum, &res)) ERR(retval);
  //  if (debug) printf("%s = %i \n",varname, res);
  return res;
}

void Parameters::putint (int ncid, const char varname[], int val) {
  int idum, retval;
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_put_var  (ncid, idum, &val)) ERR(retval);
  //  if (debug) printf("%s = %i \n",varname, val);
}

void Parameters::putbool (int ncid, const char varname[], bool val) {
  int idum, retval;
  int b = val==true ? 1 : 0 ; 
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_put_var  (ncid, idum, &b))       ERR(retval);
}

bool Parameters::getbool (int ncid, const char varname[]) {
  int idum, ires, retval;
  bool res;
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_get_var  (ncid, idum, &ires)) ERR(retval);
  res = (ires!=0) ? true : false ;
  //  if (debug) printf("%s = %i \n", varname, ires);
  return res;
}

float Parameters::get_real (int ncid, const char varname[]) {
  int idum, retval;
  float res;
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_get_var  (ncid, idum, &res)) ERR(retval);
  //  if (debug) printf("%s = %f \n",varname, res);
  return res;
}

void Parameters::put_real (int ncid, const char varname[], float val) {
  int idum, retval;
  if (retval = nc_inq_varid(ncid, varname, &idum))   ERR(retval);
  if (retval = nc_put_var  (ncid, idum, &val)) ERR(retval);
  //  if (debug) printf("%s = %f \n",varname, val);
}

void Parameters::put_wspectra (int ncid, std::vector<int> s) {

  int idum, retval;
  wspectra_start[0] = 0;
  wspectra_count[0] = nw_spectra;

  if (retval = nc_inq_varid(ncid, "wspectra", &idum))     ERR(retval);
  if (retval = nc_put_vara (ncid, idum, wspectra_start, wspectra_count, s.data())) ERR(retval);
}

void Parameters::put_pspectra (int ncid, std::vector<int> s) {

  int idum, retval;
  pspectra_start[0] = 0;
  pspectra_count[0] = np_spectra;

  if (retval = nc_inq_varid(ncid, "pspectra", &idum))     ERR(retval);
  if (retval = nc_put_vara (ncid, idum, pspectra_start, pspectra_count, s.data())) ERR(retval);
}

void Parameters::put_aspectra (int ncid, std::vector<int> s) {

  int idum, retval;
  aspectra_start[0] = 0;
  aspectra_count[0] = na_spectra;

  if (retval = nc_inq_varid(ncid, "aspectra", &idum))     ERR(retval);
  if (retval = nc_put_vara (ncid, idum, aspectra_start, aspectra_count, s.data())) ERR(retval);
}

void Parameters::putspec (int  ncid, int nspec, specie* spec) {
  int idum, retval;
  
  is_start[0] = 0;
  is_count[0] = nspec;

  // this stuff should all be in species itself!
  // reason for all this is basically legacy + cuda does not support <vector>
  
  std::vector <float> zs, ms, ns, Ts, Tps, nps, ups, nus;
  std::vector <int> types;
  
  for (int is=0; is<nspec; is++) {
    zs.push_back(spec[is].z);
    ms.push_back(spec[is].mass);
    ns.push_back(spec[is].dens);
    Ts.push_back(spec[is].temp);
    Tps.push_back(spec[is].tprim);
    nps.push_back(spec[is].fprim);
    ups.push_back(spec[is].uprim);
    nus.push_back(spec[is].nu_ss);
    types.push_back(spec[is].type);

  }
  float *z = &zs[0];
  float *m = &ms[0];
  float *n0 = &ns[0];
  float *T0 = &Ts[0];
  float *Tp = &Tps[0];
  float *np = &nps[0];
  float *up = &ups[0];
  float *nu = &nus[0];
  int *st = &types[0];
  
  if (retval = nc_inq_varid(ncid, "z", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, z))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "m", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, m))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "n0", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, n0))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "n0_prime", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, np))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "u0_prime", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, up))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "T0", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, T0))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "T0_prime", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, Tp))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "nu", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, nu))  ERR(retval);

  if (retval = nc_inq_varid(ncid, "species_type", &idum))   ERR(retval);
  if (retval = nc_put_vara (ncid, idum, is_start, is_count, st))  ERR(retval);
}



