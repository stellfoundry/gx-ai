//Globals
#ifndef NO_GLOBALS
EXTERN_SWITCH cuComplex* omega_out_h;
// EGH maybe omega_h is the same as omega_out_h?
EXTERN_SWITCH cuComplex * omega_h;
#endif

typedef struct{
    //diagnostics scalars
    float flux1, flux2, Dens, Tpar, Tprp;
    float flux1_sum, flux2_sum, Dens_sum, Tpar_sum, Tprp_sum;
} phases_struct;

typedef struct {
	// The simpledataio object containing the netcdf output file
	struct sdatio_file sdatfile;

  float mu_avg;
  float alpha_avg;

  float* val; 

	// Total integrated |field|^2 
	float phi2;

	float kphi2;
	float phi2_movav;

	// Square of fields and moments averaged over ky and integrated over z
	float* phi2_by_kx;
	
	// Square of fields and moments averaged over kx
	float* phi2_by_ky;


	// Cumulative weighted sums for calculating exponential
	// moving averages.
	
	float * phi2_by_mode_movav;
	float * phi2_zonal_by_kx_movav;

  //Zonal flow diagnostics
  float phi2_zf;
  float phi2_zf_rms;
  float phi2_zf_rms_sum;
  float phi2_zf_rms_avg;

	
	// Expectation values of wavenumbers
	//
  float expectation_ky;
  float expectation_kx;
	float expectation_kx_movav;
	float expectation_ky_movav;

	// Parallel correlation as a function of y and delta z
	float * par_corr_kydz_movav;
	

	// Total fluxes

	float hflux_tot;
	
  // Fluxes by species and by mode
	
	float * hflux_by_species;
	float * hflux_by_species_old;
	float * hflux_by_mode_movav;
	float * hflux_by_species_movav;

	float * pflux_by_species;
	float * pflux_by_species_old;
	float * pflux_by_mode_movav;
	float * pflux_by_species_movav;
	

	// Omega as a function of kx and ky
	cuComplex * omega;
  // omega_avg is not correctly normalized
  // by dtSum; omega_out is the actual 
  // average_growth rate
	cuComplex * omega_avg;
	cuComplex * omega_out;


  phases_struct phases;

	// Filenames
	

} outputs_struct;
