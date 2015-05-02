//Globals
#ifndef NO_GLOBALS
EXTERN_SWITCH cuComplex* omega_out_h;
// EGH maybe omega_h is the same as omega_out_h?
EXTERN_SWITCH cuComplex * omega_h;
#endif

typedef struct {
	// General stuff
	struct sdatio_file sdatfile;

	// Total integrated |field|^2 
	float phi2;

	// Square of fields and moments averaged over ky and integrated over z
	float* phi2_by_kx;
	
	// Square of fields and moments averaged over kx
	float* phi2_by_ky;


	// Cumulative weighted sums for calculating exponential
	// moving averages.
	
	float * phi2_by_mode_movav;
	float * phi2_zonal_by_kx_movav;

	
	// Expectation values of wavenumbers
	//
	float expectation_kx_movav;
	float expectation_ky_movav;

	// Parallel correlation as a function of y and delta z
	float * par_corr_kydz_movav;
	

	// Total fluxes

	float hflux_tot;
	
  // Fluxes by species and by mode
	
	float * hflux_by_species;
	float * hflux_by_mode_movav;
	float * hflux_by_species_movav;

	float * pflux_by_species;
	float * pflux_by_mode_movav;
	float * pflux_by_species_movav;
	

	// Omega as a function of kx and ky
	cuComplex * omega;
	cuComplex * omega_avg;



	// Filenames
	

} outputs_struct;
