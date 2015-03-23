/* Defines structs that are used by geometry_c_interface*/
#include "geo/geometry_c_interface.h"
/*#include "/global/u2/n/nmandell/noah/branches/GS2/trunk/geo/geometry_c_interface.h"*/
/*#include "/global/homes/h/highcock/Code_Carver/gs2/trunk/geo/geometry_c_interface.h"*/

/* Routines for configuring the geometry module*/
extern "C" void geometry_set_inputs_c(int * equilibrium_type,
		 											 char * eqfile,
													 int * irho,
		 											 double * rhoc, 
													 int * bishop,
													 int * nperiod,
													 int * ntheta_out);
extern "C" void geometry_vary_s_alpha_c(double * d, double*e);


/*Routine for running the geometry module*/
extern "C" void geometry_calculate_coefficients_c(int * grid_size_out);

/*Routine for getting the coefficients calculated by the geometry module*/
extern "C" void geometry_get_coefficients_c(int * grid_size_in, struct coefficients_struct * coefficients_out);

extern "C" void geometry_get_miller_parameters_c(struct miller_parameters_struct * miller_parameters_out);

extern "C" void geometry_get_constant_coefficients_c(struct constant_coefficients_struct * constant_coefficients_out);

extern "C" void geometry_mp_finish_geometry_(void);

//void get_gs2_geo(int * Nz, struct coefficients_struct * coefficients, struct constant_coefficients_struct * constant_coefficients){
//  geometry_get_nz(Nz);
//  coefficients = (struct coefficients_struct *)malloc(sizeof(struct coefficients_struct)*(*Nz));
//  geometry_get_constant_coefficients_c(constant_coefficients);
//  geometry_get_coefficients_c(&Nz, coefficients);
//}

void read_geo(int * Nz, struct coefficients_struct * coefficients, struct constant_coefficients_struct * constant_coefficients){
	double s_hat_input, beta_prime_input;
        geometry_vary_s_alpha_c(&s_hat_input, &beta_prime_input);	

	double rhoc, delrho;
	int equilibrium_type, ntheta_out, nperiod;
	int bishop, irho;
	char * eqfile;

	eqfile = (char *)malloc(sizeof(char)*800);
	equilibrium_type = 3; // CHEASE
	equilibrium_type = 1; //Miller
	//rhoc = 0.6;
	delrho = 0.01; /* Leave at this value*/
	nperiod = 1;
	eqfile = "ogyropsi.dat"; 
	ntheta_out = 16;
	bishop = 1;
	irho = 2;

	geometry_set_inputs_c(&equilibrium_type, eqfile, &irho, &rhoc, &bishop, &nperiod, &ntheta_out);

	int grid_size_out;

	geometry_calculate_coefficients_c(&grid_size_out);
	printf("Grid size out was %d\n", grid_size_out);

	

	coefficients = (struct coefficients_struct *)malloc(sizeof(struct coefficients_struct)*grid_size_out);
	geometry_get_coefficients_c(&grid_size_out, coefficients);
	printf("Got  coefficients;\n");
	geometry_get_constant_coefficients_c(constant_coefficients);

	printf("Got constant coefficients...;\n");
	
	*Nz = grid_size_out - 1;
	
	aminor = constant_coefficients->aminor;
	kxfac = constant_coefficients->kxfac;
	eps = .18;
	rmaj = constant_coefficients->rmaj;
	qsf = constant_coefficients->qsf;
	gradpar = 1./(qsf*rmaj);
	shat = constant_coefficients->shat;
	if(shat>1e-6) *&X0 = Y0*jtwist/(2*M_PI*shat);
	
	gbdrift_h = (float*) malloc(sizeof(float)**Nz);
	grho_h = (float*) malloc(sizeof(float)**Nz);
	z_h = (float*) malloc(sizeof(float)**Nz);
	cvdrift_h = (float*) malloc(sizeof(float)**Nz);
	gds2_h = (float*) malloc(sizeof(float)**Nz);
	bmag_h = (float*) malloc(sizeof(float)**Nz);
	bgrad_h = (float*) malloc(sizeof(float)**Nz);     //
	gds21_h = (float*) malloc(sizeof(float)**Nz);
	gds22_h = (float*) malloc(sizeof(float)**Nz);
	cvdrift0_h = (float*) malloc(sizeof(float)**Nz);
	gbdrift0_h = (float*) malloc(sizeof(float)**Nz); 
	jacobian_h = (float*) malloc(sizeof(float)**Nz);
  Rplot_h = (float*) malloc(sizeof(float)**Nz); 
  Zplot_h = (float*) malloc(sizeof(float)**Nz); 
  aplot_h = (float*) malloc(sizeof(float)**Nz); 
  Xplot_h = (float*) malloc(sizeof(float)**Nz); 
  Yplot_h = (float*) malloc(sizeof(float)**Nz); 
  deltaFL_h = (float*) malloc(sizeof(float)**Nz); 
    
	
	for(int k=0; k<*Nz; k++) {
	  z_h[k] = 2*M_PI*(k-*Nz/2)/ *Nz;
	  bmag_h[k] = coefficients[k].bmag;
	  //bgrad_h[k] = 1;                         //should calculate gradpar*d(bmag)/d(theta) with FFT?
	  gds2_h[k] = coefficients[k].gds2;
	  gds21_h[k] = coefficients[k].gds21;
	  gds22_h[k] = coefficients[k].gds22;
	  gbdrift_h[k] = coefficients[k].gbdrift;
	  cvdrift_h[k] = coefficients[k].cvdrift;
	  gbdrift0_h[k] = coefficients[k].gbdrift0;
	  cvdrift0_h[k] = coefficients[k].cvdrift0;
	  grho_h[k] = coefficients[k].grho;
	  jacobian_h[k] = coefficients[k].jacob; 
	}
 
  geometry_mp_finish_geometry_();
	  

	/*
	printf("\nbmag was %f %f %f %f %f etc\n ", 
			bmag_h[0],//coefficients[0].bmag,
			bmag_h[1],//coefficients[1].bmag,
			bmag_h[2],//coefficients[2].bmag,
			bmag_h[3],//coefficients[3].bmag,
			bmag_h[4] //coefficients[4].bmag
			);
	*/

	/*free(coefficients);*/

}
