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
extern "C" void geometry_set_miller_parameters_c(struct miller_parameters_struct * miller_parameters_in);

extern "C" void geometry_get_constant_coefficients_c(struct constant_coefficients_struct * constant_coefficients_out);

extern "C" void geometry_mp_finish_geometry_(void);

//void get_gs2_geo(int * Nz, struct coefficients_struct * coefficients, struct constant_coefficients_struct * constant_coefficients){
//  geometry_get_nz(Nz);
//  coefficients = (struct coefficients_struct *)malloc(sizeof(struct coefficients_struct)*(*Nz));
//  geometry_get_constant_coefficients_c(constant_coefficients);
//  geometry_get_coefficients_c(&Nz, coefficients);
//}

void read_geo(int * Nz, struct coefficients_struct * coefficients, struct constant_coefficients_struct * constant_coefficients, struct gryfx_parameters_struct * gryfxpars){
	double s_hat_input_d, beta_prime_input_d;

	double delrho; 
	int equilibrium_type, ntheta_out;
	int irho;
	char * eqfile;
  struct miller_parameters_struct millerpars;

  printf("I AM HEREEE %d\n;", iproc);
	eqfile = (char *)malloc(sizeof(char)*800);
	equilibrium_type = 3; // CHEASE
	equilibrium_type = 1; //Miller
	//rhoc = 0.6;
	delrho = 0.01; /* Leave at this value*/
	//nperiod = 1;
	eqfile = "ogyropsi.dat"; 
	ntheta_out = 16;
	//bishop = 1;
	irho = 2;

  // Need to set ntheta_out for Miller
  ntheta_out = *Nz;

  printf("nperiod = %d\n", nperiod);

	geometry_set_inputs_c(&equilibrium_type, eqfile, &irho, &gryfxpars->rhoc, &bishop, &gryfxpars->nperiod, &ntheta_out);

  geometry_get_miller_parameters_c(&millerpars);
  millerpars.rmaj=gryfxpars->rgeo_lcfs;
  millerpars.R_geo=gryfxpars->rgeo_local;
  millerpars.akappa=gryfxpars->akappa;
  millerpars.akappri=gryfxpars->akappri;
  millerpars.tri=gryfxpars->tri;
  millerpars.tripri=gryfxpars->tripri;
  millerpars.shift=gryfxpars->shift;
  millerpars.qinp=gryfxpars->qinp;
  millerpars.shat=gryfxpars->shat;

  // The effect of this function depends on the value of bishop
  // E.g., bishop = 1 ---> this function has no effect
  //       bishop = 4 ---> this function sets both shat and beta_prime using the bishop relations
  // other permutations are possible, see http://gyrokinetics.sourceforge.net/wiki/index.php/Gs2_Input_Parameters
  s_hat_input_d = s_hat_input;
  beta_prime_input_d = beta_prime_input;
  geometry_vary_s_alpha_c(&s_hat_input_d, &beta_prime_input_d);	

  printf("Nz is %d\n", *Nz);

  geometry_set_miller_parameters_c(&millerpars);

	int grid_size_out;

  grid_size_out = *Nz + 1;

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
	//eps = .18;
	rmaj = constant_coefficients->rmaj;
	qsf = constant_coefficients->qsf;
	//gradpar = 1./(qsf*rmaj);
	shat = constant_coefficients->shat;
	gradpar = coefficients[0].gradpar;
	drhodpsi = constant_coefficients->drhodpsin;
	bi = constant_coefficients->bi;

  printf("I AM HERE %d\n;", iproc);
//MPI_Bcast(&shat, 1, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&qsf, 1, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(Nz, 1, MPI_INT, 0, mpcom_global);

	printf("Adjusting jtwist, shat = %e...;\n", shat);
	if(shat>1e-6) *&X0 = Y0*jtwist/(2*M_PI*shat);

	gradpar_h = (float*) malloc(sizeof(float)**Nz);
	gbdrift_h = (float*) malloc(sizeof(float)**Nz);
	grho_h = (float*) malloc(sizeof(float)**Nz);
	z_h = (float*) malloc(sizeof(float)**Nz);
  z_regular_h = (float*) malloc(sizeof(float)**Nz);
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
//Xplot_h = (float*) malloc(sizeof(float)**Nz); 
//Yplot_h = (float*) malloc(sizeof(float)**Nz); 
//deltaFL_h = (float*) malloc(sizeof(float)**Nz); 
if (iproc==0){
	
//FILE* geofile = fopen("geo.out", "w");

  printf("Setting coefficients, Nz = %d\n", *Nz);
	for(int k=0; k<*Nz; k++) {
	  z_h[k] = 2*M_PI*(k-*Nz/2)/ *Nz;
    z_h[k] = coefficients[k].theta_eqarc;
    z_regular_h[k] = coefficients[k].theta;
	  gradpar_h[k] = coefficients[k].gradpar_eqarc;
	  bmag_h[k] = coefficients[k].bmag_eqarc;
	  //bgrad_h[k] = 1_eqarc;                         //should calculate gradpar*d(bmag)/d(theta) with FFT?
	  gds2_h[k] = coefficients[k].gds2_eqarc;
	  gds21_h[k] = coefficients[k].gds21_eqarc;
	  gds22_h[k] = coefficients[k].gds22_eqarc;
	  gbdrift_h[k] = coefficients[k].gbdrift_eqarc/4.0;
	  cvdrift_h[k] = coefficients[k].cvdrift_eqarc/4.0;
	  gbdrift0_h[k] = coefficients[k].gbdrift0_eqarc/4.0;
	  cvdrift0_h[k] = coefficients[k].cvdrift0_eqarc/4.0;
	  grho_h[k] = coefficients[k].grho_eqarc;
	  jacobian_h[k] = coefficients[k].jacob_eqarc; 
	  Rplot_h[k] = coefficients[k].Rplot_eqarc; 
	  Zplot_h[k] = coefficients[k].Zplot_eqarc; 
	  aplot_h[k] = coefficients[k].aplot_eqarc; 
//fprintf(geofile, " %f %f %f %f %f\n", z_h[k], gds2_h[k], gbdrift_h[k], cvdrift_h[k], grho_h[k]);
   
	}
//fclose(geofile);
 
  geometry_mp_finish_geometry_();
}	  
//MPI_Bcast(&z_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&bmag_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&gds2_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&gds21_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&gds22_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&gbdrift0_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&cvdrift0_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&grho_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&jacobian_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&Rplot_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&Zplot_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&aplot_h[0], *Nz, MPI_FLOAT, 0, mpcom_global);
//exit(0);

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
