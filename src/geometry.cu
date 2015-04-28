#define NO_GLOBALS true
#include "standard_headers.h"
/* Defines structs that are used by geometry_c_interface*/
#include "geo/geometry_c_interface.h"
#include "gryfx_lib.h"
#include "allocations.h"
#include "get_error.h"
#include "math.h"
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

//Defined at the bottom
void read_geo_input(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, FILE* ifile); 
void run_general_geometry_module(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, struct gryfx_parameters_struct * gryfxpars);

void set_geometry(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, struct gryfx_parameters_struct * gryfxpars){


  //Local reference for convenience
    float * z_h ;

  
  if ( pars->igeo == 0 ) // this is s-alpha
  {
    //grids->Nz = Nz;
		allocate_geo(ALLOCATE, ON_HOST, geo, &grids->z, &grids->Nz);
    z_h = grids->z;
         
    
    geo->gradpar = (float) 1./(pars->qsf*pars->rmaj);
    
    pars->drhodpsi = 1.; 
    
    for(int k=0; k<grids->Nz; k++) {
      z_h[k] = 2*M_PI*pars->Zp*(k-grids->Nz/2)/grids->Nz;
      geo->bmag[k] = 1./(1+pars->eps*cos(z_h[k]));
      geo->bgrad[k] = geo->gradpar*pars->eps*sin(z_h[k])*geo->bmag[k];            //bgrad = d/dz ln(B(z)) = 1/B dB/dz
      geo->gds2[k] = 1. + pow((pars->shat*z_h[k]-pars->shift*sin(z_h[k])),2);
      geo->gds21[k] = -pars->shat*(pars->shat*z_h[k]-pars->shift*sin(z_h[k]));
      geo->gds22[k] = pow(pars->shat,2);
      geo->gbdrift[k] = 1./(2.*pars->rmaj)*( cos(z_h[k]) + (pars->shat*z_h[k]-pars->shift*sin(z_h[k]))*sin(z_h[k]) );
      geo->cvdrift[k] = geo->gbdrift[k];
      geo->gbdrift0[k] = -1./(2.*pars->rmaj)*pars->shat*sin(z_h[k]);
      geo->cvdrift0[k] = geo->gbdrift0[k];
      geo->grho[k] = 1;
      if(pars->const_curv) {
        geo->cvdrift[k] = 1./(2.*pars->rmaj);
        geo->gbdrift[k] = 1./(2.*pars->rmaj);
        geo->cvdrift0[k] = 0.;
        geo->gbdrift0[k] = 0.;
      }
      if(pars->slab) {
        //omegad=0:
        geo->cvdrift[k] = 0.;
        geo->gbdrift[k] = 0.;       
        geo->cvdrift0[k] = 0.;
        geo->gbdrift0[k] = 0.;
        //bgrad=0:
        geo->bgrad[k] = 0.;
        //bmag=const:
        geo->bmag[k] = 1.;
      }
    }  
  }
  else if ( pars->igeo == 1) // read geometry from file 
  {
    FILE* geoFile = fopen(pars->geofilename, "r");
    printf("Reading eik geo file %s\n", pars->geofilename);
    read_geo_input(pars, grids, geo, geoFile);
  }
  else if ( pars->igeo == 2 ) // calculate geometry from geo module
  {
    
    run_general_geometry_module(pars,grids,geo,gryfxpars);
   
    // We calculate eps so that it gets the right
    // value of eps/qsf in the appropriate places.
    // However, eps also appears by itself... these
    // places need to be checked.
    //double eps_over_q;
    //bi = geometry_mp_bi_out_; 
    float eps_over_q;
    eps_over_q = 1.0 / (geo->bi * pars->drhodpsi);
    pars->eps = eps_over_q * pars->qsf;
    for(int k=0; k<grids->Nz; k++) {
      geo->Xplot[k] = geo->Rplot[k]*cos(geo->aplot[k]);
      geo->Yplot[k] = geo->Rplot[k]*sin(geo->aplot[k]);
    }
  } 
  
}

void run_general_geometry_module(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo,  struct gryfx_parameters_struct * gryfxpars){
	double s_hat_input_d, beta_prime_input_d;
  
  struct coefficients_struct * coefficients;
  struct constant_coefficients_struct constant_coefficients;

	//double delrho; 
	int equilibrium_type, ntheta_out;
	int irho;
	char * eqfile;
  struct miller_parameters_struct millerpars;

//  printf("I AM HEREEE %d\n;", iproc);
	eqfile = (char *)malloc(sizeof(char)*800);
	equilibrium_type = 3; // CHEASE
	equilibrium_type = 1; //Miller
	//rhoc = 0.6;
	//delrho = 0.01; /* Leave at this value*/
	//nperiod = 1;
	eqfile = "ogyropsi.dat"; 
	ntheta_out = 16;
	//bishop = 1;
	irho = 2;

  // Need to set ntheta_out for Miller
  ntheta_out = grids->Nz;

  printf("nperiod = %d\n", pars->nperiod);

	geometry_set_inputs_c(&equilibrium_type, eqfile, &irho, &gryfxpars->rhoc, &pars->bishop, &gryfxpars->nperiod, &ntheta_out);

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
  s_hat_input_d = pars->s_hat_input;
  beta_prime_input_d = pars->beta_prime_input;
  geometry_vary_s_alpha_c(&s_hat_input_d, &beta_prime_input_d);	

  printf("Nz is %d\n", grids->Nz);

  geometry_set_miller_parameters_c(&millerpars);

	int grid_size_out;

  grid_size_out = grids->Nz + 1;

	geometry_calculate_coefficients_c(&grid_size_out);
	printf("Grid size out was %d\n", grid_size_out);

	
	coefficients = (struct coefficients_struct *)malloc(sizeof(struct coefficients_struct)*grid_size_out);
	geometry_get_coefficients_c(&grid_size_out, coefficients);
	printf("Got  coefficients;\n");
	geometry_get_constant_coefficients_c(&constant_coefficients);

	printf("Got constant coefficients...;\n");
	
	grids->Nz = grid_size_out - 1;
	
	geo->aminor = constant_coefficients.aminor;
	pars->kxfac = constant_coefficients.kxfac;
	//eps = .18;
	pars->rmaj = constant_coefficients.rmaj;
	pars->qsf = constant_coefficients.qsf;
	//geo->gradpar = 1./(qsf*pars->rmaj);
	pars->shat = constant_coefficients.shat;
	geo->gradpar = coefficients[0].gradpar;
	pars->drhodpsi = constant_coefficients.drhodpsin;
	geo->bi = constant_coefficients.bi;

  //grids->Nz = Nz;
	allocate_geo(ALLOCATE, ON_HOST, geo, &grids->z, &grids->Nz);

//  printf("I AM HERE %d\n;", iproc);
//MPI_Bcast(&shat, 1, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(&qsf, 1, MPI_FLOAT, 0, mpcom_global);
//MPI_Bcast(Nz, 1, MPI_INT, 0, mpcom_global);

	printf("Adjusting jtwist, shat = %e...;\n", pars->shat);
	if(pars->shat>1e-6) pars->x0 = pars->y0*pars->jtwist/(2*M_PI*pars->shat);

  //z_regular_h = (float*) malloc(sizeof(float)*grids->Nz);
	
  printf("Setting coefficients, Nz = %d\n", grids->Nz);
	for(int k=0; k<grids->Nz; k++) {
	  //z_h[k] = 2*M_PI*(k-grids->Nz/2)/ grids->Nz;
    grids->z[k] = coefficients[k].theta_eqarc;
    //z_regular_h[k] = coefficients[k].theta;
	  geo->gradpar_arr[k] = coefficients[k].gradpar_eqarc;
	  geo->bmag[k] = coefficients[k].bmag_eqarc;
	  //geo->bgrad[k] = 1_eqarc;                         //should calculate gradpar*d(bmag)/d(theta) with FFT?
	  geo->gds2[k] = coefficients[k].gds2_eqarc;
	  geo->gds21[k] = coefficients[k].gds21_eqarc;
	  geo->gds22[k] = coefficients[k].gds22_eqarc;
	  geo->gbdrift[k] = coefficients[k].gbdrift_eqarc/4.0;
	  geo->cvdrift[k] = coefficients[k].cvdrift_eqarc/4.0;
	  geo->gbdrift0[k] = coefficients[k].gbdrift0_eqarc/4.0;
	  geo->cvdrift0[k] = coefficients[k].cvdrift0_eqarc/4.0;
	  geo->grho[k] = coefficients[k].grho_eqarc;
	  geo->jacobian[k] = coefficients[k].jacob_eqarc; 
	  geo->Rplot[k] = coefficients[k].Rplot_eqarc; 
	  geo->Zplot[k] = coefficients[k].Zplot_eqarc; 
	  geo->aplot[k] = coefficients[k].aplot_eqarc; 
	  geo->Rprime[k] = coefficients[k].Rplot_eqarc; 
	  geo->Zprime[k] = coefficients[k].Zplot_eqarc; 
	  geo->aprime[k] = coefficients[k].aplot_eqarc; 
   
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
void read_geo_input(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, FILE* ifile) 
{    
  int nLines;
  fpos_t* lineStartPos;
  int ch;

  int ntgrid;

  
  rewind(ifile);
  //find number of lines
  while( (ch = fgetc(ifile)) != EOF) 
  {
    if(ch == '\n') {
      nLines++;
    }  
  }
  lineStartPos = (fpos_t*) malloc(sizeof(fpos_t)*nLines);
  int i = 2;
  rewind(ifile);
  fgetpos(ifile, &lineStartPos[1]);
  while( (ch = fgetc(ifile)) != EOF) 
  {
    if(ch == '\n') {
      fgetpos(ifile, &lineStartPos[i]);
      i++;
    }
  }
  
  //lineStartPos[i] is start of line i in file... first line is i=1 (not 0)
  fsetpos(ifile, &lineStartPos[2]);
  fscanf(ifile, "%d %d %d %f %f %f %f %f", &ntgrid, &pars->nperiod, &grids->Nz, &pars->drhodpsi, &pars->rmaj, &pars->shat, &pars->kxfac, &pars->qsf);
  if(pars->debug) printf("\n\nIN READ_GEO_INPUT:\nntgrid = %d, nperiod = %d, Nz = %d, rmaj = %f\n\n\n", ntgrid, pars->nperiod, grids->Nz, pars->rmaj);

	allocate_geo(ALLOCATE, ON_HOST, geo, &grids->z, &grids->Nz);

  //Local copy for convenience
  int Nz = grids->Nz;
  

  //first block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4]);
    fscanf(ifile, "%f %f %f %f", &geo->gbdrift[i], &geo->gradpar, &geo->grho[i], &grids->z[i]);
    geo->gbdrift[i] = (1./4.)*geo->gbdrift[i];    
//    printf("z: %f \n", z[i]);
  }

  //second block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4 +1*(Nz+2)]);
    fscanf(ifile, "%f %f %f", &geo->cvdrift[i], &geo->gds2[i], &geo->bmag[i]);
    geo->cvdrift[i] = (1./4.)*geo->cvdrift[i];
  }

  //third block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4 +2*(Nz+2)]);
    fscanf(ifile, "%f %f", &geo->gds21[i], &geo->gds22[i]);
  }

  //fourth block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4 +3*(Nz+2)]);
    fscanf(ifile, "%f %f", &geo->cvdrift0[i], &geo->gbdrift0[i]);
    geo->cvdrift0[i] = (1./4.)*geo->cvdrift0[i];
    geo->gbdrift0[i] = (1./4.)*geo->gbdrift0[i];
    //if(DEBUG) printf("z: %f \n", cvdrift0[i]);  
  }
  
}         

void copy_geo_arrays_to_device(geometry_coefficents_struct * geo, geometry_coefficents_struct * geo_h, input_parameters_struct * pars, int Nz){
  //return;
  cudaMemcpy(geo->gbdrift, geo_h->gbdrift, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(geo->grho, geo_h->grho, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(geo->cvdrift, geo_h->cvdrift, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(geo->gds2, geo_h->gds2, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(geo->bmag, geo_h->bmag, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  if(pars->igeo==0) cudaMemcpy(geo->bgrad, geo_h->bgrad, sizeof(float)*Nz, cudaMemcpyHostToDevice);    //
  cudaMemcpy(geo->gds21, geo_h->gds21, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(geo->gds22, geo_h->gds22, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(geo->cvdrift0, geo_h->cvdrift0, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  cudaMemcpy(geo->gbdrift0, geo_h->gbdrift0, sizeof(float)*Nz, cudaMemcpyHostToDevice);
  if(pars->debug) getError("run_gryfx.cu, after memcpy");
}
