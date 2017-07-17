#include "geometry.h"
#include "parameters.h"
#include "device_funcs.h"
#include "cuda_constants.h"

__global__ void init_kperp2(float* kperp2, float* kx, float* ky, float* gds2, float* gds21, float* gds22, float* bmagInv, float shat) ;
__global__ void init_omegad(float* omegad, float* kx, float* ky, float* gb, float* cv, float* gb0, float* cv0, float shat) ;


Geometry::~Geometry() {
  cudaFree(z);
  cudaFree(bmag);
  cudaFree(bmagInv);
  cudaFree(bgrad);
  cudaFree(gds2);	
  cudaFree(gds21);	
  cudaFree(gds22);	
  cudaFree(gbdrift);	
  cudaFree(gbdrift0);	
  cudaFree(cvdrift);	
  cudaFree(cvdrift0);	
  cudaFree(grho);	
  cudaFree(jacobian);	

  cudaFreeHost(z_h);
  cudaFreeHost(bmag_h);
  cudaFreeHost(bmagInv_h);
  cudaFreeHost(bgrad_h);
  cudaFreeHost(gds2_h);	
  cudaFreeHost(gds21_h);	
  cudaFreeHost(gds22_h);	
  cudaFreeHost(gbdrift_h);	
  cudaFreeHost(gbdrift0_h);	
  cudaFreeHost(cvdrift_h);	
  cudaFreeHost(cvdrift0_h);	
  cudaFreeHost(grho_h);	
  cudaFreeHost(jacobian_h);	

  if(operator_arrays_allocated_) {
    cudaFree(kperp2);
    cudaFree(omegad);
  }
}

S_alpha_geo::S_alpha_geo(Parameters *pars) 
{
    operator_arrays_allocated_=false;
    size_t size = sizeof(float)*pars->nz_in;
    cudaMallocHost((void**) &z_h, size);
    cudaMallocHost((void**) &bmag_h, size);
    cudaMallocHost((void**) &bmagInv_h, size);
    cudaMallocHost((void**) &bgrad_h, size);
    cudaMallocHost((void**) &gds2_h, size);
    cudaMallocHost((void**) &gds21_h, size);
    cudaMallocHost((void**) &gds22_h, size);
    cudaMallocHost((void**) &gbdrift_h, size);
    cudaMallocHost((void**) &gbdrift0_h, size);
    cudaMallocHost((void**) &cvdrift_h, size);
    cudaMallocHost((void**) &cvdrift0_h, size);
    cudaMallocHost((void**) &grho_h, size);
    cudaMallocHost((void**) &jacobian_h, size);

    cudaMalloc((void**) &z, size);
    cudaMalloc((void**) &bmag, size);
    cudaMalloc((void**) &bmagInv, size);
    cudaMalloc((void**) &bgrad, size);
    cudaMalloc((void**) &gds2, size);
    cudaMalloc((void**) &gds21, size);
    cudaMalloc((void**) &gds22, size);
    cudaMalloc((void**) &gbdrift, size);
    cudaMalloc((void**) &gbdrift0, size);
    cudaMalloc((void**) &cvdrift, size);
    cudaMalloc((void**) &cvdrift0, size);
    cudaMalloc((void**) &grho, size);
    cudaMalloc((void**) &jacobian, size);
    
    gradpar = (float) abs(1./(pars->qsf*pars->rmaj));

    shat = pars->shat;
    
    pars->drhodpsi = 1.; 

    float qsf = pars->qsf;
    float beta_e = pars->beta;
    float rmaj = pars->rmaj;
    specie* species = pars->species;

    if(pars->shift < 0.) {
      pars->shift = 0.;
      for(int s=0; s<pars->nspec_in; s++) { 
        pars->shift += qsf*qsf*rmaj*beta_e*(species[s].temp/species[pars->nspec_in-1].temp)*(species[s].tprim + species[s].fprim);
      }
    }
    
    for(int k=0; k<pars->nz_in; k++) {
      z_h[k] = 2.*M_PI*pars->Zp*(k-pars->nz_in/2)/pars->nz_in;
      if(pars->local_limit) {z_h[k] = 0.;} // outboard-midplane
      bmag_h[k] = 1./(1.+pars->eps*cos(z_h[k]));
      bgrad_h[k] = gradpar*pars->eps*sin(z_h[k])*bmag_h[k];            //bgrad = d/dz ln(B(z)) = 1/B dB/dz
      gds2_h[k] = 1. + pow((pars->shat*z_h[k]-pars->shift*sin(z_h[k])),2);
      gds21_h[k] = -pars->shat*(pars->shat*z_h[k]-pars->shift*sin(z_h[k]));
      gds22_h[k] = pow(pars->shat,2);
      gbdrift_h[k] = 1./(2.*pars->rmaj)*( cos(z_h[k]) + (pars->shat*z_h[k]-pars->shift*sin(z_h[k]))*sin(z_h[k]) );
      cvdrift_h[k] = gbdrift_h[k];
      gbdrift0_h[k] = -1./(2.*pars->rmaj)*pars->shat*sin(z_h[k]);
      cvdrift0_h[k] = gbdrift0_h[k];
      grho_h[k] = 1;
      if(pars->const_curv) {
        cvdrift_h[k] = 1./(2.*pars->rmaj);
        gbdrift_h[k] = 1./(2.*pars->rmaj);
        cvdrift0_h[k] = 0.;
        gbdrift0_h[k] = 0.;
      }
      if(pars->slab) {
        //omegad=0:
        cvdrift_h[k] = 0.;
        gbdrift_h[k] = 0.;       
        cvdrift0_h[k] = 0.;
        gbdrift0_h[k] = 0.;
        //bgrad=0:
        bgrad_h[k] = 0.;
        //bmag=const:
        bmag_h[k] = 1.;
        //gradpar = 1.;
      }
      if(pars->local_limit) z_h[k] = 2*M_PI*pars->Zp*(k-pars->nz_in/2)/pars->nz_in;

      // calculate these derived coefficients after slab overrides
      bmagInv_h[k] = 1./bmag_h[k];
      jacobian_h[k] = 1. / abs(pars->drhodpsi*gradpar*bmag_h[k]);
    }  

    cudaMemcpy(z       , z_h        , size, cudaMemcpyHostToDevice);
    cudaMemcpy(gbdrift , gbdrift_h  , size, cudaMemcpyHostToDevice);
    cudaMemcpy(grho    , grho_h     , size, cudaMemcpyHostToDevice);
    cudaMemcpy(cvdrift , cvdrift_h  , size, cudaMemcpyHostToDevice);
    cudaMemcpy(bmag    , bmag_h     , size, cudaMemcpyHostToDevice);
    cudaMemcpy(bmagInv , bmagInv_h  , size, cudaMemcpyHostToDevice);
    cudaMemcpy(bgrad   , bgrad_h    , size, cudaMemcpyHostToDevice);
    cudaMemcpy(gds2    , gds2_h     , size, cudaMemcpyHostToDevice);
    cudaMemcpy(gds21   , gds21_h    , size, cudaMemcpyHostToDevice);
    cudaMemcpy(gds22   , gds22_h    , size, cudaMemcpyHostToDevice);
    cudaMemcpy(cvdrift0, cvdrift0_h , size, cudaMemcpyHostToDevice);
    cudaMemcpy(gbdrift0, gbdrift0_h , size, cudaMemcpyHostToDevice);
    cudaMemcpy(jacobian, jacobian_h , size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

}

Eik_geo::Eik_geo() {

}

Gs2_geo::Gs2_geo() {

}

// MFM - 07/09/17
File_geo::File_geo(Parameters *pars)
{

  operator_arrays_allocated_=false; // allocating for geo parameters, but once kperp2/omegad are set they aren't needed anymore
  size_t size = sizeof(float)*pars->nz_in; 
  cudaMallocHost((void**) &z_h, size);
  cudaMallocHost((void**) &bmag_h, size);
  cudaMallocHost((void**) &bmagInv_h, size);
  cudaMallocHost((void**) &bgrad_h, size); // why?
  cudaMallocHost((void**) &gds2_h, size);
  cudaMallocHost((void**) &gds21_h, size);
  cudaMallocHost((void**) &gds22_h, size);
  cudaMallocHost((void**) &gbdrift_h, size);
  cudaMallocHost((void**) &gbdrift0_h, size);
  cudaMallocHost((void**) &cvdrift_h, size);
  cudaMallocHost((void**) &cvdrift0_h, size);
  cudaMallocHost((void**) &grho_h, size);
  cudaMallocHost((void**) &jacobian_h, size);

  cudaMalloc((void**) &z, size);
  cudaMalloc((void**) &bmag, size);
  cudaMalloc((void**) &bmagInv, size);
  cudaMalloc((void**) &bgrad, size);
  cudaMalloc((void**) &gds2, size);
  cudaMalloc((void**) &gds21, size);
  cudaMalloc((void**) &gds22, size);
  cudaMalloc((void**) &gbdrift, size);
  cudaMalloc((void**) &gbdrift0, size);
  cudaMalloc((void**) &cvdrift, size);
  cudaMalloc((void**) &cvdrift0, size);
  cudaMalloc((void**) &grho, size);
  cudaMalloc((void**) &jacobian, size);

  FILE * geoFile = fopen(pars->geofilename, "r");

  if (geoFile == NULL) {
    printf("Cannot open file %s \n", geoFile);
    exit(0);
      }

  int nlines=0;
  fpos_t* lineStartPos;
  int ch;

  int ntgrid;
  int oldNz, oldnperiod;

  rewind(geoFile);
  nlines=0;

  // Find number of lines
  while( (ch = fgetc(geoFile)) != EOF)
    {
      if(ch == '\n') {
	nlines++;
      }
    }
  printf("Counted %d lines in geofile.\n",nlines);

  lineStartPos = (fpos_t*) malloc(sizeof(fpos_t)*nlines);
  int i=2;
  rewind(geoFile);
  fgetpos(geoFile, &lineStartPos[1]);
  while( (ch = fgetc(geoFile)) != EOF)
    {
      if(ch == '\n') {
	fgetpos(geoFile, &lineStartPos[i]);
	i++;
      }
    }

  oldNz = pars->nz_in;
  oldnperiod = pars->nperiod;
  //lineStartPos[1] is the first line, not i=0
  fsetpos(geoFile, &lineStartPos[2]);
  fscanf(geoFile, "%d %d %d %f %f %f %f %f", &ntgrid, &pars->nperiod, &pars->nz_in, &pars->drhodpsi, &pars->rmaj, &pars->shat, &pars->kxfac, &pars->qsf);
  if(pars->debug) printf("\n\nIN READ_GEO_INPUT:\nntgrid = %d, nperiod = %d, nz_in = %d, rmaj = %f\n\n\n", ntgrid, pars->nperiod, pars->nz_in, pars->rmaj);

  if(oldNz != pars->nz_in) {
    printf("You must set ntheta in the namelist equal to ntheta in the geofile. Exiting...\n");
    abort();
  }
  if(oldnperiod != pars->nperiod) {
    printf("You must set nperiod in the namelist equal to nperiod in the geofile. Exiting...\n");
    abort();
  }

  // Local copy to simplify loops
  int nz_in = pars->nz_in;

  //first block
  for(int i=0; i<nz_in; i++) {
    fsetpos(geoFile, &lineStartPos[i+4]);
    fscanf(geoFile, "%f %f %f %f", &gbdrift_h[i], &gradpar, &grho_h[i], &z_h[i]);
    gbdrift_h[i] = (1./4.)*gbdrift_h[i];
  }
  //printf("gbdrift[0]: %.7e    gbdrift[end]: %.7e\n",4.*gbdrift_h[0],4.*gbdrift_h[nz_in-1]);
  //printf("z[0]: %.7e    z[end]: %.7e\n",z_h[0],z_h[nz_in-1]);

  //second block
  for(int i=0; i<nz_in; i++) {
    fsetpos(geoFile, &lineStartPos[(i+4) + 1*(nz_in+2)]);
    fscanf(geoFile, "%f %f %f", &cvdrift_h[i], &gds2_h[i], &bmag_h[i]);
    cvdrift_h[i] = (1./4.)*cvdrift_h[i];
    bmagInv_h[i] = 1./bmag_h[i];
    jacobian_h[i] = 1. / abs(pars->drhodpsi*gradpar*bmag_h[i]);
    grho_h[i] = 1.;
    // Is bgrad necessary??
    bgrad_h[i] = gradpar*pars->eps*sin(z_h[i])*bmag_h[i];            //bgrad = d/dz ln(B(z)) = 1/B dB/dz
  }
  //printf("cvdrift[0]: %.7e    cvdrift[end]: %.7e\n",4.*cvdrift_h[0],4.*cvdrift_h[nz_in-1]);
  //printf("bmag[0]: %.7e    bmag[end]: %.7e\n",bmag_h[0],bmag_h[nz_in-1]);
  //printf("gds2[0]: %.7e    gds2[end]: %.7e\n",gds2_h[0],gds2_h[nz_in-1]);

  //third block
  for(int i=0; i<nz_in; i++) {
    fsetpos(geoFile, &lineStartPos[(i+4) + 2*(nz_in+2)]);
    fscanf(geoFile, "%f %f", &gds21_h[i], &gds22_h[i]);
  }
  //printf("gds21[0]: %.7e    gds21[end]: %.7e\n",gds21_h[0],gds21_h[nz_in-1]);
  //printf("gds22[0]: %.7e    gds22[end]: %.7e\n",gds22_h[0],gds22_h[nz_in-1]);

  //fourth block
  for(int i=0; i<nz_in; i++) {
    fsetpos(geoFile, &lineStartPos[(i+4) + 3*(nz_in+2)]);
    fscanf(geoFile, "%f %f", &cvdrift0_h[i], &gbdrift0_h[i]);
    cvdrift0_h[i] = (1./4.)*cvdrift0_h[i];
    gbdrift0_h[i] = (1./4.)*gbdrift0_h[i];
  }
  printf("All geometry information read successfully\n");
  //printf("cvdrift0[0]: %.7e    cvdrift0[end]: %.7e\n",4.*cvdrift0_h[0],4.*cvdrift0_h[nz_in-1]);
  //printf("gbdrift0[0]: %.7e    gbdrift0[end]: %.7e\n",4.*gbdrift0_h[0],4.*gbdrift0_h[nz_in-1]);

  //copy host variables to device variables
    cudaMemcpy(z, z_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gbdrift, gbdrift_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(grho, grho_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cvdrift, cvdrift_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(bmag, bmag_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(bmagInv, bmagInv_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(bgrad, bgrad_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gds2, gds2_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gds21, gds21_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gds22, gds22_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cvdrift0, cvdrift0_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gbdrift0, gbdrift0_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(jacobian, jacobian_h, size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

void Geometry::initializeOperatorArrays(Parameters* pars, Grids* grids) {

  // set this flag so we know to deallocate
  operator_arrays_allocated_ = true;

  cudaMalloc((void**) &kperp2, sizeof(float)*grids->NxNycNz);
  cudaMalloc((void**) &omegad, sizeof(float)*grids->NxNycNz);

  dim3 dimBlock(32, 4, 4);
  dim3 dimGrid(grids->Nyc/dimBlock.x+1, grids->Nx/dimBlock.y+1, grids->Nz/dimBlock.z+1);

  init_kperp2<<<dimGrid, dimBlock>>>(kperp2, grids->kx, grids->ky, gds2, gds21, gds22, bmagInv, shat);

  init_omegad<<<dimGrid, dimBlock>>>(omegad, grids->kx, grids->ky, gbdrift, cvdrift, gbdrift0, cvdrift0, shat);
}

__global__ void init_kperp2(float* kperp2, float* kx, float* ky, float* gds2, float* gds21, float* gds22, float* bmagInv, float shat) 
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  float shatInv = 1./shat;

  if(idy<nyc && idx<nx && idz<nz) {
    unsigned int idxyz = idy + nyc*idx + nx*nyc*idz;
    kperp2[idxyz] = ( ky[idy] * (ky[idy]*gds2[idz] 
                      + 2.*kx[idx]*shatInv*gds21[idz]) 
                      + pow(kx[idx]*shatInv,2)*gds22[idz] ) 
                      * pow(bmagInv[idz],2); 
  }
}

__global__ void init_omegad(float* omegad, float* kx, float* ky, float* gb, float* cv, float* gb0, float* cv0, float shat) 
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  float shatInv = 1./shat;

  if(idy<nyc && idx<nx && idz<nz) {
    unsigned int idxyz = idy + nyc*idx + nx*nyc*idz;
    omegad[idxyz] = ky[idy] * (gb[idz] + cv[idz]) 
                  + kx[idx] * shatInv * (gb0[idz] + cv0[idz]) ; 
  }
}

/* Routines for configuring the geometry module*/
//extern "C" void geometry_set_inputs_c(int * equilibrium_type,
//		 											 char * eqfile,
//													 int * irho,
//		 											 double * rhoc, 
//													 int * bishop,
//													 int * nperiod,
//													 int * ntheta_out);
//extern "C" void geometry_vary_s_alpha_c(double * d, double*e);
//
//
///*Routine for running the geometry module*/
//extern "C" void geometry_calculate_coefficients_c(int * grid_size_out);
//
///*Routine for getting the coefficients calculated by the geometry module*/
//extern "C" void geometry_get_coefficients_c(int * grid_size_in, struct coefficients_struct * coefficients_out);
//
//extern "C" void geometry_get_miller_parameters_c(struct miller_parameters_struct * miller_parameters_out);
//extern "C" void geometry_set_miller_parameters_c(struct miller_parameters_struct * miller_parameters_in);
//
//extern "C" void geometry_get_constant_coefficients_c(struct constant_coefficients_struct * constant_coefficients_out);
//
//extern "C" void geometry_mp_finish_geometry_(void);
//
////Defined at the bottom
//void read_geo_input(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, FILE* ifile); 
//void run_general_geometry_module(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, struct gryfx_parameters_struct * gryfxpars);

//void set_geometry(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, struct gryfx_parameters_struct * gryfxpars){
//
//
//  //Local reference for convenience
//    float * z_h ;
//
//  
//  if ( pars->igeo == 0 ) // this is s-alpha
//  {
//  }
//  else if ( pars->igeo == 1) // read geometry from file 
//  {
//    FILE* geoFile = fopen(pars->geofilename, "r");
//    printf("Reading eik geo file %s\n", pars->geofilename);
//    read_geo_input(pars, grids, geo, geoFile);
//  }
//  else if ( pars->igeo == 2 ) // calculate geometry from gs2 geo module
//  {
//    int len = strlen(namelistFile);
//    init_gs2_file_utils(&len, namelistFile); // needed for general geometry module
//    run_general_geometry_module(pars,grids,geo,gryfxpars);
//    finish_gs2_file_utils(); // needed for general geometry module
//   
//    // We calculate eps so that it gets the right
//    // value of eps/qsf in the appropriate places.
//    // However, eps also appears by itself... these
//    // places need to be checked.
//    //double eps_over_q;
//    //bi = geometry_mp_bi_out_; 
//    float eps_over_q;
//    eps_over_q = 1.0 / (geo->bi * pars->drhodpsi);
//    pars->eps = eps_over_q * pars->qsf;
//    for(int k=0; k<grids->Nz; k++) {
//      geo->Xplot[k] = geo->Rplot[k]*cos(geo->aplot[k]);
//      geo->Yplot[k] = geo->Rplot[k]*sin(geo->aplot[k]);
//    }
//  } 
//  
//}
//
//void run_general_geometry_module(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo,  struct gryfx_parameters_struct * gryfxpars){
//	double s_hat_input_d, beta_prime_input_d;
//  
//  struct coefficients_struct * coefficients;
//  struct constant_coefficients_struct constant_coefficients;
//
//	//double delrho; 
//	int equilibrium_type, ntheta_out;
//	int irho;
//	char * eqfile;
//  struct miller_parameters_struct millerpars;
//
////  printf("I AM HEREEE %d\n;", iproc);
//	eqfile = (char *)malloc(sizeof(char)*800);
//	equilibrium_type = 3; // CHEASE
//	equilibrium_type = 1; //Miller
//	//rhoc = 0.6;
//	//delrho = 0.01; /* Leave at this value*/
//	//nperiod = 1;
//	eqfile = "ogyropsi.dat"; 
//	ntheta_out = 16;
//	//bishop = 1;
//	irho = 2;
//
//  // Need to set ntheta_out for Miller
//  ntheta_out = grids->Nz;
//
//  printf("nperiod = %d\n", pars->nperiod);
//
//	geometry_set_inputs_c(&equilibrium_type, eqfile, &irho, &gryfxpars->rhoc, &pars->bishop, &gryfxpars->nperiod, &ntheta_out);
//
//  geometry_get_miller_parameters_c(&millerpars);
//  millerpars.rmaj=gryfxpars->rgeo_lcfs;
//  millerpars.R_geo=gryfxpars->rgeo_local;
//  millerpars.akappa=gryfxpars->akappa;
//  millerpars.akappri=gryfxpars->akappri;
//  millerpars.tri=gryfxpars->tri;
//  millerpars.tripri=gryfxpars->tripri;
//  millerpars.shift=gryfxpars->shift;
//  millerpars.qinp=gryfxpars->qinp;
//  millerpars.shat=gryfxpars->shat;
//
//  // The effect of this function depends on the value of bishop
//  // E.g., bishop = 1 ---> this function has no effect
//  //       bishop = 4 ---> this function sets both shat and beta_prime using the bishop relations
//  // other permutations are possible, see http://gyrokinetics.sourceforge.net/wiki/index.php/Gs2_Input_Parameters
//  s_hat_input_d = pars->s_hat_input;
//  beta_prime_input_d = pars->beta_prime_input;
//  geometry_vary_s_alpha_c(&s_hat_input_d, &beta_prime_input_d);	
//
//  printf("Nz is %d\n", grids->Nz);
//
//  geometry_set_miller_parameters_c(&millerpars);
//
//	int grid_size_out;
//
//  grid_size_out = grids->Nz + 1;
//
//	geometry_calculate_coefficients_c(&grid_size_out);
//	printf("Grid size out was %d\n", grid_size_out);
//
//	
//	coefficients = (struct coefficients_struct *)malloc(sizeof(struct coefficients_struct)*grid_size_out);
//	geometry_get_coefficients_c(&grid_size_out, coefficients);
//	printf("Got  coefficients;\n");
//	geometry_get_constant_coefficients_c(&constant_coefficients);
//
//	printf("Got constant coefficients...;\n");
//	
//	grids->Nz = grid_size_out - 1;
//	
//	geo->aminor = constant_coefficients.aminor;
//	pars->kxfac = constant_coefficients.kxfac;
//	//eps = .18;
//	pars->rmaj = constant_coefficients.rmaj;
//	pars->qsf = constant_coefficients.qsf;
//	//geo->gradpar = 1./(qsf*pars->rmaj);
//	pars->shat = constant_coefficients.shat;
//	geo->gradpar = coefficients[0].gradpar;
//	pars->drhodpsi = constant_coefficients.drhodpsin;
//	geo->bi = constant_coefficients.bi;
//
//  //grids->Nz = Nz;
//	allocate_geo(ALLOCATE, ON_HOST, geo, &grids->z, &grids->Nz);
//
////  printf("I AM HERE %d\n;", iproc);
////MPI_Bcast(&shat, 1, MPI_FLOAT, 0, mpcom_global);
////MPI_Bcast(&qsf, 1, MPI_FLOAT, 0, mpcom_global);
////MPI_Bcast(Nz, 1, MPI_INT, 0, mpcom_global);
//
//	printf("Adjusting jtwist, shat = %e...;\n", pars->shat);
//	if(pars->shat>1e-6) pars->x0 = pars->y0*pars->jtwist/(2*M_PI*pars->shat);
//
//  //z_regular_h = (float*) malloc(sizeof(float)*grids->Nz);
//	
//  printf("Setting coefficients, Nz = %d\n", grids->Nz);
//	for(int k=0; k<grids->Nz; k++) {
//	  //z_h[k] = 2*M_PI*(k-grids->Nz/2)/ grids->Nz;
//    grids->z[k] = coefficients[k].theta_eqarc;
//    //z_regular_h[k] = coefficients[k].theta;
//	  geo->gradpar_arr[k] = coefficients[k].gradpar_eqarc;
//	  geo->bmag[k] = coefficients[k].bmag_eqarc;
//	  //geo->bgrad[k] = 1_eqarc;                         //should calculate gradpar*d(bmag)/d(theta) with FFT?
//	  geo->gds2[k] = coefficients[k].gds2_eqarc;
//	  geo->gds21[k] = coefficients[k].gds21_eqarc;
//	  geo->gds22[k] = coefficients[k].gds22_eqarc;
//	  geo->gbdrift[k] = coefficients[k].gbdrift_eqarc/4.0;
//	  geo->cvdrift[k] = coefficients[k].cvdrift_eqarc/4.0;
//	  geo->gbdrift0[k] = coefficients[k].gbdrift0_eqarc/4.0;
//	  geo->cvdrift0[k] = coefficients[k].cvdrift0_eqarc/4.0;
//	  geo->grho[k] = coefficients[k].grho_eqarc;
//	  geo->jacobian[k] = coefficients[k].jacob_eqarc; 
//	  geo->Rplot[k] = coefficients[k].Rplot_eqarc; 
//	  geo->Zplot[k] = coefficients[k].Zplot_eqarc; 
//	  geo->aplot[k] = coefficients[k].aplot_eqarc; 
//	  geo->Rprime[k] = coefficients[k].Rprime_eqarc; 
//	  geo->Zprime[k] = coefficients[k].Zprime_eqarc; 
//	  geo->aprime[k] = coefficients[k].aprime_eqarc; 
//   
//	}
// 
//  geometry_mp_finish_geometry_();
//
//	/*
//	printf("\nbmag was %f %f %f %f %f etc\n ", 
//			bmag_h[0],//coefficients[0].bmag,
//			bmag_h[1],//coefficients[1].bmag,
//			bmag_h[2],//coefficients[2].bmag,
//			bmag_h[3],//coefficients[3].bmag,
//			bmag_h[4] //coefficients[4].bmag
//			);
//	*/
//
//	/*free(coefficients);*/
//
//}
//void read_geo_input(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, FILE* ifile) 
//{    
//  int nLines=0;
//  fpos_t* lineStartPos;
//  int ch;
//
//  int ntgrid;
//
//  
//  rewind(ifile);
//  //find number of lines
//  while( (ch = fgetc(ifile)) != EOF) 
//  {
//    if(ch == '\n') {
//      nLines++;
//    }  
//  }
//  lineStartPos = (fpos_t*) malloc(sizeof(fpos_t)*nLines);
//  int i = 2;
//  rewind(ifile);
//  fgetpos(ifile, &lineStartPos[1]);
//  while( (ch = fgetc(ifile)) != EOF) 
//  {
//    if(ch == '\n') {
//      fgetpos(ifile, &lineStartPos[i]);
//      i++;
//    }
//  }
//  
//  //lineStartPos[i] is start of line i in file... first line is i=1 (not 0)
//  fsetpos(ifile, &lineStartPos[2]);
//  fscanf(ifile, "%d %d %d %f %f %f %f %f", &ntgrid, &pars->nperiod, &grids->Nz, &pars->drhodpsi, &pars->rmaj, &pars->shat, &pars->kxfac, &pars->qsf);
//  if(pars->debug) printf("\n\nIN READ_GEO_INPUT:\nntgrid = %d, nperiod = %d, Nz = %d, rmaj = %f\n\n\n", ntgrid, pars->nperiod, grids->Nz, pars->rmaj);
//
//	allocate_geo(ALLOCATE, ON_HOST, geo, &grids->z, &grids->Nz);
//
//  //Local copy for convenience
//  int Nz = grids->Nz;
//  
//
//  //first block
//  for(int i=0; i<Nz; i++) {
//    fsetpos(ifile, &lineStartPos[i+4]);
//    fscanf(ifile, "%f %f %f %f", &geo->gbdrift[i], &geo->gradpar, &geo->grho[i], &grids->z[i]);
//    geo->gbdrift[i] = (1./4.)*geo->gbdrift[i];    
////    printf("z: %f \n", z[i]);
//  }
//
//  //second block
//  for(int i=0; i<Nz; i++) {
//    fsetpos(ifile, &lineStartPos[i+4 +1*(Nz+2)]);
//    fscanf(ifile, "%f %f %f", &geo->cvdrift[i], &geo->gds2[i], &geo->bmag[i]);
//    geo->cvdrift[i] = (1./4.)*geo->cvdrift[i];
//  }
//
//  //third block
//  for(int i=0; i<Nz; i++) {
//    fsetpos(ifile, &lineStartPos[i+4 +2*(Nz+2)]);
//    fscanf(ifile, "%f %f", &geo->gds21[i], &geo->gds22[i]);
//  }
//
//  //fourth block
//  for(int i=0; i<Nz; i++) {
//    fsetpos(ifile, &lineStartPos[i+4 +3*(Nz+2)]);
//    fscanf(ifile, "%f %f", &geo->cvdrift0[i], &geo->gbdrift0[i]);
//    geo->cvdrift0[i] = (1./4.)*geo->cvdrift0[i];
//    geo->gbdrift0[i] = (1./4.)*geo->gbdrift0[i];
//    //if(DEBUG) printf("z: %f \n", cvdrift0[i]);  
//  }
//  
//}         
//
//void copy_geo_arrays_to_device(geometry_coefficents_struct * geo, geometry_coefficents_struct * geo_h, input_parameters_struct * pars, int Nz){
//  //return;
//  cudaMemcpy(geo->gbdrift, geo_h->gbdrift, sizeof(float)*Nz, cudaMemcpyHostToDevice);
//  cudaMemcpy(geo->grho, geo_h->grho, sizeof(float)*Nz, cudaMemcpyHostToDevice);
//  cudaMemcpy(geo->cvdrift, geo_h->cvdrift, sizeof(float)*Nz, cudaMemcpyHostToDevice);
//  cudaMemcpy(geo->gds2, geo_h->gds2, sizeof(float)*Nz, cudaMemcpyHostToDevice);
//  cudaMemcpy(geo->bmag, geo_h->bmag, sizeof(float)*Nz, cudaMemcpyHostToDevice);
//  if(pars->igeo==0) cudaMemcpy(geo->bgrad, geo_h->bgrad, sizeof(float)*Nz, cudaMemcpyHostToDevice);    //
//  cudaMemcpy(geo->gds21, geo_h->gds21, sizeof(float)*Nz, cudaMemcpyHostToDevice);
//  cudaMemcpy(geo->gds22, geo_h->gds22, sizeof(float)*Nz, cudaMemcpyHostToDevice);
//  cudaMemcpy(geo->cvdrift0, geo_h->cvdrift0, sizeof(float)*Nz, cudaMemcpyHostToDevice);
//  cudaMemcpy(geo->gbdrift0, geo_h->gbdrift0, sizeof(float)*Nz, cudaMemcpyHostToDevice);
//  if(pars->debug) getError("run_gryfx.cu, after memcpy");
//}
