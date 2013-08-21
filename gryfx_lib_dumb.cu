#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "sys/stat.h"
#include "cufft.h"
#include "cuda_profiler_api.h"

int Nx, Ny, Nz, zThreads, totalThreads;
float X0, Y0, Zp;
__constant__ int nx,ny,nz,zthreads,totalthreads;
__constant__ float X0_d,Y0_d,Zp_d;

dim3 dimBlock, dimGrid;

#define ION 0
#define ELECTRON 1
#define PHI 0
#define DENS 1

//species variables defined in gs2 namelist
int nSpecies;
typedef struct {
  float z;
  float mass;
  float dens;
  float temp;
  float tprim;
  float fprim;
  float uprim;
  float zstm;
  float tz;
  float zt;
  float nu_ss;
  float rho;           
  float vt;    
  char* type;        
} specie;

specie *species;



//global variables
int nClasses;
int *nLinks;
int *nChains;
cuComplex nu[11];
cuComplex mu[11];

float endtime;
float dt=.02;

//globals defined in eik.out
int ntgrid, nperiod;
float drhodpsi, rmaj, shat, kxfac, qsf, gradpar, eps, aminor;

//globals defined in gs2 namelist
int nwrite;
int navg;
int nstop;
float cfl;
float maxdt= .02;  //.02;
float g_exb;
int jtwist;
float tau;
int nSteps=100000;
float fluxDen;


//global host arrays from eik.out
float *gbdrift_h, *grho_h, *z_h, *cvdrift_h, *gds2_h, *bmag_h, *bgrad_h;
float *gds21_h, *gds22_h, *cvdrift0_h, *gbdrift0_h;

//global device arrays from eik.out
float *gbdrift, *grho, *z, *cvdrift, *gds2, *bmag, *bgrad;
float *gds21, *gds22, *cvdrift0, *gbdrift0;

//calculated globals
float D_par;
float D_prp;
float Beta_par;

//other global device arrays
float *kx, *ky, *kz;
float *bmagInv;
cuComplex *bmag_complex;
cuComplex *deriv_nlps;
float *derivR1_nlps, *derivR2_nlps, *resultR_nlps;
float* jacobian;

float *kx_h, *ky_h;

//plans
cufftHandle NLPSplanR2C, NLPSplanC2R, ZDerivBplanR2C, ZDerivBplanC2R, ZDerivplan;

bool DEBUG = false;
bool LINEAR = false;  //set in read_namelist
bool RESTART = false;
bool SCAN = true;
bool NO_ZDERIV = false;
bool NO_ZDERIV_COVERING = false;
bool NO_ZDERIV_B = false;
bool NO_OMEGAD = false;
bool CONST_CURV = false;
bool WRITE_OUT = false;

bool PM = false;
bool varenna = true;

int init = DENS;

//#include "device_funcs.cu"
#include "c_fortran_namelist3.c"
#include "getfcn.cu"
#include "read_namelist.cu"
#include "read_input.cu"
#include "definitions.cu"
#include "read_geo.cu"
#include "gryfx_lib.h"


void gryfx_get_default_parameters_(struct gryfx_parameters_struct * gryfxpars){

	printf("starting gryfx_get_default_parameters_\n");
  bool DEFAULT_INPUTS = true;
  
  
   
  // fopen returns 0, the NULL pointer, on failure 
  if ( DEFAULT_INPUTS )
  {
    kxfac = 1;
    eps = .18;
    rmaj = 1;
    qsf = 1.4;
    gradpar = rmaj/qsf;
    shat = .776;
    drhodpsi = 1;
    float shift = 0;    
    tau = 1.;
    *&Nz =32;
    float epsl = 1;
    
	printf("allocating arrays\n");
    gbdrift_h = (float*) malloc(sizeof(float)*Nz);
    grho_h = (float*) malloc(sizeof(float)*Nz);
    z_h = (float*) malloc(sizeof(float)*Nz);
    cvdrift_h = (float*) malloc(sizeof(float)*Nz);
    gds2_h = (float*) malloc(sizeof(float)*Nz);
    bmag_h = (float*) malloc(sizeof(float)*Nz);
    bgrad_h = (float*) malloc(sizeof(float)*Nz);     //
    gds21_h = (float*) malloc(sizeof(float)*Nz);
    gds22_h = (float*) malloc(sizeof(float)*Nz);
    cvdrift0_h = (float*) malloc(sizeof(float)*Nz);
    gbdrift0_h = (float*) malloc(sizeof(float)*Nz); 
    
    if(NO_OMEGAD) rmaj = 0;
    
	printf("setting arrays\n");
    for(int k=0; k<Nz; k++) {
			printf("allocating arrays: k = %d\n", k);
      z_h[k] = 2*M_PI*(k-Nz/2)/Nz;
      bmag_h[k] = 1./(1+eps*cos(z_h[k]));
      bgrad_h[k] = gradpar*eps*sin(z_h[k])*bmag_h[k];            //
      gds2_h[k] = 1. + pow((shat*z_h[k]-shift*sin(z_h[k])),2);
      gds21_h[k] = -shat*(shat*z_h[k]-shift*sin(z_h[k]));
      gds22_h[k] = pow(shat,2);
      gbdrift_h[k] = (rmaj/2)*(cos(z_h[k]) + (shat*z_h[k]-shift*sin(z_h[k]))*sin(z_h[k]));
      cvdrift_h[k] = gbdrift_h[k];
      gbdrift0_h[k] = -(rmaj/2)*shat*sin(z_h[k]);
      cvdrift0_h[k] = gbdrift0_h[k];
      grho_h[k] = 1;
      if(CONST_CURV) {
        cvdrift_h[k] = (rmaj/2)*epsl;
	gbdrift_h[k] = (rmaj/2)*epsl;
	cvdrift0_h[k] = 0;
	gbdrift0_h[k] = 0;
      }
    }  
	printf("allocating arrays\n");
  }
  else 
  {
    /*if(DEBUG) getError("gryfx.cu, before input file read");
    if(ifile == 0)
      printf("could not open %s\n", ifileName);
    else*/
    
    coefficients_struct *coefficients;
    constant_coefficients_struct * constant_coefficients;
    read_geo(&Nz,coefficients,constant_coefficients);
    
    /*if(DEBUG) getError("gryfx.cu, after input file read");
    fclose(ifile);
    if(DEBUG) getError("gryfx.cu, after input file close");*/
  } 
 
	printf("reading namelist\n");
  if(DEBUG) getError("before namelist read");
  //use "./blank" to use default namelist values
  read_namelist("./blank");
	printf("read namelist\n");
  //read_namelist("./inputs/linear.in");
  //read_namelist("./inputs/cyclone_miller_ke.in");
  if(DEBUG) getError("after namelist read");
}
  

void gryfx_get_fluxes_(struct gryfx_parameters_struct *  gryfxpars, 
											struct gryfx_outputs_struct * gryfxouts)
{
	printf("Let's pretend the heat flux was 3!\n");
	gryfxouts->qflux[0] = 3.0;
}  	

