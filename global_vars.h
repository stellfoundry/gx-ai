//#include "simpledataio_cuda.h"

int Nx, Ny, Nz, zThreads, totalThreads;
float X0, Y0;
int Zp;
__constant__ int nx,ny,nz,zthreads,totalthreads;
__constant__ float X0_d,Y0_d;
__constant__ int Zp_d;

dim3 dimBlock, dimGrid;

#define ION 0
#define ELECTRON 1
#define PHI 0
#define DENS 1

//species variables defined in gs2 namelist
int nSpecies=1;
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
  char type[100];        
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
int ntgrid;
float drhodpsi, rmaj, shat, kxfac, qsf, gradpar, eps, aminor, epsl;

//other Miller parameters;
float r_geo, akappa, akappri, tri, tripri, shift, asym, asympri;

//Greene & Chance/Bishop parameters
float beta_prime_input, s_hat_input;


//globals defined in gs2 namelist
int nwrite;
int nsave;
int navg;
int nstop;
int converge_stop = 10000;
float converge_bounds = 20;
float cfl;
float maxdt= .02;  //.02;
float g_exb;
int jtwist;
float tau;
int nSteps=10000;
float fluxDen;
float cflx;
float cfly;
int Nx_unmasked;
int Ny_unmasked;

//input parameters for geometry
int equilibrium_type;
int bishop;
int irho = 2;
int nperiod;
float rhoc;

//global host arrays from eik.out
float *gbdrift_h, *grho_h, *z_h, *cvdrift_h, *gds2_h, *bmag_h, *bgrad_h;
float *gds21_h, *gds22_h, *cvdrift0_h, *gbdrift0_h, *jacobian_h;

//global device arrays from eik.out
float *gbdrift, *grho, *z, *cvdrift, *gds2, *bmag, *bgrad;
float *gds21, *gds22, *cvdrift0, *gbdrift0;

//calculated globals
float D_par;
float D_prp;
float Beta_par;
float diffusion;
float nu_hyper=1.;
int p_hyper=2;
float kperp2_max_Inv;

float dnlpm = 1.;
int inlpm = 2;

//other global device arrays
float *kx, *ky, *kz;
float *bmagInv;
cuComplex *bmag_complex;
cuComplex *deriv_nlps;
float *derivR1_nlps, *derivR2_nlps, *resultR_nlps;
float* jacobian;

float *kx_h, *ky_h;

//plans
cufftHandle NLPSplanR2C, NLPSplanC2R, ZDerivBplanR2C, ZDerivBplanC2R, ZDerivplan, XYplanC2R;

//streams and events
cudaStream_t* streams;
cudaEvent_t end_of_zderiv;

bool DEBUG = false;

bool LINEAR = false;  
bool RESTART = false;
bool CHECK_FOR_RESTART = false;
bool SCAN = true;
bool NO_ZDERIV = false;
bool NO_ZDERIV_COVERING = false;
bool NO_ZDERIV_B = false;
bool SLAB = false;
bool CONST_CURV = false;
bool write_omega = true;
bool write_phi = true;
bool S_ALPHA = true;

bool NLPM = false;
bool varenna = false;
bool SMAGORINSKY = false;
bool HYPER = false;
bool zero_restart_avg = false;

int init = DENS;
float init_amp;

//char* fluxfileName;
//char stopfileName[60];
//char restartfileName[60];
char* scan_type;
char out_stem[40];
int scan_number;

char* run_name;
char restartfileName[60];

//diagnostics stuff
float phi2;
float* phi2_by_mode;
float hflux_tot;
float hflux_sp;
float hflux_tot_av;
float phi2_av;
float ky_mean;
float kx_mean;
float* l_parallel;
cuComplex* phi_h;
cuComplex* dens_h;
cuComplex* upar_h;
cuComplex* tpar_h;
cuComplex* tprp_h;
cuComplex* qpar_h;
cuComplex* qprp_h;

//struct sdatio_file sdatfile;
