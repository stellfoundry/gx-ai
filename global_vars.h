//#include "simpledataio_cuda.h"


int Nx, Ny, Nz, zThreads, totalThreads, zBlockThreads;
float X0, Y0;
int Zp;
int nSpecies;
__constant__ int nx,ny,nz, zthreads, nspecies;
__constant__ float X0_d,Y0_d;
__constant__ int Zp_d;

dim3 dimBlock, dimGrid;
dim3 dimBlockCovering;
dim3 *dimGridCovering;

#define ION 0
#define ELECTRON 1
#define PHI 0
#define DENS 1
#define FORCE 2
#define RH_equilibrium 3

//species variables defined in gs2 namelist
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
int icovering = 1;
cuComplex nu[11];
cuComplex mu[11];

float endtime;
float dt = .02;
float dt_cfl;
bool cfl_flag = true;

int reset;
int iproc;
int mpcom;

extern "C" double run_parameters_mp_code_delt_max_;
extern "C" int kt_grids_mp_naky_;
extern "C" double gs2_time_mp_code_dt_;
extern "C" double gs2_time_mp_code_dt_cfl_;
extern "C" int* mp_mp_proc0_;
extern "C" int* mp_mp_iproc_;

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
float *gbdrift_h, *grho_h, *z_h; 
float *cvdrift_h, *gds2_h, *bmag_h, *bgrad_h;
float *gds21_h, *gds22_h, *cvdrift0_h, *gbdrift0_h, *jacobian_h;


//global device arrays from eik.out
float *gbdrift, *grho, *z, *cvdrift, *gds2, *bmag, *bgrad;
float *gds21, *gds22, *cvdrift0, *gbdrift0;

extern "C" double *geometry_mp_gbdrift_, *geometry_mp_grho_, *geometry_mp_cvdrift_, *geometry_mp_gds2_, *geometry_mp_bmag_;
extern "C" double *geometry_mp_gds21_, *geometry_mp_gds22_, *geometry_mp_cvdrift0_, *geometry_mp_gbdrift0_, *geometry_mp_jacob_, *geometry_mp_gradpar_;
extern "C" int geometry_mp_ntheta_, geometry_mp_nperiod_;
extern "C" double geometry_mp_qsf_, geometry_mp_kxfac_, geometry_mp_shat_, geometry_mp_rmaj_, theta_grid_mp_drhodpsi_, geometry_mp_rhoc_;

//calculated globals
float D_par;
float D_prp;
float Beta_par;
float diffusion;
float D_hyper=0.1;
int p_hyper=2;
float kx_max;
float ky_max;
float kx4_max;
float ky4_max;
float ky_max_Inv;
float kx4_max_Inv;
float kperp2_max;
float kperp4_max_Inv;

float dnlpm = 1.;
int inlpm = 2;
float low_cutoff = .01;
float high_cutoff = .1;
float dnlpm_max = 1.;
char* nlpm_option = "constant";
float tau_nlpm = 10.;
bool nlpm_zonal_kx1_only = false;
bool dorland_nlpm = false;
bool dorland_nlpm_phase = true;
bool dorland_phase_complex = false;
int dorland_phase_ifac = 1;

int ivarenna = 1;
bool varenna_fsa = false;
bool new_varenna_fsa = false;
int zonal_dens_switch = 0;
int q0_dens_switch = 0;

bool tpar_omegad_corrections = true;
bool tperp_omegad_corrections = true;
bool qpar_gradpar_corrections = false;
bool qpar_bgrad_corrections = false;
bool qperp_gradpar_corrections = false;
bool qperp_bgrad_corrections = false;
bool qpar0_switch = true;
bool qprp0_switch = true;

int iphi00 = 2;
int igeo = 0;
float shaping_ps = 1.6;
char* geoFileName;

bool secondary_test;
cuComplex phi_test;
float NLdensfac, NLuparfac, NLtparfac, NLtprpfac, NLqparfac, NLqprpfac;
char* secondary_test_restartfileName;

//other global device arrays
float *kx, *ky, *kz, *kz_complex;
float *bmagInv;
cuComplex *bmag_complex;
cuComplex *deriv_nlps;
float *derivR1_nlps, *derivR2_nlps, *resultR_nlps;
float* jacobian;
float* PhiAvgDenom;

float *kx_h, *ky_h, *kz_h;

cuComplex *field_h;

char filename[200];

//plans
cufftHandle NLPSplanR2C, NLPSplanC2R, ZDerivBplanR2C, ZDerivBplanC2R, ZDerivplan, XYplanC2R;

//streams and events
cudaStream_t* zstreams;
cudaStream_t copystream;
cudaEvent_t* end_of_zderiv;
//cudaEvent_t end_of_zderiv;

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
bool write_phase = true;
bool S_ALPHA = true;
bool no_landau_damping = false;
bool turn_off_gradients_test = false;

bool NLPM = false;
bool varenna = false;
bool new_varenna = false;
bool new_catto = false;
bool SMAGORINSKY = false;
bool HYPER = false;
bool isotropic_shear = false;
bool zero_restart_avg = false;

int init = DENS;
float init_amp;
float phiext=-1.;

//char* fluxfileName;
//char stopfileName[60];
//char restartfileName[60];
char* scan_type;
char out_stem[200];
int scan_number;

char* run_name;
char restartfileName[200];

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
