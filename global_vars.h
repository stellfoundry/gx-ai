
//irho=2;

int zThreads, totalThreads, zBlockThreads;
__constant__ int nx,ny,nz, zthreads, nspecies;
__constant__ float X0_d,Y0_d;
__constant__ int Zp_d;

dim3 dimBlock, dimGrid;
dim3 dimBlockCovering;
dim3 *dimGridCovering;





//global variables
int nClasses;
int *nLinks;
int *nChains;
cuComplex nu[11];
cuComplex mu[11];

float endtime;
double dt_cfl;
bool cfl_flag = true;

int reset;
int iproc;
int mpcom;
int mpcom_global;
int gpuID;

extern "C" double run_parameters_mp_code_delt_max_;
extern "C" int kt_grids_mp_naky_;
extern "C" double gs2_time_mp_code_dt_;
extern "C" double gs2_time_mp_code_dt_cfl_;
extern "C" double gs2_time_mp_code_time_;
extern "C" int* mp_mp_proc0_;
extern "C" int* mp_mp_iproc_;



//globals defined in gs2 namelist
int converge_stop = 10000;
float converge_bounds = 20;
float fluxDen;
float cflx;
float cfly;
int Nx_unmasked;
int Ny_unmasked;


extern "C" double *geometry_mp_gbdrift_, *geometry_mp_grho_, *geometry_mp_cvdrift_, *geometry_mp_gds2_, *geometry_mp_bmag_;
extern "C" double *geometry_mp_gds21_, *geometry_mp_gds22_, *geometry_mp_cvdrift0_, *geometry_mp_gbdrift0_, *geometry_mp_jacob_, *geometry_mp_gradpar_;
extern "C" int geometry_mp_ntheta_, geometry_mp_nperiod_;
extern "C" double geometry_mp_qsf_, geometry_mp_kxfac_, geometry_mp_shat_, geometry_mp_rmaj_, theta_grid_mp_drhodpsi_, geometry_mp_rhoc_;
extern "C" double *geometry_mp_rplot_, *geometry_mp_zplot_, *geometry_mp_aplot_;

//calculated globals
float D_par;
float D_prp;
float Beta_par;
float diffusion;
float kx_max;
float ky_max;
float kx4_max;
float ky4_max;
float ky_max_Inv;
float kx4_max_Inv;
float kperp2_max;
float kperp4_max_Inv;




//other global device arrays
cuComplex *deriv_nlps;
float *derivR1_nlps, *derivR2_nlps, *resultR_nlps;
float* PhiAvgDenom;


cuComplex *field_h;

char filename[200];

//plans
cufftHandle NLPSplanR2C, NLPSplanC2R, ZDerivBplanR2C, ZDerivBplanC2R, ZDerivplan, XYplanC2R;

//streams and events
cudaStream_t* zstreams;
cudaStream_t copystream;
cudaEvent_t* end_of_zderiv;
//cudaEvent_t end_of_zderiv;




//char* fluxfileName;
//char stopfileName[60];
//char restartfileName[60];
char out_stem[200];

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



//struct sdatio_file sdatfile;
