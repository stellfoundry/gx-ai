//This lists all global variables that haven't yet made into sub modules/headers

//irho=2;

EXTERN_SWITCH int zThreads, totalThreads, zBlockThreads;

EXTERN_SWITCH dim3 dimBlock, dimGrid;
EXTERN_SWITCH dim3 dimBlockCovering;
EXTERN_SWITCH dim3 *dimGridCovering;
EXTERN_SWITCH dim3 dimGridCovering_all;





//global variables
EXTERN_SWITCH int nClasses;
EXTERN_SWITCH int *nLinks;
EXTERN_SWITCH int *nChains;
EXTERN_SWITCH cuComplex * nu;
EXTERN_SWITCH cuComplex * mu;

EXTERN_SWITCH float endtime;
EXTERN_SWITCH double dt_cfl;
EXTERN_SWITCH bool cfl_flag;

EXTERN_SWITCH int reset;
EXTERN_SWITCH int iproc;
EXTERN_SWITCH int mpcom;
EXTERN_SWITCH int mpcom_global;
EXTERN_SWITCH int gpuID;

//extern "C" double run_parameters_mp_code_delt_max_;
//extern "C" int kt_grids_mp_naky_;
//extern "C" double gs2_time_mp_code_dt_;
//extern "C" double gs2_time_mp_code_dt_cfl_;
//extern "C" double gs2_time_mp_code_time_;
//extern "C" int* mp_mp_proc0_;
//extern "C" int* mp_mp_iproc_;



//globals defined in gs2 namelist
EXTERN_SWITCH int converge_stop;
EXTERN_SWITCH float converge_bounds;
EXTERN_SWITCH float fluxDen;
EXTERN_SWITCH float cflx;
EXTERN_SWITCH float cfly;
EXTERN_SWITCH int Nx_unmasked;
EXTERN_SWITCH int Ny_unmasked;


//extern "C" double *geometry_mp_gbdrift_, *geometry_mp_grho_, *geometry_mp_cvdrift_, *geometry_mp_gds2_, *geometry_mp_bmag_;
//extern "C" double *geometry_mp_gds21_, *geometry_mp_gds22_, *geometry_mp_cvdrift0_, *geometry_mp_gbdrift0_, *geometry_mp_jacob_, *geometry_mp_gradpar_;
// extern "C" int geometry_mp_ntheta_, geometry_mp_nperiod_;
//extern "C" double geometry_mp_qsf_, geometry_mp_kxfac_, geometry_mp_shat_, geometry_mp_rmaj_, theta_grid_mp_drhodpsi_, geometry_mp_rhoc_;
//extern "C" double *geometry_mp_rplot_, *geometry_mp_zplot_, *geometry_mp_aplot_;

//calculated globals
EXTERN_SWITCH float D_par;
EXTERN_SWITCH float D_prp;
EXTERN_SWITCH float Beta_par;
EXTERN_SWITCH float diffusion;
EXTERN_SWITCH float kx_max;
EXTERN_SWITCH float ky_max;
EXTERN_SWITCH float kx4_max;
EXTERN_SWITCH float ky4_max;
EXTERN_SWITCH float ky_max_Inv;
EXTERN_SWITCH float kx4_max_Inv;
EXTERN_SWITCH float kperp2_max;
EXTERN_SWITCH float kperp4_max_Inv;




//other global device arrays
EXTERN_SWITCH cuComplex *deriv_nlps;
EXTERN_SWITCH float *derivR1_nlps, *derivR2_nlps, *derivR3_nlps, *resultR_nlps;
EXTERN_SWITCH float* PhiAvgDenom;
//EXTERN_SWITCH cuComplex** g_covering_d;


EXTERN_SWITCH cuComplex *field_h;

EXTERN_SWITCH char filename[200];

//plans
EXTERN_SWITCH cufftHandle NLPSplanR2C, NLPSplanC2R, ZDerivBplanR2C, ZDerivBplanC2R, ZDerivplan, XYplanC2R;

//streams and events
EXTERN_SWITCH cudaStream_t* zstreams;
//EXTERN_SWITCH cudaStream_t copystream;
EXTERN_SWITCH cudaEvent_t* end_of_zderiv;
//cudaEvent_t end_of_zderiv;




//char* fluxfileName;
//char stopfileName[60];
//char restartfileName[60];
EXTERN_SWITCH char * out_stem;

EXTERN_SWITCH char* run_name;
EXTERN_SWITCH char * restartfileName;

//diagnostics stuff
EXTERN_SWITCH float phi2;
EXTERN_SWITCH float* phi2_by_mode;
EXTERN_SWITCH float hflux_tot;
EXTERN_SWITCH float hflux_sp;
EXTERN_SWITCH float hflux_tot_av;
EXTERN_SWITCH float phi2_av;
EXTERN_SWITCH float ky_mean;
EXTERN_SWITCH float kx_mean;
EXTERN_SWITCH float* l_parallel;



//struct sdatio_file sdatfile;
void initialize_globals();
void set_globals_after_gryfx_lib(everything_struct * everything);
