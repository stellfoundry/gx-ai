#include <math.h>
#include "simpledataio_cuda.h"
#include "cufft.h"
#include "sys/stat.h"
#include "cuda_profiler_api.h"
#include "libgen.h"
#include "mpi.h"
#include "nvToolsExt.h"


// EGH: Until we do something more fancy with cudaGetSymbolAddress the following have
// to have file scope... ie they have to be in the same aggregate source file as the 
// kernels.

__constant__ int nx,ny,nz, zthreads, nspecies;
__constant__ float X0_d,Y0_d;
__constant__ int Zp_d;

#define EXTERN_SWITCH extern
#include "everything_struct.h"
#include "global_variables.h"
#include "allocations.h"
#include "write_data.h"
#include "gs2.h"
#include "get_error.h"
#include "device_funcs.cu"
#include "operations_kernel.cu"
#include "diagnostics_kernel.cu"
#include "exb_kernel.cu"
#include "nlps_kernel.cu"
#include "zderiv_kernel.cu"
#include "covering_kernel.cu"
#include "reduc_kernel.cu"
#include "cudaReduc_kernel.cu"
#include "init_kernel.cu"
#include "omega_kernel.cu"
#include "phi_kernel.cu"
#include "qneut_kernel.cu"
#include "nlpm_kernel.cu"
#include "zonal_kernel.cu"
#include "getfcn.cu"
#include "maxReduc.cu"
#include "sumReduc.cu"
#include "diagnostics.cu"
#include "coveringSetup.cu"
#include "exb.cu"
#include "qneut.cu"
#include "ztransform_covering.cu"
#include "zderiv.cu"
#include "zderivB.cu"
#include "zderiv_covering.cu"
#include "nlps.cu"
#include "nlpm.cu"
#include "hyper.cu"
#include "courant.cu"
#include "energy.cu"
#include "timestep_gryfx.cu"


#ifdef GS2_zonal
//extern "C" void fields_implicit_mp_advance_implicit_(int* gs2_counter, cuComplex* dens_ky0_h, cuComplex* upar_ky0_h, cuComplex* tpar_ky0_h, cuComplex* tprp_ky0_h, cuComplex* qpar_ky0_h, cuComplex* qprp_ky0_h);
//extern "C" void fields_mp_advance_(int* gs2_counter);
//extern "C" void gs2_main_mp_advance_gs2_(int* gs2_counter, cuComplex* dens_ky0_h, cuComplex* upar_ky0_h, cuComplex* tpar_ky0_h, cuComplex* tprp_ky0_h, cuComplex* qpar_ky0_h, cuComplex* qprp_ky0_h, cuComplex* phi_ky0_h, int* first_half_flag);
extern "C" void advance_gs2(int* gs2_counter, cuComplex* dens_ky0_h, cuComplex* upar_ky0_h, cuComplex* tpar_ky0_h, cuComplex* tprp_ky0_h, cuComplex* qpar_ky0_h, cuComplex* qprp_ky0_h, cuComplex* phi_ky0_h, int* first_half_flag);
//extern "C" void gs2_time_mp_update_time_();
//extern "C" void gs2_reinit_mp_check_time_step_(int* reset, int* exit);
//extern "C" void gs2_reinit_mp_reset_time_step_(int* gs2_counter, int* exit);
extern "C" void getmoms_gryfx(cuComplex* dens, cuComplex* upar, cuComplex* tpar, cuComplex* tprp, cuComplex* qpar, cuComplex* qprp, cuComplex* phi);
//extern "C" void gs2_diagnostics_mp_loop_diagnostics_(int* gs2_counter, int* exit);
extern "C" void broadcast_integer(int* a);
//extern "C" void broadcast_integer(int* a);
//extern "C" void species_mp_reinit_species_(int* nSpecies, double* dens, double* temp, double* fprim, double* tprim, double* nu);
#endif




void set_cuda_constants(){
  cudaMemcpyToSymbol(nx, &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ny, &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nz, &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nspecies, &nSpecies, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(X0_d, &X0, sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Y0_d, &Y0, sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Zp_d, &Zp, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(zthreads, &zThreads, sizeof(int),0,cudaMemcpyHostToDevice);
 // cudaMemcpyToSymbol(zblockthreads, &zBlockThreads, sizeof(int),0,cudaMemcpyHostToDevice);

  if(DEBUG) getError("run_gryfx.cu, copied constants");
}

void run_gryfx(everything_struct * ev_h, double * pflux, double * qflux, FILE* outfile)//, FILE* omegafile,FILE* gammafile, FILE* energyfile, FILE* fluxfile, FILE* phikyfile, FILE* phikxfile, FILE* phifile)
{

    set_globals_after_gryfx_lib(ev_h);
    if (iproc==0) set_cuda_constants();
    // Open some text files for output

    //if (iproc==0) printf("dimGrid2 = (%d, %d, %d)     dimBlock = (%d, %d, %d)\n", dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z);
    //FILE *omegafile;
    //FILE *gammafile;
    //FILE *phifile;
    //host variables


    cudaEvent_t start, stop,  nonlin_halfstep, H2D, D2H, GS2start, GS2stop;
    
    int naky, ntheta0;// nshift;
    naky = 1 + (Ny-1)/3;
    ntheta0 = 1 + 2*((Nx-1)/3);     //MASK IN MIDDLE OF ARRAY
    
    float* Dnlpm_d;
    float* Phi_zf_kx1_d;
    float Dnlpm = 0;
    float Dnlpm_avg = 0;
    float Dnlpm_sum = 0;
    //float* phiVal; 
    float* val;
    //float phiVal0;
  
    float Phi_zf_kx1 = 0.;
    float Phi_zf_kx1_old = 0.;
    //float Phi_zf_kx1_sum = 0.;
    float Phi_zf_kx1_avg = 0.;
    float alpha_nlpm = 0.;
    float mu_nlpm = 0.;
    //float tau_nlpm = 50.;
    cuComplex *init_h;
    //float Phi_energy;
    //cuComplex *omega_h;  
    //float dtBox[navg];
    cuComplex* omegaAvg_h;
    float wpfx[nSpecies];
    float pflx[nSpecies];
    float wpfx_old[nSpecies];
    float pflx_old[nSpecies];
    //float wpfx_sum[nSpecies];
    float tmpX_h[Nx];
    float tmpY_h[Ny/2+1];
    float tmpXY_h[Nx*(Ny/2+1)];
    float tmpYZ_h[(Ny/2+1)*Nz];
    float tmpXY_R_h[Nx*Ny];
    //float phi0_X[Nx];
    //cuComplex CtmpX_h[Nx];
    cuComplex field_h[Nx*(Ny/2+1)*Nz];
    cuComplex CtmpZ_h[Nz];
   
    float Phi2_zf;
    float Phi_zf_rms;
    float kx2Phi_zf_rms;
    float kx2Phi_zf_rms_old;
    float Phi_zf_rms_sum;
    float Phi_zf_rms_avg;
    float kx2Phi_zf_rms_sum;
    float kx2Phi_zf_rms_avg;
    
    //for secondary test
    cuComplex *phi_fixed, *dens_fixed, *upar_fixed, *tpar_fixed, *tprp_fixed, *qpar_fixed, *qprp_fixed;
    float S_fixed = 1.;

    char filename[200];
  
    //int exit_flag = 0;  
  
    double runtime;
    int counter;
    int gs2_counter;


    //nSteps = 100;
    
    int Stable[Nx*(Ny/2+1)*2];
    //cuComplex stability[Nx*(Ny/2+1)];
    for(int i=0; i<Nx*(Ny/2+1); i++) {
      //stability[i].x = 0;
      //stability[i].y = 0;
      Stable[i] = 0;
      Stable[i +Nx*(Ny/2+1)] = 0;
    }
    //bool STABLE_STOP=false;  
    int stableMax = 500;
      
    //float wpfxmax=0.;
    //float wpfxmin=0.;
    int converge_count=0;
    
    //bool startavg=false;
  
    //float dt_start = .02;
    float alpha_avg = (float) 2./(navg+1.);
    float mu_avg = exp(-alpha_avg);
    //float navg_nlpm;

    /*
    float ky_spectrum1_h[Ny/2+1];
    float ky_spectrum2_h[Ny/2+1];
    float ky_spectrum3_h[Ny/2+1];
    float ky0_kx_spectrum_h[Nx];
    float kxky_spectrum1_h[Nx*(Ny/2+1)];  
    float kxky_spectrum2_h[Nx*(Ny/2+1)];
    float kxky_spectrum3_h[Nx*(Ny/2+1)];
    */
    //device variables; main device arrays will be capitalized
    if(DEBUG) getError("run_gryfx.cu, before device alloc");
    
    //tmps for timestep routine
    cuComplex *field, *tmp;
    float *tmpZ;  
    float *tmpX;
    float *tmpX2;
    float *tmpY;
    float *tmpY2;
    float *tmpXY;
    float *tmpXY2;
    float *tmpXY3;
    float *tmpXY4;
    float *tmpXY_R;
    float *tmpXZ;
    float *tmpYZ;
    float *tmpXYZ;
    cuComplex *CtmpX;
    cuComplex *CtmpX2;
    cuComplex *CtmpXZ;
   
    cuComplex *omega;
    cuComplex *omegaBox[navg];
    
    cuComplex *omegaAvg;
    
    //float *Phi2_XYBox[navg];
    specie* species_d;
    
    //double dt_old;
    //double avgdt;
    float totaltimer;
    float timer;
    //float GS2timer;
    
    //diagnostics scalars
    float flux1,flux2;
    float flux1_phase, flux2_phase, Dens_phase, Tpar_phase, Tprp_phase;
    float flux1_phase_sum, flux2_phase_sum, Dens_phase_sum, Tpar_phase_sum, Tprp_phase_sum;
   
    float Phi2, kPhi2;
    float Phi2_sum;//, kPhi2_sum;
    float expectation_ky;
    float expectation_ky_sum;
    float expectation_kx;
    float expectation_kx_sum;
    float dtSum;
    //diagnostics arrays
    float wpfxAvg[nSpecies];
    float pflxAvg[nSpecies];
    float *Phi2_kxky_sum;
    float *wpfxnorm_kxky_sum;
    float *Phi2_zonal_sum;
    //cuComplex *Phi_sum;
    float *zCorr_sum;
    
    float *kx_shift;
    int *jump;
    
    float *nu_nlpm;
    float *nu1_nlpm;
    float *nu22_nlpm;
    float nu1_nlpm_max;
    float nu22_nlpm_max;
    cuComplex *nu1_nlpm_complex;
    cuComplex *nu22_nlpm_complex;
    float *shear_rate_z;
    float *shear_rate_z_nz;
    float *shear_rate_nz;  
  
    cuComplex *dens_ky0_h, *upar_ky0_h, *tpar_ky0_h, *tprp_ky0_h, *qpar_ky0_h, *qprp_ky0_h, *phi_ky0_h;
    cuComplex *dens_ky0_d[nSpecies], *upar_ky0_d[nSpecies], *tpar_ky0_d[nSpecies], *tprp_ky0_d[nSpecies], *qpar_ky0_d[nSpecies], *qprp_ky0_d[nSpecies], *phi_ky0_d;

    printf("At the beginning of run_gryfx, gs2 time is %f\n", gs2_time_mp_code_time_/sqrt(2.0));
    
////////////  
#ifdef GS2_zonal
    //allocate these host arrays on all procs
    cudaMallocHost((void**) &dens_ky0_h, sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    cudaMallocHost((void**) &upar_ky0_h, sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    cudaMallocHost((void**) &tpar_ky0_h, sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    cudaMallocHost((void**) &tprp_ky0_h, sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    cudaMallocHost((void**) &qpar_ky0_h, sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    cudaMallocHost((void**) &qprp_ky0_h, sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    cudaMallocHost((void**) &phi_ky0_h, sizeof(cuComplex)*ntheta0*Nz);

/*    dens_ky0_h = (cuComplex*) malloc(sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    upar_ky0_h = (cuComplex*) malloc(sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    tpar_ky0_h = (cuComplex*) malloc(sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    tprp_ky0_h = (cuComplex*) malloc(sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    qpar_ky0_h = (cuComplex*) malloc(sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    qprp_ky0_h = (cuComplex*) malloc(sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    phi_ky0_h = (cuComplex*) malloc(sizeof(cuComplex)*ntheta0*Nz);
*/
			if(iproc==0) {

#endif

    //omega_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)); 
    omega_out_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)); 
    omegaAvg_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1));

} //end if iproc

	/* ev_hd is on the host but the pointers point to memory on the device*/
	everything_struct * ev_hd;
	/* ev_d is on the device (and the pointers point to memory on the device)*/
	everything_struct * ev_d;
  setup_everything_structs(ev_h, &ev_hd, &ev_d);

// The file local_pointers.cu
//declares and assigns local 
// pointers to members of the everything
// structs, for example Phi... it should
// eventually be unnecessary
#include "local_pointers.cu"


   //omegaAvg_h = ev_h->outs.omega;

if (iproc==0){






    //kx_h = (float*) malloc(sizeof(float)*Nx);
    //ky_h = (float*) malloc(sizeof(float)*(Ny/2+1));

    kx_h = ev_h->grids.kx;
    ky_h = ev_h->grids.ky;
    kz_h = (float*) malloc(sizeof(float)*Nz);  

  
    //zero dtBox array
    //for(int t=0; t<navg; t++) {  dtBox[t] = 0;  }
      
    init_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    //Phi_energy = (float*) malloc(sizeof(float));
    
  //#ifdef GS2_zonal
    for(int s=0; s<nSpecies; s++) {
  
      cudaMalloc((void**) &dens_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz);
      cudaMalloc((void**) &upar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz);
      cudaMalloc((void**) &tpar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz);
      cudaMalloc((void**) &tprp_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz);
      cudaMalloc((void**) &qpar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz);
      cudaMalloc((void**) &qprp_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz);
  
    }
    cudaMalloc((void**) &phi_ky0_d, sizeof(cuComplex)*ntheta0*Nz);
  //#endif
    
    
    
    //cudaMalloc((void**) &Phi, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    //cudaMalloc((void**) &Phi1, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    
    cudaMalloc((void**) &Phi2_kxky_sum, sizeof(float)*Nx*(Ny/2+1)); 
    cudaMalloc((void**) &wpfxnorm_kxky_sum, sizeof(float)*Nx*(Ny/2+1));
    cudaMalloc((void**) &Phi2_zonal_sum, sizeof(float)*Nx); 
    //cudaMalloc((void**) &Phi_sum, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &zCorr_sum, sizeof(float)*(Ny/2+1)*Nz);
    
    cudaMalloc((void**) &field, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &tmp,   sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &tmpZ, sizeof(float)*Nz);
    cudaMalloc((void**) &tmpX, sizeof(float)*Nx);
    cudaMalloc((void**) &tmpX2, sizeof(float)*Nx);
    cudaMalloc((void**) &tmpY, sizeof(float)*(Ny/2+1));
    cudaMalloc((void**) &tmpY2, sizeof(float)*(Ny/2+1));
    cudaMalloc((void**) &tmpXY, sizeof(float)*Nx*(Ny/2+1));
    cudaMalloc((void**) &tmpXY2, sizeof(float)*Nx*(Ny/2+1));
    cudaMalloc((void**) &tmpXY3, sizeof(float)*Nx*(Ny/2+1));
    cudaMalloc((void**) &tmpXY4, sizeof(float)*Nx*(Ny/2+1));
    cudaMalloc((void**) &tmpXY_R, sizeof(float)*Nx*Ny);
    cudaMalloc((void**) &tmpXZ, sizeof(float)*Nx*Nz);
    cudaMalloc((void**) &tmpYZ, sizeof(float)*(Ny/2+1)*Nz);
    cudaMalloc((void**) &CtmpX, sizeof(cuComplex)*Nx);
    cudaMalloc((void**) &CtmpX2, sizeof(cuComplex)*Nx);
    cudaMalloc((void**) &CtmpXZ, sizeof(cuComplex)*Nx*Nz);
    cudaMalloc((void**) &tmpXYZ, sizeof(float)*Nx*(Ny/2+1)*Nz);
   
    cudaMalloc((void**) &deriv_nlps, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &derivR1_nlps, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &derivR2_nlps, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &resultR_nlps, sizeof(float)*Nx*Ny*Nz);
  
    cudaMalloc((void**) &kx, sizeof(float)*Nx);
    cudaMalloc((void**) &kx_abs, sizeof(float)*Nx);
    cudaMalloc((void**) &ky, sizeof(float)*(Ny/2+1));
    cudaMalloc((void**) &kz, sizeof(float)*(Nz));
    cudaMalloc((void**) &kz_complex, sizeof(float)*(Nz/2+1));
  
    cudaMalloc((void**) &bmagInv, sizeof(float)*Nz); 
    cudaMalloc((void**) &bmag_complex, sizeof(cuComplex)*(Nz/2+1));
    cudaMalloc((void**) &jacobian, sizeof(float)*Nz);
    cudaMalloc((void**) &PhiAvgDenom, sizeof(float)*Nx);
    
    cudaMalloc((void**) &species_d, sizeof(specie)*nSpecies);
  
    
    cudaMalloc((void**) &omega, sizeof(cuComplex)*Nx*(Ny/2+1));
    for(int t=0; t<navg; t++) {
      if(LINEAR || secondary_test) cudaMalloc((void**) &omegaBox[t], sizeof(cuComplex)*Nx*(Ny/2+1));
      //cudaMalloc((void**) &Phi2_XYBox[t], sizeof(float)*Nx*(Ny/2+1));
    }
    cudaMalloc((void**) &omegaAvg, sizeof(cuComplex)*Nx*(Ny/2+1));
    
    cudaMalloc((void**) &kx_shift, sizeof(float)*(Ny/2+1));
    cudaMalloc((void**) &jump, sizeof(int)*(Ny/2+1));
    
    cudaMalloc((void**) &nu_nlpm, sizeof(float)*Nz);
    cudaMalloc((void**) &nu1_nlpm, sizeof(float)*Nz);
    cudaMalloc((void**) &nu22_nlpm, sizeof(float)*Nz);
    cudaMalloc((void**) &nu1_nlpm_complex, sizeof(cuComplex)*Nz);
    cudaMalloc((void**) &nu22_nlpm_complex, sizeof(cuComplex)*Nz);
    cudaMalloc((void**) &shear_rate_z, sizeof(float)*Nz);  
    cudaMalloc((void**) &shear_rate_nz, sizeof(float)*Nz);  
    cudaMalloc((void**) &shear_rate_z_nz, sizeof(float)*Nz);  
  
    cudaMalloc((void**) &Dnlpm_d, sizeof(float));
    cudaMalloc((void**) &Phi_zf_kx1_d, sizeof(float));

    cudaMalloc((void**) &phi_fixed, sizeof(cuComplex)*Nz);
    cudaMalloc((void**) &dens_fixed, sizeof(cuComplex)*Nz);
    cudaMalloc((void**) &upar_fixed, sizeof(cuComplex)*Nz);
    cudaMalloc((void**) &tpar_fixed, sizeof(cuComplex)*Nz);
    cudaMalloc((void**) &tprp_fixed, sizeof(cuComplex)*Nz);
    cudaMalloc((void**) &qpar_fixed, sizeof(cuComplex)*Nz);
    cudaMalloc((void**) &qprp_fixed, sizeof(cuComplex)*Nz);
  
  
    if(DEBUG) getError("run_gryfx.cu, after device alloc");
  
    cudaMemcpy(species_d, species, sizeof(specie)*nSpecies, cudaMemcpyHostToDevice);
    
    cudaMemcpy(z, z_h, sizeof(float)*Nz, cudaMemcpyHostToDevice);

    copy_geo_arrays_to_device(&ev_hd->geo, &ev_h->geo, &ev_h->pars, ev_h->grids.Nz); 
    cudaMalloc((void**) &val, sizeof(float));
    //phiVal = (float*) malloc(sizeof(float));
    
    
    //set up plans for NLPS, ZDeriv, and ZDerivB
    //plan for ZDerivCovering done below
    int NLPSfftdims[2] = {Nx, Ny};
    cufftPlanMany(&NLPSplanR2C, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, Nz);
    cufftPlanMany(&NLPSplanC2R, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, Nz);
    cufftPlan1d(&ZDerivBplanR2C, Nz, CUFFT_R2C, 1);
    cufftPlan1d(&ZDerivBplanC2R, Nz, CUFFT_C2R, 1);  
    cufftPlan2d(&XYplanC2R, Nx, Ny, CUFFT_C2R);  //for diagnostics
    int n[1] = {Nz};
    int inembed[1] = {(Ny/2+1)*Nx*Nz};
    int onembed[1] = {(Ny/2+1)*Nx*(Nz)};
    cufftPlanMany(&ZDerivplan,1,n,inembed,(Ny/2+1)*Nx,1,
                                  onembed,(Ny/2+1)*Nx,1,CUFFT_C2C,(Ny/2+1)*Nx);	
                       //    n rank  nembed  stride   dist
    if(DEBUG) getError("after plan");
      
    // INITIALIZE ARRAYS AS NECESSARY
    zero<<<dimGrid,dimBlock>>>(nu22_nlpm, 1, 1, Nz);
    zero<<<dimGrid,dimBlock>>>(nu1_nlpm, 1, 1, Nz);
    zero<<<dimGrid,dimBlock>>>(nu_nlpm, 1, 1, Nz);
  
    kInit  <<< dimGrid, dimBlock >>> (kx, ky, kz, kx_abs, NO_ZDERIV);
    kx_max = (float) ((int)((Nx-1)/3))/X0;
    ky_max = (float) ((int)((Ny-1)/3))/Y0;
    kperp2_max = pow(kx_max,2) + pow(ky_max,2);
    kx4_max = pow(kx_max,4);
    ky4_max = pow(ky_max,4);
    ky_max_Inv = 1. / ky_max;
    kx4_max_Inv = 1. / kx4_max;
    kperp4_max_Inv = 1. / pow(kperp2_max,2);
    if(DEBUG) printf("kperp4_max_Inv = %f\n", kperp4_max_Inv);
    bmagInit <<<dimGrid,dimBlock>>>(bmag,bmagInv);
    jacobianInit<<<dimGrid,dimBlock>>> (jacobian,drhodpsi,gradpar,bmag);
    if(igeo != 0) {

      cudaMemset(bmag_complex, 0, sizeof(cuComplex)*(Nz/2+1));
      //calculate bgrad = d/dz ln(B(z)) = 1/B dB/dz
      if(DEBUG) printf("calculating bgrad\n");
      ZDerivB(bgrad, bmag, bmag_complex, kz);
      //cufftExecR2C(ZDerivBplanR2C, bmag, bmag_complex);
      //cudaMemcpy(CtmpZ_h, bmag_complex, sizeof(cuComplex)*(Nz/2+1), cudaMemcpyDeviceToHost);
      //for(int i=0; i<(Nz/2+1); i++) {
      //  printf("bmag_complex(kz=%d) = %f + i %f\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
      //}
      multdiv<<<dimGrid,dimBlock>>>(bgrad, bgrad, bmagInv, 1, 1, Nz, 1);
    }  
    if(DEBUG) getError("before cudaMemset");  
    //cudaMemset(jump, 0, sizeof(float)*Ny);
    //cudaMemset(kx_shift,0,sizeof(float)*Ny);
    if(DEBUG) getError("after cudaMemset"); 
  
    //for flux calculations
    multdiv<<<dimGrid,dimBlock>>>(tmpZ, jacobian, grho,1,1,Nz,1);
    fluxDen = sumReduc(tmpZ,Nz,false);
  
    //PhiAvg denominator for qneut
    cudaMemset(PhiAvgDenom, 0, sizeof(float)*Nx);
    phiavgdenom<<<dimGrid,dimBlock>>>(PhiAvgDenom, tmpXZ, jacobian, species_d, kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);  
  
    if(DEBUG) getError("run_gryfx.cu, after init"); 
   
    cudaMemcpy(kx_h,kx, sizeof(float)*Nx, cudaMemcpyDeviceToHost);
  
    if(DEBUG) getError("after k memcpy 1");
  
    cudaMemcpy(ky_h,ky, sizeof(float)*(Ny/2+1), cudaMemcpyDeviceToHost);
    
    if(DEBUG) getError("after k memcpy 2");
    
    setup_files(&ev_h->files, &ev_h->pars, &ev_h->grids, ev_h->info.run_name);
    writedat_beginning(ev_h);
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //set up kxCover and kyCover for covering space z-transforms
    //nshift = Nx - ntheta0;
  
#ifdef GS2_zonal
			}
#endif
  
    int idxRight[naky*ntheta0];
    int idxLeft[naky*ntheta0];
  
    int linksR[naky*ntheta0];
    int linksL[naky*ntheta0];
    int n_k[naky*ntheta0];

#ifdef GS2_zonal
			if(iproc==0) {
#endif
    getNClasses(&nClasses, idxRight, idxLeft, linksR, linksL, n_k, naky, ntheta0, jtwist);
    
#ifdef GS2_zonal
			}
#endif

    int *kxCover[nClasses];
    int *kyCover[nClasses];
    cuComplex *g_covering[nClasses];
    float *kz_covering[nClasses];
    cufftHandle plan_covering[nClasses];
    int *kxCover_h[nClasses];
    int *kyCover_h[nClasses];

#ifdef GS2_zonal
			if(iproc==0) {
#endif

    if(DEBUG) getError("run_gryfx.cu, after nclasses");
  
    nLinks = (int*) malloc(sizeof(int)*nClasses);
    nChains = (int*) malloc(sizeof(int)*nClasses);
  
    getNLinksChains(nLinks, nChains, n_k, nClasses, naky, ntheta0);
  
    
    //int **kxCover_h, **kyCover_h;
    //kxCover_h = (int**) malloc(sizeof(int)*nClasses);
    //kyCover_h = (int**) malloc(sizeof(int)*nClasses);
    
    for(int c=0; c<nClasses; c++) {   
      kyCover_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
      kxCover_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    }  
  
    kFill(nClasses, nChains, nLinks, kyCover_h, kxCover_h, linksL, linksR, idxRight, naky, ntheta0); 
    
    if(DEBUG) getError("run_gryfx.cu, after kFill");
  
    //these are the device arrays... cannot be global because jagged!
    //also set up a stream for each class.
    zstreams = (cudaStream_t*) malloc(sizeof(cudaStream_t)*nClasses);
    end_of_zderiv = (cudaEvent_t*) malloc(sizeof(cudaEvent_t)*nClasses);
    dimGridCovering = (dim3*) malloc(sizeof(dim3)*nClasses);
    dimBlockCovering.x = 8;
    dimBlockCovering.y = 8;
    dimBlockCovering.z = 8;



    for(int c=0; c<nClasses; c++) {    
      int n[1] = {nLinks[c]*Nz*icovering};
      cudaStreamCreate(&(zstreams[c]));
      cufftPlanMany(&plan_covering[c],1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,nChains[c]);
      //cufftSetStream(plan_covering[c], zstreams[0]);
  
      dimGridCovering[c].x = (Nz+dimBlockCovering.x-1)/dimBlockCovering.x;
      dimGridCovering[c].y = (nChains[c]+dimBlockCovering.y-1)/dimBlockCovering.y;
      dimGridCovering[c].z = (nLinks[c]*icovering+dimBlockCovering.z-1)/dimBlockCovering.z;
  
      if(DEBUG) kPrint(nLinks[c], nChains[c], kyCover_h[c], kxCover_h[c]); 
      cudaMalloc((void**) &g_covering[c], sizeof(cuComplex)*icovering*Nz*nLinks[c]*nChains[c]);
      cudaMalloc((void**) &kz_covering[c], sizeof(float)*icovering*Nz*nLinks[c]);
      cudaMalloc((void**) &kxCover[c], sizeof(int)*nLinks[c]*nChains[c]);
      cudaMalloc((void**) &kyCover[c], sizeof(int)*nLinks[c]*nChains[c]);    
      cudaMemcpy(kxCover[c], kxCover_h[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyHostToDevice);
      cudaMemcpy(kyCover[c], kyCover_h[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyHostToDevice);    
    }    
    //printf("nLinks[0] = %d  nChains[0] = %d\n", nLinks[0],nChains[0]);
    
  
    if(DEBUG) getError("run_gryfx.cu, after kCover");
    ///////////////////////////////////////////////////////////////////////////////////////////////////
      
  
    
  
    
    ////////////////////////////////////////////////
    // set up some diagnostics/control flow files //
    ////////////////////////////////////////////////
    
    
    //set up stopfile
    //strcpy(stopfileName, out_stem);
    //strcat(stopfileName, "stop");
    
    
    ////////////////////////////////////////////
    
  
    //////////////////////////////
    // initial conditions set here
    //////////////////////////////
    
  
    if(DEBUG) getError("run_gryfx.cu, before initial condition"); 
    
    //if running nonlinear part of secondary test...
    if(secondary_test && !LINEAR && RESTART) { 
      restartRead(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi, pflxAvg, wpfxAvg, Phi2_kxky_sum, Phi2_zonal_sum,
      			zCorr_sum,&expectation_ky_sum, &expectation_kx_sum, &Phi_zf_kx1_avg,
      			&dtSum, &counter,&runtime,&dt,&totaltimer,secondary_test_restartfileName);
  			
       fieldWrite(Phi, field_h, "phi_restarted.field", filename);

       get_fixed_mode<<<dimGrid,dimBlock>>>(phi_fixed, Phi, 1, 0);
       get_fixed_mode<<<dimGrid,dimBlock>>>(dens_fixed, Dens[ION], 1, 0);
       get_fixed_mode<<<dimGrid,dimBlock>>>(upar_fixed, Upar[ION], 1, 0);
       get_fixed_mode<<<dimGrid,dimBlock>>>(tpar_fixed, Tpar[ION], 1, 0);
       get_fixed_mode<<<dimGrid,dimBlock>>>(tprp_fixed, Tprp[ION], 1, 0);
       get_fixed_mode<<<dimGrid,dimBlock>>>(qpar_fixed, Qpar[ION], 1, 0);
       get_fixed_mode<<<dimGrid,dimBlock>>>(qprp_fixed, Qprp[ION], 1, 0);

       printf("Before set_fixed_amplitude\n");
       cudaMemcpy(CtmpZ_h, phi_fixed, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("phi_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }

       cudaMemcpy(CtmpZ_h, dens_fixed, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("dens_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       if(SLAB) set_fixed_amplitude<<<dimGrid,dimBlock>>>(phi_fixed, dens_fixed, upar_fixed, tpar_fixed, tprp_fixed, qpar_fixed, qprp_fixed, phi_test);
       else set_fixed_amplitude_withz<<<dimGrid,dimBlock>>>(phi_fixed, dens_fixed, upar_fixed, tpar_fixed, tprp_fixed, qpar_fixed, qprp_fixed, phi_test);

       printf("After set_fixed_amplitude\n");
       cudaMemcpy(CtmpZ_h, phi_fixed, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("phi_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }

       cudaMemcpy(CtmpZ_h, dens_fixed, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("dens_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       cudaMemcpy(CtmpZ_h, upar_fixed, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("upar_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       cudaMemcpy(CtmpZ_h, tpar_fixed, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("tpar_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       cudaMemcpy(CtmpZ_h, tprp_fixed, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("tprp_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       cudaMemcpy(CtmpZ_h, qpar_fixed, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("qpar_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       cudaMemcpy(CtmpZ_h, qprp_fixed, sizeof(cuComplex)*Nz, cudaMemcpyDeviceToHost);
 
       for(int i=0; i<Nz; i++) {
         printf("qprp_fixed(idz=%d) = (%e,%e)\n", i, CtmpZ_h[i].x, CtmpZ_h[i].y);
       }
       //initialize density with noise
       RESTART = false; 
       init = DENS;

       //restart run from t=0
       counter = 0;
       runtime = 0.;
       dtSum = 0.;

       maxdt = .1/(phi_test.x*kx_h[1]*ky_h[1]);
    } 
    
    //float amp;
    
    if(!RESTART) {
      
        for(int index=0; index<Nx*(Ny/2+1)*Nz; index++) 
        {
  	init_h[index].x = 0;
  	init_h[index].y = 0;
        }
        //amp = 1.e-5; //e-20;
        
        srand(22);
        float samp;
  
  	for(int j=0; j<Nx; j++) {
  	  for(int i=0; i<(Ny/2+1); i++) {
  
  	    //int index = i + (Ny/2+1)*j + (Ny/2+1)*Nx*k;
  	    //int idxy = i + (Ny/2+1)*j;
  
  	      //if(i==0) amp = 1.e-5;
  	      //else amp = 1.e-5;
  
  	      //if(i==0) {
  	      //	samp = 0.;//init_amp;//*1.e-8;  //initialize zonal flows at much
  	      //  			 //smaller amplitude
  	      //}
  	      //else {
  	      //	samp = init_amp;
  	      //}
  
//  	      if(j==0 && secondary_test) 
//		samp = 0.;
//             else 
		samp = init_amp;
  
  	      float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
  	      float rb = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
  	      //printf("%e\n", ra);
  
  	      //loop over z here to get rid of randomness in z in initial condition
  	      for(int k=0; k<Nz; k++) {
  	        int index = i + (Ny/2+1)*j + (Ny/2+1)*Nx*k;
  		  init_h[index].x = samp;//*cos(1*z_h[k]);
  	          init_h[index].y = samp;//init_amp;//*cos(1*z_h[k]);
  	      }
  	      
  	      
  	        
  	      
  	  }
  	}
        
      if(init == DENS) {
        if(DEBUG) getError("initializing density");    
        for(int s=0; s<nSpecies; s++) {
          cudaMemset(Dens[s], 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
        }
        
        
  
        cudaMemcpy(Dens[ION], init_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
        if(DEBUG) getError("after copy");    
  
        //enforce reality condition -- this is CRUCIAL when initializing in k-space
        reality<<<dimGrid,dimBlock>>>(Dens[ION]);
        
        mask<<<dimGrid,dimBlock>>>(Dens[ION]);  
  
        cudaMemset(Phi, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
        
      }
      
      if(init == PHI) {
        
        
        
        cudaMemset(Phi, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz); 
        
        cudaMemcpy(Phi, init_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
      
        cudaMemset(Dens[ION], 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);  
  
        mask<<<dimGrid,dimBlock>>>(Phi);
  
        reality<<<dimGrid,dimBlock>>>(Phi);
      } 
  
      if(init == FORCE) {
        cudaMemset(Phi, 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
        cudaMemset(Dens[ION], 0, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
      }   
     
  
      for(int s=0; s<nSpecies; s++) {
        zeroC <<< dimGrid, dimBlock >>> (Dens1[s]);
        if(s!=0) zeroC<<<dimGrid,dimBlock>>>(Dens[s]);
  
        zeroC <<< dimGrid, dimBlock >>> (Upar[s]);
        
        zeroC <<< dimGrid, dimBlock >>> (Tpar[s]);
  
        zeroC <<< dimGrid, dimBlock >>> (Qpar[s]);
  
        zeroC <<< dimGrid, dimBlock >>> (Tprp[s]);
  
        zeroC <<< dimGrid, dimBlock >>> (Qprp[s]);
        
        zeroC <<< dimGrid, dimBlock >>> (Upar1[s]);
        
        zeroC <<< dimGrid, dimBlock >>> (Tpar1[s]);
  
        zeroC <<< dimGrid, dimBlock >>> (Qpar1[s]);
  
        zeroC <<< dimGrid, dimBlock >>> (Tprp1[s]);
  
        zeroC <<< dimGrid, dimBlock >>> (Qprp1[s]);
      }
      
      zeroC<<<dimGrid,dimBlock>>>(Phi1);
      if(DEBUG) getError("run_gryfx.cu, after zero");
       
      if(init == RH_equilibrium) {
         
        zeroC<<<dimGrid,dimBlock>>>(Dens[0]);
        zeroC<<<dimGrid,dimBlock>>>(Phi);
  
        RH_equilibrium_init<<<dimGrid,dimBlock>>>(Dens[0], Upar[0], Tpar[0], Tprp[0], Qpar[0], Qprp[0], kx, gds22, qsf, eps, bmagInv, shat, species[0]);
  
        qneut(Phi, Dens, Tprp, tmp, tmp, field, species, species_d);
  
      }
      
   
      if(DEBUG) getError("after initial qneut");
      
      
      
      
      for(int s=0; s<nSpecies; s++) {
        //wpfx_sum[s]= 0.;
      }
      expectation_ky_sum= 0.;
      expectation_kx_sum= 0.;
      dtSum= 0.;
      flux1_phase_sum = 0.;
      flux2_phase_sum = 0.;
      Dens_phase_sum = 0.;
      Tpar_phase_sum = 0.;
      Tprp_phase_sum = 0.;
      zero<<<dimGrid,dimBlock>>>(Phi2_kxky_sum,Nx,Ny/2+1,1);
      zero<<<dimGrid,dimBlock>>>(wpfxnorm_kxky_sum, Nx, Ny/2+1, 1);
      zero<<<dimGrid,dimBlock>>>(Phi2_zonal_sum, Nx, 1,1);
      //zeroC<<<dimGrid,dimBlock>>>(Phi_sum);
      zero<<<dimGrid,dimBlock>>>(zCorr_sum, 1, Ny/2+1, Nz);
      
          
      
    } 
    else {
      restartRead(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi, pflxAvg, wpfxAvg, Phi2_kxky_sum, Phi2_zonal_sum,
      			zCorr_sum,&expectation_ky_sum, &expectation_kx_sum, &Phi_zf_kx1_avg,
      			&dtSum, &counter,&runtime,&dt,&totaltimer,restartfileName);
  			
      if(zero_restart_avg) {
        printf("zeroing avg sums...\n");
        for(int s=0; s<nSpecies; s++) {
          //wpfx_sum[s] = 0.;
        }
        expectation_ky_sum = 0.;
        expectation_kx_sum = 0.;
        dtSum = 0.;
        zero<<<dimGrid,dimBlock>>>(Phi2_kxky_sum, Nx, Ny/2+1, 1);
        zero<<<dimGrid,dimBlock>>>(Phi2_zonal_sum, Nx, 1, 1);
        zero<<<dimGrid,dimBlock>>>(zCorr_sum, 1, Ny/2+1, Nz);
      }


  	
    
    }
  


   
    
    //printf("phi file is %s\n", phifileName);


    //writedat_beginning();
      
    cudaEventCreate(&start);
    cudaEventCreate(&stop);		
    cudaEventCreate(&nonlin_halfstep);
    cudaEventCreate(&H2D);
    cudaEventCreateWithFlags(&D2H, cudaEventBlockingSync);
    for(int c=0; c<nClasses; c++) {
      cudaEventCreate(&end_of_zderiv[c]); 
    }
    cudaEventCreate(&GS2start);
    cudaEventCreate(&GS2stop);		

    //cudaEventCreate(&end_of_zderiv);  

    cudaStreamCreate(&copystream);

    //int copystream = 0;
  
    //cudaEventCreate(&start1); 			    
    //cudaEventCreate(&stop1);
    //cudaEventRecord(start,0);
    
    //cudaProfilerStart();
    
    /*
    float step_timer=0.;
    float step_timer_total=0.;
    float diagnostics_timer=0.;
    float diagnostics_timer_total=0.;
    */ 

#ifdef GS2_zonal    
			} //end of iproc if    
    //MPI_Barrier(MPI_COMM_WORLD);
#endif 

      //strcpy(stopfileName, out_stem);
      //strcat(stopfileName, "stop");
      runtime=0.;
      counter=0;
      gs2_counter=1;
      totaltimer=0.;
      timer=0.;
      //GS2timer=0.;
    int stopcount = 0;
    int nstop = 10;
    int first_half_flag;

#ifdef GS2_zonal

if(iproc==0) {
    cudaMemset(phi_ky0_d, 0., sizeof(cuComplex)*ntheta0*Nz);
    for(int s=0; s<nSpecies; s++) {
      cudaMemset(dens_ky0_d[s], 0., sizeof(cuComplex)*ntheta0*Nz);
      cudaMemset(upar_ky0_d[s], 0., sizeof(cuComplex)*ntheta0*Nz);
      cudaMemset(tpar_ky0_d[s], 0., sizeof(cuComplex)*ntheta0*Nz);
      cudaMemset(tprp_ky0_d[s], 0., sizeof(cuComplex)*ntheta0*Nz);
      cudaMemset(qpar_ky0_d[s], 0., sizeof(cuComplex)*ntheta0*Nz);
      cudaMemset(qprp_ky0_d[s], 0., sizeof(cuComplex)*ntheta0*Nz);
    }
}    

    memset(phi_ky0_h, 0., sizeof(cuComplex)*ntheta0*Nz);
    memset(dens_ky0_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    memset(upar_ky0_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    memset(tpar_ky0_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    memset(tprp_ky0_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    memset(qpar_ky0_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);
    memset(qprp_ky0_h, 0., sizeof(cuComplex)*ntheta0*Nz*nSpecies);

    //set initial condition from GS2 for ky=0 modes

    getmoms_gryfx(dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h);
      
if(iproc==0) {
    cudaMemcpyAsync(phi_ky0_d, phi_ky0_h, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
    for(int s=0; s<nSpecies; s++) {
      cudaMemcpyAsync(dens_ky0_d[s], dens_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(upar_ky0_d[s], upar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(tpar_ky0_d[s], tpar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(tprp_ky0_d[s], tprp_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(qpar_ky0_d[s], qpar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(qprp_ky0_d[s], qprp_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
    }
    
  
    cudaEventRecord(H2D, copystream);
    cudaStreamWaitEvent(0, H2D, 0);
   
    fieldWrite_nopad_h(phi_ky0_h, "phi0.field", filename, Nx, 1, Nz, ntheta0, 1);

    //replace ky=0 modes with results from GS2
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(Phi, phi_ky0_d);
    for(int s=0; s<nSpecies; s++) {
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Dens[s], dens_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Upar[s], upar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Tpar[s], tpar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Tprp[s], tprp_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Qpar[s], qpar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Qprp[s], qprp_ky0_d[s]);
      reality<<<dimGrid,dimBlock>>>(Dens[s]);
      reality<<<dimGrid,dimBlock>>>(Upar[s]);
      reality<<<dimGrid,dimBlock>>>(Tpar[s]);
      reality<<<dimGrid,dimBlock>>>(Tprp[s]);
      reality<<<dimGrid,dimBlock>>>(Qpar[s]);
      reality<<<dimGrid,dimBlock>>>(Qprp[s]);
      mask<<<dimGrid,dimBlock>>>(Dens[s]);
      mask<<<dimGrid,dimBlock>>>(Upar[s]);
      mask<<<dimGrid,dimBlock>>>(Tpar[s]);
      mask<<<dimGrid,dimBlock>>>(Tprp[s]);
      mask<<<dimGrid,dimBlock>>>(Qpar[s]);
      mask<<<dimGrid,dimBlock>>>(Qprp[s]);
    }
    reality<<<dimGrid,dimBlock>>>(Phi);
    mask<<<dimGrid,dimBlock>>>(Phi);

    fieldWrite(Phi, field_h, "phi_1.field", filename); 
    getky0_nopad<<<dimGrid,dimBlock>>>(phi_ky0_d, Phi);
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(Phi, phi_ky0_d);
    fieldWrite(Phi, field_h, "phi_2.field", filename); 
   
}  
    //gs2_main_mp_advance_gs2_();
    //printf("finished gs2\n");
    //exit(1);
  
#endif
    
#ifdef GS2_zonal
			if(iproc==0) {  
#endif
      if(secondary_test && !LINEAR) {
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Phi, phi_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Dens[ION], dens_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Upar[ION], upar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Tpar[ION], tpar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Tprp[ION], tprp_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Qpar[ION], qpar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Qprp[ION], qprp_fixed, 1, 0, S_fixed);
      }

      if(init == DENS) {
        // Solve for initial phi
        // assumes the initial conditions have been moved to the device
        qneut(Phi, Dens, Tprp, tmp, tmp, field, species, species_d);
      }


  if(DEBUG) getError("about to start timestep loop");
    fieldWrite(Dens[ION], field_h, "dens0.field", filename); 
    fieldWrite(Upar[ION], field_h, "upar0.field", filename); 
    fieldWrite(Tpar[ION], field_h, "tpar0.field", filename); 
    fieldWrite(Tprp[ION], field_h, "tprp0.field", filename); 
    fieldWrite(Qpar[ION], field_h, "qpar0.field", filename); 
    fieldWrite(Qprp[ION], field_h, "qprp0.field", filename); 
    fieldWrite(Phi, field_h, "phi0.field", filename); 

  volflux(Phi,Phi,tmp,tmpXY);
  sumY_neq_0<<<dimGrid,dimBlock>>>(tmpX, tmpXY);
  kxWrite(tmpX, tmpX_h, filename, "phi2_0.kx");

#ifdef GS2_zonal
			}
#endif 

    //uint64_t diff;
    struct timespec clockstart;//, clockend;
    //MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
   first_half_flag = 1;
   // //cudaEventRecord(GS2Istart,0);
//    clock_gettime(CLOCK_MONOTONIC, &clockstart);
//   // //getError("Before GS2start");
//    while(gs2_counter<=1) { 
//   //   //printf("gs2 step\t");
//   //   first_half_flag=1;
//    MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
//      gs2_main_mp_advance_gs2_(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h, &first_half_flag);
//      gs2_counter++;
//   //   first_half_flag=0;
//    MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
//      gs2_main_mp_advance_gs2_(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h, &first_half_flag);
//      gs2_counter++;
//    }
//    clock_gettime(CLOCK_MONOTONIC, &clockend);
//    diff = 1000000000L * (clockend.tv_sec - clockstart.tv_sec) + clockend.tv_nsec - clockstart.tv_nsec;
//    printf("elapsed time = %llu ns\n", (long long unsigned int) diff);
//   // //GS2timer = ((double) (clockend - clockstart)) / CLOCKS_PER_SEC;
//   // //cudaEventRecord(GS2stop,0);
//   // //getError("After GS2stop");
//   // //cudaEventElapsedTime(&GS2timer, GS2start, GS2stop);
//   // //getError("After GS2timer");
//    printf("proc%d: GS2 time/timestep is %e ns\n", iproc, ((double) diff)/1.);
//   // 
//    gs2_counter=1;

    clock_gettime(CLOCK_MONOTONIC, &clockstart);
#ifdef GS2_zonal
    if(iproc==0) {
#endif
      cudaEventRecord(start,0);
#ifdef GS2_zonal
    }
#endif

    /////////////////////
    // Begin timestep loop
    /////////////////////
   
    while(/*counter < 1 &&*/ 
        counter<nSteps &&
  	stopcount<nstop 
  	/*&& converge_count<2*navg*/
  	)
     {

      //if(counter==9) cudaProfilerStart();
#ifdef GS2_zonal
//     if(turn_off_gradients_test && counter==nSteps/2) {
//       species[ION].tprim = 0.;
//       species[ION].fprim = 0.;
//       double den[1], tem[1], fp[1], tp[1], vnewk[1];
//       den[0] = (double) species[ION].dens;
//       tem[0] = (double) species[ION].temp;
//       fp[0] = (double) species[ION].fprim;
//       tp[0] = (double) species[ION].tprim;
//       vnewk[0] = (double) (species[ION].nu_ss/sqrt(2.));
//       
//       species_mp_reinit_species_(&nSpecies, den, tem, fp, tp, vnewk);
//       if(iproc==0) printf("\n\n%d: GryfX gradients have been reset to tprim = %f, fprim = %f; runtime=%f\n\n", counter, species[ION].tprim, species[ION].fprim, runtime);
//     }
//if(DEBUG && counter==0) printf("proc %d has entered the timestep loop\n", iproc);
			if(iproc==0) {
#endif
  //    cudaEventRecord(start1,0);


#ifndef GS2_zonal
      //dt_old = dt;
      //if(!LINEAR && !secondary_test) dt = courant(Phi, tmp, field, resultR_nlps, species);   
#endif
      //avgdt = .5*(dt_old+dt);    

#ifdef GS2_zonal

      if(gs2_time_mp_code_time_/sqrt(2.) - runtime > .0001) printf("\n\nRuntime mismatch! GS2 time is %f, GryfX time is %f\n\n", gs2_time_mp_code_time_/sqrt(2.), runtime); 

#endif

      //if(counter<100 && dt>dt_start) dt = dt_start;
      
  
      //EXBshear bug fixed, need to check if correct
      //ExBshear(Phi,Dens,Upar,Tpar,Tprp,Qpar,Qprp,kx_shift,jump,avgdt);  
      
      
      //if(DEBUG) getError("after exb");
     
      //if(LINEAR) { 
      //volflux_zonal<<<dimGrid,dimBlock>>>(tmpX, Phi, Phi, jacobian, 1./fluxDen); 
      //  getPhiVal<<<dimGrid,dimBlock>>>(val, Phi1, 0, 4, Nz/2);
     //   fprintf(phifile, "\n\t%f\t%e", runtime, val);
      
      //getPhiVal<<<dimGrid,dimBlock>>>(val, tmpX, 4);
      
      //getky0z0<<<dimGrid,dimBlock>>>(tmpX, Phi);
  
      //volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Phi, jacobian, 1./fluxDen);
      //get_real_X<<<dimGrid,dimBlock>>>(tmpX, CtmpX);
  
      //phiVal[0] = 0;
      //cudaMemcpy(phiVal, val, sizeof(float), cudaMemcpyDeviceToHost);
      //cudaMemcpy(tmpX_h, tmpX, sizeof(float)*Nx, cudaMemcpyDeviceToHost);
      /*if(runtime == 0) {
        phiVal0 = phiVal[0];
      }*/
  
    //  if(counter==0){
    //    fprintf(phifile,"t");
    //    for(int i=0; i<Nx; i++) {
    //      fprintf(phifile, "\tkx=%g", kx_h[i]);
    //      if(init==DENS || init==PHI) phi0_X[i] = tmpX_h[i];
    //    }
  
    //  }
    //  if(init== DENS || init==PHI) fprintf(phifile, "\n\t%f\t%e", runtime, (float) tmpX_h[0]);///phi0_X[0]);
    //  else fprintf(phifile, "\n\t%f\t%e", runtime, (float) 1-tmpX_h[0]); 
    //  for(int i=1; i<Nx; i++) {
    //    if(init== DENS || init==PHI) fprintf(phifile, "\t%e", (float) tmpX_h[i]);///phi0_X[i]);
    //    else fprintf(phifile, "\t%e", (float) 1-tmpX_h[i]); 
    //  }
 //     }
  
      
      //calculate diffusion here... for now we just set it to 1
//      diffusion = 1.;
       
      //cudaProfilerStart();

#ifdef GS2_zonal
			} //end of iproc if
#endif

  /////////////////////////////////
  //FIRST HALF OF GRYFX TIMESTEP
  /////////////////////////////////
  
      //////nvtxRangePushA("Gryfx t->t+dt/2");

      first_half_flag = 1;
#ifdef GS2_zonal
			if(iproc==0) {
#endif
      if(!LINEAR) {
        for(int s=0; s<nSpecies; s++) {
          //calculate NL(t) = NL(Moment)
          nonlinear_timestep(Dens[s], Dens[s], Dens1[s], 
                 Upar[s], Upar[s], Upar1[s], 
                 Tpar[s], Tpar[s], Tpar1[s], 
                 Qpar[s], Qpar[s], Qpar1[s], 
                 Tprp[s], Tprp[s], Tprp1[s], 
                 Qprp[s], Qprp[s], Qprp1[s], 
                 Phi, 
                 dens_ky0_d[s], upar_ky0_d[s], tpar_ky0_d[s], tprp_ky0_d[s], qpar_ky0_d[s], qprp_ky0_d[s],
                 &dt, species[s],
  	       field,field,field,field,field,field,
  	       tmp,tmp, tmpX, CtmpX, first_half_flag,
	       field_h, counter, kx2Phi_zf_rms, kx2Phi_zf_rms_avg);
  
          //Moment1 = Moment + (dt/2)*NL(Moment)
  


#ifdef GS2_zonal
  
          //copy NL(t)_ky=0 from D2H
  
          if(s==nSpecies-1) {  //Only after all species have been done
            cudaEventRecord(nonlin_halfstep, 0); //record this after all streams (ie the default stream) reach this point
            cudaStreamWaitEvent(copystream, nonlin_halfstep,0); //wait for all streams before copying
            for(int i=0; i<nSpecies; i++) {
              cudaMemcpyAsync(dens_ky0_h + s*ntheta0*Nz, dens_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
              cudaMemcpyAsync(upar_ky0_h + s*ntheta0*Nz, upar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
              cudaMemcpyAsync(tpar_ky0_h + s*ntheta0*Nz, tpar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
              cudaMemcpyAsync(tprp_ky0_h + s*ntheta0*Nz, tprp_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
              cudaMemcpyAsync(qpar_ky0_h + s*ntheta0*Nz, qpar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
              cudaMemcpyAsync(qprp_ky0_h + s*ntheta0*Nz, qprp_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
            }
            cudaEventRecord(D2H, copystream);
          }

          if(counter==0) fieldWrite_nopad_h(dens_ky0_h, "NLdens.field", filename, Nx, 1, Nz, ntheta0, 1);
  
#endif
  
          //calculate L(t) = L(Moment)
          linear_timestep(Dens1[s], Dens[s], Dens1[s], 
                 Upar1[s], Upar[s], Upar1[s], 
                 Tpar1[s], Tpar[s], Tpar1[s], 
                 Qpar1[s], Qpar[s], Qpar1[s], 
                 Tprp1[s], Tprp[s], Tprp1[s], 
                 Qprp1[s], Qprp[s], Qprp1[s], 
                 Phi, kxCover,kyCover, g_covering, kz_covering, species[s], dt/2.,
  	       field,field,field,field,field,field,
  	       tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmpZ,plan_covering,
  	       nu_nlpm, tmpX, tmpXZ, CtmpX, CtmpX2);
   
          //Moment1 = Moment1 + (dt/2)*L(Moment)

  	}         
      }
      else { //if only linear
        for(int s=0; s<nSpecies; s++) {
  
          //calculate L(t) = L(Moment)
          linear_timestep(Dens[s], Dens[s], Dens1[s], 
                 Upar[s], Upar[s], Upar1[s], 
                 Tpar[s], Tpar[s], Tpar1[s], 
                 Qpar[s], Qpar[s], Qpar1[s], 
                 Tprp[s], Tprp[s], Tprp1[s], 
                 Qprp[s], Qprp[s], Qprp1[s], 
                 Phi, kxCover,kyCover, g_covering, kz_covering, species[s], dt/2.,
  	       field,field,field,field,field,field,
  	       tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmpZ,plan_covering,
  	       nu_nlpm, tmpX, tmpXZ, CtmpX, CtmpX2);
           
          //Moment1 = Moment + (dt/2)*L(Moment)

        }
      }

//if(DEBUG && counter==0) getError("after linear step"); 

      qneut(Phi1, Dens1, Tprp1, tmp, tmp, field, species, species_d);
  
      if(secondary_test && !LINEAR) {
        if(runtime < .02/maxdt/M_PI) S_fixed = 1.;// sin(.01/maxdt * runtime);
        else S_fixed = 1.;
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Phi1, phi_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Dens1[ION], dens_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Upar1[ION], upar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Tpar1[ION], tpar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Tprp1[ION], tprp_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Qpar1[ION], qpar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Qprp1[ION], qprp_fixed, 1, 0, S_fixed);
      }

      //f1 = f(t+dt/2)
  
  /*
    if(DEBUG) {*/
    //if(counter==0) fieldWrite(Dens1[ION], field_h, "dens.5.field", filename); 
  /*  if(counter==0) fieldWrite(Upar1[ION], field_h, "upar.5.field", filename); 
    if(counter==0) fieldWrite(Tpar1[ION], field_h, "tpar.5.field", filename); 
    if(counter==0) fieldWrite(Tprp1[ION], field_h, "tprp.5.field", filename); 
    if(counter==0) fieldWrite(Qpar1[ION], field_h, "qpar.5.field", filename); 
    if(counter==0) fieldWrite(Qprp1[ION], field_h, "qprp.5.field", filename); 
    }
  */
  
  
#ifdef GS2_zonal
			} //end of iproc if
#endif

      //////nvtxRangePop();
#ifdef GS2_zonal
    
    //MPI_Barrier(MPI_COMM_WORLD); //make all procs wait


    if(!LINEAR) {
      if(iproc==0) cudaEventSynchronize(D2H); //have proc 0 wait for NL(t) to be copied D2H before advancing GS2 if running nonlinearly
    }
  
      //MPI_Barrier(MPI_COMM_WORLD); //make all procs wait
    //nvtxRangePushA("GS2 t->t+dt/2");
    //cudaEventRecord(GS2start,0);

    //advance GS2 t -> t + dt/2  
    advance_gs2(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h, &first_half_flag);
    gs2_counter++;

    //nvtxRangePop();
    //cudaEventRecord(GS2stop,0);
    //cudaEventElapsedTime(&GS2timer, GS2start, GS2stop);
    //printf("100 calls of advance_gs2 took %f ms\n", GS2timer);

/*
    if(!LINEAR) {
      fields_implicit_mp_advance_implicit_(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h);
    } else {
      fields_mp_advance_(&gs2_counter);
    }
    if(DEBUG) printf("step %d of %d: proc %d advanced gs2 first half step\n", counter, nSteps, iproc);
    gs2_time_mp_update_time_();
    //gs2_diagnostics_mp_loop_diagnostics_(&gs2_counter, &exit_flag);
    gs2_counter++;
  
    MPI_Barrier(MPI_COMM_WORLD); 

    dist_fn_mp_getmoms_gryfx_(dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h);
*/

    //////nvtxRangePushA("copy moms(t+dt/2)_ky=0 from H2D");
if(iproc==0) {  
    //copy moms(t+dt/2)_ky=0 from H2D
    cudaMemcpyAsync(phi_ky0_d, phi_ky0_h, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
    for(int s=0; s<nSpecies; s++) {
      cudaMemcpyAsync(dens_ky0_d[s], dens_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(upar_ky0_d[s], upar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(tpar_ky0_d[s], tpar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(tprp_ky0_d[s], tprp_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(qpar_ky0_d[s], qpar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(qprp_ky0_d[s], qprp_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
    }
    
  
    cudaEventRecord(H2D, copystream);
    cudaStreamWaitEvent(0, H2D, 0);
   
    //replace ky=0 modes with results from GS2
    replace_ky0_nopad<<<dimGrid,dimBlock>>>(Phi1, phi_ky0_d);
    reality<<<dimGrid,dimBlock>>>(Phi1);
    mask<<<dimGrid,dimBlock>>>(Phi1);

    for(int s=0; s<nSpecies; s++) {
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Dens1[s], dens_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Upar1[s], upar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Tpar1[s], tpar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Tprp1[s], tprp_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Qpar1[s], qpar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Qprp1[s], qprp_ky0_d[s]);
      reality<<<dimGrid,dimBlock>>>(Dens1[s]);
      reality<<<dimGrid,dimBlock>>>(Upar1[s]);
      reality<<<dimGrid,dimBlock>>>(Tpar1[s]);
      reality<<<dimGrid,dimBlock>>>(Tprp1[s]);
      reality<<<dimGrid,dimBlock>>>(Qpar1[s]);
      reality<<<dimGrid,dimBlock>>>(Qprp1[s]);
      mask<<<dimGrid,dimBlock>>>(Dens1[s]);
      mask<<<dimGrid,dimBlock>>>(Upar1[s]);
      mask<<<dimGrid,dimBlock>>>(Tpar1[s]);
      mask<<<dimGrid,dimBlock>>>(Tprp1[s]);
      mask<<<dimGrid,dimBlock>>>(Qpar1[s]);
      mask<<<dimGrid,dimBlock>>>(Qprp1[s]);
    }
 } 
    //////nvtxRangePop();
  #endif
         

    //////nvtxRangePushA("Gryfx t->t+dt");

      first_half_flag = 0;
#ifdef GS2_zonal 
			if(iproc==0) {
#endif

      //NLPM calculated AFTER ky=0 quantities passed back from GS2!
      if(!LINEAR && NLPM && dorland_phase_complex) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPM(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s], 
        		tmpX, CtmpX, tmpXZ, CtmpXZ, tmpYZ, nu_nlpm, nu1_nlpm_complex, nu22_nlpm_complex, tmpZ, species[s], dt/2., Dnlpm_d, Phi_zf_kx1_avg, kx2Phi_zf_rms);
        }	    
      }
      else if(!LINEAR && NLPM) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPM(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s], 
        		tmpX, tmpXZ, tmpYZ, nu_nlpm, nu1_nlpm, nu22_nlpm, species[s], dt/2., Dnlpm_d, Phi_zf_kx1_avg, kx2Phi_zf_rms, tmp);
        }	    
      }  
      //hyper too...
      if(HYPER) {
        if(isotropic_shear) {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_iso(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s], 
  			tmpXYZ, shear_rate_nz, dt/2.);
  		    
          }  
        }
        else {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_aniso(Phi1, Dens1[s], Upar1[s], Tpar1[s], Tprp1[s], Qpar1[s], Qprp1[s],
                          tmpXYZ, shear_rate_nz, shear_rate_z, shear_rate_z_nz, dt/2.);
          }
        }
      }
 
  /////////////////////////////////
  //SECOND HALF OF GRYFX TIMESTEP
  /////////////////////////////////
      if(!LINEAR) {
        for(int s=0; s<nSpecies; s++) {
          //calculate NL(t+dt/2) = NL(Moment1)
          nonlinear_timestep(Dens[s], Dens1[s], Dens[s], 
                 Upar[s], Upar1[s], Upar[s], 
                 Tpar[s], Tpar1[s], Tpar[s], 
                 Qpar[s], Qpar1[s], Qpar[s], 
                 Tprp[s], Tprp1[s], Tprp[s], 
                 Qprp[s], Qprp1[s], Qprp[s], 
                 Phi1, 
                 dens_ky0_d[s], upar_ky0_d[s], tpar_ky0_d[s], tprp_ky0_d[s], qpar_ky0_d[s], qprp_ky0_d[s],
                 &dt, species[s],
  	       field,field,field,field,field,field,
  	       tmp,tmp, tmpX, CtmpX, first_half_flag,
	       field_h, counter, kx2Phi_zf_rms, kx2Phi_zf_rms_avg);
  
          //Moment = Moment + dt * NL(Moment1)
  
  #ifdef GS2_zonal
  
          //copy NL(t+dt/2)_ky=0 from D2H
  
          if(s==nSpecies-1) {  //Only after all species have been done
            cudaEventRecord(nonlin_halfstep, 0); //record this after all streams (ie the default stream) reach this point
            cudaStreamWaitEvent(copystream, nonlin_halfstep,0); //wait for all streams before copying
            for(int i=0; i<nSpecies; i++) {
              cudaMemcpyAsync(dens_ky0_h + s*ntheta0*Nz, dens_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
              cudaMemcpyAsync(upar_ky0_h + s*ntheta0*Nz, upar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
              cudaMemcpyAsync(tpar_ky0_h + s*ntheta0*Nz, tpar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
              cudaMemcpyAsync(tprp_ky0_h + s*ntheta0*Nz, tprp_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
              cudaMemcpyAsync(qpar_ky0_h + s*ntheta0*Nz, qpar_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
              cudaMemcpyAsync(qprp_ky0_h + s*ntheta0*Nz, qprp_ky0_d[s], sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyDeviceToHost, copystream);
            }
            cudaEventRecord(D2H, copystream);
          }
  
  #endif
  
          //calculate L(t+dt/2)=L(Moment1) 
          linear_timestep(Dens[s], Dens1[s], Dens[s], 
                 Upar[s], Upar1[s], Upar[s], 
                 Tpar[s], Tpar1[s], Tpar[s], 
                 Qpar[s], Qpar1[s], Qpar[s], 
                 Tprp[s], Tprp1[s], Tprp[s], 
                 Qprp[s], Qprp1[s], Qprp[s], 
                 Phi1, kxCover,kyCover, g_covering, kz_covering, species[s], dt,
  	       field,field,field,field,field,field,
  	       tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmpZ,plan_covering,
  	       nu_nlpm, tmpX, tmpXZ, CtmpX, CtmpX2);
  
          //Moment = Moment + dt * L(Moment1)
  	}         
      }
      else { //if only linear
        for(int s=0; s<nSpecies; s++) {
  
          linear_timestep(Dens[s], Dens1[s], Dens[s], 
                 Upar[s], Upar1[s], Upar[s], 
                 Tpar[s], Tpar1[s], Tpar[s], 
                 Qpar[s], Qpar1[s], Qpar[s], 
                 Tprp[s], Tprp1[s], Tprp[s], 
                 Qprp[s], Qprp1[s], Qprp[s], 
                 Phi1, kxCover,kyCover, g_covering, kz_covering, species[s], dt,
  	       field,field,field,field,field,field,
  	       tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmpZ,plan_covering,
  	       nu_nlpm, tmpX, tmpXZ, CtmpX, CtmpX2);
        }
      }
      

        if(!LINEAR && !secondary_test && !write_omega) qneut(Phi, Dens, Tprp, tmp, tmp, field, species, species_d); //don't need to keep Phi=Phi(t) when running nonlinearly, overwrite with Phi=Phi(t+dt)
        else qneut(Phi1, Dens, Tprp, tmp, tmp, field, species, species_d); //don't overwrite Phi=Phi(t), use Phi1=Phi(t+dt); for growth rate calculation
      
      if(secondary_test && !LINEAR) {
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Phi1, phi_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Dens[ION], dens_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Upar[ION], upar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Tpar[ION], tpar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Tprp[ION], tprp_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Qpar[ION], qpar_fixed, 1, 0, S_fixed);
        replace_fixed_mode<<<dimGrid,dimBlock>>>(Qprp[ION], qprp_fixed, 1, 0, S_fixed);
      }
      //f = f(t+dt)
  
  
#ifdef GS2_zonal
			} //end of iproc if
#endif

    //////nvtxRangePop();

#ifdef GS2_zonal
    
    
    if(!LINEAR) {
      if(iproc==0) cudaEventSynchronize(D2H); //wait for NL(t+dt/2) to be copied D2H before advancing GS2 if running nonlinearly
    }
    //advance GS2 t+dt/2 -> t+dt
      //MPI_Barrier(MPI_COMM_WORLD);
    //nvtxRangePushA("GS2 t+dt/2->t+dt");
    advance_gs2(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h, &first_half_flag);
    gs2_counter++;
    //nvtxRangePop();
  
/*
    MPI_Barrier(MPI_COMM_WORLD);
    if(!LINEAR) {
      fields_implicit_mp_advance_implicit_(&gs2_counter, dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h);
    } else {
      fields_mp_advance_(&gs2_counter);
    }
    gs2_time_mp_update_time_();
    //gs2_diagnostics_mp_loop_diagnostics_(&gs2_counter, &exit_flag);
  
    MPI_Barrier(MPI_COMM_WORLD);
    dist_fn_mp_getmoms_gryfx_(dens_ky0_h, upar_ky0_h, tpar_ky0_h, tprp_ky0_h, qpar_ky0_h, qprp_ky0_h, phi_ky0_h);
*/

    //////nvtxRangePushA("copy moms(t+dt)_ky=0 from H2D");
if(iproc==0) {  
    //copy moms(t+dt)_ky=0 from H2D
    cudaMemcpyAsync(phi_ky0_d, phi_ky0_h, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
    for(int s=0; s<nSpecies; s++) {
      cudaMemcpyAsync(dens_ky0_d[s], dens_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(upar_ky0_d[s], upar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(tpar_ky0_d[s], tpar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(tprp_ky0_d[s], tprp_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(qpar_ky0_d[s], qpar_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
      cudaMemcpyAsync(qprp_ky0_d[s], qprp_ky0_h + s*ntheta0*Nz, sizeof(cuComplex)*ntheta0*Nz, cudaMemcpyHostToDevice, copystream);
    }
    
  
    cudaEventRecord(H2D, copystream);
    cudaStreamWaitEvent(0, H2D, 0);
   
    if(!LINEAR && !secondary_test && !write_omega) {
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Phi, phi_ky0_d);
      mask<<<dimGrid,dimBlock>>>(Phi);
      reality<<<dimGrid,dimBlock>>>(Phi);
    } else {
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Phi1, phi_ky0_d);
      mask<<<dimGrid,dimBlock>>>(Phi1);
      reality<<<dimGrid,dimBlock>>>(Phi1);
    }
    for(int s=0; s<nSpecies; s++) {
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Dens[s], dens_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Upar[s], upar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Tpar[s], tpar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Tprp[s], tprp_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Qpar[s], qpar_ky0_d[s]);
      replace_ky0_nopad<<<dimGrid,dimBlock>>>(Qprp[s], qprp_ky0_d[s]);

      reality<<<dimGrid,dimBlock>>>(Dens[s]);
      reality<<<dimGrid,dimBlock>>>(Upar[s]);
      reality<<<dimGrid,dimBlock>>>(Tpar[s]);
      reality<<<dimGrid,dimBlock>>>(Tprp[s]);
      reality<<<dimGrid,dimBlock>>>(Qpar[s]);
      reality<<<dimGrid,dimBlock>>>(Qprp[s]);

      mask<<<dimGrid,dimBlock>>>(Dens[s]);
      mask<<<dimGrid,dimBlock>>>(Upar[s]);
      mask<<<dimGrid,dimBlock>>>(Tpar[s]);
      mask<<<dimGrid,dimBlock>>>(Tprp[s]);
      mask<<<dimGrid,dimBlock>>>(Qpar[s]);
      mask<<<dimGrid,dimBlock>>>(Qprp[s]);
    }
}  

    //////nvtxRangePop();
  
#endif
  
    //////nvtxRangePushA("Diagnostics");
    
#ifdef GS2_zonal
			if(iproc==0) {
#endif  
  
      //NLPM
      if(!LINEAR && NLPM && dorland_phase_complex) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPM(Phi, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s], 
        		tmpX, CtmpX, tmpXZ, CtmpXZ, tmpYZ, nu_nlpm, nu1_nlpm_complex, nu22_nlpm_complex, tmpZ, species[s], dt, Dnlpm_d, Phi_zf_kx1_avg, kx2Phi_zf_rms);
        }	    
      }
      else if(!LINEAR && NLPM) {
        for(int s=0; s<nSpecies; s++) {
          filterNLPM(Phi, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s], 
        		tmpX, tmpXZ, tmpYZ, nu_nlpm, nu1_nlpm, nu22_nlpm, species[s], dt, Dnlpm_d, Phi_zf_kx1_avg, kx2Phi_zf_rms, tmp);
        }	    
      }  
          
      
      if(HYPER) {
        if(isotropic_shear) {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_iso(Phi, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s], 
  			tmpXYZ, shear_rate_nz, dt);
  		    
          }  
        }
        else {
          for(int s=0; s<nSpecies; s++) {
            filterHyper_aniso(Phi, Dens[s], Upar[s], Tpar[s], Tprp[s], Qpar[s], Qprp[s],
                          tmpXYZ, shear_rate_nz, shear_rate_z, shear_rate_z_nz, dt);
          }
        }
      }



  /*
      cudaEventRecord(stop1,0);
      cudaEventSynchronize(stop1);
      cudaEventElapsedTime(&step_timer,start1,stop1);
      step_timer_total += step_timer;
  */    //cudaProfilerStop();
      
      
      //DIAGNOSTICS
       
     // if(LINEAR) { 
     //   getPhiVal<<<dimGrid,dimBlock>>>(val, Phi1, 0, 4, Nz/2);
     //   cudaMemcpy(phiVal, val, sizeof(float), cudaMemcpyDeviceToHost);
     //   fprintf(phifile, "\n\t%f\t%e", runtime, phiVal[0]);
     // }
    //  cudaEventRecord(start1,0);  
      if(LINEAR || secondary_test || write_omega) {
        growthRate<<<dimGrid,dimBlock>>>(omega,Phi1,Phi,dt);    
        //cudaMemcpy(omega_h, omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);     
        //weighted average of omega over 'navg' timesteps
        //boxAvg(omegaAvg, omega, omegaBox, dt, dtBox, navg, counter);
        //cudaMemcpy(omegaAvg_h, omegaAvg, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);      
  
        //copy Phi for next timestep
        cudaMemcpy(Phi, Phi1, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
        mask<<<dimGrid,dimBlock>>>(Phi1);
        mask<<<dimGrid,dimBlock>>>(Phi);
  
        
        //print growth rates to files   
        //omegaWrite(omegafile,gammafile,omegaAvg_h,runtime); 
        
  
        //if(counter>2*navg) {
  	//omegaStability(omega_h, omegaAvg_h, stability,Stable,stableMax);
  	//STABLE_STOP = stabilityCheck(Stable,stableMax);
        //}
      }
      
  
  
  //DIAGNOSTICS
  
      if( strcmp(nlpm_option,"constant") == 0) Dnlpm = dnlpm;
      else cudaMemcpy(&Dnlpm, Dnlpm_d, sizeof(float), cudaMemcpyDeviceToHost);
      
      /*
      if(counter%nwrite==0) {
        cudaMemcpy(phi_h, Phi, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
      }
      */
      
      for(int s=0; s<nSpecies; s++) {    
        wpfx_old[s] = wpfx[s];
        pflx_old[s] = pflx[s];
      }
  
      //calculate instantaneous heat flux
      for(int s=0; s<nSpecies; s++) {  
        fluxes(&pflx[s], &wpfx[s],flux1,flux2,Dens[s],Tpar[s],Tprp[s],Phi,
               tmp,tmp,tmp,field,field,field,tmpZ,tmpXY,species[s],runtime,
               &flux1_phase, &flux2_phase, &Dens_phase, &Tpar_phase, &Tprp_phase);        
      }
       
      volflux_zonal(Phi,Phi,tmpX);  //tmpX = Phi_zf**2(kx)
      get_kx1_rms<<<1,1>>>(Phi_zf_kx1_d, tmpX);
      Phi_zf_kx1_old = Phi_zf_kx1;
      cudaMemcpy(&Phi_zf_kx1, Phi_zf_kx1_d, sizeof(float), cudaMemcpyDeviceToHost);
      
      //volflux_zonal(Phi,Phi,tmpX);  //tmpX = Phi_zf**2(kx)
      kx2Phi_zf_rms_old = kx2Phi_zf_rms;
      multKx4<<<dimGrid,dimBlock>>>(tmpX2, tmpX, kx); 
      kx2Phi_zf_rms = sumReduc(tmpX2, Nx, false);
      kx2Phi_zf_rms = sqrt(kx2Phi_zf_rms);
  
      //volflux_zonal(Phi,Phi,tmpX);  //tmpX = Phi_zf**2(kx)
      Phi2_zf = sumReduc(tmpX, Nx, false);
      Phi_zf_rms = sqrt(Phi2_zf);   

      //calculate tmpXY = Phi**2(kx,ky)
      volflux(Phi,Phi,tmp,tmpXY);
      //if(!LINEAR && write_phi2kxky_time) {
      //  cudaMemcpy(tmpXY_h, tmpXY, sizeof(float)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
      //  kxkyTimeWrite(phikxkyfile, tmpXY_h, runtime);
      //}
      sumX<<<dimGrid,dimBlock>>>(tmpY, tmpXY);
      cudaMemcpy(tmpY_h, tmpY, sizeof(float)*(Ny/2+1), cudaMemcpyDeviceToHost);
      cudaMemcpy(ev_h->outs.phi2_by_ky, tmpY, sizeof(float)*(Ny/2+1), cudaMemcpyDeviceToHost);
      sumY <<< dimGrid, dimBlock >>>(tmpX2, tmpXY);
      cudaMemcpy(ev_h->outs.phi2_by_kx, tmpX2, sizeof(float)*Nx,cudaMemcpyDeviceToHost);

      if(!LINEAR && turn_off_gradients_test) {
        kyTimeWrite(ev_h->files.phifile, tmpY_h, runtime);
      }
      //calculate <kx> and <ky>
      expect_k<<<dimGrid,dimBlock>>>(tmpXY2, tmpXY, ky);
      kPhi2 = sumReduc(tmpXY2, Nx*(Ny/2+1), false);
      Phi2 = sumReduc(tmpXY, Nx*(Ny/2+1), false);

      ev_h->outs.phi2 = Phi2;

      expectation_ky = (float) Phi2/kPhi2;
  
      expect_k<<<dimGrid,dimBlock>>>(tmpXY2, tmpXY, kx);
      kPhi2 = sumReduc(tmpXY2, Nx*(Ny/2+1), false);
      expectation_kx = (float) Phi2/kPhi2;
      
      //calculate z correlation function = tmpYZ (not normalized)
      zCorrelation<<<dimGrid,dimBlock>>>(tmpYZ, Phi);
     // volflux(Phi,Phi,tmp,tmpXY);

      nu1_nlpm_max = maxReduc(nu1_nlpm, Nz, false);
      nu22_nlpm_max = maxReduc(nu22_nlpm, Nz, false); 
    
      if(counter>0) { 
        //we use an exponential moving average
        // wpfx_avg[t] = alpha_avg*wpfx[t] + (1-alpha_avg)*wpfx_avg[t-1]
        // now with time weighting...
        // wpfx_sum[t] = alpha_avg*dt*wpfx[t] + (1-alpha_avg)*wpfx_avg[t-1]
        // dtSum[t] = alpha_avg*dt[t] + (1-alpha_avg)*dtSum[t-1]
        // wpfx_avg[t] = wpfx_sum[t]/dtSum[t]
   
        // keep a running total of dt, phi**2(kx,ky), expectation values, etc.
        dtSum = dtSum*(1.-alpha_avg) + dt*alpha_avg;
        add_scaled<<<dimGrid,dimBlock>>>(Phi2_kxky_sum, 1.-alpha_avg, Phi2_kxky_sum, dt*alpha_avg, tmpXY, Nx, Ny, 1);
        add_scaled<<<dimGrid,dimBlock>>>(Phi2_zonal_sum, 1.-alpha_avg, Phi2_zonal_sum, dt*alpha_avg, tmpX, Nx, 1, 1);
        add_scaled<<<dimGrid,dimBlock>>>(zCorr_sum, 1.-alpha_avg, zCorr_sum, dt*alpha_avg, tmpYZ, 1, Ny, Nz);
        if(LINEAR || write_omega || secondary_test) {
          if(counter>0) {
            add_scaled<<<dimGrid,dimBlock>>>(omegaAvg, 1.-alpha_avg, omegaAvg, dt*alpha_avg, omega, Nx, Ny, 1);
            cudaMemcpy(omegaAvg_h, omegaAvg, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);      
            //print growth rates to files   
            omegaWrite(ev_h->files.omegafile,ev_h->files.gammafile,omegaAvg_h,dtSum,runtime); 
          }
          else {
            cudaMemcpy(omegaAvg_h, omega, sizeof(cuComplex)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);      
            //print growth rates to files   
            omegaWrite(ev_h->files.omegafile,ev_h->files.gammafile,omegaAvg_h,runtime); 
          }             

        }  
        expectation_kx_sum = expectation_kx_sum*(1.-alpha_avg) + expectation_kx*dt*alpha_avg;
        expectation_ky_sum = expectation_ky_sum*(1.-alpha_avg) + expectation_ky*dt*alpha_avg;
        Phi2_sum = Phi2_sum*(1.-alpha_avg) + Phi2*dt*alpha_avg;
        Phi_zf_rms_sum = Phi_zf_rms_sum*(1.-alpha_avg) + Phi_zf_rms*dt*alpha_avg;
        kx2Phi_zf_rms_sum = kx2Phi_zf_rms_sum*(1.-alpha_avg) + kx2Phi_zf_rms*dt*alpha_avg;
        flux1_phase_sum = flux1_phase_sum*(1.-alpha_avg) + flux1_phase*dt*alpha_avg;
        flux2_phase_sum = flux2_phase_sum*(1.-alpha_avg) + flux2_phase*dt*alpha_avg;
        Dens_phase_sum = Dens_phase_sum*(1.-alpha_avg) + Dens_phase*dt*alpha_avg;
        Tpar_phase_sum = Tpar_phase_sum*(1.-alpha_avg) + Tpar_phase*dt*alpha_avg;
        Tprp_phase_sum = Tprp_phase_sum*(1.-alpha_avg) + Tprp_phase*dt*alpha_avg;
        Dnlpm_sum = Dnlpm_sum*(1.-alpha_avg) + Dnlpm*dt*alpha_avg;
  
        
        // **_sum/dtSum gives time average of **
        Phi_zf_rms_avg = Phi_zf_rms_sum/dtSum;
        //kx2Phi_zf_rms_avg = kx2Phi_zf_rms_sum/dtSum;
        Dnlpm_avg = Dnlpm_sum/dtSum;
  
        for(int s=0; s<nSpecies; s++) {
          wpfxAvg[s] = mu_avg*wpfxAvg[s] + (1-mu_avg)*wpfx[s] + (mu_avg - (1-mu_avg)/alpha_avg)*(wpfx[s] - wpfx_old[s]);
          pflxAvg[s] = mu_avg*pflxAvg[s] + (1-mu_avg)*pflx[s] + (mu_avg - (1-mu_avg)/alpha_avg)*(pflx[s] - pflx_old[s]);
        }
  
        alpha_nlpm = dt/tau_nlpm;
        mu_nlpm = exp(-alpha_nlpm);
        if(runtime<20) {
          Phi_zf_kx1_avg = Phi_zf_kx1; //allow a build-up time of tau_nlpm
          kx2Phi_zf_rms_avg = kx2Phi_zf_rms;
        }
        else { 
          Phi_zf_kx1_avg = mu_nlpm*Phi_zf_kx1_avg + (1-mu_nlpm)*Phi_zf_kx1 + (mu_nlpm - (1-mu_nlpm)/alpha_nlpm)*(Phi_zf_kx1 - Phi_zf_kx1_old);
          kx2Phi_zf_rms_avg = mu_nlpm*kx2Phi_zf_rms_avg + (1-mu_nlpm)*kx2Phi_zf_rms + (mu_nlpm - (1-mu_nlpm)/alpha_nlpm)*(kx2Phi_zf_rms - kx2Phi_zf_rms_old);
        }
  /*
        // try to autostop when wpfx converges
        // look at min and max of wpfxAvg over time... if wpfxAvg stays within certain bounds for a given amount of
        // time, it is converged
        if(counter >= navg*1.2) {
          //set bounds to be +/- .05*wpfxAvg
  	converge_bounds = .1*wpfxAvg[ION];
  	//if counter reaches navg/3, recenter bounds
  	if(counter == navg || converge_count == navg/3) {
  	  wpfxmax = wpfxAvg[ION] + .5*converge_bounds;
  	  wpfxmin = wpfxAvg[ION] - .5*converge_bounds;
  	  converge_count++;
  	}
  	//if wpfxAvg goes outside the bounds, reset the bounds.
  	else if(wpfxAvg[ION] > wpfxmax) {
            wpfxmax = wpfxAvg[ION] + .3*converge_bounds;
  	  wpfxmin = wpfxmax - converge_bounds;
  	  converge_count=0;
  	}
  	else if(wpfxAvg[ION] < wpfxmin) {
            wpfxmin = wpfxAvg[ION] - .3*converge_bounds;
  	  wpfxmax = wpfxmin + converge_bounds;
  	  converge_count=0;
  	}
  	//if wpfxAvg stays inside the bounds, increment the convergence counter.
  	else converge_count++; 
        }    
  */
      }
  
      fluxWrite(ev_h->files.fluxfile,pflx, pflxAvg, wpfx,wpfxAvg, Dnlpm, Dnlpm_avg, Phi_zf_kx1, Phi_zf_kx1_avg, kx2Phi_zf_rms, kx2Phi_zf_rms_avg, nu1_nlpm_max,nu22_nlpm_max,converge_count,runtime,species);
    
  	     
      if(counter%nsave==0 && write_phi) phiR_historyWrite(Phi,omega,tmpXY_R,tmpXY_R_h, runtime, ev_h->files.phifile); //save time history of Phi(x,y,z=0)          
      
      
      // print wpfx to screen if not printing growth rates
      if(!write_omega && counter%nwrite==0) printf("%d: wpfx = %f, dt = %f, dt_cfl =  %f, Dnlpm = %f\n", gpuID, wpfx[0],dt, dt_cfl, Dnlpm);
      
      // write flux to file
      if(counter%nsave==0) fflush(NULL);
               
      
      
      
      //print growth rates to screen every nwrite timesteps if write_omega
      if(write_omega) {
        if (counter%nwrite==0 || stopcount==nstop-1 || counter==nSteps-1) {
  	printf("ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<((Nx-1)/3+1); i++) {
  	  for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      printf("%.4f\t%.4f\t\t%.6f\t%.6f", ky_h[j], kx_h[i], omegaAvg_h[index].x/dtSum, omegaAvg_h[index].y/dtSum);
        ev_h->outs.omega[index].x = omegaAvg_h[index].x/dtSum;
        ev_h->outs.omega[index].y = omegaAvg_h[index].y/dtSum;
  	      if(Stable[index] >= stableMax) printf("\tomega");
  	      if(Stable[index+Nx*(Ny/2+1)] >= stableMax) printf("\tgamma");
  	      printf("\n");
  	    }
  	  }
  	  printf("\n");
  	}
  	//for(int i=2*Nx/3+1; i<2*Nx/3+1; i++) {
        for(int i=2*Nx/3+1; i<Nx; i++) {
            for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	    printf("%.4f\t%.4f\t\t%.6f\t%.6f", ky_h[j], kx_h[i], omegaAvg_h[index].x/dtSum, omegaAvg_h[index].y/dtSum);
        ev_h->outs.omega[index].x = omegaAvg_h[index].x/dtSum;
        ev_h->outs.omega[index].y = omegaAvg_h[index].y/dtSum;
  	    if(Stable[index] >= stableMax) printf("\tomega");
  	    if(Stable[index+Nx*(Ny/2+1)] >= stableMax) printf("\tgamma");
  	    printf("\n");
  	  }
  	  printf("\n");
  	}	
        }            
      }
  /*    
      cudaEventRecord(stop1,0);
      cudaEventSynchronize(stop1);
      cudaEventElapsedTime(&diagnostics_timer, start1, stop1);
      diagnostics_timer_total += diagnostics_timer;    
  */    
      //writedat_each();
      
	if (counter%nwrite==0){
  writedat_each(&ev_h->grids, &ev_h->outs, &ev_h->fields, &ev_h->time);
}
        ev_h->time.runtime = runtime;
#ifdef GS2_zonal
			} //end of iproc if
#endif
      //////nvtxRangePop();
      //if(counter==9) cudaProfilerStop();
      runtime+=dt;
      counter++;       
      //checkstop
      if(FILE* checkstop = fopen(ev_h->files.stopfileName, "r") ) {
        fclose(checkstop);
        stopcount++;
      }     
    
#ifdef GS2_zonal
			if(iproc==0) {
      if(counter%nsave==0 || stopcount==nstop-1 || counter==nSteps-1) {
        printf("%d: %f    %f     dt=%f   %d: %s\n",gpuID,runtime,gs2_time_mp_code_time_/sqrt(2.), dt,counter,cudaGetErrorString(cudaGetLastError()));
      }
#endif
      
      
      
      //check for problems with run
      if(!LINEAR && !secondary_test && (isnan(wpfx[ION]) || isinf(wpfx[ION]) || wpfx[ION] < -100 || wpfx[ION] > 100000) ) {
        printf("\n-------------\n--RUN ERROR--\n-------------\n\n");
        
        stopcount=100;
#ifdef GS2_zonal
        abort();
        broadcast_integer(&stopcount);
#endif
/*
        if(counter>nsave) {	
          printf("RESTARTING FROM LAST RESTART FILE...\n");
  	restartRead(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,pflxAvg, wpfxAvg, Phi2_kxky_sum, Phi2_zonal_sum,
      			zCorr_sum,&expectation_ky_sum, &expectation_kx_sum, &Phi_zf_kx1_avg,
      			&dtSum, &counter,&runtime,&dt,&totaltimer,restartfileName);
        
          printf("cfl was %f. maxdt was %f.\n", cfl, maxdt);
          cfl = .8*cfl;
          maxdt = .8*maxdt;    
          printf("cfl is now %f. maxdt is now %f.\n", cfl, maxdt);
        }
        else{
  	printf("ERROR: FLUX IS NAN AT BEGINNING OF RUN\nENDING RUN...\n\n");
          stopcount=100;
        } */
      }
  
      else if(counter%nsave==0) {
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timer,start,stop);
        totaltimer+=timer;
        cudaEventRecord(start,0);
        restartWrite(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi,pflxAvg,wpfxAvg,Phi2_kxky_sum, Phi2_zonal_sum,
        			zCorr_sum, expectation_ky_sum, expectation_kx_sum, Phi_zf_kx1_avg,
        			dtSum,counter,runtime,dt,totaltimer,restartfileName);
      }
      
      
      
      if(counter%nsave == 0) gryfx_finish_diagnostics(Dens, Upar, Tpar, Tprp, Qpar, Qprp, 
                          Phi, tmp, tmp, field, tmpZ, CtmpX,
                          tmpXY, tmpXY, tmpXY, tmpXY2, tmpXY3, tmpXY4, tmpYZ, tmpYZ,
    			tmpX, tmpX2, tmpY, tmpY, tmpY, tmpY, tmpY2, tmpY2, tmpY2, 
                          kxCover, kyCover, tmpX_h, tmpY_h, tmpXY_h, tmpYZ_h, field_h, 
                          kxCover_h, kyCover_h, omegaAvg_h, qflux, &expectation_ky, &expectation_kx,
  			Phi2_kxky_sum, wpfxnorm_kxky_sum, Phi2_zonal_sum, zCorr_sum, expectation_ky_sum, 
  			expectation_kx_sum, dtSum,
  			counter, runtime, false,
  			&Phi2, &flux1_phase, &flux2_phase, &Dens_phase, &Tpar_phase, &Tprp_phase,
  			Phi2_sum, flux1_phase_sum, flux2_phase_sum, Dens_phase_sum, Tpar_phase_sum, Tprp_phase_sum);
  
#ifdef GS2_zonal
			} //end of iproc if
#endif


#ifdef GS2_zonal
/*  
    gs2_reinit_mp_check_time_step_(&reset, &exit_flag); //use gs2_time_mp_code_dt_cfl to check gs2_time_mp_code_dt
    if(reset) {
      gs2_reinit_mp_reset_time_step_(&gs2_counter, &exit_flag); //change gs2_time_mp_code_dt
      dt = gs2_time_mp_code_dt_ * 2. / sqrt(2.);  // pass GS2 dt to GryfX, with appropriate normalization
    }
*/  
    //dt_old = dt;
    dt = gs2_time_mp_code_dt_ * 2. / sqrt(2.);

    //gs2_counter++; //each proc has its own gs2_counter
//if(DEBUG) printf("proc %d thinks counter is %d, gs2_counter is %d, nsteps is %d\n", iproc, counter, gs2_counter, nSteps);
  
#endif

    } //end of timestep loop 
 
#ifdef GS2_zonal
//if(DEBUG) printf("proc %d exited the timestep loop\n", iproc);
    //clock_gettime(CLOCK_MONOTONIC, &clockend);
    //diff = 1000000000L * (clockend.tv_sec - clockstart.tv_sec) + clockend.tv_nsec - clockstart.tv_nsec;
    //printf("elapsed time = %llu ns\n", (long long unsigned int) diff);
    ////GS2timer = ((double) (clockend - clockstart)) / CLOCKS_PER_SEC;
    ////cudaEventRecord(GS2stop,0);
    ////getError("After GS2stop");
    ////cudaEventElapsedTime(&GS2timer, GS2start, GS2stop);
    ////getError("After GS2timer");
    //printf("proc%d: GS2 time/timestep is %e ns\n", iproc, ((double) diff)/nSteps);

			if(iproc==0) {
#endif
  
    //exit_flag = 1; 
   
    //gs2_diagnostics_mp_loop_diagnostics_(&gs2_counter, &exit_flag);
  
    if(DEBUG) getError("just finished timestep loop");
    
    //cudaProfilerStop();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer,start,stop);
    totaltimer+=timer;
    
    nSteps = counter;     //counter at which fields were last calculated
    endtime = runtime;    //time at which fields were last calculated
    
    for(int s=0; s<nSpecies; s++) {
      qflux[s] = wpfxAvg[s];
      pflux[s] = pflxAvg[s];
    }
    
    ////////////////////////////////////////////////////////////
  
    if(DEBUG) getError("before restartWrite");  
    
    // save for restart run
    restartWrite(Dens,Upar,Tpar,Tprp,Qpar,Qprp,Phi, pflxAvg, wpfxAvg, Phi2_kxky_sum, Phi2_zonal_sum, 
    			zCorr_sum, expectation_ky_sum, expectation_kx_sum, Phi_zf_kx1_avg,
    			dtSum,counter,runtime,dt,totaltimer,restartfileName);
    
    if(DEBUG) getError("after restartWrite");
    
    //phiR_historyWrite(Phi1,omega,tmpXY_R,tmpXY_R_h, runtime, ev_h->files.phifile); //save time history of Phi(x,y,z=0)      
    
    gryfx_finish_diagnostics(Dens, Upar, Tpar, Tprp, Qpar, Qprp, 
                          Phi, tmp, tmp, field, tmpZ, CtmpX,
                          tmpXY, tmpXY, tmpXY, tmpXY2, tmpXY3, tmpXY4, tmpYZ, tmpYZ,
    			tmpX, tmpX2, tmpY, tmpY, tmpY, tmpY, tmpY2, tmpY2, tmpY2, 
                          kxCover, kyCover, tmpX_h, tmpY_h, tmpXY_h, tmpYZ_h, field_h, 
                          kxCover_h, kyCover_h, omegaAvg_h, qflux, &expectation_ky, &expectation_kx,
  			Phi2_kxky_sum, wpfxnorm_kxky_sum, Phi2_zonal_sum, zCorr_sum, expectation_ky_sum, 
  			expectation_kx_sum, dtSum,
  			counter, runtime, true,
  			&Phi2, &flux1_phase, &flux2_phase, &Dens_phase, &Tpar_phase, &Tprp_phase,
  			Phi2_sum, flux1_phase_sum, flux2_phase_sum, Dens_phase_sum, Tpar_phase_sum, Tprp_phase_sum);
  
  
    
    
    //if(write_omega) stabilityWrite(stability,Stable,stableMax);
      
    //Timing       
    printf("Total steps: %d\n", counter);
    printf("Total time (min): %f\n",totaltimer/60000.);    //convert ms to minutes
    printf("Avg time/timestep (s): %f\n",totaltimer/counter/1000.);   //convert ms to s
    //printf("Advance steps:\t%f min\t(%f%)\n", step_timer_total/60000., 100*step_timer_total/totaltimer);
    //printf("Diagnostics:\t%f min\t(%f%)\n", diagnostics_timer_total/60000., 100*diagnostics_timer_total/totaltimer);
  
    
    fprintf(outfile,"expectation val of ky = %f\n", expectation_ky);
    fprintf(outfile,"expectation val of kx = %f\n", expectation_kx);
    fprintf(outfile,"Q_i = %f\n Phi_zf_rms = %f\n Phi2 = %f\n", qflux[ION],Phi_zf_rms_avg, Phi2);
    fprintf(outfile, "flux1_phase = %f \t\t flux2_phase = %f\nDens_phase = %f \t\t Tpar_phase = %f \t\t Tprp_phase = %f\n", flux1_phase, flux2_phase, Dens_phase, Tpar_phase, Tprp_phase);
    fprintf(outfile,"\nTotal time (min): %f\n",totaltimer/60000);
    fprintf(outfile,"Total steps: %d\n", counter);
    fprintf(outfile,"Avg time/timestep (s): %f\n",totaltimer/counter/1000);
      
    //cleanup  
    for(int s=0; s<nSpecies; s++) {
      cudaFree(Dens[s]), cudaFree(Dens1[s]);
      cudaFree(Upar[s]), cudaFree(Upar1[s]);
      cudaFree(Tpar[s]), cudaFree(Tpar1[s]);
      cudaFree(Tprp[s]), cudaFree(Tprp1[s]);
      cudaFree(Qpar[s]), cudaFree(Qpar1[s]);
      cudaFree(Qprp[s]), cudaFree(Qprp1[s]);
      cudaFree(dens_ky0_d[s]);
      cudaFree(upar_ky0_d[s]);
      cudaFree(tpar_ky0_d[s]);
      cudaFree(tprp_ky0_d[s]);
      cudaFree(qpar_ky0_d[s]);
      cudaFree(qprp_ky0_d[s]);
    }  
    cudaFree(Phi);
    cudaFree(Phi1);
    cudaFree(Phi2_kxky_sum);
    //cudaFree(Phi_sum);
    cudaFree(phi_ky0_d);
    
    cudaFree(tmp);
    cudaFree(field);      
    cudaFree(tmpZ);
    cudaFree(tmpX);
    cudaFree(tmpY);
    cudaFree(tmpY2);
    cudaFree(tmpXY);
    cudaFree(tmpXY2);
    cudaFree(tmpYZ);
    
    cudaFree(bmag), cudaFree(bmagInv), cudaFree(bmag_complex), cudaFree(bgrad);
    cudaFree(omega); cudaFree(omegaAvg);
    for(int t=0; t<navg; t++) {
      //cudaFree(Phi2_XYBox[t]);
      if(LINEAR) cudaFree(omegaBox[t]);
    }
    
    cudaFree(kx_shift), cudaFree(jump);
    
    cudaFree(jacobian);
      
    cudaFree(kx), cudaFree(ky), cudaFree(kz);
    
    cudaFree(deriv_nlps);
    cudaFree(derivR1_nlps);
    cudaFree(derivR2_nlps);
    cudaFree(resultR_nlps);

    cudaFree(nu_nlpm);
    cudaFree(nu1_nlpm);
    cudaFree(nu22_nlpm);
    
    cufftDestroy(NLPSplanR2C);
    cufftDestroy(NLPSplanC2R);
    cufftDestroy(ZDerivBplanR2C);
    cufftDestroy(ZDerivBplanC2R);
    cufftDestroy(ZDerivplan);
    for(int c=0; c<nClasses; c++) {
      cufftDestroy(plan_covering[c]);
      cudaFree(kxCover[c]);
      cudaFree(kyCover[c]);
      cudaFree(g_covering[c]); 
      cudaFree(kz_covering[c]);
    }
    
    close_files(&ev_h->files);
    //fclose(fluxfile);
    //fclose(omegafile);
    //fclose(gammafile);
    //fclose(ev_h->files.phifile);
   writedat_end(ev_h->outs);
    
    //cudaProfilerStop();
#ifdef GS2_zonal
  			} //end of iproc if  
#endif
    cudaFreeHost(dens_ky0_h);
    cudaFreeHost(upar_ky0_h);
    cudaFreeHost(tpar_ky0_h);
    cudaFreeHost(tprp_ky0_h);
    cudaFreeHost(qpar_ky0_h);
    cudaFreeHost(qprp_ky0_h);
    cudaFreeHost(phi_ky0_h);
}    
    
    
    
    
