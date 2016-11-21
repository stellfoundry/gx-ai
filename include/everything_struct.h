//This header file defines the everything struct , which 
// contains all parameters and all dynamically allocated
// memory in the simulation (basically everything except
// local variables).

#include "species.h"
#include "grids.h"
#include "fields.h"
#include "outputs.h"
#include "input_parameters_struct.h"
#include "geometry.h"
#include "info.h"
#include "files.h"


typedef struct {
	// Dimensionless simulation time
	double runtime;
	// Timestep counter
	int counter;
  int gs2_counter;
	// Used to calculate exponential moving averages
	float dtSum;
	// Total runtime in minutes
	float totaltimer;
	// Step runtime in minutes
	float timer;

  double end_time; // The time at which the simulation must end in seconds since the epoch
  double start_time; // The start time of the first gryfx call in this program execution in seconds since the epoch
  //time_t this_start_time; // The start time of this gryfx call 

  int first_half_flag;

  double dt;

  float cflx;
  float cfly;

  // For use with trinity; allows certain netcdf variables 
  // to be written as a function of trinity call.
  int trinity_timestep;
  int trinity_iteration;
  int trinity_conv_count;
} time_struct;

typedef struct {
  int zThreads, totalThreads, zBlockThreads;
	dim3 dimBlock;
  dim3 dimGrid;
	dim3 dimBlockCovering;
  dim3 * dimGridCovering;
  dim3 dimGridCovering_all;
} cuda_dimensions_struct;

typedef struct {
  cuComplex *phi, *dens, *upar, *tpar, *tprp, *qpar, *qprp;
  float S;
  int iky_fixed;
  int ikx_fixed;
} secondary_fixed_arrays_struct;

typedef struct {
  cudaStream_t * zstreams;
  cudaStream_t copystream;
} cuda_streams_struct;

typedef struct {
  cudaEvent_t start, stop,  nonlin_halfstep, H2D, D2H, GS2start, GS2stop;
  cudaEvent_t * end_of_zderiv;
} cuda_events_struct;

typedef struct{
	cufftHandle NLPSplanR2C;
	cufftHandle	NLPSplanC2R;
 	cufftHandle	ZDerivBplanR2C;
 	cufftHandle	ZDerivBplanC2R;
  cufftHandle	ZDerivplan;
  cufftHandle	XYplanC2R;
  cufftHandle   XplanC2C;
  cufftHandle   XYplanC2C;
  cufftHandle   XYplanZ_C2C;

	/* Allocated in initialize_z_covering */
	cufftHandle * plan_covering;
} cuffts_struct;


typedef struct {
  float D_par;
  float D_prp;
  float Beta_par;
  cuComplex nu[11];
  cuComplex mu[11];
} damping_coefficients_struct;
  
typedef struct {
  int mpcom;
  int iproc;
} mpi_info_struct;

typedef struct {

	cuComplex * phi;
	cuComplex ** dens;
	cuComplex ** upar;
	cuComplex ** tpar;
	cuComplex ** tprp;
	cuComplex ** qpar;
	cuComplex ** qprp;
	cuComplex * dens_h;
	cuComplex * upar_h;
	cuComplex * tpar_h;
	cuComplex * tprp_h;
	cuComplex * qpar_h;
	cuComplex * qprp_h;
	
} hybrid_zonal_arrays_struct;


/* A bunch of device arrays to be reused continuously */
typedef struct {
	cuComplex * CXYZ;
        cuComplex * CXYZ2;
	float * X;
	float * X2;
	float * Y;
	float * Y2;
	float * Z;
	cuComplex * CZ;
	cuComplex * CX;
	cuComplex * CX2;

	cuComplex * CXY;
	cuComplex * CXZ;


	float * XY;
	float * XY2;
	float * XY3;
	float * XY4;
	float * XY_R;

	float * XZ;
	float * XZ2;
	float * YZ;
	float * XYZ;
} temporary_arrays_struct;	

typedef struct {
  float kx2Phi_zf_rms;
  float kx2Phi_zf_rms_avg;
  float kx2Phi_zf_rms_old;
  float Phi_zf_kx1;
  float Phi_zf_kx1_old;
  float Phi_zf_kx1_avg;
  float *nu;
  float *nu1;
  float *nu22;
  float nu1_max;
  float nu22_max;
  cuComplex *nu1_complex;
  cuComplex *nu22_complex;
  float D;
  float D_avg;
  float D_sum;
  float alpha;
  float mu;
} nlpm_struct;


typedef struct {
    float *shear_rate_z;
    float *shear_rate_z_nz;
    float *shear_rate_nz;  
} hyper_struct;

/* Information about how the run
 * is progressing and whether it has
 * converged */
typedef struct {
  int * stable;
  int stable_max;
  int converge_count;
  int stopcount;
  int nstop;
} run_control_struct;


typedef struct {

  fields_struct  fields;

  //This is used only on the device and contains
  //pointers which point to fields, e.g. fields1.dens = fields.dens1
  //Eventually we will get rid of fields.*1 and replace them entirely
  //with fields1.*
  fields_struct  fields1;
	geometry_coefficents_struct  geo;
	outputs_struct  outs;
	input_parameters_struct pars;
	time_struct time;
	grids_struct grids;
	info_struct info;
	temporary_arrays_struct tmp;
	files_struct files;
	cuda_dimensions_struct cdims;
	cuffts_struct ffts;
  damping_coefficients_struct damps;
  mpi_info_struct mpi;
  cuda_streams_struct streams;
  cuda_events_struct events;
  secondary_fixed_arrays_struct sfixed;
  hybrid_zonal_arrays_struct hybrid;
  nlpm_struct nlpm;
  hyper_struct hyper;
  run_control_struct ctrl;
  

	/* Specifies whether the pointers in the struct point 
	 * to host or device memory */
	int memory_location;

} everything_struct;



void setup_everything_structs(everything_struct * ev_h, everything_struct ** ev_hd_ptr, everything_struct ** ev_d_ptr);
