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
	// Used to calculate exponential moving averages
	float dtSum;
	// Total runtime in minutes
	float totaltimer;
	// Step runtime in minutes
	float timer;

  int first_half_flag;

  double dt;

  float cflx;
  float cfly;
} time_struct;

typedef struct {
  int zThreads, totalThreads, zBlockThreads;
	dim3 dimBlock;
  dim3 dimGrid;
	dim3 dimBlockCovering;
  dim3 * dimGridCovering;
} cuda_dimensions_struct;

typedef struct {
  cuComplex *phi, *dens, *upar, *tpar, *tprp, *qpar, *qprp;
  float S;
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
	float * X;
	float * X2;
	float * Y;
	float * Y2;
	float * Z;
	cuComplex * CZ;
	cuComplex * CX;
	cuComplex * CX2;

	cuComplex * CXZ;


	float * XY;
	float * XY2;
	float * XY3;
	float * XY4;
	float * XY_R;

	float * XZ;
	float * YZ;
	float * XYZ;
} temporary_arrays_struct;	

typedef struct {
  float kx2Phi_zf_rms;
  float kx2Phi_zf_rms_avg;
  float Phi_zf_kx1_avg;
  float *nu;
  float *nu1;
  float *nu22;
  cuComplex *nu1_complex;
  cuComplex *nu22_complex;
  float D;
} nlpm_struct;


typedef struct {
    float *shear_rate_z;
    float *shear_rate_z_nz;
    float *shear_rate_nz;  
} hyper_struct;


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
  

	/* Specifies whether the pointers in the struct point 
	 * to host or device memory */
	int memory_location;

} everything_struct;



void setup_everything_structs(everything_struct * ev_h, everything_struct ** ev_hd_ptr, everything_struct ** ev_d_ptr);
