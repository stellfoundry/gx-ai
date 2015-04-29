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
  cudaStream_t * zstreams;
  cudaStream_t copystream;
} cuda_streams_struct;

typedef struct {
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



/* A bunch of device arrays to be reused continuously */
typedef struct {
	cuComplex * CXYZ;
	float * X;
	float * X2;
	float * Y;
	float * Y2;
	float * Z;

	float * XY;
	float * XY2;
	float * XY3;
	float * XY4;
	float * XY_R;

	float * XZ;
	float * YZ;
} temporary_arrays_struct;	




typedef struct {

  fields_struct  fields;
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
  

	/* Specifies whether the pointers in the struct point 
	 * to host or device memory */
	int memory_location;

} everything_struct;



void setup_everything_structs(everything_struct * ev_h, everything_struct ** ev_hd_ptr, everything_struct ** ev_d_ptr);
