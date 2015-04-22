//This header file defines the everything struct , which 
// contains all parameters and all dynamically allocated
// memory in the simulation (basically everything except
// local variables).

#include "species.h"
#include "grids.h"
#include "fields.h"
#include "outputs.h"
#include "geometry.h"
#include "input_parameters_struct.h"


typedef struct {
	// Dimensionless simulation time
	float runtime;
	// Timestep counter
	int counter;
	// Used to calculate exponential moving averages
	float dtSum;
	// Total runtime in minutes
	float totaltimer;
	// Step runtime in minutes
	float timer;
} time_struct;

typedef struct {
	dim3 dimBlock;
  dim3 dimGrid;
} cuda_dimensions_struct;

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
	FILE * fluxfile;
	FILE * omegafile; 
	FILE * gammafile;
	FILE * phifile;
	char stopfileName[60];
} files_struct;


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
	char * run_name;
} run_info_struct;



typedef struct {

  fields_struct  fields;
	geometry_coefficents_struct  geo;
	outputs_struct  outs;
	input_parameters_struct pars;
	time_struct time;
	grids_struct grids;
	run_info_struct info;
	temporary_arrays_struct tmp;
	files_struct files;
	cuda_dimensions_struct cdims;
	cuffts_struct ffts;

	/* Specifies whether the pointers in the struct point 
	 * to host or device memory */
	int memory_location;

} everything_struct;



