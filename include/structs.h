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
  int mpcom;
  int iproc;
} mpi_info_struct;


