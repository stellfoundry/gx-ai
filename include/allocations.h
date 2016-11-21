// The functions in the allocations file handle
// the allocation of the everything struct, which 
// contains all parameters and all dynamically allocated
// memory in the simulation (basically everything except
// local variables).

#define  TYPE_INT 0
#define  TYPE_FLOAT 1
#define  TYPE_DOUBLE 2
#define  TYPE_CUCOMPLEX 3
#define  TYPE_CUCOMPLEXDOUBLE 4
#define  TYPE_CUCOMPLEX_PTR 5

#define ON_HOST 0
#define ON_DEVICE 1
#define ON_HOST_AND_DEVICE 2

#define ALLOCATE 0
#define DEALLOCATE 1

// Possibly defunct?
input_parameters_struct * inps_str(everything_struct * ev_d);


//Allocate/deallocate geometry section of the everything struct (allocation has to be
// done separately from the rest, deallocation is done in allocate_or_deallocate_everything)
// aod is either ALLOCATE or DEALLOCATE, ml is either ON_HOST or ON_DEVICE, z_array is the 
// pointer to the theta section of the grids struct);
void allocate_geo(int aod, int ml, geometry_coefficents_struct * geo, float ** z_array, int *Nz);

//This is called separately from gryfx_lib.cu
void allocate_info(int aod, int ml, info_struct * info, int run_name_size, int restart_name_size);

//Allocate or deallocate the everything struct. allocate_or_deallocate is either
// ALLOCATE or DEALLOCATE; 
void allocate_or_deallocate_everything(int allocate_or_deallocate, everything_struct * ev);
