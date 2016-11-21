#define NO_GLOBALS true
#include "standard_headers.h"
#include "allocations.h"

void setup_everything_structs(everything_struct * ev_h, everything_struct ** ev_hd_ptr, everything_struct ** ev_d_ptr){
	/* ev_hd is on the host but the pointers point to memory on the device*/
	/* ev_d is on the device (and the pointers point to memory on the device)*/
  /* ev_d is only allocated for proc0 */

  allocate_or_deallocate_everything(ALLOCATE, ev_h);
  *ev_hd_ptr = (everything_struct*)malloc(sizeof(everything_struct));
  if (ev_h->mpi.iproc==0){
    cudaMalloc((void**) ev_d_ptr, sizeof(everything_struct));

    /* Copy all the immediate values such as input parameters*/ 
    **ev_hd_ptr = *ev_h;

    /* Set the memory_location of ev_hd to let the allocate_or_deallocate_everything 
       function know to allocate its arrays on the device*/
    (*ev_hd_ptr)->memory_location = ON_DEVICE;

    allocate_or_deallocate_everything(ALLOCATE, *ev_hd_ptr);
    /* This has to be done separately */
    allocate_geo(ALLOCATE, ON_DEVICE, &(*ev_hd_ptr)->geo, &(*ev_hd_ptr)->grids.z, &(*ev_hd_ptr)->grids.Nz);

    cudaMemcpy(*ev_d_ptr, *ev_hd_ptr, sizeof(everything_struct), cudaMemcpyHostToDevice);
    /*(No need to allocate ev_d as its pointers point to the same arrays as ev_hd)*/
  }
}
