#include <math.h>
#include <float.h>
#include "cufft.h"
#include "mpi.h"
#define EXTERN_SWITCH extern
#include "cufft.h"
#include "simpledataio_cuda.h"
#include "everything_struct.h"
#include "allocations.h"
#include "read_namelist.h"
#include "write_data.h"
#include "unit_tests.h"

 

int main(int argc, char* argv[])
{

  //float shat_correct=0.634;
  //float eps_correct=0.18;
  //bool smagorinsky_correct = false;



  everything_struct ev;

  ev.memory_location = ON_HOST;

  read_namelist(&ev.pars, &ev.grids, "test_write_data.in");

  set_grid_masks_and_unaliased_sizes(&ev.grids);  

  allocate_or_deallocate_everything(ALLOCATE, &ev);

  allocate_geo(ALLOCATE, ON_HOST, &ev.geo, &ev.grids.z, &ev.grids.Nz);

  writedat_set_run_name(&ev.info.run_name, "test_write_data.in");

  writedat_beginning(&ev);
  writedat_each(&ev.outs, &ev.fields, &ev.time);
  writedat_end(ev.outs);

  allocate_or_deallocate_everything(DEALLOCATE, &ev);


   
	return 0;

}
