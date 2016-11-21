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


  int proc;

  char run_name[] = "test_write_data.in";

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);

  if (proc==0){
  everything_struct ev;

  ev.memory_location = ON_HOST;

  read_namelist(&ev.pars, &ev.grids, run_name);

  set_grid_masks_and_unaliased_sizes(&ev.grids);  

  allocate_or_deallocate_everything(ALLOCATE, &ev);

  allocate_geo(ALLOCATE, ON_HOST, &ev.geo, &ev.grids.z, &ev.grids.Nz);

  setup_info(run_name, &ev.pars, &ev.info);

  //ev.info.run_name  = run_name;
  //ev.info.run_name  = run_name;

  //writedat_set_run_name(&ev.info.run_name, "test_write_data.in");

  writedat_beginning(&ev);
  writedat_each(&ev.grids, &ev.outs, &ev.fields, &ev.time);
  writedat_end(ev.outs);

  //ev.info.restart_file_name = (char*)malloc(sizeof(char)*1);
  allocate_or_deallocate_everything(DEALLOCATE, &ev);

  }

  MPI_Finalize();
   
	return 0;

}
