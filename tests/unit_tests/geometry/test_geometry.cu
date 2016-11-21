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
#include "gryfx_lib.h"
#include "unit_tests.h"

 

int main(int argc, char* argv[])
{

  //float shat_correct=0.634;
  //float eps_correct=0.18;
  //bool smagorinsky_correct = false;



	struct gryfx_parameters_struct gryfxpars;

  int proc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);

	gryfx_get_default_parameters_(&gryfxpars, argv[1], MPI_COMM_WORLD);

  everything_struct * ev = (everything_struct *) gryfxpars.everything_struct_address;

  if (proc==0) set_geometry(&ev->pars, &ev->grids, &ev->geo, &gryfxpars);


  MPI_Finalize();

   
	return 0;

}
