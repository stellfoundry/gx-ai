#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "gx_lib.h"

int main(int argc, char* argv[])
{
  int mpcom_ftn = 0;
  MPI_Init(&argc, &argv);
  // for legacy reasons, need to use Fortran handle for communicator
  mpcom_ftn = MPI_Comm_c2f(MPI_COMM_WORLD);
        
  gx_main(argc, argv, mpcom_ftn);


  MPI_Finalize();
}
