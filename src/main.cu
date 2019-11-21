#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "gx_lib.h"

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm mpcom_ftn = MPI_COMM_WORLD;
        
  gx_main(argc, argv, mpcom_ftn);

  MPI_Finalize();
}
