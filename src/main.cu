#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "gx_lib.h"
#include "version.h"

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm mpcom_ftn = MPI_COMM_WORLD;

  printf("Version: %s \t Compiled: %s \n", build_git_sha, build_git_time);

  gx_main(argc, argv, mpcom_ftn);

  MPI_Finalize();
  cudaDeviceReset();
}
