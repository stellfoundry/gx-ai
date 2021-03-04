#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "species.h"
#include "version.h"

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm mpcom_ftn = MPI_COMM_WORLD;

  printf("Version: %s \t Compiled: %s \n", build_git_sha, build_git_time);

  Species * species_h = nullptr;
  
  species_h = new Species(1);

  species_h -> load(0, 1., 1., 1., 1., 0., 0., 2., 0.1, 0);

  printf("mass = %f \n",species_h->mass[0]);
  

  MPI_Finalize();
  cudaDeviceReset();
}
