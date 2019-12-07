#include <stdlib.h>
#include <stdio.h>
#include <string.h> 
#include <mpi.h>

void namelistRead(char *fn, char *afn);

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  char *run_name;
  char nml_file [255] = " ";
  
  run_name = argv[1];

  strcpy(nml_file, run_name);
  strcat(nml_file, ".in");

  char nc_file[255];
  strcpy(nc_file, run_name);
  strcat(nc_file, ".nc");

  namelistRead(nml_file, nc_file);

}
