#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "gryfx_lib.h"

// device constants
__constant__ int nx,ny,nyc,nz, nspecies, nhermite, nlaguerre, zp;
__constant__ float dx, dy;

int main(int argc, char* argv[])
{
        int mpcom_ftn = 0;
        MPI_Init(&argc, &argv);
	// for legacy reasons, need to use Fortran handle for communicator
        mpcom_ftn = MPI_Comm_c2f(MPI_COMM_WORLD);
        
        gryfx_main(argc, argv, mpcom_ftn);


        MPI_Finalize();
}
