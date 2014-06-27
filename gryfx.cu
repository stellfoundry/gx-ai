#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "sys/stat.h"
#include "cufft.h"
#include "cuda_profiler_api.h"
//#include "global_vars.h"
#include "gryfx_lib.h"
#include "mpi.h"


int main(int argc, char* argv[])
{
        int mpcom;
#ifdef GS2_zonal
        MPI_Init(&argc, &argv);
        mpcom = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif
        
	struct gryfx_parameters_struct gryfxpars;
	struct gryfx_outputs_struct gryfxouts;
//	printf("argc = %d\nargv[0] = %s   argv[1] = %s\n", argc, argv[0],argv[1]); 
  	char* namelistFile;
	if(argc == 2) {
	  namelistFile = argv[1];
//	  printf("namelist = %s\n", namelistFile);
	}
	else {
	  namelistFile = "inputs/cyclone_miller_ke.in";
	}
	gryfx_get_default_parameters_(&gryfxpars, namelistFile, mpcom);
	gryfx_get_fluxes_(&gryfxpars, &gryfxouts, namelistFile);

#ifdef GS2_zonal
        MPI_Finalize();
#endif
}
