#include <stdlib.h>
#include <stdio.h>
//#include "global_vars.h"
#include "gryfx_lib.h"


int main(int argc, char* argv[])
{
        int mpcom;
#ifdef GS2_zonal
        MPI_Init(&argc, &argv);
        mpcom = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif
        
        gryfx_main(argc, argv, mpcom);


#ifdef GS2_zonal
        MPI_Finalize();
#endif
}
