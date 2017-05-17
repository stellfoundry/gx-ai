#include <math.h>
#include <float.h>
#include "cufft.h"
#include "mpi.h"
#include "unit_tests.h"

extern cuComplex * omega_out_h;
extern void gryfx_main(int argc, char* argv[], int mpcom);

int mpcom_glob;
int argc_glob;
char** argv_glob;

 

int main(int argc, char* argv[])
{

  int proc;
  cuComplex omega_out_h_correct[] =  {
{ -0.034995, 0.089934},
{ -0.085987, 0.174519},
{ -0.151499, 0.243774},
{ -0.236446, 0.300161},
{ -0.340213, 0.344247},
{ -0.460121, 0.376258},
{ -0.592453, 0.396383},
{ -0.733279, 0.404926},
{ -0.878807, 0.402574},
{ -1.025300, 0.390381},
{ -1.169105, 0.369408},
{ -1.306994, 0.340533},
{ -1.436442, 0.304612},
{ -1.555587, 0.262704},
{ -1.663118, 0.216176},
                };
  MPI_Init(&argc, &argv);
  mpcom_glob = MPI_Comm_c2f(MPI_COMM_WORLD);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  
  argc_glob = argc;
  argv_glob = argv;
  gryfx_main(argc_glob, argv_glob, mpcom_glob);

  if(proc==0) {
  if (agrees_with_cuComplex_imag(omega_out_h, omega_out_h_correct, 14, 1.0e-2)==0){
    printf("Growth rates don't match!\n");
    exit(1);
   }
   }


        MPI_Finalize();
   
	return 0;

}
