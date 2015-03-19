#include <math.h>
#include <float.h>
#include "cufft.h"
#include "mpi.h"

extern cuComplex * omega_h;
extern void gryfx_main(int argc, char* argv[], int mpcom);

int mpcom_glob;
int argc_glob;
char** argv_glob;

int agrees_with_float(float * val, float * correct, const int size, const float eps){
  int result, i;

  result = 1;

  for (i=0;i<size;i++)
    if( !( fabsf((val[i]-correct[i])/correct[i]) < eps) ) {
      result = 0;
      printf("Error: %e should be %e\n", val[i], correct[i]);
    }

  return result;
}

int agrees_with_cuComplex_imag(cuComplex * val, cuComplex * correct, const int size, const float eps){
  int result, i;

  result = 1;

  for (i=0;i<size;i++)
    result = result && agrees_with_float(&val[i].y, &correct[i].y, 1, eps);
  return result;
}
 

int main(int argc, char* argv[])
{

  cuComplex omega_h_correct[] = 
    {{0.0,0.0} {0.000006, 2.488627} {0.000004, 2.489097} {-0.000006, 2.488627} {-0.000001, 2.489098};
#ifdef GS2_zonal
  MPI_Init(&argc, &argv);
  mpcom_glob = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif
  
  argc_glob = argc;
  argv_glob = argv;
  gryfx_main(argc_glob, argv_glob, mpcom_glob);

  if (!agrees_with_cuComplex_imag(omega_h, omega_h_correct, 1, 1.0e-4)){
    printf("Growth rates don't match!\n");
    exit(1);
   }




#ifdef GS2_zonal
        MPI_Finalize();
#endif
   
	return 0;

}
