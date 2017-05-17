#include <math.h>
#include <float.h>
#include "cufft.h"
#include "mpi.h"

extern cuComplex * omega_out_h;
extern void gryfx_main(int argc, char* argv[], int mpcom);

int mpcom_glob;
int argc_glob;
char** argv_glob;

int agrees_with_float(float * val, float * correct, const int size, const float eps){
  int result;

  result = 1;

  for (int i=0;i<size;i++) {
    if(
        (
          (fabsf(correct[i]) < FLT_MIN) && !(fabsf(val[i]) < eps) 
        ) || (
          (fabsf(correct[i]) > FLT_MIN) && 
          !( fabsf((val[i]-correct[i])/correct[i]) < eps) 
        ) 
      ) {
      result = 0;
      printf("Error: %e should be %e\n", val[i], correct[i]);
    }
    else
      printf("Value %e agrees with correct value %e\n", val[i], correct[i]);
  }
  return result;
}

int agrees_with_cuComplex_imag(cuComplex * val, cuComplex * correct, const int size, const float eps){
  int result;

  result = 1;

  for (int i=0;i<size;i++)
    result = agrees_with_float(&val[i].y, &correct[i].y, 1, eps) && result;
    printf("result = %d\n", result);
  return result;
}
 

int main(int argc, char* argv[])
{

  int proc;
  cuComplex omega_out_h_correct[] = 
    {{0.0,0.0}, {0.000006, 2.488744}, {0.000004, 2.488744}, {-0.000006, 2.488744}, {-0.000001, 2.488744}};
  MPI_Init(&argc, &argv);
  mpcom_glob = MPI_Comm_c2f(MPI_COMM_WORLD);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  
  argc_glob = argc;
  argv_glob = argv;
  gryfx_main(argc_glob, argv_glob, mpcom_glob);

  if(proc==0) {
  if (agrees_with_cuComplex_imag(omega_out_h, omega_out_h_correct, 5, 1.0e-2)==0){
    printf("Growth rates don't match!\n");
    exit(1);
   }
   }


        MPI_Finalize();
   
	return 0;

}
