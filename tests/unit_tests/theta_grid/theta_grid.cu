#include <math.h>
#include <float.h>
#include "cufft.h"
#include "mpi.h"

extern int Nz;
extern float *z_h, *z_regular_h;
extern float *Rplot_h, *Zplot_h, *aplot_h, *gbdrift_h, *bmag_h, *gradpar_arr_h;
extern float *Xplot_h, *Yplot_h, *deltaFL_h;
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
  MPI_Init(&argc, &argv);
  mpcom_glob = MPI_Comm_c2f(MPI_COMM_WORLD);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  
  argc_glob = argc;
  argv_glob = argv;
  gryfx_main(argc_glob, argv_glob, mpcom_glob);
  if(proc==0) {
  for(int k=0; k<Nz; k++) {
      if(k>0) {
        deltaFL_h[k] = sqrt(pow(Xplot_h[k-1]-Xplot_h[k],2.) + pow(Yplot_h[k-1]-Yplot_h[k],2.) + pow(Zplot_h[k-1]-Zplot_h[k],2.));
        //printf("%d: deltaFL = %f\tRplot = %f\tZplot = %f\taplot = %f\ttheta = %f\tgbdrift=%f\n", k, deltaFL_h[k],Rplot_h[k],Zplot_h[k],aplot_h[k], z_h[k], gbdrift_h[k]);
        printf("%d: deltaFL = %f\ttheta = %f\n", k, deltaFL_h[k], z_h[k]);
      }
  }
  for(int k=0; k<Nz; k++) {
        //printf("%d: gp=%f th=%f thr=%f\n", k, gradpar_arr_h[k], z_h[k], z_regular_h[k]);
        printf("%d: gp=%f th=%f thr=%f\n", k, gradpar_arr_h[k], z_h[k], 0.0);
  }
  for(int k=0; k<Nz; k++) {
        printf("%d: gb=%f bm=%f th=%f\n", k, gbdrift_h[k]*4.0, bmag_h[k], z_h[k]);
  }
  }



        MPI_Finalize();
   
	return 0;

}
