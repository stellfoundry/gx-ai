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
  cuComplex omega_out_h_correct[] =  {
//		{-0.086062,	0.094208  } ,
//		{-0.184508,	0.178823  } ,
//		{-0.299080,	0.236866  } ,
//		{-0.433454,	0.270937  } ,
//		{-0.583825,	0.283765  } ,
//		{-0.742790,	0.278278  } ,
//		{-0.902706,	0.256830  } ,
//		{-1.057670,	0.221767  } ,
//		{-1.203555,	0.175838  } ,
//		{-1.337483,	0.122050  } ,
//		{-1.457503,	0.063530  } ,
//		{-1.562514,	0.003359  } ,
//		{-1.652453,	-0.055654 }

//		{-0.154642,	0.101132  } ,
//		{-0.433891,	0.195182  } , 
//		{-0.763951,	0.237872  } , 
//		{-1.079684,	0.206392  } , 
//		{-1.354246,	0.119322  } , 
//		{-1.573511,	0.005262  } 
		{-0.052147,	0.038303 },
		{-0.154592,	0.101112 },
		{-0.278026,	0.159875 },
		{-0.433889,	0.195188 },
		{-0.599848,	0.229016 },
		{-0.763952,	0.237874 },
		{-0.926517,	0.230921 },
		{-1.079683,	0.206394 },
		{-1.223355,	0.167966 },
		{-1.354247,	0.119321 },
		{-1.471318,	0.063769 },
		{-1.573508,	0.005260 },
		{-1.660872,	-0.053048}
                };
  MPI_Init(&argc, &argv);
  mpcom_glob = MPI_Comm_c2f(MPI_COMM_WORLD);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  
  argc_glob = argc;
  argv_glob = argv;
  gryfx_main(argc_glob, argv_glob, mpcom_glob);

  if(proc==0) {
  if (agrees_with_cuComplex_imag(omega_out_h, omega_out_h_correct, 13, 1.0e-1)==0){
    printf("Growth rates don't match!\n");
    exit(1);
   }
   }


        MPI_Finalize();
   
	return 0;

}
