#include <math.h>
#include <float.h>
#include "cufft.h"
#include "mpi.h"
#define EXTERN_SWITCH extern
#include "species.h"
#include "input_parameters_struct.h"
#include "grids.h"
#include "read_namelist.h"
#include "unit_tests.h"

 

int main(int argc, char* argv[])
{

  float shat_correct=0.634;
  float eps_correct=0.18;
  bool smagorinsky_correct = false;

  input_parameters_struct inps;
  grids_struct grids;


  read_namelist(&inps, &grids, "test_read_namelist.in");

  if (!agrees_with_float(&inps.shat, &shat_correct, 1, 1.0e-7))
  {
    printf("shat doesn't agree\n");
    abort();
  } 
  // eps is not specified in the input file
  // and so this tests the default
  if (!agrees_with_float(&inps.eps, &eps_correct, 1, 1.0e-7))
  {
    printf("eps doesn't agree\n");
    abort();
  } 
  if (!agrees_with_bool(&inps.smagorinsky, &smagorinsky_correct, 1))
  {
    printf("smagorinsky doesn't agree\n");
    abort();
  } 

   
	return 0;

}
