
#include "cufft.h"
#include "species.h"
#define EXTERN_SWITCH 
#include "simpledataio_cuda.h"
#include "grids.h"
#include "geometry.h"
#include "fields.h"
#include "outputs.h"
#include "input_parameters_struct.h"
#include "global_variables.h"

bool globals_initialized = false;
void initialize_globals(){

  if (!globals_initialized) {
    irho = 2;
    maxdt = 0.02; 
    converge_stop = 10000;
    converge_bounds = 20;
    cfl_flag = true;
  globals_initialized = true;
}
}
