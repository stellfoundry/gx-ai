
#include "cufft.h"
#include "species.h"
#define EXTERN_SWITCH 
#include "simpledataio_cuda.h"
#include "grids.h"
#include "geometry.h"
#include "fields.h"
#include "outputs.h"
#include "input_parameters_struct.h"

bool globals_initialized = false;
void initialize_globals(){

  if (!globals_initialized) {
    irho = 2;
    maxdt = 0.02; 
  globals_initialized = true;
}
}
