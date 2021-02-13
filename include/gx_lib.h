#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <assert.h>
#include "mpi.h"
#include "external_parameters.h"
#include "grids.h"
#include "parameters.h"
#include "diagnostics.h"
#include "run_gx.h"
#include "geometry.h"
#include "get_error.h"

#pragma once

//char *runname;

void gx_get_default_parameters_(struct external_parameters_struct *, char *run_name, MPI_Comm mpcom);
extern "C"
void gx_get_fluxes_(struct external_parameters_struct *, struct gx_outputs_struct*,  MPI_Comm mpcom);

void gx_main(int argc, char **argv, MPI_Comm mpcom);
