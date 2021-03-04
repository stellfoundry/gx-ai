#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <assert.h>
#include "mpi.h"
#include "grids.h"
#include "parameters.h"
#include "diagnostics.h"
#include "run_gx.h"
#include "geometry.h"
#include "get_error.h"
#pragma once

extern "C"
void gx_get_fluxes_(MPI_Comm mpcom, char* run_name);

void gx_main(int argc, char **argv, MPI_Comm mpcom);
