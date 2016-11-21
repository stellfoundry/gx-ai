#include <math.h>
#include "simpledataio_cuda.h"
#include "cufft.h"
#include "sys/stat.h"
#include "cuda_profiler_api.h"
#include "libgen.h"
#include "mpi.h"
#include "nvToolsExt.h"

// EGH: Until we do something more fancy with cudaGetSymbolAddress the following have
// to have file scope... ie they have to be in the same aggregate source file as the 
// kernels.

__constant__ int nx,ny,nz, zthreads, nspecies;
__constant__ float X0_d,Y0_d;
__constant__ int Zp_d;

#define EXTERN_SWITCH extern

#include "everything_struct.h"
#include "global_variables.h"
#include "allocations.h"
#include "write_data.h"
#include "gs2.h"
#include "get_error.h"
#include "profile.h"
#include "device_funcs.cu"
#include "operations_kernel.cu"
#include "diagnostics_kernel.cu"
#include "exb_kernel.cu"
#include "nlps_kernel.cu"
#include "zderiv_kernel.cu"
#include "covering_kernel.cu"
#include "reduc_kernel.cu"
#include "cudaReduc_kernel.cu"
#include "init_kernel.cu"
#include "omega_kernel.cu"
#include "phi_kernel.cu"
#include "qneut_kernel.cu"
#include "nlpm_kernel.cu"
#include "zonal_kernel.cu"
#include "timestep_kernel.cu"
#include "getfcn.cu"
#include "maxReduc.cu"
#include "sumReduc.cu"
#include "diagnostics.cu"
#include "coveringSetup.cu"
#include "exb.cu"
#include "qneut.cu"
#include "ztransform_covering.cu"
#include "zderiv.cu"
#include "zderivB.cu"
#include "zderiv_covering.cu"
#include "nlps.cu"
#include "nlpm.cu"
#include "hyper.cu"
#include "courant.cu"
#include "energy.cu"
#include "timestep_gryfx.cu"
#include "gryfx_run_diagnostics.cu"
#include "run_gryfx_functions.cu"


void set_cuda_constants(){
  cudaMemcpyToSymbol(nx, &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(ny, &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nz, &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(nspecies, &nSpecies, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(X0_d, &X0, sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Y0_d, &Y0, sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Zp_d, &Zp, sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(zthreads, &zThreads, sizeof(int),0,cudaMemcpyHostToDevice);
 // cudaMemcpyToSymbol(zblockthreads, &zBlockThreads, sizeof(int),0,cudaMemcpyHostToDevice);

  if(DEBUG) getError("run_gryfx.cu, copied constants");
}
