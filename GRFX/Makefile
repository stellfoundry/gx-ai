EXECUTABLE := timestep

CUFILES := timestep_driver.cu

CU_DEPS := \
      c_fortran_namelist.cu \
      timestep_kernel.cu \
      timestep.cu \
      timestep_test.cu \
      nlps.cu \
      nlps_kernel.cu \
      zderiv.cu \
      zderiv_kernel.cu \
      energy.cu \
      courant.cu \
      maxReduc.cu \
      sumReduc.cu \
      reduc_kernel.cu \
      z_covering.cu \
      ztransform.cu \
      coveringRoutines.cu \
      advance.cu \
      getfcn.cu \


USECUFFT := 1

OMIT_CUTIL_LIB := 1

verbose = 1

fastmath = 1

ptxas = 1 

include ../../../common/common.mk
