EXECUTABLE := ORS-TANG_TEST.out

CUFILES := ors-tang_driver.cu

CU_DEPS := \
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

verbose = 1

valgrind = 0

cudagdb = 0

OMIT_CUTIL_LIB := 1

include ../common.mk

test:
	@echo $(CUBINS)
