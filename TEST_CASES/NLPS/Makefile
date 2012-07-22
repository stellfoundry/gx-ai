EXECUTABLE := NLPSTEST.out

CUFILES := nlps_driver.cu

CU_DEPS := \
      getfcn.cu \
      nlps_kernel.cu \
      timestep_kernel.cu \
      zderiv_kernel.cu \
      nlps.cu \
      nlpstest.cu \

USECUFFT := 1

verbose = 1

valgrind = 0

cudagdb = 0

OMIT_CUTIL_LIB := 1

include ../common.mk

test:
	@echo $(CUBINS)
