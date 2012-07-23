EXECUTABLE := ORS-TANG_TEST.out

CUFILES := ors-tang_driver.cu

CU_DEPS := \
      timestep_kernel.cu \
      timestep_test.cu \
      nlps.cu \
      nlps_kernel.cu \
      operations_kernel.cu \
      zderiv.cu \
      zderiv_kernel.cu \
      energy.cu \
      courant.cu \
      maxReduc.cu \
      sumReduc.cu \
      reduc_kernel.cu \
      zderiv_covering.cu \
      ztransform_covering.cu \
      coveringSetup.cu \
      covering_kernel.cu \
      advance.cu \
      getfcn.cu \
      kInit_kernel.cu \

USECUFFT := 1

verbose = 1

valgrind = 0

cudagdb = 0

OMIT_CUTIL_LIB := 1

include ../common.mk

test:
	@echo $(CUBINS)
