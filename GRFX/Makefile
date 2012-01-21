EXECUTABLE := timestep

CUFILES := timestep_driver.cu

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
      

USECUFFT := 1

include ../../../common/common.mk
