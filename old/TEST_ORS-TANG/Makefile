##################################################
#           Makefile of gryfx for CUDA           #
#                                                #
#  NOTE: environmental variables                 #
#     CUDAARCH, SDKDIR                           #
#  need to be properly defined                   #
##################################################
TARGET    = ors-tang
CU_DEPS := \
      ../device_funcs.cu \
      ../operations_kernel.cu \
      ../diagnostics_kernel.cu \
      ../exb_kernel.cu \
      ../nlps_kernel.cu \
      ../zderiv_kernel.cu \
      ../covering_kernel.cu \
      ../reduc_kernel.cu \
      ../cudaReduc_kernel.cu \
      ../init_kernel.cu \
      ../omega_kernel.cu \
      ../phi_kernel.cu \
      ../getfcn.cu \
      ../c_fortran_namelist3.c \
      ../read_namelist.cu \
      ../read_input.cu \
      ../definitions.cu \
      ../diagnostics.cu \
      ../coveringSetup.cu \
      ../exb.cu \
      ../nlps.cu \
      ../sumReduc.cu \
      energy.cu \
      timestep_ors-tang.cu \
      run_ors-tang.cu \
      timestep_kernel.cu \
      #../ztransform_covering.cu \
      ../zderiv.cu \
      ../zderivB.cu \
      ../zderiv_covering.cu \

#courant.cu and maxReduc.cu

FILES     = *.cu *.c *.cpp Makefile
VER       = `date +%y%m%d`

NVCC      = nvcc
NVCCFLAGS = -arch=compute_$(CUDAARCH) -use_fast_math -D ORSTANG
NVCCINCS  = 
NVCCLIBS  = -lcufft -lcudart

ifeq ($(debug),on)
  NVCCFLAGS += -g -G2
else
  NVCCFLAGS += -O3
endif

.SUFFIXES:
.SUFFIXES: .cu .o

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $(NVCCINCS) $<

# main program
$(TARGET): ors-tang.o
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(NVCCLIBS) 

gryfx.o: $(CU_DEPS)

test_make:
	@echo TARGET=    $(TARGET)
	@echo SDKDIR=    $(SDKDIR)
	@echo NVCC=      $(NVCC)
	@echo NVCCFLAGS= $(NVCCFLAGS)
	@echo NVCCINCS=  $(NVCCINCS)
	@echo NVCCLIBS=  $(NVCCLIBS)

clean:
	rm -rf *.o *~ \#*

distclean: clean
	rm -rf $(TARGET)

tar:
	@echo $(TARGET)-$(VER) > .package
	@-rm -fr `cat .package`
	@mkdir `cat .package`
	@ln $(FILES) `cat .package`
	tar cvf - `cat .package` | bzip2 -9 > `cat .package`.tar.bz2
	@-rm -fr `cat .package` .package
