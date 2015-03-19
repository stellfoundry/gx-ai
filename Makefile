#######################################
#        Makefile for GRYFX
#######################################
#
# To build GRYFX, use the build_gryfx
# script which is included with this
# distribution. It automatically loads
# the required modules for a given
# system and checks you have set the
# right environment variables
#
#######################################

TARGET    = gryfx
FILES     = *.cu *.c *.cpp Makefile
VER       = `date +%y%m%d`

#######################################
# Include system-dependent make variables
#######################################

ifndef GK_SYSTEM
	ifdef SYSTEM
$(warning SYSTEM environment variable is obsolete)
$(warning use GK_SYSTEM instead)
	GK_SYSTEM = $(SYSTEM)
	else
$(error GK_SYSTEM is not set)
	endif
endif
include Makefile.$(GK_SYSTEM)

#####################################
# Specify dependencies of gryfx_lib.o
#####################################

include Makefile.depend

#############################
# Include test suite Makefile
#############################

export GK_HEAD_DIR=$(PWD)
TEST_DIR=$(PWD)/tests
include tests/Makefile.tests_and_benchmarks

############################
## Setup Compiler Flags
###########################

CC = $(NVCC)
LD = $(NVCC)
GEO_LIBS=${GS2}/geometry_c_interface.o  # ${GS2}/utils.a #${GS2}/geo.a
GS2_CUDA_FLAGS=-I ${GS2} ${GS2}/libgs2.a ${GS2}/libsimpledataio.a 

CFLAGS=-I$(PWD)/tests/unity $(NVCCFLAGS) $(NVCCINCS)  ${FFT_INC} ${NETCDF_LIB}
LDFLAGS=$(NVCCLIBS) ${GEO_LIBS} ${FFT_LIB} ${NETCDF_LIB} ${FORTRAN_LIBS} ${GS2_CUDA_FLAGS}


ifeq ($(GS2_zonal),on)
  NVCCFLAGS += -D GS2_zonal 
endif

ifeq ($(GS2_all),on)
  NVCCFLAGS += -D GS2_all -I ${GS2}
  GS2_CUDA_FLAGS=${GS2}/libgs2.a
endif

##########################
## Suffix Build Rules
#############################

.SUFFIXES:
.SUFFIXES: .cu .o
.DEFAULT_GOAL := $(TARGET)

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $(NVCCINCS) $< 

#####################################
# Rule for building the system_config
# used by the build script
#####################################

ifdef STANDARD_SYSTEM_CONFIGURATION
system_config: Makefiles/Makefile.$(GK_SYSTEM) Makefile
	@echo "#!/bin/bash " > system_config
	@echo "$(STANDARD_SYSTEM_CONFIGURATION)" >> system_config
	@sed -i 's/^ //' system_config

else
.PHONY: system_config
system_config:
	$(error "STANDARD_SYSTEM_CONFIGURATION is not defined for this system")
endif

#######################################
# Rules for building gryfx
####################################

libgryfx.a: gryfx_lib.o 
	ar cr $@ $<
	ranlib $@

gryfx_lib.o: gryfx_lib.h $(CU_DEPS)

# main program
$(TARGET): gryfx.o libgryfx.a $(GS2)/libgs2.a
	$(NVCC)  -o $@  $^ $(CFLAGS) $(LDFLAGS) 

gryfx.o: gryfx_lib.h

#######################################
# Rules for building gryfx dumb
# used for testing Trinity interface
####################################
    
gryfx_lib_dumb.o: gryfx_lib.h

gryfx_lib_dumb.a: gryfx_lib_dumb.o
	ar cr $@ $<
	ranlib $@

gryfx_dumb: gryfx.o gryfx_lib_dumb.a
	$(NVCC)  -o $@ $(NVCCFLAGS) $(NVCCLIBS) $< gryfx_lib_dumb.a ${GEO_LIBS} ${FORTRAN_LIBS} 


gryfx_hybrid.o: 
	$(NVCC) -c $(NVCCFLAGS) -D GS2_zonal -I ${GS2} $(NVCCINCS) $< gryfx.cu


########################
# Cleaning up
########################


clean: 
	rm -rf *.o *~ \#*

distclean: clean clean_tests
	rm -rf $(TARGET)



#########################
# Misc
#######################


test_make:
	@echo TARGET=    $(TARGET)
	@echo SDKDIR=    $(SDKDIR)
	@echo NVCC=      $(NVCC)
	@echo NVCCFLAGS= $(NVCCFLAGS)
	@echo NVCCINCS=  $(NVCCINCS)
	@echo NVCCLIBS=  $(NVCCLIBS)


tar:
	@echo $(TARGET)-$(VER) > .package
	@-rm -fr `cat .package`
	@mkdir `cat .package`
	@ln $(FILES) `cat .package`
	tar cvf - `cat .package` | bzip2 -9 > `cat .package`.tar.bz2
	@-rm -fr `cat .package` .package

