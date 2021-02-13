#######################################
#        Makefile for GX
#######################################

TARGET    = gx

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
include Makefiles/Makefile.$(GK_SYSTEM)

############################
## Setup Compiler Flags
###########################

CC = $(NVCC)
LD = $(NVCC)
GEO_LIBS=${GS2}/geometry_c_interface.o 
GS2_CUDA_FLAGS=-I ${GS2} ${GS2}/libgs2.a ${GS2}/libsimpledataio.a 

CFLAGS= ${CUDA_INC} ${MPI_INC} ${GSL_INC} ${CUTENSOR_INC}
LDFLAGS= $(CUDA_LIB) ${MPI_LIB} ${GSL_LIB} ${NETCDF_LIB} ${CUTENSOR_LIB}

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

VPATH=.:src

##########################
## Suffix Build Rules
#############################

.SUFFIXES:
.SUFFIXES: .c .cpp .cu .o .d
.DEFAULT_GOAL := $(TARGET)

HEADERS=$(wildcard include/*.h) 

obj/%.o: %.cu $(HEADERS) 
	$(NVCC) -w -dc -o $@ $< $(CFLAGS) $(NVCCFLAGS) -I. -I include 

obj/%.o: %.cpp $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS) -I. -I include

#######################################
# Rules for building gx
####################################

OBJS = main.o run_gx.o gx_lib.o parameters.o geometry.o grids.o reductions.o moments.o fields.o solver.o linear.o timestepper.o diagnostics.o device_funcs.o grad_parallel.o grad_parallel_linked.o closures.o cuda_constants.o smith_par_closure.o forcing.o laguerre_transform.o nonlinear.o grad_perp.o ncdf.o

# header dependencies
# ncdf.h: ncarr.h
# device_funcs.h: species.h cuda_constants.h
# parameters.h: species.h external_parameters.h toml.hpp
# grids.h: parameters.h device_funcs.h
# reductions.h: grids.h
# grad_perp.h: grids.h 
# fields.h: grids.h 
# moments.h: grids.h 
# forcing.h: moments.h
# grad_parallel.h: moments.h
# geometry.h: grad_parallel.h
# laguerre_transform.h: moments.h
# ncdf.h: geometry.h reductions.h
# solver.h: fields.h moments.h geometry.h 
# closures.h: moments.h geometry.h smith_par_closure.h
# linear.h: fields.h closures.h 
# nonlinear.h: fields.h grad_perp.h geometry.h laguerre_transform.h reductions.h 
# timestepper.h: linear.h nonlinear.h solver.h forcing.h
# diagnostics.h: fields.h geometry.h ncdf.h reductions.h 
# run_gx.h: timestepper.h diagnostics.h 
# gx_lib.h: run_gx.h

# main program
$(TARGET): $(addprefix obj/, $(OBJS)) 
	$(NVCC) -o $@ $(addprefix obj/, $(OBJS)) $(NVCCFLAGS) $(LDFLAGS)

########################
# Cleaning up
########################

clean: 
	rm -rf obj/*.o *~ \#*

distclean: clean clean_tests
	rm -rf $(TARGET)



#########################
# Misc
#######################


test_make:
	@echo TARGET=    $(TARGET)
	@echo SDKDIR=    $(SDKDIR)
	@echo NVCC=      $(NVCC)
	@echo CFLAGS= 	 $(CFLAGS)
	@echo LDFLAGS= 	 $(LDFLAGS)
	@echo NVCCFLAGS= $(NVCCFLAGS)
	@echo CUDA_INC=  $(CUDA_INC)
	@echo CUDA_LIB=  $(CUDA_LIB)
	@echo CUTENSOR_LIB=  $(CUTENSOR_LIB)



