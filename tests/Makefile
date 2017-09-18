# Points to the root of Google Test, relative to where this file is.
# Remember to tweak this if you move this file.
GTEST_DIR = googletest/googletest
GRYFX_DIR = ..

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
include $(GRYFX_DIR)/Makefiles/Makefile.$(GK_SYSTEM)

############################
## Setup Compiler Flags
###########################

CC = $(NVCC)
LD = $(NVCC)

# Flags passed to nvcc compiler.
CFLAGS= ${CUDA_INC} ${MPI_INC} ${PYTHON_INC}
LDFLAGS=$(CUDA_LIB) ${MPI_LIB} ${PYTHON_LIB}

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
GTEST_FLAGS += -isystem $(GTEST_DIR)/include

# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.
UNIT_TESTS = test_parameters test_grids test_geometry test_grad_parallel_periodic test_grad_parallel_linked test_moments test_laguerre_transform test_grad_perp
REGRESSION_TESTS = test_cyclone_linear_beer
TESTS = $(UNIT_TESTS) $(REGRESSION_TESTS)

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

GRYFX_HEADERS = $(GRYFX_DIR)/include/*.h

GRYFX_OBJS := run_gx.o gx_lib.o parameters.o geometry.o grids.o moments.o fields.o solver.o linear.o timestepper.o diagnostics.o device_funcs.o grad_parallel.o grad_parallel_linked.o closures.o cuda_constants.o smith_par_closure.o forcing.o laguerre_transform.o nonlinear.o grad_perp.o
GRYFX_OBJS := $(addprefix $(GRYFX_DIR)/obj/, $(GRYFX_OBJS))

VPATH=.:src

# House-keeping build targets.

all : unit_tests

clean :
	rm -f test_* obj/*.a obj/*.o

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
obj/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
	            $(GTEST_DIR)/src/gtest-all.cc -o $@

obj/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(GTEST_FLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
	            $(GTEST_DIR)/src/gtest_main.cc -o $@

obj/gtest.a : obj/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

obj/gtest_main.a : obj/gtest-all.o obj/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

$(GRYFX_DIR)/obj/%.o: $(GRYFX_DIR)/src/%.cu
	cd $(GRYFX_DIR); \
	make obj/$*.o; \
	cd tests;

# Build tests
obj/test_%.o: test_%.cu 
	$(NVCC) -dc -o $@ $< $(CFLAGS) $(NVCCFLAGS) -I $(GRYFX_DIR)/include $(GTEST_FLAGS)

obj/device_test.o: device_test.cu
	$(NVCC) -dc -o $@ $< $(CFLAGS) $(NVCCFLAGS) $(GTEST_FLAGS)

test_%: obj/test_%.o obj/gtest_main.a $(GRYFX_OBJS) obj/device_test.o
	$(NVCC) -o $@ $^ $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) 

unit_tests: $(addprefix obj/, $(UNIT_TESTS:=.o)) obj/gtest_main.a $(GRYFX_OBJS) obj/device_test.o
	$(NVCC) -o $@ $^ $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) 

regression_tests: $(addprefix obj/, $(REGRESSION_TESTS:=.o)) obj/gtest_main.a $(GRYFX_OBJS) obj/device_test.o
	$(NVCC) -o $@ $^ $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) 

test_each: $(TESTS)

run_tests: unit_tests
	./unit_tests;
