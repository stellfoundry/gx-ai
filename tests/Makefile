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
CFLAGS= -g -G ${CUDA_INC} ${MPI_INC} 
LDFLAGS=$(CUDA_LIB) ${MPI_LIB} 

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
GTEST_FLAGS += -isystem $(GTEST_DIR)/include

# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.
TESTS = test_parameters

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

GRYFX_HEADERS = $(GRYFX_DIR)/include/*.h

VPATH=.:src

# House-keeping build targets.

all : test_parameters

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

$(GRYFX_DIR)/obj/%.o: 
	cd $(GRYFX_DIR); \
	make obj/$*.o; \
	cd tests;

# Build tests
obj/test_%.o: test_%.cu
	$(NVCC) -dc -o $@ $< $(CFLAGS) $(NVCCFLAGS) -I $(GRYFX_DIR)/include $(GTEST_FLAGS)

test_%: obj/test_%.o obj/gtest_main.a $(GRYFX_DIR)/obj/%.o
	$(NVCC) $^ -o $@ $(CFLAGS) $(NVCCFLAGS) $(LDFLAGS) 

