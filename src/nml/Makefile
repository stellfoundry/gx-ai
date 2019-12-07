#######################################
#        Makefile for GX
#######################################

TARGET    = nml
FORT=gfortran
NVCC=nvcc
CUDAARCH=70

CUDA_LIB = 
NVCC_FLAGS = -arch=compute_${CUDAARCH} -code=sm_${CUDAARCH} 
NETCDF_INC = -I/usr/include 
NETCDF_LIB = -L/usr/lib/x86_64-linux-gnu -lnetcdff -lnetcdf
MPI_INC= -I/usr/lib/x86_64-linux-gnu/openmpi/include/
MPI_LIB= -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi

FC = $(FORT)
CC = $(NVCC)
LD = $(NVCC)

CFLAGS= -std=c++03 --disable-warnings ${MPI_INC}
LDFLAGS=$(CUDA_LIB) ${NETCDF_LIB} ${MPI_LIB} -lgfortran

.SUFFIXES:
.SUFFIXES: .c .cpp .cu .o
.DEFAULT_GOAL := $(TARGET)

HEADERS=$(wildcard include/*.h) 

%.o: %.cpp $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS) -I. -I include

%.o: %.f90
	$(FC) $(NETCDF_INC) -w -c -o $@ $< 

OBJS = nmlread_f.o nmlread_c.o 

# main program
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $(OBJS) $(NVCCFLAGS) $(LDFLAGS)




