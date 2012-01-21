#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"


__constant__ int Nx, Ny, Nz, zThreads, X0;
float endtime;
float nu, eta;

#include "nlps_kernel.cu"
#include "zderiv_kernel.cu"
#include "timestep_kernel.cu"
#include "reduc_kernel.cu"
#include "zderiv.cu"
#include "nlps.cu"
#include "maxReduc.cu"
#include "sumReduc.cu"
#include "courant.cu"
#include "energy.cu"
#include "timestep.cu"
#include "timestep_test.cu"




int main(int argc, char* argv[])
{
    
    
    cufftReal *f, *g;
    
    
    int ct, dev;
    struct cudaDeviceProp prop;
    cudaGetDeviceCount(&ct);
    printf("Device Count: %d\n",ct);
    cudaGetDevice(&dev);
    printf("Device ID: %d\n",dev);
    cudaGetDeviceProperties(&prop,dev);
    printf("Device Name: %s\n", prop.name);
    printf("Global Memory (bytes): %lu\n", (unsigned long)prop.totalGlobalMem);
    printf("Shared Memory per Block (bytes): %lu\n", (unsigned long)prop.sharedMemPerBlock);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Warp Size (threads): %d\n", prop.warpSize); 
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Size of Block Dimension (threads): %d * %d * %d\n", prop.maxThreadsDim[0], 
                        prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Size of Grid Dimension (blocks): %d * %d * %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    
    
    
    
    if ( argc != 4 ) // argc should be 2 for correct execution 
    {
        //We print argv[0] assuming it is the program name 
        printf( "usage: %s inputfile outputfile", argv[0] );
    }
    else 
    {
        // We assume argv[1] is a filename to open
        FILE *ifile = fopen( argv[1], "r" );
	FILE *ofile = fopen( argv[2], "w+");
	FILE *pipe = popen( "gnuplot -persist", "w");
	FILE *plotfile = fopen( argv[3], "w+");

        // fopen returns 0, the NULL pointer, on failure 
        if ( ifile == 0 )
        {
            printf( "Could not open file\n" );
        }
        else 
        {
            fscanf(ifile, "%d %d %d %d", &Nx, &Ny, &Nz, &X0);
            fscanf(ifile, "%f %f %f", &endtime, &nu, &eta);
	    fclose( ifile );
        } 
		
	f = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
        g = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);				//
	
	
	cudaMemcpyToSymbol("Nx", &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("Ny", &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("Nz", &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("X0", &X0, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("zThreads", &prop.maxThreadsDim[2], sizeof(int),0,cudaMemcpyHostToDevice);
	
	timestep_test(f, g, ofile);
        
	
	//make a gnuplot script called plotTime for later plotting
	fprintf(plotfile, "set terminal x11 size 600,600\n");
	fprintf(plotfile, "set size .8,.8\n");
	fprintf(plotfile, "set origin .07,.15\n");
	fprintf(plotfile, "set title \"ENERGY PLOTS\"\n");
	fprintf(plotfile, "set xlabel \"Time (s)\"\nset ylabel \"Energy\"\n");
	fprintf(plotfile, "set xtics 0, .5, %f\nset mxtics 5\nset ytics 0, 1\nset mytics 5\nset tics scale 3\n",endtime);
	fprintf(plotfile, "set label \"Nx=%d   Ny=%d   Nz=%d   Boxsize=2pi*%d\\n\\n\\\n", Nx, Ny, Nz, X0);
	fprintf(plotfile, "nu=%g   eta=%g\" at 0,-.9\n",nu,eta);
	fprintf(plotfile, "plot [ ] [0:] \"%s\" using 1:2 title \"total energy\" with lines, \\\n", argv[2]);
	fprintf(plotfile, "\"%s\" using 1:3 title \"kinetic energy\" with lines, \\\n", argv[2]);
	fprintf(plotfile, "\"%s\" using 1:4 title \"magnetic energy\" with lines\n", argv[2]);
	fprintf(plotfile, "pause -1 \"press any key\"");
	
	fclose(plotfile);
	
	//pipe to gnuplot to instantly plot data
	fprintf(pipe, "set terminal x11 size 600,600\n");
	fprintf(pipe, "set size .8,.8\n");
	fprintf(pipe, "set origin .07,.15\n");
	fprintf(pipe, "set title \"ENERGY PLOTS\"\n");
	fprintf(pipe, "set xlabel \"Time (s)\"\nset ylabel \"Energy\"\n");
	fprintf(pipe, "set xtics 0, .5, %f\nset mxtics 5\nset ytics 0, 1\nset mytics 5\nset tics scale 3\n",endtime);
	fprintf(pipe, "set label \"Nx=%d   Ny=%d   Nz=%d   Boxsize=2pi*%d\\n\\n\\\n", Nx, Ny, Nz, X0);
	fprintf(pipe, "nu=%g   eta=%g\" at 0,-.9\n",nu,eta);
	fprintf(pipe, "plot [ ] [0:] \"%s\" using 1:2 title \"total energy\" with lines, \\\n", argv[2]);
	fprintf(pipe, "\"%s\" using 1:3 title \"kinetic energy\" with lines, \\\n", argv[2]);
	fprintf(pipe, "\"%s\" using 1:4 title \"magnetic energy\" with lines\n", argv[2]);
	fprintf(pipe, "pause -1 \"press any key\"");
	
	fclose(pipe);
	
	
	
	printf("\nNx=%d   Ny=%d  Nz=%d  BoxSize=2pi*%d\n", Nx, Ny, Nz, X0);
	
	fclose(ofile);
	 
    }
}  	
