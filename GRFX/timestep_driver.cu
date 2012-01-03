#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"
#include "constants.h"
#include "nlps_kernel.cu"
#include "zderiv_kernel.cu"
#include "timestep_kernel.cu"
#include "zderiv.cu"
#include "nlps.cu"
#include "energy.cu"
#include "timestep.cu"
#include "timestep_test.cu"




int main(int argc, char* argv[])
{
    int fkx, fky, fkz, gkx, gky, gkz, fsin, fcos, gsin, gcos;
    cufftReal *f, *g;
    float *x, *y, *z;
    
    
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
    
    
    
    
    
    if ( argc != 3 ) // argc should be 2 for correct execution 
    {
        //We print argv[0] assuming it is the program name 
        printf( "usage: %s inputfile outputfile", argv[0] );
    }
    else 
    {
        // We assume argv[1] is a filename to open
        FILE *ifile = fopen( argv[1], "r" );
	FILE *ofile = fopen( argv[2], "w+");

        // fopen returns 0, the NULL pointer, on failure 
        if ( ifile == 0 )
        {
            printf( "Could not open file\n" );
        }
        else 
        {
            fscanf(ifile, "%d %d %d %d %d %d %d %d %d %d %d %d %d", &fkx, &fky, &fkz, &fsin, &fcos, &gkx, &gky, &gkz, &gsin, &gcos, &Nx, &Ny, &Nz);
            fclose( ifile );
        } 
	
	y = (float*) malloc(sizeof(float)*Ny);					//
        x = (float*) malloc(sizeof(float)*Nx);
	z = (float*) malloc(sizeof(float)*Nz);	
	f = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
        g = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);				//
	
	
	cudaMemcpyToSymbol("Nx", &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("Ny", &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("Nz", &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("zThreads", &prop.maxThreadsDim[2], sizeof(int),0,cudaMemcpyHostToDevice);
	
	timestep_test(f, g, fkx, fky, fkz, fsin, fcos, gkx, gky, gkz, gsin, gcos, ofile);
        
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
	  for(int i=0; i<Ny; i++) {
	    y[i] = 2*M_PI*(float)(i-Ny/2)/Ny;					//
            x[j] = 2*M_PI*(float)(j-Nx/2)/Nx;					//
            z[k] = 2*M_PI*(float)(k-Nz/2)/Nz;
	    int index = i + Ny*j + Nx*Ny*k;
	    
	    
	   
	    
	  }
	 }     
	}
	
	
	/*
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
	  for(int i=0; i<Ny; i++) {
	    int index = i + Ny*j + Nx*Ny*k;
	    fprintf(ofile,"zp(%.2fPI,%.2fPI,%.2fPI)=%.3f: %d  ",y[i]/M_PI,
	               x[j]/M_PI, z[k]/M_PI, f[index], index);
	  }
	  fprintf(ofile, "\n");	       
	 }
	}
	
	fprintf(ofile, "\n");
	
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
	  for(int i=0; i<Ny; i++) {
	    int index = i + Ny*j + Nx*Ny*k;
	    fprintf(ofile,"zm(%.2fPI,%.2fPI,%.2fPI)=%.3f: %d  ",y[i]/M_PI,
	               x[j]/M_PI, z[k]/M_PI, g[index], index);
	  }
	  fprintf(ofile, "\n");	       
	 }
	} */
	
	printf("\nfkx=%d  fky=%d  fkz=%d  fcos=%d  fsin=%d\n", fkx, fky,fkz,fcos,fsin);
	printf("gkx=%d  gky=%d  gkz=%d  gcos=%d  gsin=%d\nf=zp, g=zm\n", gkx,
	                                gky,gkz,gcos,gsin);
	
	fclose(ofile);
	 
    }
}  	
