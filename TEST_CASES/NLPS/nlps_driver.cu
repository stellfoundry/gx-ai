#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cufft.h>

__constant__ int Nx,Ny,Nz,zThreads, X0, Y0, Z0;
dim3 dimGrid;
dim3 dimBlock;
int totalThreads;
bool MASK=false;
bool debug=false;
bool quiet=false;

//make sure GRFX, GPU_NLPS, and TEST_CASES are in same directory
#include "getfcn.cu"
#include "nlps_kernel.cu"
#include "timestep_kernel.cu"
#include "zderiv_kernel.cu"
#include "nlps.cu"
#include "nlpstest.cu"


int main(int argc, char* argv[])
{
    int fkx, fky, fkz, gkx, gky, gkz, fsin, fcos, gsin, gcos;
    cufftReal *nlps;
    cufftReal *nlpscheck, *fdxcheck, *fdycheck, *gdxcheck, *gdycheck;
    float *x, *y, *z;
    
    if (argc == 4 && strcmp(argv[3],"-quiet")==0) {quiet = true; }
    
    
    int ct, dev;
    struct cudaDeviceProp prop;
    cudaGetDeviceCount(&ct);
    if(!quiet) printf("Device Count: %d\n",ct);
    cudaGetDevice(&dev);
    if(!quiet) printf("Device ID: %d\n",dev);
    cudaGetDeviceProperties(&prop,dev);
    if(!quiet) {
      printf("Device Name: %s\n", prop.name);
      printf("Global Memory (bytes): %lu\n", (unsigned long)prop.totalGlobalMem);
      printf("Shared Memory per Block (bytes): %lu\n", (unsigned long)prop.sharedMemPerBlock);
      printf("Registers per Block: %d\n", prop.regsPerBlock);
      printf("Warp Size (threads): %d\n", prop.warpSize); 
      printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
      printf("Max Size of Block Dimension (threads): %d * %d * %d\n", prop.maxThreadsDim[0], 
                          prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
      printf("Max Size of Grid Dimension (blocks): %d * %d * %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    
    
    
    
    if ( argc != 3 && argc !=4 )  
    {
        //We print argv[0] assuming it is the program name 
        printf( "usage: %s inputfile outputfile", argv[0] );
    }
    else 
    {
        // We assume argv[1] is a filename to open
        FILE *ifile = fopen( argv[1], "r" );
	
	if (argc == 4 && strcmp(argv[3],"-quiet")==0) {quiet = true; }
	else if (argc == 4 && strcmp(argv[3],"-debug")==0) {debug = true;}
	else if (argc == 4) printf("invalid option\n");

        // fopen returns 0, the NULL pointer, on failure 
        if ( ifile == 0 )
        {
            printf( "Could not open file\n" );
        }
        else 
        {
            fscanf(ifile, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", &fkx, &fky, &fkz, &fsin, &fcos, &gkx, &gky, &gkz, &gsin, &gcos, &Nx, &Ny, &Nz, &X0,&Y0,&Z0);
            fclose( ifile );
        } 
	nlpscheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);        
        nlps = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
	y = (float*) malloc(sizeof(float)*Ny);					//
        x = (float*) malloc(sizeof(float)*Nx);
	z = (float*) malloc(sizeof(float)*Nz);					//
	fdxcheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
	fdycheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
	gdxcheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
	gdycheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
	
	
	
	cudaMemcpyToSymbol("Nx", &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("Ny", &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("Nz", &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("zThreads", &prop.maxThreadsDim[2], sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("X0", &X0, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("Y0", &Y0, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol("Z0", &Z0, sizeof(int),0,cudaMemcpyHostToDevice);
	
	
	*&zThreads = prop.maxThreadsDim[2];
        totalThreads = prop.maxThreadsPerBlock;
        int xy = totalThreads/Nz;
        int blockxy = sqrt(xy);
        //dimBlock = threadsPerBlock, dimGrid = numBlocks
        dimBlock.x = blockxy;
        dimBlock.y = blockxy;
        dimBlock.z = Nz;
        if(Nz>zThreads) {
          dimBlock.x = sqrt(totalThreads/zThreads);
          dimBlock.y = sqrt(totalThreads/zThreads);
          dimBlock.z = zThreads;
        }  
    
        dimGrid.x = Nx/dimBlock.x+1;
        dimGrid.y = Ny/dimBlock.y+1;
        dimGrid.z = 1;    
	
	
	printf("\nStarting NLPS test...\n");
	nlps = NLPStest(fkx, fky, fkz, fsin, fcos, gkx, gky, gkz, gsin, gcos);
	printf("\nExecuted NLPS test. Checking......\n");
        
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
	  for(int i=0; i<Ny; i++) {
	    y[i] = 2*M_PI*(float)(i-Ny/2)/Ny;					//
            x[j] = 2*M_PI*(float)(j-Nx/2)/Nx;					//
            z[k] = 2*M_PI*(float)(k-Nz/2)/Nz;
	    int index = i + Ny*j + Nx*Ny*k;
	    
	    //(df/dx)(dg/dy)-(df/dy)(dg/dx) 
	    fdxcheck[index] = -fkx*fcos*sin(fky*y[i] + fkx*x[j]+ fkz*z[k]) + fkx*fsin*cos(fky*y[i] + fkx*x[j]+ fkz*z[k]);	// only x and y changed,
	    fdycheck[index] = -fky*fcos*sin(fky*y[i] + fkx*x[j]+ fkz*z[k]) + fky*fsin*cos(fky*y[i] + fkx*x[j]+ fkz*z[k]);	// fkx,fky,gkx,gky,i,j same
	    gdxcheck[index] = -gkx*gcos*sin(gky*y[i] + gkx*x[j]+ gkz*z[k]) + gkx*gsin*cos(gky*y[i] + gkx*x[j]+ gkz*z[k]);	//
	    gdycheck[index] = -gky*gcos*sin(gky*y[i] + gkx*x[j]+ gkz*z[k]) + gky*gsin*cos(gky*y[i] + gkx*x[j]+ gkz*z[k]);	//
	    
	    
	    
	    nlpscheck[index] = fdxcheck[index]*gdycheck[index] - fdycheck[index]*gdxcheck[index];
	    
	  }
	 }     
	}
	
	FILE *ofile = fopen( argv[2], "w+");
	
	fprintf(ofile,"f(y,x)= %d*cos(%dy + %dx + %dz) + %d*sin(%dy + %dx + %dz)\n",fcos,fky,fkx,fkz,fsin,fky,fkx,fkz);
	fprintf(ofile,"g(y,x)= %d*cos(%dy + %dx) + %d*sin(%dy + %dx)\n",gcos,gky,gkx,gkz,gsin,gky,gkx,gkz);
	fprintf(ofile,"Nx=%d, Ny=%d, Nz=%d\n\nOutputs:\nNLPS BRACKET\n",Nx,Ny,Nz);		      
	
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
          for(int i=0; i<Ny; i++) {
            int index = i + Ny*j + Nx*Ny*k;
            fprintf(ofile,"N(%.2fPI,%.2fPI,%.2fPI)=%.3f: %d  ", y[i]/M_PI,                      //
	              x[j]/M_PI, z[k]/M_PI, nlps[index], index);     			        	 //
          }
          fprintf(ofile,"\n");
         }
	 
	} 
	
	fprintf(ofile,"\nExpected values:\n(df/dx)(dg/dy)-(df/dy)(dg/dx)\n");
	                       
	
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
          for(int i=0; i<Ny; i++) {
            int index = i + Ny*j + Nx*Ny*k;
            fprintf(ofile,"N(%.2fPI,%.2fPI,%.2fPI)=%.3f: %d  ", y[i]/M_PI, 			//
	              x[j]/M_PI, z[k]/M_PI, nlpscheck[index], index);     			 	//
          }	
          fprintf(ofile,"\n");
	 } 
	 fprintf(ofile,"\n");
        }
	
	int errorCounter=0;
	int errorSum=0;
	float errorAvg;
	
	bool equal = true;
	for(int k=0; k<Nz; k++) { 
	 for(int j=0; j<Nx; j++) {
	  for(int i=0; i<Ny; i++) {
	    int index = i + Ny*j + Nx*Ny*k;
	    if(abs(nlpscheck[index] - nlps[index]) > .4) { 
	      fprintf(ofile, "Element %d, off by %f\n",index,abs(nlpscheck[index] - nlps[index]));
	      if(debug) printf("Element %d, off by %f\n",index,abs(nlpscheck[index] - nlps[index]));
	      errorCounter++;
	      errorSum += abs(nlpscheck[index] - nlps[index]);	      
	    }
	    if(nlps[index] != nlps[index]) {equal = false;}      //check for nan
	  }	  
	 }	 
	}     
	
	if(errorCounter!=0) {
	  errorAvg = errorSum/errorCounter;
	  if(debug) printf("Error Count: %d   Avg Error: %f", errorCounter, errorAvg);
	  if(errorAvg > 1) {equal = false;}
	}  
	
	if(equal == true) {fprintf(ofile, "\nNLPS CHECKS\n"); printf("\nNLPS CHECKS\n");}
	else {fprintf(ofile, "\nNLPS DOES NOT CHECK\n"); printf("NLPS DOES NOT CHECK\n");}
	
	printf("fkx=%d  fky=%d  fkz=%d  fsin=%d  fcos=%d  gkx=%d  gky=%d  gkz=%d  gsin=%d  gcos=%d  Nx=%d  Ny=%d  Nz=%d\n\n", fkx, fky,fkz,fsin,fcos,gkx,gky,gkz,gsin,gcos,Nx, Ny, Nz);
	
	fclose(ofile);
	        
    }
}
  
