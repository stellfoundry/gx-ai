#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "sys/stat.h"
#include "cufft.h"
#include "cuda_profiler_api.h"

int Nx, Ny, Nz, zThreads, totalThreads;
float X0, Y0, Zp;
__constant__ int nx,ny,nz,zthreads,totalthreads;
__constant__ float X0_d,Y0_d,Zp_d;

dim3 dimBlock, dimGrid;

float endtime;
float nu, eta;

int nClasses;
int *nLinks;
int *nChains;

bool MASK=true;         //defaults to on unless set otherwise
bool DEBUG=true;
bool QUIET=false;
bool NO_ZDERIV=false;
bool NO_ZDERIV_COVERING=false;

float *kx, *ky, *kz, *kPerp2, *kPerp2Inv;
cuComplex *deriv_nlps;
float *derivR1_nlps, *derivR2_nlps, *resultR_nlps;

float *kx_h, *ky_h;

float kxfac = 1;
float gradpar = 1;

//plans
cufftHandle NLPSplanR2C, NLPSplanC2R, ZDerivBplanR2C, ZDerivBplanC2R, ZDerivplan;


//make sure GRFX and TEST_CASES are in same directory

//GRYFX files
#include "../device_funcs.cu"
#include "../operations_kernel.cu"
#include "../nlps_kernel.cu"
#include "../zderiv_kernel.cu"
#include "../covering_kernel.cu"
#include "../reduc_kernel.cu"
#include "../cudaReduc_kernel.cu"
#include "../init_kernel.cu"
#include "../omega_kernel.cu"
#include "../phi_kernel.cu"
#include "../getfcn.cu"
//#include "maxReduc.cu"
#include "../sumReduc.cu"
#include "../coveringSetup.cu"
//#include "../ztransform_covering.cu"
//#include "../zderiv.cu"
//#include "../zderiv_covering.cu"
#include "../nlps.cu"
//#include "courant.cu"


//ORS-TANG test files
#include "timestep_kernel.cu"
#include "energy.cu"
#include "timestep_ors-tang.cu"
#include "run_ors-tang.cu"


int main(int argc, char* argv[])
{   
  cufftReal *f, *g;
  
  if (argc == 5 && strcmp(argv[4],"-quiet")==0) {QUIET = true; }
    
  int ct, dev;
  struct cudaDeviceProp prop;

  cudaGetDeviceCount(&ct);
  if(!QUIET) printf("Device Count: %d\n",ct);
  cudaGetDevice(&dev);
  if(!QUIET) printf("Device ID: %d\n",dev);
  cudaGetDeviceProperties(&prop,dev);
  if(!QUIET) {
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
  
  
  if ( argc != 4 && argc != 5 ) // argc should be 2 for correct execution 
    {
      //We print argv[0] assuming it is the program name 
      printf( "usage: %s inputfile outputfile\n\n", argv[0] );
    }
  else 
    {
      // We assume argv[1] is a filename to open
      FILE *ifile = fopen( argv[1], "r" );
      FILE *ofile = fopen( argv[2], "w+");
      FILE *pipe = popen( "gnuplot -persist", "w");
      FILE *plotfile = fopen( argv[3], "w+");
      
      if (argc == 5 && strcmp(argv[4],"-quiet")==0) {QUIET = true; }
      else if (argc == 5 && strcmp(argv[4],"-debug")==0) {DEBUG = true;}
      else if (argc == 5) printf("invalid option\n");

      // fopen returns 0, the NULL pointer, on failure 
      if ( ifile == 0 )
        {
	  printf( "Could not open file\n" );
        }
      else 
        {

            fscanf(ifile, "%d %d %d %f %f %f", &Nx, &Ny, &Nz, &X0, &Y0, &Zp);
            fscanf(ifile, "%f %f %f", &endtime, &nu, &eta);
	    fclose( ifile );

        } 
		
        f = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
        g = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);				//
	
	

	
	
	
	//////////////////////////////////////////////////////////
  
        //set up normal dimGrid/dimBlock config  
        
	int zBlockThreads = prop.maxThreadsDim[2];
	*&zThreads = zBlockThreads*prop.maxGridSize[2];
	*&totalThreads = prop.maxThreadsPerBlock;     

	if(Nz>zBlockThreads) dimBlock.z = zBlockThreads;
	else dimBlock.z = Nz;
  float otherThreads = totalThreads/dimBlock.z;
  int xy = floorf(otherThreads);
  if( (xy%2) != 0 ) xy = xy - 1; // make sure xy is even and less than totalThreads/dimBlock.z
  //find middle factors of xy
  int fx, fy;
  for(int f1 = 1; f1<xy; ++f1) {
    float f2 = (float) xy/f1;
    if(f2 == floorf(f2)) {
      fy = f1; fx = f2;
    }
    if(f2<=f1) break;
  }
  dimBlock.x = fx;
  dimBlock.y = fy;

  dimGrid.x = (Nx+dimBlock.x-1)/dimBlock.x;
  dimGrid.y = (Ny+dimBlock.y-1)/dimBlock.y;
  if(prop.maxGridSize[2] == 1) dimGrid.z = 1;
  else dimGrid.z = (Nz+dimBlock.z-1)/dimBlock.z;

 printf("%d %d %d     %d %d %d\n", dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z);
	
	if(DEBUG) getError("After dimGrid/dimBlock setup");
	
	//X0 = Y0 = Zp = 1;
	
	cudaMemcpyToSymbol(nx, &Nx, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(ny, &Ny, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(nz, &Nz, sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(X0_d, &X0, sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Y0_d, &Y0, sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Zp_d, &Zp, sizeof(float),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(zthreads, &zThreads, sizeof(int),0,cudaMemcpyHostToDevice);
        
	
	if(DEBUG) getError("After memcpyToSymbol");
	
	float *x, *y, *z;    
        x = (float*) malloc(sizeof(float)*Nx);
        y = (float*) malloc(sizeof(float)*Ny);
        z = (float*) malloc(sizeof(float)*Nz);
		
        for(int k=0; k<Nz; k++) {
     	 for(int j=0; j<Nx; j++) {
      	  for(int i=0; i<Ny; i++) {
      
      
      	   y[i] = Y0*2*M_PI*(float)(i-Ny/2)/Ny;                            
      	   x[j] = X0*2*M_PI*(float)(j-Nx/2)/Nx;	  			    
      	   z[k] = Zp*2*M_PI*(float)(k-Nz/2)/Nz;	  			    
      	   int index = i + Ny*j + Ny*Nx*k;
      
      
           //we use the Orszag-Tang initial conditions
           //phi = -2(cosx + cosy)
           //A = 2cosy + cos2x
           //f = z+ = phi + A
      	   //g = z- = phi - A
	
     	   f[index] = 2*(-cos(x[j]) - cos(y[i])) + 2*(cos(y[i]) + .5*cos(2*x[j]));
   	   g[index] = 2*(-cos(x[j]) - cos(y[i])) - 2*(cos(y[i]) + .5*cos(2*x[j]));  
      	        		   
           f[index] = f[index]*1.e-19;
           g[index] = g[index]*1.e-19;
    	  }
   	 }
   	} 
	
	if(DEBUG) getError("After f,g host init");
	
	
    
        //////////////////////////////////////////////////////////
	
	printf("\nStarting Orszag-Tang test...\n");	
        timestep_test(f, g, ofile);
	printf("\nPlotting......\n");
        	
	//////////////////////////////////////////////////////////
	
	
	
	//make a gnuplot script called plotTime for later plotting
	fprintf(plotfile, "set terminal x11 size 600,600\n");
	fprintf(plotfile, "set size .8,.8\n");
	fprintf(plotfile, "set origin .07,.15\n");
	fprintf(plotfile, "set title \"ENERGY PLOTS\"\n");
	fprintf(plotfile, "set xlabel \"Time (s)\"\nset ylabel \"Energy\"\n");
	fprintf(plotfile, "set xtics 0, .5, %f\nset mxtics 5\nset ytics 0, 1\nset mytics 5\nset tics scale 3\n",endtime);
	fprintf(plotfile, "set label \"Nx=%d   Ny=%d   Nz=%d  Boxsize=2pi*(%f,%f,%f)\\n\\n\\\n", Nx, Ny, Nz, X0, Y0, Zp);
	fprintf(plotfile, "nu=%g   eta=%g\" at 0,-.9\n",nu,eta);
	fprintf(plotfile, "plot [ ] [0:] \"../%s\" using 1:2 title \"total energy\" with lines, \\\n", argv[2]);
	fprintf(plotfile, "\"../%s\" using 1:3 title \"kinetic energy\" with lines, \\\n", argv[2]);
	fprintf(plotfile, "\"../%s\" using 1:4 title \"magnetic energy\" with lines\n", argv[2]);
	fprintf(plotfile, "pause -1 \"press any key\"");

	
        fclose(plotfile);
	

	//pipe to gnuplot to instantly plot data
	fprintf(pipe, "set terminal x11 size 600,600\n");
	fprintf(pipe, "set size .8,.8\n");
	fprintf(pipe, "set origin .07,.15\n");
	fprintf(pipe, "set title \"ENERGY PLOTS\"\n");
	fprintf(pipe, "set xlabel \"Time (s)\"\nset ylabel \"Energy\"\n");
	//fprintf(pipe, "set xtics 0, .5, %f\nset mxtics 5\nset ytics 0, 1\nset mytics 5\nset tics scale 3\n",endtime);
	fprintf(pipe, "set label \"Nx=%d   Ny=%d   Nz=%d  Boxsize=2pi*(%f,%f,%f)\\n\\n\\\n", Nx, Ny, Nz, X0, Y0, Zp);
	fprintf(pipe, "nu=%g   eta=%g\" at 0,-.9\n",nu,eta);
	fprintf(pipe, "plot [ ] [0:] \"%s\" using 1:2 title \"total energy\" with lines, \\\n", argv[2]);
	fprintf(pipe, "\"%s\" using 1:3 title \"kinetic energy\" with lines, \\\n", argv[2]);
	fprintf(pipe, "\"%s\" using 1:4 title \"magnetic energy\" with lines\n", argv[2]);
	fprintf(pipe, "pause -1 \"press any key\"");

	
        fclose(pipe);
	
	
	
        printf("\nNx=%d   Ny=%d  Nz=%d  BoxSize=2pi*(%f,%f,%f) endtime=%f eta=%f nu=%f\n\n", Nx, Ny, Nz, X0, Y0, Zp, endtime, eta, nu);


	
        fclose(ofile);
	 
    }
}  	
