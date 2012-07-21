/***** LINES CHANGED FOR X <-> Y MARKED BY '//' *******/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>
#include <cufft.h>


// includes, kernels
#include "nlpstest.cu"
//#include "constants.h"



int main(int argc, char* argv[])
{
    int fkx, fky, gkx, gky, fsin, fcos, gsin, gcos;
    cufftReal *nlps;
    cufftReal *nlpscheck, *fdxcheck, *fdycheck, *gdxcheck, *gdycheck;
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

        // fopen returns 0, the NULL pointer, on failure 
        if ( ifile == 0 )
        {
            printf( "Could not open file\n" );
        }
        else 
        {
            fscanf(ifile, "%d %d %d %d %d %d %d %d %d %d %d", &fkx, &fky, &fsin, &fcos, &gkx, &gky, &gsin, &gcos, &Nx, &Ny, &Nz);
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
	
	nlps = NLPStest(fkx, fky, fsin, fcos, gkx, gky, gsin, gcos);
        
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
	  for(int i=0; i<Ny; i++) {
	    y[i] = 2*M_PI*(float)(i-Ny/2)/Ny;					//
            x[j] = 2*M_PI*(float)(j-Nx/2)/Nx;					//
            z[k] = 2*M_PI*(float)(k-Nz/2)/Nz;
	    int index = i + Ny*j + Nx*Ny*k;
	    
	    //(df/dx)(dg/dy)-(df/dy)(dg/dx) 
	    fdxcheck[index] = -fkx*fcos*sin(fky*y[i] + fkx*x[j]) + fkx*fsin*cos(fky*y[i] + fkx*x[j]);	// only x and y changed,
	    fdycheck[index] = -fky*fcos*sin(fky*y[i] + fkx*x[j]) + fky*fsin*cos(fky*y[i] + fkx*x[j]);	// fkx,fky,gkx,gky,i,j same
	    gdxcheck[index] = -gkx*gcos*sin(gky*y[i] + gkx*x[j]) + gkx*gsin*cos(gky*y[i] + gkx*x[j]);	//
	    gdycheck[index] = -gky*gcos*sin(gky*y[i] + gkx*x[j]) + gky*gsin*cos(gky*y[i] + gkx*x[j]);	//
	    
	    /*f and g
	    f[index]= -2*cos(y[i])-.864*cos(y[i]+2*x[j])+.26*cos(2*y[i]+2*x[j])
                +.41846*cos(3*y[i]+2*x[j])+.28*cos(4*y[i]+2*x[j])
		-.264*cos(2*y[i]+4*x[j])+.56*cos(3*y[i]+4*x[j])
		+.3*cos(4*y[i]+4*x[j]);
      
            g[index]= cos(y[i])+4*cos(2*x[j])-2.08*cos(y[i]+2*x[j])-4.32*cos(2*y[i]+2*x[j])
                -1.8*cos(3*y[i]+2*x[j])
		-9.12*cos(2*y[i]+4*x[j])+11*cos(3*y[i]+4*x[j])
		-13.44*cos(4*y[i]+4*x[j]);
	    */
	    
	    /*fdxcheck[index]= 2*.864*sin(y[i]+2*x[j])-2*.26*sin(2*y[i]+2*x[j])
                -2*.41846*sin(3*y[i]+2*x[j])-2*.28*sin(4*y[i]+2*x[j])
		+4*.264*sin(2*y[i]+4*x[j])-4*.56*sin(3*y[i]+4*x[j])
		-4*.3*sin(4*y[i]+4*x[j]);
	    fdycheck[index]= 2*sin(y[i])+.864*sin(y[i]+2*x[j])-2*.26*sin(2*y[i]+2*x[j])
                -3*.41846*sin(3*y[i]+2*x[j])-4*.28*sin(4*y[i]+2*x[j])
		+2*.264*sin(2*y[i]+4*x[j])-3*.56*sin(3*y[i]+4*x[j])
		-4*.3*sin(4*y[i]+4*x[j]);
	    gdxcheck[index]= -2*4*sin(2*x[j])+2*2.08*sin(y[i]+2*x[j])+2*4.32*sin(2*y[i]+2*x[j])
                +2*1.8*sin(3*y[i]+2*x[j])
		+4*9.12*sin(2*y[i]+4*x[j])-4*11*sin(3*y[i]+4*x[j])
		+4*13.44*sin(4*y[i]+4*x[j]);
	    gdycheck[index]= -sin(y[i])+2.08*sin(y[i]+2*x[j])+2*4.32*sin(2*y[i]+2*x[j])
                +3*1.8*sin(3*y[i]+2*x[j])
		+2*9.12*sin(2*y[i]+4*x[j])-3*11*sin(3*y[i]+4*x[j])
		+4*13.44*sin(4*y[i]+4*x[j]);	*/
	    
	    //gdxcheck[index]= -32*sin(2*y[i]+2*x[j])+5.6*sin(3*y[i]+4*x[j]);
	    //gdycheck[index]= -32*sin(2*y[i]+2*x[j])-.2*sin(y[i])+4.2*sin(3*y[i]+4*x[j]);
	    //fdxcheck[index]= -2*sin(y[i]+2*x[j])+.176*sin(3*y[i]+4*x[j]);
	    //fdycheck[index]= -sin(y[i]+2*x[j])+.1*sin(y[i])+.132*sin(3*y[i]+4*x[j]);
	    
	    //fdxcheck[index]= -4*sin(2*y[i]+2*x[j]);
	    //fdycheck[index]= -4*sin(2*y[i]+2*x[j]);
	    //gdxcheck[index]= -10*sin(y[i]+2*x[j]);
	    //gdycheck[index]= -5*sin(y[i]+2*x[j]);
	    
	    nlpscheck[index] = fdxcheck[index]*gdycheck[index] - fdycheck[index]*gdxcheck[index];
	    
	  }
	 }     
	}
	FILE *ofile = fopen( argv[2], "w+");
	
	fprintf(ofile,"f(y,x)= %d*cos(%dy + %dx) + %d*sin(%dy + %dx)\ng(y,x)= %d*cos(%dy + %dx) + %d*sin(%dy + %dx)\nNx=%d, Ny=%d, Nz=%d\n\nOutputs:\nNLPS BRACKET\n",
	                      fcos,fky,fkx,fsin,fky,fkx,gcos,gky,gkx,gsin,gky,gkx,Nx,Ny,Nz);
	
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
	
	
	
	bool equal = true;
	for(int k=0; k<Nz; k++) { 
	 for(int j=0; j<Nx; j++) {
	  for(int i=0; i<Ny; i++) {
	    int index = i + Ny*j + Nx*Ny*k;
	    if(abs(nlpscheck[index] - nlps[index]) > .1) {equal = false; fprintf(ofile, "%d\n",index);}
	  }
	  
	 }
	 
	}     
	if(equal == true) {fprintf(ofile, "\nNLPS CHECKS\n"); printf("NLPS CHECKS\n");}
	else {fprintf(ofile, "\nNLPS DOES NOT CHECK\n"); printf("NLPS DOES NOT CHECK\n");}
	
	printf("fkx=%d  fky=%d  fsin=%d  fcos=%d  gkx=%d  gky=%d  gsin=%d  gcos=%d  Nx=%d  Ny=%d  Nz=%d\n", fkx, fky, fsin,fcos,gkx, gky,gsin,gcos,Nx, Ny, Nz);
	
	fclose(ofile);
	
	
	  
	
	
        
    }
}


