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
#include <fftderiv1.cu>


int main(int argc, char* argv[])
{
    int k_x, k_y, Nx, Ny, sincoef, coscoef;
    cufftReal *fcndx, *fcndy;
    cufftReal *dxcheck, *dycheck;
    float *x, *y;
    
    
    
    
    
    
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
            fscanf(ifile, "%d %d %d %d %d %d", &k_x, &k_y, &Nx, &Ny, &sincoef, &coscoef);
            fclose( ifile );
        } 
	dxcheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny);
        dycheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny);
        x = (float*) malloc(sizeof(float)*Nx*Ny);
        y = (float*) malloc(sizeof(float)*Nx*Ny);
	
	fcndx = fftderiv(k_x, k_y, Nx, Ny, sincoef, coscoef, 1);
        fcndy = fftderiv(k_x, k_y, Nx, Ny, sincoef, coscoef, 2);
	
	for(int j=0; j<Ny; j++) {
	  for(int i=0; i<Nx; i++) {
	    x[i] = 2*M_PI*(float)(i-Nx/2)/Nx;
            y[j] = 2*M_PI*(float)(j-Ny/2)/Ny;
            int index = i + Nx*j;
	    
	    dxcheck[index] = -k_x*coscoef*sin(k_x*x[i] + k_y*y[j]) + k_x*sincoef*cos(k_x*x[i] + k_y*y[j]);
	    dycheck[index] = -k_y*coscoef*sin(k_x*x[i] + k_y*y[j]) + k_y*sincoef*cos(k_x*x[i] + k_y*y[j]);
	  }
	}     
	
	FILE *ofile = fopen( argv[2], "w");
	
	fprintf(ofile,"f(x,y)= %d*cos(%dx + %dy) + %d*sin(%dx + %dy)\n\nOutputs:\n\ndf/dx\n",
	                      coscoef,k_x,k_y,sincoef,k_x,k_y);
	for(int j=0; j<Ny; j++) {
          for(int i=0; i<Nx; i++) {
            int index = i + Nx*j;
            fprintf(ofile,"f(%.2fPI,%.2fPI)=%.3f: %d  ", (float)2*(i-Nx/2)/Nx, 
	              (float)2*(j-Ny/2)/Ny, fcndx[index], index);     
          }
          fprintf(ofile,"\n");
        }
	fprintf(ofile,"\ndf/dy\n");
	for(int j=0; j<Ny; j++) {
          for(int i=0; i<Nx; i++) {
            int index = i + Nx*j; 
            fprintf(ofile,"f(%.2fPI,%.2fPI)=%.3f: %d  ", (float)2*(i-Nx/2)/Nx, 
	              (float)2*(j-Ny/2)/Ny, fcndy[index], index);     
          }
          fprintf(ofile,"\n");
        }
	fprintf(ofile,"\nExpected values:\ndf/dx= -%d*%d*sin(%dx + %dy) + %d*%d*cos(%dx + %dy)\n", 
	                       k_x, coscoef, k_x, k_y, k_x, sincoef, k_x, k_y);
	
	for(int j=0; j<Ny; j++) {
          for(int i=0; i<Nx; i++) {
            int index = i + Nx*j;
            fprintf(ofile,"f(%.2fPI,%.2fPI)=%.3f: %d  ", (float)2*(i-Nx/2)/Nx, 
	              (float)2*(j-Ny/2)/Ny, dxcheck[index], index);     
          }
          fprintf(ofile,"\n");
        }
	fprintf(ofile,"\ndf/dy= -%d*%d*sin(%dx + %dy) + %d*%d*cos(%dx + %dy)\n",
	                       k_y, coscoef, k_x, k_y, k_y, sincoef, k_x, k_y);
	for(int j=0; j<Ny; j++) {
          for(int i=0; i<Nx; i++) {
            int index = i + Nx*j;
            fprintf(ofile,"f(%.2fPI,%.2fPI)=%.3f: %d  ", (float)2*(i-Nx/2)/Nx, 
	              (float)2*(j-Ny/2)/Ny, dycheck[index], index);     
          }
          fprintf(ofile,"\n");
        }  
	
	bool equal = false;
	for(int j=0; j<Ny; j++) {
	  for(int i=0; i<Nx; i++) {
	    int index = i + Nx*j;
	    if(abs(dxcheck[index] - fcndx[index]) < .0001) { equal = true;}
	    else {equal = false; fprintf(ofile, "\n%d\n",index); break;}
	  }
	}    
	if(equal == true) {fprintf(ofile, "\nCHECKS\n"); printf("CHECKS\n");}
	else {fprintf(ofile, "\nDOES NOT CHECK\n"); printf("DOES NOT CHECK\n");}
	
	fclose(ofile);
        
    }
}


