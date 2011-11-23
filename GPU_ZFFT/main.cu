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
#include <zderivtest.cu>
#include <zhilberttest.cu>
//#include <zfftb.cu>
//#include <zfftc.cu>
//#include <zfft_kernel.cu>

int main(int argc, char* argv[])
{
    int akx, aky, akz, Ny, Nx, Nz, asin, acos;
    cufftReal *b, *c;
    cufftReal *bcheck, *ccheck;
    float *x, *y, *z;
    
    
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
            fscanf(ifile, "%d %d %d %d %d %d %d %d", &akx, &aky, &akz, &asin, &acos, &Nx, &Ny, &Nz);
            fclose( ifile );
        } 
	
	bcheck = (cufftReal*) malloc(sizeof(cufftReal)*Ny*Nx*Nz);
	b = (cufftReal*) malloc(sizeof(cufftReal)*Ny*Nx*Nz);
	ccheck = (cufftReal*) malloc(sizeof(cufftReal)*Ny*Nx*Nz);
	c = (cufftReal*) malloc(sizeof(cufftReal)*Ny*Nx*Nz);
	y = (float*) malloc(sizeof(float)*Ny);					//
        x = (float*) malloc(sizeof(float)*Nx);
	z = (float*) malloc(sizeof(float)*Nz);
	
	b = ZDERIVtest(akx, aky, akz, asin, acos, Ny, Nx, Nz);
	c = ZHILBERTtest(akx, aky, akz, asin, acos, Ny, Nx, Nz);
	
	
	
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
	  for(int i=0; i<Ny; i++) {
	    y[i] = 2*M_PI*(float)(i-Ny/2)/Ny;					//
            x[j] = 2*M_PI*(float)(j-Nx/2)/Nx;
	    z[k] = 2*M_PI*(float)(k-Nz/2)/Nz;					//
            int index = i + Ny*j + Ny*Nx*k;
	    
	    bcheck[index] = -akz*acos*sin(aky*y[i] + akx*x[j] + akz*z[k]) + 
	                     akz*asin*cos(aky*y[i] + akx*x[j] + akz*z[k]);
	    
	    ccheck[index] = abs(akz)*asin*sin(aky*y[i] + akx*x[j] + akz*z[k]) + 
	                     abs(akz)*acos*cos(aky*y[i] + akx*x[j] + akz*z[k]);
	  }  
	 }
	}
	
	
	FILE *ofile = fopen( argv[2], "w+");
	
	fprintf(ofile,"a(y,x,z)= %d*cos(%dy + %dx + %dz) + %d*sin(%dy + %dx + %dz)\nNy=%d, Nx=%d, Nz=%d\n\nOutputs:\nb= df/dz\n",
	                      acos,aky,akx,akz,asin,aky,akx,akz,Ny,Nx,Nz);
	 
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
          for(int i=0; i<Ny; i++) {
            int index = i + Ny*j + Ny*Nx*k;
            fprintf(ofile,"b(%.2fPI,%.2fPI,%.2fPI)=%.3f: %d  ", y[i]/M_PI,                      //
	              x[j]/M_PI, z[k]/M_PI, b[index], index);     			        	 //
          }
          fprintf(ofile,"\n");
         }
	 fprintf(ofile,"\n");
	} 
	
	fprintf(ofile,"\nExpected values:\n");
	                       
	
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
          for(int i=0; i<Ny; i++) {
            int index = i + Ny*j + Ny*Nx*k;
            fprintf(ofile,"b(%.2fPI,%.2fPI,%.2fPI)=%.3f: %d  ", y[i]/M_PI, 			//
	              x[j]/M_PI, z[k]/M_PI, bcheck[index], index);     			 	//
          }	
          fprintf(ofile,"\n");
	 } 
	 fprintf(ofile,"\n");
        }
	
	fprintf(ofile,"\nc= HC(a)\n");
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
          for(int i=0; i<Ny; i++) {
            int index = i + Ny*j + Ny*Nx*k;
            fprintf(ofile,"c(%.2fPI,%.2fPI,%.2fPI)=%.3f: %d  ", y[i]/M_PI,                      //
	              x[j]/M_PI, z[k]/M_PI, c[index], index);     			        	 //
          }
          fprintf(ofile,"\n");
         }
	 fprintf(ofile,"\n");
	} 
	
	fprintf(ofile,"\nExpected values:\n");
	
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Nx; j++) {
          for(int i=0; i<Ny; i++) {
            int index = i + Ny*j + Ny*Nx*k;
            fprintf(ofile,"c(%.2fPI,%.2fPI,%.2fPI)=%.3f: %d  ", y[i]/M_PI, 			//
	              x[j]/M_PI, z[k]/M_PI, ccheck[index], index);     			 	//
          }	
          fprintf(ofile,"\n");
	 } 
	 fprintf(ofile,"\n");
        }
	
	
	
	bool equal = true;
	for(int k=0; k<Nz; k++) { 
	 for(int j=0; j<Nx; j++) {
	  for(int i=0; i<Ny; i++) {
	    int index = i + Ny*j + Ny*Nx*k;
	    if(abs(bcheck[index] - b[index]) > .1) {equal = false; fprintf(ofile, "%d\n",index);}
	  }
	  
	 }
	 
	}     
	if(equal == true) {fprintf(ofile, "\nB CHECKS\n"); printf("B CHECKS\n");}
	else {fprintf(ofile, "\nB DOES NOT CHECK\n"); printf("B DOES NOT CHECK\n");}
	
	bool equal2 = true;
	for(int k=0; k<Nz; k++) { 
	 for(int j=0; j<Nx; j++) {
	  for(int i=0; i<Ny; i++) {
	    int index = i + Ny*j + Ny*Nx*k;
	    if(abs(ccheck[index] - c[index]) > .1) {equal2 = false; fprintf(ofile, "%d\n",index);}
	  }
	  
	 }
	 
	}     
	if(equal2 == true) {fprintf(ofile, "\nC CHECKS\n"); printf("C CHECKS\n");}
	else {fprintf(ofile, "\nC DOES NOT CHECK\n"); printf("C DOES NOT CHECK\n");}
	
	
	
	fclose(ofile);
	
	  
	
	
        
    }
}

	
	
	
	
	
	
	
	
	
	
    
