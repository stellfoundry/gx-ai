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
#include <nlpstest.cu>




int main(int argc, char* argv[])
{
    int fkx, fky, gkx, gky, Nx, Ny, Nz, fsin, fcos, gsin, gcos;
    cufftReal *nlps;
    cufftReal *nlpscheck, *fdxcheck, *fdycheck, *gdxcheck, *gdycheck;
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
	
	nlps = NLPStest(fkx, fky, fsin, fcos, gkx, gky, gsin, gcos, Ny, Nx, Nz);
        
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
	
	
	
	fclose(ofile);
	
	  
	
	
        
    }
}


