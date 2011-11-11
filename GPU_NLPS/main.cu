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
            fscanf(ifile, "%d %d %d %d %d %d %d %d %d %d %d", &fkx, &fky, &fsin, &fcos, &gkx, &gky, &gsin, &gcos, &Nx, &Ny, &Nz);
            fclose( ifile );
        } 
	nlpscheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);        
        nlps = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
	x = (float*) malloc(sizeof(float)*Nx);
        y = (float*) malloc(sizeof(float)*Ny);
	fdxcheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
	fdycheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
	gdxcheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
	gdycheck = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
	
	nlps = NLPStest(fkx, fky, fsin, fcos, gkx, gky, gsin, gcos, Nx, Ny, Nz);
        
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Ny; j++) {
	  for(int i=0; i<Nx; i++) {
	    x[i] = 2*M_PI*(float)(i-Nx/2)/Nx;
            y[j] = 2*M_PI*(float)(j-Ny/2)/Ny;
            int index = i + Nx*j + Nx*Ny*k;
	    
	    //(df/dx)(dg/dy)-(df/dy)(dg/dx) 
	    fdxcheck[index] = -fkx*fcos*sin(fkx*x[i] + fky*y[j]) + fkx*fsin*cos(fkx*x[i] + fky*y[j]);
	    fdycheck[index] = -fky*fcos*sin(fkx*x[i] + fky*y[j]) + fky*fsin*cos(fkx*x[i] + fky*y[j]);
	    gdxcheck[index] = -gkx*gcos*sin(gkx*x[i] + gky*y[j]) + gkx*gsin*cos(gkx*x[i] + gky*y[j]);
	    gdycheck[index] = -gky*gcos*sin(fkx*x[i] + gky*y[j]) + gky*gsin*cos(gkx*x[i] + gky*y[j]);
	    nlpscheck[index] = fdxcheck[index]*gdycheck[index] - fdycheck[index]*gdxcheck[index];
	    
	  }
	 }     
	}
	FILE *ofile = fopen( argv[2], "w+");
	
	fprintf(ofile,"f(x,y)= %d*cos(%dx + %dy) + %d*sin(%dx + %dy)\ng(x,y)= %d*cos(%dx + %dy) + %d*sin(%dx + %dy)\n\nOutputs:\nNLPS BRACKET\n",
	                      fcos,fkx,fky,fsin,fkx,fky,gcos,gkx,gky,gsin,gkx,gky);
	
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Ny; j++) {
          for(int i=0; i<Nx; i++) {
            int index = i + Nx*j + Nx*Ny*k;
            fprintf(ofile,"N(%.2fPI,%.2fPI)=%.3f: %d  ", (float)2*(i-Nx/2)/Nx, 
	              (float)2*(j-Ny/2)/Ny, nlps[index], index);     
          }
          fprintf(ofile,"\n");
         }
	} 
	
	fprintf(ofile,"\nExpected values:\n(df/dx)(dg/dy)-(df/dy)(dg/dx)\n");
	                       
	
	for(int k=0; k<Nz; k++) {
	 for(int j=0; j<Ny; j++) {
          for(int i=0; i<Nx; i++) {
            int index = i + Nx*j + Nx*Ny*k;
            fprintf(ofile,"N(%.2fPI,%.2fPI)=%.3f: %d  ", (float)2*(i-Nx/2)/Nx, 
	              (float)2*(j-Ny/2)/Ny, nlpscheck[index], index);     
          }
          fprintf(ofile,"\n");
	 } 
        }
	
	
	bool equal = false;
	for(int k=0; k<Nz; k++) { 
	 for(int j=0; j<Ny; j++) {
	  for(int i=0; i<Nx; i++) {
	    int index = i + Nx*j + Nx*Ny*k;
	    if(abs(nlpscheck[index] - nlps[index]) < .0001) { equal = true;}
	    else {equal = false; fprintf(ofile, "\n%d\n",index); break;}
	  }
	  if(equal == false) { break;}
	 }
	 if(equal == false) {break;}
	}     
	if(equal == true) {fprintf(ofile, "\nNLPS CHECKS\n"); printf("NLPS CHECKS\n");}
	else {fprintf(ofile, "\nNLPS DOES NOT CHECK\n"); printf("NLPS DOES NOT CHECK\n");}
	
	
	
	fclose(ofile);
	
	  
	
	
        
    }
}


