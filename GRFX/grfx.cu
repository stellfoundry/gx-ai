#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"

// includes, project
#include <cutil_inline.h>

void grfx(bool trinity, FILE *infile, float dvdrho, float grhoavg_out, 
          float *pflux, float *qflux, float *heat)
{

  //initialize GPU
  
  //find value of trinity
  
  if(trinity) {
  
    //initialize MPI layer
  }
    
  //prepare to read infile
  
  itg(infile,trinity);
  
  save();
  
  if(trinity) {
  
    //calculate pflux, qflux, heat
  
  }
  
  //clean up memory, release GPU
  
}      
	  
