#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"

// includes, project
#include <cutil_inline.h>

void itg(FILE *infile, bool trinity) 
{
  
  cufftComplex* old_density, old_upar, old_tpar, old_tperp, old_qperp;
  cufftComplex* old_potential;
  old_density = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz*Ns);
  old_upar = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz*Ns);
  old_tpar = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz*Ns);
  old_tperp = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz*Ns);
  old_qperp = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz*Ns);
  old_potential = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
  
  //get input data from infile
  
  init_physics();
  init_local_memory();
  
  for(int i=0; i<imax; i++) {
   
    //need to define imax, dt
    
    nonlinear(density);
    courant();
    double time += dt;
    double hdt = dt*.5;
    timestep(old_vars,vars,hdt);
    get_fields(old_potential,old_density,old_tperp);
    
    nonlinear(old_density,old_upar,old_tpar,old_tperp,old_qperp,old_potential);
    timestep(vars,old_vars,dt);
    getfields(potential,density,tperp);
    
    //calculate diagnostic quantities, do incremental outputs, look for control
    //flags
    
    //exit if exit flag is set
    
  }

}

void init_physics() {
  //Based on input file, allocate memory for main variables in each component of
  //the code, initialize FFTs, and set up some basic arrays of data that will not
  //change over the course of the run
}

void init_local_memory() {
  cufftComplex* old_density, old_upar, old_tpar, old_tperp, old_qperp;
  cufftComplex* old_potential;
  cudaMalloc((void**) &old_density, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz*Ns);
  cudaMalloc((void**) &old_upar, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz*Ns);
  cudaMalloc((void**) &old_tpar, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz*Ns);
  cudaMalloc((void**) &old_tperp, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz*Ns);
  cudaMalloc((void**) &old_qperp, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz*Ns);
  cudaMalloc((void**) &old_potential, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
}    

