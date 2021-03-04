#pragma once
#include "device_funcs.h"
#include "parameters.h"
#include "get_error.h"

class Grids {

 public:
  Grids(Parameters* pars);
  ~Grids();

  int Nx;
  int Ny;
  int Nz;
  int Nspecies; 
  int Nm;
  int Nl;
  int Nj;
  int Nyc;
  int Naky;
  int Nakx;
  int NxNyc;
  int NxNy;
  int NxNycNz;
  int NxNyNz;
  int NxNz;
  int NycNz;
  int Nmoms;
  size_t size_G;
  
  float * ky;       float * kx;    float * kz;
  float * ky_h;     float * kx_h;  float * kz_h;
  float * kx_outh;  float * kz_outh; 
  //  float * theta0_h ;

  /* Flow shear arrays*/
  //  float * kx_shift ;
  //  int * jump ;
  
 private:
  Parameters * pars_ ; 
};

