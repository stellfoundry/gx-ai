#pragma once

#include "parameters.h"

class Grids {

 public:
  Grids(Parameters* pars);
  ~Grids();

  const int Nx;
  const int Ny;
  const int Nz;
  const int Nspecies; 
  const int Nm;
  const int Nl;
  const int Nj;
  const int Nyc;
  const int Naky;
  const int Nakx;
  const int NxNyc;
  const int NxNy;
  const int NxNycNz;
  const int NxNyNz;
  const int NxNz;
  const int NycNz;
  const int Nmoms;
  const size_t size_G;
  
  float * ky;       float * kx;    float * kz;
  float * ky_h;     float * kx_h;  float * kz_h;
  float * kx_outh;
  float * theta0_h ;

  /* Flow shear arrays*/
  //  float * kx_shift ;
  //  int * jump ;
  
 private:
  Parameters * pars_ ; 
};

