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
  //  const int Nakz;
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
  
  float * ky = NULL;
  float * kx = NULL;
  float * kz = NULL;

  float * ky_h = NULL;
  float * kx_h = NULL;
  float * kx_outh = NULL;
  float * kz_h = NULL;

  float * theta0_h = NULL;

  /* A grid the size of kx, true if in the dealiased zone*/
  bool * kx_mask = NULL;
  
  /* Flow shear arrays*/
  float * kx_shift = NULL;
  int * jump = NULL;
  
  /* Stuff for z covering */
  int nClasses = NULL;
  /* Note: the arrays below are allocated in initialize_z_covering, not
   * in the main allocation routines. Also note that the ** 
   * pointers to pointers point to arrays of pointers on the
   * host. The pointers themselves then may point to arrays 
   * on either the device or host */
  int * nLinks = NULL;
  int * nChains = NULL;
  int ** kxCover;
  int ** kyCover;
  //cuComplex ** g_covering;
  float ** kz_covering;
  
  int ** kxCover_d;
  int ** kyCover_d;
  //cuComplex ** g_covering_d;
  float ** kz_covering_d;
  //int * nLinks_d;
  //int * nChains_d;
  
  float * covering_scaler = NULL;
  
 private:
  Parameters * pars_ = NULL; //local Parameters object for convenience
};

