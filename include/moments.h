#pragma once

#include "grids.h"
#include "fields.h"
#include "parameters.h"
#include "geometry.h"

class Geometry; // Forward Declaration

class MomentsG {
 public:
  MomentsG(Parameters* pars, Grids* grids);
  ~MomentsG();

  // accessor function to get pointer to specific l,m,s of G array
  // calling with no arguments gives pointer to beginning of G_lm
  cuComplex* G(int l=0, int m=0, int s=0) {
    return &G_lm[grids_->NxNycNz*l + grids_->NxNycNz*grids_->Nl*m + grids_->NxNycNz*grids_->Nmoms*s];
    // glm[ky, kx, z]
  }

  cuComplex * Gm(int m, int s=0) {   return G(0,m,s);   }

  void qvar (int N);
  void apply_mask(void);
  int initialConditions(Geometry* geo, double* time);
  int initialConditions(double* time);
  int restart_write(double* time);
  int restart_read(double* time);
  
  int add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2);
  int add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, double c3, MomentsG* G3);
  int add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, double c3, MomentsG* G3,
		 double c4, MomentsG* G4);
  int add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, double c3, MomentsG* G3,
		 double c4, MomentsG* G4, double c5, MomentsG* G5);

  int scale(double scalar);
  int scale(cuComplex scalar);

  int reality(int ngz);
  
  inline void copyFrom(MomentsG* source) {
    cudaMemcpy(this->G(), source->G(), LHsize_, cudaMemcpyDeviceToDevice);
  }
 
  dim3 dimGrid, dimBlock, dG_all, dB_all;

  cuComplex ** dens_ptr;
  cuComplex ** upar_ptr;
  cuComplex ** tpar_ptr;
  cuComplex ** tprp_ptr;
  cuComplex ** qpar_ptr;
  cuComplex ** qprp_ptr;
 
 private:
  cuComplex  * G_lm;
  Parameters * pars_;
  Grids      * grids_;
  size_t LHsize_;
  size_t Momsize_;

};
