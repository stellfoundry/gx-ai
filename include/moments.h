#pragma once
#include "netcdf.h"
#include "grids.h"
#include "parameters.h"
#include "device_funcs.h"
#include "get_error.h"
#include "cuda_constants.h"

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
  void initialConditions(float* z_h, double* time);
  void initialConditions(double* time);
  void restart_write(double* time);
  void restart_read(double* time);
  
  void add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2);
  void add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, double c3, MomentsG* G3);
  void add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, double c3, MomentsG* G3,
		  double c4, MomentsG* G4);
  void add_scaled(double c1, MomentsG* G1, double c2, MomentsG* G2, double c3, MomentsG* G3,
		  double c4, MomentsG* G4, double c5, MomentsG* G5);

  void scale(double scalar);
  void scale(cuComplex scalar);

  //  void dz(MomentsG* G);
  void reality(int ngz);
  
  inline void copyFrom(MomentsG* source) {
    cudaMemcpy(this->G(), source->G(), grids_->size_G, cudaMemcpyDeviceToDevice);
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
  Grids      * grids_;
  Parameters * pars_;
};
