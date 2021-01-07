#ifndef GEO_H
#define GEO_H

#include "cufft.h"
#include "parameters.h"
#include "grids.h"
#include "grad_parallel.h" // MFM

class GradParallel; // Forward Declaration

class Geometry {
 public:
  Geometry() {operator_arrays_allocated_=false;};
  ~Geometry();
  
  float * z_h = NULL;
  float * gbdrift_h = NULL;
  float * grho_h = NULL;
  float * cvdrift_h = NULL;
  float * bmag_h = NULL;
  float * bmagInv_h = NULL;
  float * bgrad_h = NULL;
  float * gds2_h = NULL;
  float * gds21_h = NULL;
  float * gds22_h = NULL;
  float * cvdrift0_h = NULL;
  float * gbdrift0_h = NULL;
  float * jacobian_h = NULL;

  float * z = NULL;
  float * gbdrift = NULL;
  float * grho = NULL;
  float * cvdrift = NULL;
  float * bmag = NULL;
  float * bmagInv = NULL;
  float * bgrad = NULL;
  float * gds2 = NULL;
  float * gds21 = NULL;
  float * gds22 = NULL;
  float * cvdrift0 = NULL;
  float * gbdrift0 = NULL;
  float * jacobian = NULL;

  float * gradpar_arr = NULL;
  float * Rplot = NULL;
  float * Zplot = NULL;
  float * aplot = NULL;
  float * Xplot = NULL;
  float * Yplot = NULL;
  float * Rprime = NULL;
  float * Zprime = NULL;
  float * aprime = NULL;
  float * deltaFL = NULL;

  float drhodpsi;
  float gradpar;
  float bi;
  float aminor;
  float shat;
  
  cuComplex * bmag_complex = NULL;
  float * bgrad_temp = NULL;

  // operator arrays
  float * kperp2 = NULL;
  float * omegad = NULL; // still used in closures. Should be cleaned out. 
  float * cv_d = NULL;
  float * gb_d = NULL; 
  float * kperp2_h = NULL;
  
  void initializeOperatorArrays(Grids* grids);
  void calculate_bgrad(Grids* grids); // MFM

  bool operator_arrays_allocated_;
};

class S_alpha_geo : public Geometry {
 public:
  S_alpha_geo(Parameters* parameters, Grids* grids);

};

class Eik_geo : public Geometry {
 public:
  Eik_geo();
};

class Gs2_geo : public Geometry {
 public:
  Gs2_geo();
};

// MFM
class File_geo : public Geometry {
 public:
  File_geo(Parameters* parameters, Grids* grids);

};

//void set_geometry(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, struct gx_parameters_struct * gxpars);

//void copy_geo_arrays_to_device(geometry_coefficents_struct * geo, geometry_coefficents_struct * geo_h, input_parameters_struct * pars, int Nz);


#endif
