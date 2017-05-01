#ifndef GEO_H
#define GEO_H

#include "cufft.h"

struct geometry_coefficents_struct{
  float *gradpar_arr;
  float *gbdrift;
  float *grho;
  float *cvdrift;
  float *bmag;
  float *bgrad;
  float *gds2;
  float *gds21;
  float *gds22;
  float *cvdrift0;
  float *gbdrift0;
  float *jacobian;
  float * Rplot;
  float * Zplot;
  float * aplot;
  float * Xplot;
  float * Yplot;
  float * Rprime;
  float * Zprime;
  float * aprime;
  float * deltaFL;

  //float drhodpsi;
  float gradpar;
  float bi;
  float aminor;


  float fluxDen;
  
  cuComplex * bmag_complex;
  float * bmagInv;
};

class Geometry {
 public:
  virtual ~Geometry() {}
  geometry_coefficents_struct* get_geo_coeffs() {
    return geo_;
  }

 private:
  geometry_coefficents_struct *geo_;
};

class S_alpha_geo : public Geometry {
 public:
  S_alpha_geo(int Nz);
};

class Eik_geo : public Geometry {
 public:
  Eik_geo();
};

class Gs2_geo : public Geometry {
 public:
  Gs2_geo();
};

//void set_geometry(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, struct gryfx_parameters_struct * gryfxpars);
//
//void copy_geo_arrays_to_device(geometry_coefficents_struct * geo, geometry_coefficents_struct * geo_h, input_parameters_struct * pars, int Nz);


#endif
