#pragma once
#include "device_funcs.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "linear.h"
#include "nonlinear.h"
#include "fields.h"
#include "ncdf.h"
#include "grad_parallel.h"
#include "grad_perp.h"
#include "reservoir.h"
#include "diagnostic_classes.h"
#include "spectra_calc.h"
#include <memory>

using namespace std;

class Diagnostics {
 public:
  virtual ~Diagnostics() {};
  virtual bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) = 0 ;
  virtual void finish(MomentsG** G, Fields* fields, double time) = 0;  
  void restart_write(MomentsG** G, double *time);

 protected:
  Parameters   * pars_         ;
  Grids        * grids_        ;

};

class Diagnostics_GK : public Diagnostics {
 public:
  Diagnostics_GK(Parameters *pars, Grids *grids, Geometry *geo, Linear *linear, Nonlinear *nonlinear);
  ~Diagnostics_GK();

  bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) ;
  void finish(MomentsG** G, Fields* fields, double time);  

private:
  float* P2(int s=0) {return &P2s[grids_->NxNycNz*s];}
  float* G2(int s=0) {return &G2s[grids_->NxNycNz*grids_->Nmoms*s];}

  int ndiag; 
  int ikx_local, iky_local, iz_local;
  dim3 dG_spectra, dB_spectra, dG_all, dB_all, dbp, dgp; //, dG_scale, dB_scale;
  dim3 dGk, dBk;
  bool checkstop();
 
  float Dks; 
  float fluxDenom; float * flux_fac; 
  float  volDenom; float * vol_fac ;
  float * kvol_fac;
  
  cuComplex valphi;

  Geometry     * geo_          ;
  GradPerp     * grad_perp     ; 
  GradParallel * grad_par      ;
  Fields       * fields_old    ;
  MomentsG     ** G_old    ;
  NetCDF       * ncdf_         ;
  NetCDF       * ncdf_big_         ;
  Reservoir    * rc            ;
  AllSpectraCalcs * allSpectra_;
  Linear * linear_;
  Nonlinear * nonlinear_;

  float * tmpf;
  cuComplex * tmpC;
  float * tmpG;
  float *P2s;
  float *G2s;
  
  float        * val           ;
  cuComplex    * omg_d         ;
  cuComplex    * tmp_omg_h     ;
  cuComplex    * t_bar         ;
  cuComplex    * favg          ;
  cuComplex    * df            ;
  cuComplex    * amom_d        ;
  cuComplex    * vEk           ;
  float        * phi_max       ; 

  float *gy_d, *gy_h;
  double *ry_h;
  
  void print_omg (cuComplex *W);
  void get_rh    (Fields* f);
  void print_growth_rates_to_screen (cuComplex *w);
  void write_Wtot  (float   Wh, bool endrun);

  char stopfilename_[2000];

  vector<unique_ptr<SpectraDiagnostic>> spectraDiagnosticList;
  GrowthRateDiagnostic *growthRateDiagnostic;
  vector<unique_ptr<MomentsDiagnostic>> momentsDiagnosticList;
  FieldsDiagnostic *fieldsDiagnostic;
};

class Diagnostics_KREHM : public Diagnostics {
 public:
  Diagnostics_KREHM(Parameters *pars, Grids *grids);
  ~Diagnostics_KREHM();

  bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) ;
  void finish(MomentsG** G, Fields* fields, double time);  

private:
  float* P2(int s=0) {return &P2s[grids_->NxNycNz*s];}
  float* G2(int s=0) {return &G2s[grids_->NxNycNz*s];}

  int ndiag; 
  int ikx_local, iky_local, iz_local;
  dim3 dG_spectra, dB_spectra, dG_all, dB_all, dbp, dgp; //, dG_scale, dB_scale;
  dim3 dGk, dBk;
  bool checkstop();
 
  float Dks; 
  float fluxDenom; float * flux_fac; 
  float  volDenom; float * vol_fac ;
  float * kvol_fac;
  
  cuComplex valphi;

  GradParallel * grad_par      ;
  Fields       * fields_old    ;
  NetCDF_ids   * id            ;
  Reservoir    * rc            ;
  
  float        * G2s           ;
  float        * P2s           ;
  float        * Phi2          ;
  float        * val           ;
  cuComplex    * field_d       ;
  cuComplex    * field_h       ;
  cuComplex    * omg_d         ;
  cuComplex    * tmp_omg_h     ;
  cuComplex    * t_bar         ;
  cuComplex    * favg          ;
  cuComplex    * df            ;
  cuComplex    * amom_d        ;
  cuComplex    * vEk           ;
  float        * phi_max       ; 

  float *gy_d, *gy_h;
  double *ry_h;
  
  void print_omg (cuComplex *W);
  void get_rh    (Fields* f);
  void print_growth_rates_to_screen (cuComplex *w);
  void write_Wtot  (float   Wh, bool endrun);

  char stopfilename_[2000];
};

class Diagnostics_cetg : public Diagnostics {
 public:
  Diagnostics_cetg(Parameters *pars, Grids *grids, Geometry *geo);
  ~Diagnostics_cetg();

  bool loop(MomentsG** G, Fields* fields, double dt, int counter, double time) ;
  void finish(MomentsG** G, Fields* fields, double time);  

private:
  float* P2(int s=0) {return &P2s[grids_->NxNycNz*s];}
  float* G2(int s=0) {return &G2s[grids_->NxNycNz*s];}

  int ndiag; 
  int ikx_local, iky_local, iz_local;
  dim3 dG_spectra, dB_spectra, dG_all, dB_all, dbp, dgp; 
  dim3 dGk, dBk;
  bool checkstop();
 
  float fluxDenom; float * flux_fac; 
  float  volDenom; float * vol_fac ;
  
  Geometry     * geo_          ;  
  Fields       * fields_old    ;
  NetCDF_ids   * id            ;
  
  float        * G2s           ;
  float        * P2s           ;
  cuComplex    * omg_d         ;
  cuComplex    * tmp_omg_h     ;
  cuComplex    * vEk           ;

  void print_omg (cuComplex *W);
  void print_growth_rates_to_screen (cuComplex *w);

  char stopfilename_[2000];
};
