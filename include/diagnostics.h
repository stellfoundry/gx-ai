#pragma once
#include "device_funcs.h"
// #include "gx_lib.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "fields.h"
#include "ncdf.h"
#include "reductions.h"
#include "grad_parallel.h"

class Diagnostics {
 public:
  Diagnostics(Parameters *pars, Grids *grids, Geometry *geo);
  ~Diagnostics();

  bool loop(MomentsG* G, Fields* fields, double dt, int counter, double time) ;
  void finish(MomentsG* G, Fields* fields);  
  void write_init(MomentsG* G, Fields* f);

private:
  float* P2(int s=0) {return &P2s[grids_->NxNycNz*s];}

  int ikx_local, iky_local, iz_local;;
  dim3 dG_spectra, dB_spectra, dG_all, dB_all; //, dG_scale, dB_scale;
  bool checkstop();
 
  float fluxDenom; float * flux_fac; 
  float  volDenom; float * vol_fac ;
  float * kvol_fac;
 
  cuComplex valphi;

  Parameters   * pars_         ;
  Grids        * grids_        ;
  Geometry     * geo_          ;  

  GradParallel * grad_par      ;
  Fields       * fields_old    ;
  NetCDF_ids   * id            ;
  
  float        * G2            ;
  float        * P2s           ;
  float        * Phi2          ;
  float        * val           ;
  cuComplex    * omg_d         ;
  cuComplex    * tmp_omg_h     ;
  cuComplex    * t_bar         ;
  cuComplex    * favg          ;
  cuComplex    * df            ;
  cuComplex    * amom_d        ; 
  //  void write_omg (cuComplex *W, bool endrun);
  void print_omg (cuComplex *W);
  void get_rh    (Fields* f);
  //  void reduce2z  (float* fk, cuComplex* f);
  //  void writeMomOrField(cuComplex* m, nca lid);
  void print_growth_rates_to_screen (cuComplex *w);

  void write_Wtot  (float   Wh, bool endrun);
  //  void pzt(MomentsG* G, Fields* f);

  char stopfilename_[2000];
};
