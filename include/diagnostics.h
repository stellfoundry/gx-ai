#pragma once
#include "device_funcs.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "fields.h"
#include "grad_parallel.h"
#include "ncdf.h"
#include "reductions.h"

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
  int maxThreadsPerBlock_;
  dim3 dimGrid_xy, dimBlock_xy, dG_spec, dB_spec, dG_all, dB_all, dG_scale, dB_scale;
  bool checkstop();
 
  float fluxDenom, fluxDenomInv;
  float  volDenom,  volDenomInv;
 
  cuComplex valphi;

  Parameters   * pars_         = NULL;
  Grids        * grids_        = NULL;
  Geometry     * geo_          = NULL;  

  Fields       * fields_old    = NULL;
  Red          * red           = NULL;
  Red          * pot           = NULL;
  Red          * all_red       = NULL;
  GradParallel * grad_parallel = NULL;
  NetCDF_ids   * id            = NULL;
  
  float        * primary       = NULL;
  float        * secondary     = NULL;
  float        * tertiary      = NULL;
  
  float        * amom_h        = NULL;
  float        * omg_h         = NULL;
  float        * G2            = NULL;
  float        * P2s           = NULL;
  float        * val           = NULL;
  float        * pflux         = NULL;
  float        * qflux         = NULL;
  cuComplex    * tmp_omg_h     = NULL;
  cuComplex    * omg_d         = NULL;
  cuComplex    * tmp_amom_h    = NULL;
  cuComplex    * amom_d        = NULL;
  cuComplex    * t_bar         = NULL;
       
  void fluxes(MomentsG* G, Fields* f, bool endrun);
  void freqs (Fields* f, Fields* f_old, double dt);
  void write_omg (cuComplex *W, bool endrun);
  void write_Q(float* Q, bool endrun);
  void writeMomOrField(cuComplex* m, nca lid);
  void print_growth_rates_to_screen(cuComplex *w);
  void pzt(MomentsG* G, Fields* f);
  void get_rh(Fields* f);
  void reduce2k(float* fk, cuComplex* f);
  void reduce2z(float* fk, cuComplex* f);
  void write_nc    (int ncid, nca D, const float  *f, const bool endrun);
  void write_nc    (int ncid, nca D, const double  f, const bool endrun);

  void write_Wm    (float * G2, bool endrun);
  void write_Wl    (float * G2, bool endrun);
  void write_Wlm   (float * G2, bool endrun);

  void write_Wtot  (float * Wh, bool endrun);
  void write_Ws    (float * G2, bool endrun);
  void write_Wz    (float * G2, bool endrun);
  void write_Wky   (float * G2, bool endrun);
  void write_Wkx   (float * G2, bool endrun);
  void write_Wkxky (float * G2, bool endrun);

  void write_Ps    (float * P2, bool endrun);
  void write_Pz    (float * P2, bool endrun);
  void write_Pky   (float * P2, bool endrun);
  void write_Pkx   (float * P2, bool endrun);
  void write_Pkxky (float * P2, bool endrun);
  
  float * Wm_d        = NULL;
  float * Wm_h        = NULL;
  float * Wl_d        = NULL;
  float * Wl_h        = NULL;
  float * Wlm_d       = NULL;
  float * Wlm_h       = NULL;
  float * Ws_d        = NULL;
  float * Ws_h        = NULL;
  float * Wz_d        = NULL;
  float * Wz_h        = NULL;
  float * Wky_d       = NULL;
  float * Wky_h       = NULL;
  float * tmp_Wky_h   = NULL;
  float * Wkx_d       = NULL;
  float * Wkx_h       = NULL;
  float * tmp_Wkx_h   = NULL;
  float * Ps_d        = NULL;
  float * Ps_h        = NULL;
  float * Pz_d        = NULL;
  float * Pz_h        = NULL;
  float * Pky_d       = NULL;
  float * Pky_h       = NULL;
  float * tmp_Pky_h   = NULL;
  float * Pkx_d       = NULL;
  float * Pkx_h       = NULL;
  float * tmp_Pkx_h   = NULL;
  float * Wkxky_d     = NULL;
  float * Wkxky_h     = NULL;
  float * tmp_Wkxky_h = NULL;
  float * Pkxky_d     = NULL;
  float * Pkxky_h     = NULL;
  float * tmp_Pkxky_h = NULL;
  float * qs_d        = NULL;
  float * qs_h        = NULL;
  float * Wtot_h      = NULL;
   
  char stopfilename_[2000];
};
