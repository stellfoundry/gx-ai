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
  float totW; 

private:
  float* P2(int s=0) {return &P2s[grids_->NxNycNz*s];}

  int ikx_local, iky_local, iz_local;;
  int maxThreadsPerBlock_;
  dim3 dimGrid_xy, dimBlock_xy, dG_spec, dB_spec, dG_all, dB_all, dG_scale, dB_scale;
  bool checkstop();
 
  float fluxDenom, fluxDenomInv;
  float  volDenom,  volDenomInv;
 
  cuComplex valphi;

  Parameters   * pars_         ;
  Grids        * grids_        ;
  Geometry     * geo_          ;  

  Fields       * fields_old    ;
  Red          * red           ;
  Red          * pot           ;
  Red          * ph2           ;
  Red          * all_red       ;
  GradParallel * grad_parallel ;
  NetCDF_ids   * id            ;
  
  float        * primary       ;
  float        * secondary     ;
  float        * tertiary      ;
  
  float        * amom_h        ;
  float        * omg_h         ;
  float        * G2            ;
  float        * P2s           ;
  float        * Phi2          ;
  float        * val           ;
  float        * pflux         ;
  float        * qflux         ;
  cuComplex    * tmp_omg_h     ;
  cuComplex    * omg_d         ;
  cuComplex    * tmp_amom_h    ;
  cuComplex    * amom_d        ;
  cuComplex    * t_bar         ;
       
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

  void write_Wtot  (float   Wh, bool endrun);
  void write_Ws    (float * G2, bool endrun);
  void write_Wz    (float * G2, bool endrun);
  void write_Wky   (float * G2, bool endrun);
  void write_Wkx   (float * G2, bool endrun);
  void write_Wkxky (float * G2, bool endrun);

  void write_As    (float * P2, bool endrun);
  void write_Az    (float * P2, bool endrun);
  void write_Aky   (float * P2, bool endrun);
  void write_Akx   (float * P2, bool endrun);
  void write_Akxky (float * P2, bool endrun);

  void write_Ps    (float * P2, bool endrun);
  void write_Pz    (float * P2, bool endrun);
  void write_Pky   (float * P2, bool endrun);
  void write_Pkx   (float * P2, bool endrun);
  void write_Pkxky (float * P2, bool endrun);
  
  float * Wm_d        ;
  float * Wm_h        ;
  float * Wl_d        ;
  float * Wl_h        ;
  float * Wlm_d       ;
  float * Wlm_h       ;

  float * Ws_d        ;
  float * Ws_h        ;
  float * Wz_d        ;
  float * Wz_h        ;
  float * Wky_d       ;
  float * Wky_h       ;
  float * tmp_Wky_h   ;
  float * Wkx_d       ;
  float * Wkx_h       ;
  float * tmp_Wkx_h   ;
  float * Wkxky_d     ;
  float * Wkxky_h     ;
  float * tmp_Wkxky_h ;

  float * Ps_d        ;
  float * Ps_h        ;
  float * Pz_d        ;
  float * Pz_h        ;
  float * Pky_d       ;
  float * Pky_h       ;
  float * tmp_Pky_h   ;
  float * Pkx_d       ;
  float * Pkx_h       ;
  float * tmp_Pkx_h   ;
  float * Pkxky_d     ;
  float * Pkxky_h     ;
  float * tmp_Pkxky_h ;

  float * As_d        ;
  float * As_h        ;
  float * Az_d        ;
  float * Az_h        ;
  float * Aky_d       ;
  float * Aky_h       ;
  float * tmp_Aky_h   ;
  float * Akx_d       ;
  float * Akx_h       ;
  float * tmp_Akx_h   ;
  float * Akxky_d     ;
  float * Akxky_h     ;
  float * tmp_Akxky_h ;

  float * qs_d        ;
  float * qs_h        ;
   
  char stopfilename_[2000];
};
