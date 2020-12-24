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

  Fields *fields_old;
  Red *red, *pot, *all_red;
  GradParallel* grad_parallel;
  NetCDF_ids* id;
  cuComplex *t_bar;
  
  float *primary;
  float *secondary;
  float *tertiary;
  
  float *amom_h;
  cuComplex *tmp_amom_h, *amom_d;
  cuComplex valphi;
  float *G2, *P2s;
  float *val, *pflux, *qflux;
  cuComplex *omg_d, *tmp_omg_h;
  float *omg_h;
  
  int ikx_local, iky_local, iz_local;;
     
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  
  int maxThreadsPerBlock_;
  dim3 dimGrid_xy, dimBlock_xy;
  dim3 dG_spec, dB_spec;
  dim3 dG_all, dB_all;
  dim3 dG_scale, dB_scale;
  
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

  void write_Wm    (float* G2, bool endrun);
  void write_Wl    (float* G2, bool endrun);
  void write_Wlm   (float* G2, bool endrun);

  void write_Wtot  (float* Wh, bool endrun);
  void write_Ws    (float* G2, bool endrun);
  void write_Wz    (float* G2, bool endrun);
  void write_Wky   (float* G2, bool endrun);
  void write_Wkx   (float* G2, bool endrun);
  void write_Wkxky (float* G2, bool endrun);

  void write_Ps    (float* P2, bool endrun);
  void write_Pz    (float* P2, bool endrun);
  void write_Pky   (float* P2, bool endrun);
  void write_Pkx   (float* P2, bool endrun);
  void write_Pkxky (float* P2, bool endrun);
  
  
  bool checkstop();
 
  float fluxDenom, fluxDenomInv;
  float  volDenom,  volDenomInv;

  float *Wm_d, *Wm_h, *Wl_d, *Wl_h, *Wlm_d, *Wlm_h;
  float *Ws_d, *Ws_h, *Wz_d, *Wz_h, *Wky_d, *Wky_h, *tmp_Wky_h, *Wkx_d, *Wkx_h, *tmp_Wkx_h; 
  float *Ps_d, *Ps_h, *Pz_d, *Pz_h, *Pky_d, *Pky_h, *tmp_Pky_h, *Pkx_d, *Pkx_h, *tmp_Pkx_h;
  float *Wkxky_d, *Wkxky_h, *tmp_Wkxky_h;  
  float *Pkxky_d, *Pkxky_h, *tmp_Pkxky_h;
  float *qs_d, *qs_h;
  float *Wtot_h;
   
  char stopfilename_[2000];
  FILE* timefile;
};
