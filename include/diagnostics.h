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

  bool loop_diagnostics(MomentsG* G, Fields* fields, double dt, int counter, double time) ;
  void final_diagnostics(MomentsG* G, Fields* fields);
  //  void close_netcdf_diag(NetCDF_ids* id);
  //  void write_something(NetCDF_ids* id);
  //  void close_netcdf_diag();
  void write_something();
  void reduce2k(float* fk, cuComplex* f);
  void reduce2z(float* fk, cuComplex* f);
  
  void pzt(MomentsG* G, Fields* f);
  void fluxes(MomentsG* G, Fields* f);
  void write_init(MomentsG* G, Fields* f);
  
  void writeMomOrField(cuComplex* m, int handle);
  void writeLHspectrum(MomentsG* G, bool endrun);
  void writeLspectrum (MomentsG* G, bool endrun);
  void writeHspectrum (MomentsG* G, bool endrun);
  void writeGrowthRates();
  void writeTimeHistory(cuComplex* f, float time, int i, int j, int k, FILE* out);
  
  float *pflux;
  float *qflux;
  float *primary;
  float *secondary;
  float *tertiary;
  
  //  cuDoubleComplex *growth_rates, *growth_rates_h;
  cuComplex *growth_rates, *growth_rates_h;

 private:
  Fields *fields_old;
  Red *red;
  GradParallel* grad_parallel;
  NetCDF_ids* id;
  cuComplex *t_bar;
  
  cuComplex *amom_h;
  cuComplex *amom;
  cuComplex valphi;
  float *rmom; 
  float *val;
  float *val1;
  
  int ikx_local;
  int iky_local;
  int iz_local;
     
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  
  int maxThreadsPerBlock_;
  dim3 dimGrid_xy, dimBlock_xy;
  dim3 dG_spec, dB_spec;
  
  void print_growth_rates_to_screen();

  void LHspectrum(MomentsG* G, float* f);
  void  Lspectrum(MomentsG* G, float* f);
  void  Hspectrum(MomentsG* G, float* f);

  bool checkstop();
 
  float fluxDenom;
  float fluxDenomInv;

  //  bool mask;

  char stopfilename_[2000];
  FILE* timefile;
};
