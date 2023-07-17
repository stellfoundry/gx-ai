#pragma once
#include "parameters.h"
#include "grids.h"
#include "geometry.h"
#include "moments.h"
#include "fields.h"
#include "ncdf.h"
#include "reductions.h"
#include "spectra_calc.h"
#include "device_funcs.h"
#include "netcdf_par.h"
#include "netcdf.h"
#include "get_error.h"
#include <string>

using namespace std;

// a SpectraDiagnostic is a diagnostic for which 
// spectra can be computed, i.e. a diagnostic quantity 
// that can be summed over various dimensions.
// for example, |Phi|**2 is a diagnostic quantity that 
// can be summed over various indices to get
// e.g. |Phi|**2(kx), |Phi|**2(ky), etc. 
class SpectraDiagnostic {
 public:
  ~SpectraDiagnostic() {};
  virtual void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf) = 0;
 protected:
  void add_spectra(SpectraCalc *spectra);
  void write_spectra(float* data);
  void set_kernel_dims();

  vector<SpectraCalc*> spectraList;
  vector<int> spectraIds;
  string varname;
  int nc_group, nc_type;
  bool isMoments;
  bool skipWrite = false;
  dim3 dG, dB;
  Parameters* pars_;
  Grids* grids_;
  Geometry* geo_;
  NetCDF* ncdf_;
};

// The following classes each define a particular diagnostic quantity

// |Phi|**2
class Phi2Diagnostic : public SpectraDiagnostic {
 public:
  Phi2Diagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// Wphi = (1-Gamma0(b_s))|Phi|**2
class WphiDiagnostic : public SpectraDiagnostic {
 public:
  WphiDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// Wg = |G|^2
class WgDiagnostic : public SpectraDiagnostic {
 public:
  WgDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// HeatFlux (Q)
class HeatFluxDiagnostic : public SpectraDiagnostic {
 public:
  HeatFluxDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

// ParticleFlux (Gamma)
class ParticleFluxDiagnostic : public SpectraDiagnostic {
 public:
  ParticleFluxDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* nc, AllSpectraCalcs* allSpectra);
  void calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf);
};

class GrowthRateDiagnostic {
 public:
  GrowthRateDiagnostic(Parameters* pars, Grids* grids, NetCDF* ncdf);
  ~GrowthRateDiagnostic();
  void calculate(Fields* f, Fields* f_old, double dt);
  void write();
 private:
  void dealias_and_reorder(cuComplex* fold, float* fnew);

  string tag;
  int ndim, N, Nwrite;
  int dims[6];
  size_t count[6] = {0};
  size_t start[6] = {0};
  int varid;

  string varname;
  int nc_group, nc_type;
  dim3 dG, dB;
  Parameters* pars_;
  Grids* grids_;
  NetCDF* ncdf_;

  cuComplex *omg_d, *omg_h;
  float *cpu;
};

class EigenfunctionDiagnostic {
 public:
  EigenfunctionDiagnostic(Parameters* pars, Grids* grids, NetCDF* ncdf);
  ~EigenfunctionDiagnostic();
  void calculate_and_write(Fields* f);
 private:
  void dealias_and_reorder(cuComplex* fold, float* fnew);

  string tag;
  int ndim, N, Nwrite;
  int dims[6];
  size_t count[6] = {0};
  size_t start[6] = {0};
  int varids[3];

  string varnames[3];
  int nc_group, nc_type;
  dim3 dG, dB;
  Parameters* pars_;
  Grids* grids_;
  NetCDF* ncdf_;

  cuComplex *f_h;
  float *cpu;
};

