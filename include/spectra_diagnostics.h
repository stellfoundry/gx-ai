#pragma once
#include "grids.h"
#include "ncdf.h"
#include "reductions.h"
#include "netcdf_par.h"
#include "netcdf.h"
#include "get_error.h"
#include <string>

using namespace std;
class Spectra;

// a SpectraDiagnostic is a diagnostic for which 
// spectra can be computed, i.e. a diagnostic quantity 
// that can be summed over various dimensions.
// for example, |Phi|**2 is a diagnostic quantity that 
// can be summed over various indices to get
// e.g. |Phi|**2(kx), |Phi|**2(ky), etc. 
class SpectraDiagnostic {
 public:
  SpectraDiagnostic(string varname, int nc_group, int nc_type);
  ~SpectraDiagnostic() {};
  void add_spectra(Spectra *spectra);
  void write(float* data, bool isMoments);
 private:
  vector<Spectra> spectraList;
  vector<int> spectraIds;
  string varname;
  int nc_group, nc_type;
};

// base class for a particular kind of Spectra
class Spectra {
 public:
  ~Spectra();
  void allocate();
  int define_variable(string varstem, int nc_group);
  void write(float *fullData, int varid, int nc_group, bool isMoments);
  virtual void dealias_and_reorder(float *fold, float *fnew) {};

 protected:
  string tag;
  int ndim, N, Nwrite;
  int dims[6];
  size_t count[6] = {0};
  size_t start[6] = {0};

  Grids *grids_;
  Red *field_reduce, *moments_reduce;
 
  float *data, *tmp, *cpu;
};

// class for spectra of form f(ky, s, t)
class Spectra_kyst : public Spectra {
 public:
  Spectra_kyst(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};

// class for spectra of form f(kx, ky, s, t)
class Spectra_kxkyst : public Spectra {
 public:
  Spectra_kxkyst(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};

// class for spectra of form f(ky, t)
class Spectra_kyt : public Spectra {
 public:
  Spectra_kyt(Grids *grids, NcDims *nc_dims);
  void dealias_and_reorder(float *fold, float *fnew);
};
 
