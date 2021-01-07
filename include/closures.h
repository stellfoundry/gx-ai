#pragma once

#include "grids.h"
#include "parameters.h"
#include "moments.h"
#include "grad_parallel.h"
#include "geometry.h"

class Closures {
 public:
  virtual ~Closures() {};
  virtual int apply_closures(MomentsG* G, MomentsG* GRhs) = 0;
};

class Beer42 : public Closures {
 public:
  Beer42(Parameters* pars, Grids* grids, Geometry* geo, GradParallel* grad_par);
  ~Beer42();
  int apply_closures(MomentsG* G, MomentsG* GRhs);

 private:
  float gpar_;
  Grids        * grids_   = NULL;
  Parameters   * pars_    = NULL;
  GradParallel * grad_par = NULL;
  float        * omegad_  = NULL;
  cuComplex    * tmp      = NULL;

  // closure coefficients
  float Beta_par;
  float D_par;
  float D_perp;
  cuComplex * nu = NULL;

  dim3 dimGrid, dimBlock;
};

class SmithPerp : public Closures {
 public: 
  SmithPerp(Parameters* pars, Grids* grids, Geometry* geo);
  ~SmithPerp();
  int apply_closures(MomentsG* G, MomentsG* GRhs);

 private:
  Grids      * grids_  = NULL;
  Parameters * pars_   = NULL;
  float      * omegad_ = NULL;
  
  // closure coefficent array, to be allocated
  cuComplex * Aclos_   = NULL;
  int q_;

  dim3 dimGrid, dimBlock; 
};

class SmithPar : public Closures {
 public: 
  SmithPar(Parameters* pars, Grids* grids, Geometry* geo, GradParallel* grad_par);
  ~SmithPar();
  int apply_closures(MomentsG* G, MomentsG* GRhs);

 private:
  dim3 dimGrid, dimBlock;  
  int q_;
  float gpar_;
  Grids        * grids_   = NULL;
  Parameters   * pars_    = NULL;
  GradParallel * grad_par = NULL;
  cuComplex    * tmp      = NULL;
  cuComplex    * tmp_abs  = NULL;

  // closure coefficent array
  cuComplex * a_coefficients_ = NULL;

  // closure array
  cuComplex * clos = NULL;
};
