#pragma once

#include "grids.h"
#include "moments.h"
#include "grad_parallel.h"
#include "geometry.h"

class Closures {
 public:
  virtual ~Closures() {};
  virtual int apply_closures(Moments* m, Moments* mRhs) = 0;
};

class Beer42 : public Closures {
 public:
  Beer42(Grids* grids, const Geometry* geo);
  ~Beer42();
  int apply_closures(Moments* m, Moments* mRhs);

 private:
  Grids* grids_;
  GradParallel* grad_par;
  GradParallel* abs_grad_par;

  float* omegad_;
  float gradpar_;

  cuComplex* tmp;

  // closure coefficients
  float Beta_par;
  float D_par;
  float D_perp;
  cuComplex* nu;


  dim3 dimGrid, dimBlock;
};

class SmithPerp : public Closures {
 public: 
  SmithPerp(Grids* grids, const Geometry* geo, int q, cuComplex w0);
  ~SmithPerp();
  int apply_closures(Moments* m, Moments* mRhs);

 private:
  Grids* grids_;
  float* omegad_;
  
  // closure coefficent array, to be allocated
  cuComplex* Aclos_;
  const int q_;

  dim3 dimGrid, dimBlock;
 
};
