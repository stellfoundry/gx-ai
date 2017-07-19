#pragma once

#include "grids.h"
#include "moments.h"
#include "geometry.h"
#include "parameters.h"

class Forcing {
  public:
   virtual ~Forcing() {};
   virtual void stir(Moments *m) = 0;
};

class ZForcing : public Forcing {
 public:
  ZForcing(Parameters *pars, Grids *grids, Geometry *geo); 
  ~ZForcing();
  void stir(Moments *m);
 
 private:
  float forcing_amp_;
  
  Parameters *pars_;
  Grids *grids_;
  Geometry *geo_;
 
  cuComplex *random_force;
  
  dim3 dimBlock, dimGrid;
};


