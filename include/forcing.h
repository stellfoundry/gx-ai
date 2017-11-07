#pragma once

#include "grids.h"
#include "moments.h"
#include "geometry.h"
#include "parameters.h"

class Forcing {
  public:
   virtual ~Forcing() {};
   virtual void stir(MomentsG *G) = 0;
};

class KzForcing : public Forcing {
 public:
  KzForcing(Parameters *pars, Grids *grids, Geometry *geo); 
  ~KzForcing();
  void stir(MomentsG *G);

 private:
  float forcing_amp_;
  cuComplex random_force;

  Parameters *pars_;
  Grids *grids_;
  Geometry *geo_;
};


