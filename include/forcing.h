#pragma once
#include "moments.h"
#include "parameters.h"
#include "device_funcs.h"

class Forcing {
  public:
   virtual ~Forcing() {};
   virtual void stir(MomentsG *G) = 0;
};

class KzForcing : public Forcing {
 public:
  KzForcing(Parameters *pars);
  ~KzForcing();
  void stir(MomentsG *G);

 protected:
  float forcing_amp_;
  cuComplex rf;
  Parameters * pars_ ;

};

class KzForcingImpulse : public KzForcing {
 public:
  KzForcingImpulse(Parameters *pars);
    void stir(MomentsG *G);

    private:
    bool stirring_done;
};

class genForcing : public Forcing {
 public:
  genForcing(Parameters *pars);
  ~genForcing();
  void stir(MomentsG *G);

 protected:
  float forcing_amp_;
  cuComplex rf;
  Parameters * pars_ ;
};


