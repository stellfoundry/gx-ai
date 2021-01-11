#pragma once

#include "moments.h"
#include "fields.h"
#include "grids.h"
#include "linear.h"
#include "nonlinear.h"
#include "solver.h"
#include "forcing.h"

class Timestepper {
 public:
  virtual ~Timestepper() {};
  virtual int advance(double* t, MomentsG* G, Fields* fields) = 0;
  virtual double get_dt() = 0;
};

class RungeKutta2 : public Timestepper {
 public:
  RungeKutta2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~RungeKutta2();
  int advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};
  
 private:
  void EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* G, MomentsG* GRhs,
		 Fields* f, double adt, bool setdt);

  double dt_;
  const double dt_max;
  Linear     * linear_    = NULL;
  Nonlinear  * nonlinear_ = NULL;
  Solver     * solver_    = NULL;
  Parameters * pars_      = NULL;
  Grids      * grids_     = NULL;
  Forcing    * forcing_   = NULL;
  MomentsG   * GRhs       = NULL;
  MomentsG   * G1         = NULL;
};

class RungeKutta4 : public Timestepper {
 public:
  RungeKutta4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~RungeKutta4();
  int advance(double* t, MomentsG* G, Fields* fields);
  void partial(MomentsG* G, MomentsG* Gt, Fields *f,
	       MomentsG* Rhs, MomentsG *Gnew, double adt, bool setdt);
  double get_dt() {return dt_;};

 private:
  const double dt_max;
  double dt_;

  Linear     * linear_    = NULL;
  Nonlinear  * nonlinear_ = NULL;
  Solver     * solver_    = NULL;
  Parameters * pars_      = NULL;
  Grids      * grids_     = NULL;
  Forcing    * forcing_   = NULL;
  MomentsG   * GStar      = NULL;
  MomentsG   * GRhs       = NULL;
  MomentsG   * G_q1       = NULL;
  MomentsG   * G_q2       = NULL;
};
/*
class SDCe : public Timestepper {
 public:
  SDCe(Linear *linear, Nonlinear *nonlinear, Solver *solver,
       Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~SDCe();
  int advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};
  
 private:
  void full_rhs(MomentsG* G_q1, MomentsG* GRhs, Fields* f, MomentsG* GStar);
  Linear *linear_;
  Nonlinear *nonlinear_;
  Solver *solver_;
  Parameters *pars_;
  Grids *grids_;
  Forcing *forcing_;
  const double dt_max;
  
  double dt_;
}
*/

class Ketcheson10 : public Timestepper {
 public:
  Ketcheson10(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~Ketcheson10();
  int advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G_q1, MomentsG* GRhs, Fields* f,  bool setdt);
  const double dt_max;
  double dt_;

  Linear     * linear_    = NULL;
  Nonlinear  * nonlinear_ = NULL;
  Solver     * solver_    = NULL;
  Parameters * pars_      = NULL;
  Grids      * grids_     = NULL;
  Forcing    * forcing_   = NULL;
  MomentsG   * G_q1       = NULL;
  MomentsG   * G_q2       = NULL;
};

class K2 : public Timestepper {
 public:
  K2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~K2();
  int advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G_q1, MomentsG* GRhs, Fields* f, MomentsG* GStar, bool setdt);
  const double dt_max;
  double dt_;

  Linear     * linear_    = NULL;
  Nonlinear  * nonlinear_ = NULL;
  Solver     * solver_    = NULL;
  Parameters * pars_      = NULL;
  Grids      * grids_     = NULL;
  Forcing    * forcing_   = NULL;

  MomentsG   * GRhs       = NULL;
  MomentsG   * GStar      = NULL;
  MomentsG   * G_q1       = NULL;
  MomentsG   * G_q2       = NULL;
};

class SSPx2 : public Timestepper {
 public:
  SSPx2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~SSPx2();
  int advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* GRhs, Fields* f, bool setdt);
  const double dt_max;
  double dt_;

  Linear     * linear_    = NULL;
  Nonlinear  * nonlinear_ = NULL;
  Solver     * solver_    = NULL;
  Parameters * pars_      = NULL;
  Grids      * grids_     = NULL;
  Forcing    * forcing_   = NULL;
  MomentsG   * G1         = NULL;
  MomentsG   * G2         = NULL;
  MomentsG   * GRhs       = NULL;
};

class SSPx3 : public Timestepper {
 public:
  SSPx3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~SSPx3();
  int advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* GRhs, Fields* f, bool setdt);

  const double dt_max;
  const double adt = pow(1./6., 1./3.);
  const double wgtfac = sqrt(9. - 2.* pow(6.,2./3.));
  const double w1 = 0.5 * (wgtfac - 1.);
  const double w2 = 0.5 * (pow(6.,2./3.) - 1 - wgtfac);
  const double w3 = 1./adt - 1. - w2*(w1+1.);
 
  Linear      * linear_    = NULL;
  Nonlinear   * nonlinear_ = NULL;
  Solver      * solver_    = NULL;
  Parameters  * pars_      = NULL;
  Grids       * grids_     = NULL;
  Forcing     * forcing_   = NULL;
  MomentsG    * G1         = NULL;
  MomentsG    * G2         = NULL;
  MomentsG    * G3         = NULL;
  MomentsG    * GRhs       = NULL;
  double dt_;
};

