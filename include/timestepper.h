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
		 Fields* f, MomentsG* GStar, double adt, bool setdt);
  Linear  *linear_;
  Nonlinear *nonlinear_;
  Solver *solver_;
  Parameters *pars_;
  Grids  *grids_;
  Forcing *forcing_;
  const double dt_max;
  
  MomentsG* GStar;
  MomentsG* GRhs;
  MomentsG* G1;
  double dt_;
};

class RungeKutta3 : public Timestepper {
 public:
  RungeKutta3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~RungeKutta3();
  int advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};
  
 private:
  Linear  *linear_;
  Nonlinear *nonlinear_;
  Solver *solver_;
  Parameters *pars_;
  Grids  *grids_;
  Forcing *forcing_;
  const double dt_max;

  MomentsG* GRhs1;
  MomentsG* GRhs2;
  MomentsG* GStar;
  double dt_;
};

class RungeKutta4 : public Timestepper {
 public:
  RungeKutta4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	      Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~RungeKutta4();
  int advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  Linear  *linear_;
  Nonlinear *nonlinear_;
  Solver *solver_;
  Parameters *pars_;
  Grids  *grids_;
  Forcing *forcing_;
  const double dt_max;

  MomentsG* GStar;
  MomentsG* GRhs1;
  MomentsG* GRhs3;
  MomentsG* GRhs4;
  double dt_;
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
  void EulerStep(MomentsG* G_q1, MomentsG* GRhs, Fields* f, MomentsG* GStar, bool setdt);
  Linear  *linear_;
  Nonlinear *nonlinear_;
  Solver *solver_;
  Parameters *pars_;
  Grids  *grids_;
  Forcing *forcing_;
  const double dt_max;

  MomentsG* GRhs;
  MomentsG* GStar;
  MomentsG* G_q1;
  MomentsG* G_q2;
  double dt_;
};

class SSPx2 : public Timestepper {
 public:
  SSPx2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	Parameters *pars, Grids *grids, Forcing *forcing, double dt_in);
  ~SSPx2();
  int advance(double* t, MomentsG* G, Fields* fields);
  double get_dt() {return dt_;};

 private:
  void EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* GRhs, Fields* f, MomentsG* GStar, bool setdt);
  Linear  *linear_;
  Nonlinear *nonlinear_;
  Solver *solver_;
  Parameters *pars_;
  Grids  *grids_;
  Forcing *forcing_;
  const double dt_max;

  MomentsG* G1;
  MomentsG* GStar;
  MomentsG* GRhs;
  double dt_;
};
  
