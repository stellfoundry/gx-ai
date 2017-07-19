#pragma once

#include "moments.h"
#include "fields.h"
#include "grids.h"
#include "linear.h"
#include "solver.h"
#include "forcing.h"

class Timestepper {
 public:
  virtual ~Timestepper() {};
  virtual int advance(double* t, Moments* moms, Fields* fields) = 0;
  virtual double get_dt() = 0;
};

class RungeKutta2 : public Timestepper {
 public:
  RungeKutta2(Linear *linear, Solver *solver, Grids *grids, Forcing *forcing, const double dt_in);
  ~RungeKutta2();
  int advance(double* t, Moments* moms, Fields* fields);
  double get_dt() {return dt_;};

 private:
  Linear  *linear_;
  Solver *solver_;
  Grids  *grids_;
  Forcing *forcing_;

  Moments* mStar;
  Moments* mRhs;
  double dt_;
};

class RungeKutta4 : public Timestepper {
 public:
  RungeKutta4(Linear *linear, Solver *solver, Grids *grids, Forcing *forcing, const double dt_in);
  ~RungeKutta4();
  int advance(double* t, Moments* moms, Fields* fields);
  double get_dt() {return dt_;};

 private:
  Linear  *linear_;
  Solver *solver_;
  Grids  *grids_;
  Forcing *forcing_;

  Moments* mStar;
  Moments* mRhs1;
  Moments* mRhs2;
  Moments* mRhs3;
  Moments* mRhs4;
  double dt_;
};
