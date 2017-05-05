#pragma once

#include "moments.h"
#include "fields.h"
#include "grids.h"
#include "model.h"
#include "solver.h"

class Timestepper {
 public:
  virtual ~Timestepper() {};
  virtual int advance(double t, Moments* moms, Fields* fields) = 0;
};

class RungeKutta2 : public Timestepper {
 public:
  RungeKutta2(Model *model, Solver *solver, Grids *grids, const double dt_in);
  ~RungeKutta2();
  int advance(double t, Moments* moms, Fields* fields);
  double get_dt() {return dt_;};

 private:
  Model  *model_;
  Solver *solver_;
  Grids  *grids_;

  Moments* mStar;
  Moments* mRhs;
  double dt_;
};
