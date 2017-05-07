#include "timestepper.h"
#include "get_error.h"

RungeKutta2::RungeKutta2(Linear *linear, Solver *solver, Grids *grids, const double dt_in) :
  linear_(linear), solver_(solver), grids_(grids)
{
  // new objects for temporaries
  mStar = new Moments(grids);
  mRhs = new Moments(grids);
  dt_ = dt_in;
}

RungeKutta2::~RungeKutta2()
{
  delete mStar;
  delete mRhs;
}

int RungeKutta2::advance(double *t, Moments* m, Fields* f) {
  linear_->rhs(m, f, mRhs);
  //if(linear_->nonlinear) dt_ = linear_->cfl();
  mStar->add_scaled(1., m, dt_/2., mRhs);
  solver_->fieldSolve(mStar, f);

  linear_->rhs(mStar, f, mRhs);
  mStar->add_scaled(1., m, dt_, mRhs);
  m->copyFrom(mStar);
  solver_->fieldSolve(m, f);
  *t+=dt_;
  return 0;
}
