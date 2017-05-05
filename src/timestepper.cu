#include "timestepper.h"
#include "get_error.h"

RungeKutta2::RungeKutta2(Model *model, Solver *solver, Grids *grids, const double dt_in) :
  model_(model), solver_(solver), grids_(grids)
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

int RungeKutta2::advance(double t, Moments* m, Fields* f) {
  model_->rhs(m, f, mRhs);
  //if(model_->nonlinear) dt_ = model_->cfl();
  mStar->add_scaled(1., m, dt_/2., mRhs);
  solver_->fieldSolve(mStar, f);

  model_->rhs(mStar, f, mRhs);
  mStar->add_scaled(1., m, dt_, mRhs);
  m->copyFrom(mStar);
  solver_->fieldSolve(m, f);
  t+=dt_;
  return 0;
}
