#include "timestepper.h"
#include "get_error.h"

// ======= RK2 =======
RungeKutta2::RungeKutta2(Linear *linear, Nonlinear *nonlinear, Solver *solver, Grids *grids, Forcing *forcing, const double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), forcing_(forcing), dt_(dt_in), dt_max(dt_in)
{
  // new objects for temporaries
  GStar = new MomentsG(grids);
  GRhs = new MomentsG(grids);
}

RungeKutta2::~RungeKutta2()
{
  delete GStar;
  delete GRhs;
}

int RungeKutta2::advance(double *t, MomentsG* G, Fields* f) {
  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(G, f, GRhs);
    dt_ = nonlinear_->cfl(dt_max);
    GStar->add_scaled(1., G, dt_/2., GRhs);
    linear_->rhs(G, f, GRhs);
    GStar->add_scaled(1., GStar, dt_/2., GRhs);
  } else {
    linear_->rhs(G, f, GRhs);
    GStar->add_scaled(1., G, dt_/2., GRhs);
  }
  solver_->fieldSolve(GStar, f);

  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(GStar, f, GRhs);
    G->add_scaled(1., G, dt_, GRhs);
  }
  linear_->rhs(GStar, f, GRhs);
  G->add_scaled(1., G, dt_, GRhs);

  if (forcing_ != NULL) {
    forcing_->stir(G);
  }
  solver_->fieldSolve(G, f);
  *t+=dt_;
  return 0;
}

// ============= RK4 =============
RungeKutta4::RungeKutta4(Linear *linear, Solver *solver, Grids *grids, Forcing *forcing, const double dt_in) :
  linear_(linear), solver_(solver), grids_(grids), forcing_(forcing)
{
  // new objects for temporaries
  GStar = new MomentsG(grids);
  GRhs1 = new MomentsG(grids);
  GRhs2 = new MomentsG(grids);
  GRhs3 = new MomentsG(grids);
  GRhs4 = new MomentsG(grids);
  dt_ = dt_in;
}

RungeKutta4::~RungeKutta4()
{
  delete GStar;
  delete GRhs1;
  delete GRhs2;
  delete GRhs3;
  delete GRhs4;
}

int RungeKutta4::advance(double *t, MomentsG* G, Fields* f) {
  linear_->rhs(G, f, GRhs1);
  GStar->add_scaled(1., G, dt_/2., GRhs1);
  solver_->fieldSolve(GStar, f);

  linear_->rhs(GStar, f, GRhs2);
  GStar->add_scaled(1., G, dt_/2., GRhs2);
  solver_->fieldSolve(GStar, f);

  linear_->rhs(GStar, f, GRhs3);
  GStar->add_scaled(1., G, dt_/2., GRhs3);
  solver_->fieldSolve(GStar, f);

  linear_->rhs(GStar, f, GRhs4);
  GStar->add_scaled(1., G, dt_/6., GRhs1, dt_/3., GRhs2, dt_/3., GRhs3, dt_/6., GRhs4);
 
  if (forcing_ != NULL) {
    forcing_->stir(GStar);
  }
  
  G->copyFrom(GStar);
  solver_->fieldSolve(G, f);
  *t+=dt_;
  return 0;
}
