#include "timestepper.h"
#include "get_error.h"

// ======= RK2 =======
RungeKutta2::RungeKutta2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), 
  forcing_(forcing), dt_max(dt_in), dt_(dt_in)
{
  // new objects for temporaries
  GStar = new MomentsG(pars, grids);
  GRhs  = new MomentsG(pars, grids);
}

RungeKutta2::~RungeKutta2()
{
  delete GStar;
  delete GRhs;
}

int RungeKutta2::advance(double *t, MomentsG* G, Fields* f)
{
  linear_->rhs(G, f, GRhs);
  if(nonlinear_ != NULL) {

    nonlinear_->nlps5d(G, f, GStar);
    GRhs->add_scaled(1., GRhs, 1., GStar);

    // choose the timestep
    dt_ = nonlinear_->cfl(dt_max);
  }
  
  GStar->add_scaled(1., G, dt_/2., GRhs);
  solver_->fieldSolve(GStar, f);

  //////////////////////////////////////////
  //////////////////////////////////////////
  //////////////////////////////////////////
  
  linear_->rhs(GStar, f, GRhs);
      
  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(GStar, f, GStar);

    // In slab test case the RHS is zero
    //    printf("GRhs: (should be zero) \n");
    //    GRhs->qvar(grids_->Nmoms*grids_->NxNycNz);

    GRhs->add_scaled(1., GRhs, 1., GStar);

    // There should be values here
    //    printf("Gnl: \n");
    //    Gnl->qvar(grids_->Nmoms*grids_->NxNycNz);

    // And now GRhs should = Gnl
    //    printf("GRhs: \n");
    //    GRhs->qvar(grids_->Nmoms*grids_->NxNycNz);
  }

  G->add_scaled(1., G, dt_, GRhs);

  if (forcing_ != NULL) forcing_->stir(G);  

  solver_->fieldSolve(G, f);

  *t += dt_;
  return 0;
}

// ============= RK4 =============
// why is dt double precision? 
RungeKutta4::RungeKutta4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), 
  forcing_(forcing), dt_max(dt_in), dt_(dt_in)
{
  // new objects for temporaries
  GStar = new MomentsG(pars, grids);
  GRhs1 = new MomentsG(pars, grids);
  GRhs3 = new MomentsG(pars, grids);
  GRhs4 = new MomentsG(pars, grids);
}


RungeKutta4::~RungeKutta4()
{
  delete GStar;
  delete GRhs1;
  delete GRhs3;
  delete GRhs4;
}

int RungeKutta4::advance(double *t, MomentsG* G, Fields* f)
{
  // do not check cfl for now:
  //  dt_ = ... 
  
  //////////////////////////////////////////
  /// In: G                             ////
  /// Out: GRhs1, GRhs3_(GStar)         ////
  //////////////////////////////////////////

  // GRhs1 = RHS(t)
  linear_->rhs(G, f, GRhs1);

  if (nonlinear_ != NULL) {  
    nonlinear_->nlps5d(G, f, GRhs3);         // GRhs3 is a tmp array here
    GRhs1->add_scaled(1., GRhs1, 1., GRhs3); // GRhs3 is a tmp array here

      // choose the timestep
    dt_ = nonlinear_->cfl(dt_max);
  }
  
  // GStar = G(t) + RHS(t) * dt/2 = G(t+dt/2) (first approximation)
  GRhs3->add_scaled(1., G, dt_/2., GRhs1);  // GRhs3 is a tmp array here
  solver_->fieldSolve(GRhs3, f);            // GRhs3 is a tmp array here

  //////////////////////////////////////////
  /// In: G, GRhs3_(GStar)              /////// ???
  /// In: G, Gstar_(GRhs3)              ////
  /// Out: GStar_(GRhs2), GRhs4_(GStar) ////
  //////////////////////////////////////////

  // GRhs2 = RHS(t+dt/2)  (using first approx from t+dt/2)
  linear_->rhs(GRhs3, f, GStar);            // GRhs3 is a tmp array here

  if (nonlinear_ != NULL) {  
    nonlinear_->nlps5d(GRhs3, f, GRhs4);    // GRhs3 and GRhs4 are tmp arrays here
    GStar->add_scaled(1., GStar, 1., GRhs4);// GRhs3 and GRhs4 are tmp arrays here
  }
  
  // GStar = G + linear(t+dt/2) * dt/2   (this is second approx for t+dt/2)
  GRhs4->add_scaled(1., G, dt_/2., GStar);  // GRhs4 is a tmp array here
  solver_->fieldSolve(GRhs4, f);            // GRhs4 is a tmp array here

  /////////////////////////////////////////////////
  /// First: GRhs1 == dt/6 GRhs1 + dt/3 Gstar  ////
  /// In: G, GRhs4_(GStar)                     ////
  /// Out: GRhs3, GStar                        ////
  /////////////////////////////////////////////////

  // Do a partial accumulation of final update to save memory
  GRhs1->add_scaled(dt_/6., GRhs1, dt_/3., GStar);
  // GStar is now available as a tmp array
  
  // GRhs3 = linear(t+dt/2)  (using second approx from t+dt/2)
  linear_->rhs(GRhs4, f, GRhs3);            // GRhs4 is a tmp array here

  if (nonlinear_ != NULL) {  
    nonlinear_->nlps5d(GRhs4, f, GStar);    // GRhs4 is a tmp array here
    GRhs3->add_scaled(1., GRhs3, 1., GStar);
  }
  // GStar = G + linear(t+dt/2) * dt  (this is first approx at t+dt)
  GStar->add_scaled(1., G, dt_, GRhs3);
  solver_->fieldSolve(GStar, f);       

  /////////////////////////////////////////////////
  /// In: G, GStar                             ////
  /// Out: GRhs4                               ////
  /////////////////////////////////////////////////

  // GRhs4 = linear(t+dt)  (using first approx from t+dt)
  linear_->rhs(GStar, f, GRhs4);             

  if(nonlinear_ != NULL) {  
    nonlinear_->nlps5d(GStar, f, GStar);     
    GRhs4->add_scaled(1., GRhs4, 1., GStar); 
  }
  
  /////////////////////////////////////////////////
  /// In: G, GRhs1, GRhs3, GRhs4               ////
  /// Out: G                                   ////
  /////////////////////////////////////////////////

  // G = G + linear(t,            ) dt/6
  //       + linear(t+dt/2, first ) dt/3
  //       + linear(t+dt/2, second) dt/3
  //       + linear(t+dt,   first ) dt/6
  //  G->add_scaled(1., G, dt_/6., GRhs1, dt_/3., GRhs2, dt_/3., GRhs3, dt_/6., GRhs4);
  G->add_scaled(1., G, 1., GRhs1, dt_/3., GRhs3, dt_/6., GRhs4);
 
  if (forcing_ != NULL) forcing_->stir(G);
  
  solver_->fieldSolve(G, f);
  *t += dt_;
  return 0;
}

// ============= RK3 SSP =============
RungeKutta3::RungeKutta3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), 
  forcing_(forcing), dt_max(dt_in), dt_(dt_in)
{
  // new objects for temporaries
  GRhs1 = new MomentsG(pars, grids);
  GRhs2 = new MomentsG(pars, grids);
  if(nonlinear_ != NULL) GStar = new MomentsG(pars, grids);
}

RungeKutta3::~RungeKutta3()
{
  delete GRhs1;
  delete GRhs2;
  if(nonlinear_ != NULL) delete GStar;
}

int RungeKutta3::advance(double *t, MomentsG* G, Fields* f)
{

  // need to include nonlinear terms for RK3

  // GRhs1 = linear(t)
  linear_->rhs(G, f, GRhs1);

  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(G, f, GStar);        
    GRhs1->add_scaled(1., GRhs1, 1., GStar);

    // choose the timestep
    dt_ = nonlinear_->cfl(dt_max);
  }    
  // GRhs1 = G(t) + linear(t) * dt  
  GRhs1->add_scaled(1., G, dt_, GRhs1);
  solver_->fieldSolve(GRhs1, f);

  /////////////////////////////////////////////
  /////////////////////////////////////////////
  /////////////////////////////////////////////
  
  // GRhs2 = RHS(g1)  
  linear_->rhs(GRhs1, f, GRhs2);

  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(GRhs1, f, GStar);    
    GRhs2->add_scaled(1., GRhs2, 1., GStar);
  }    

  // GRhs2 = g1 + RHS(g1) * dt
  GRhs2->add_scaled(1., GRhs1, dt_, GRhs2);
  solver_->fieldSolve(GRhs2, f);

  /////////////////////////////////////////////
  /////////////////////////////////////////////
  /////////////////////////////////////////////
  
  // save memory by getting partial sum, freeing GRhs1:
  G->add_scaled(1./3, G, 1./2., GRhs1);
  
  // reuse GRhs1 as tmp array
  // GRhs1 = linear(g2)
  linear_->rhs(GRhs2, f, GRhs1); // GRhs1 is used for GRhs3

  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(GRhs2, f, GStar);    
    GRhs1->add_scaled(1., GRhs1, 1., GStar);
  }
  // GRhs3 = g2 + linear(g2) * dt
  GRhs1->add_scaled(1., GRhs2, dt_, GRhs1);

  G->add_scaled(1., G, 1./6., GRhs1);

  if (forcing_ != NULL) forcing_->stir(G);
  
  solver_->fieldSolve(G, f);
  *t += dt_;
  return 0;
}


// ============= K10,4 ============
Ketcheson10::Ketcheson10(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in)
{
  // new objects for temporaries
  G_q1 = new MomentsG(pars, grids);
  G_q2 = new MomentsG(pars, grids);
  GRhs = new MomentsG(pars, grids);
  GStar = new MomentsG(pars, grids);
}

Ketcheson10::~Ketcheson10()
{
  delete G_q1;
  delete G_q2;
  delete GRhs;
  delete GStar;
}

int Ketcheson10::advance(double *t, MomentsG* G, Fields* f)
{

  G_q1->copyFrom(G);
  G_q2->copyFrom(G);

  // Need to make f consistent with G here? No, it starts out that way
  
  // First five iterations
  for(int i=1; i<6; i++) {

    linear_->rhs(G_q1, f, GRhs);
    
    if(nonlinear_ != NULL) {
      nonlinear_->nlps5d(G_q1, f, GStar);    

      GRhs->add_scaled(1., GRhs, 1., GStar);

      //            printf("GStar (K10):\n");
      //            GStar->qvar(grids_->Nmoms*grids_->NxNycNz);

	    //
      //      printf("GRhs (K10):\n");
      //      GRhs->qvar(grids_->Nmoms*grids_->NxNycNz);

      // choose the timestep
    dt_ = nonlinear_->cfl(dt_max);
    }

    G_q1->add_scaled(1., G_q1, dt_/6., GRhs);

    //    printf("another G_q1 (K10):\n");
    //    G_q1->qvar(grids_->Nmoms*grids_->NxNycNz);

    solver_->fieldSolve(G_q1, f);
 
  }

  G_q2->add_scaled(0.04, G_q2, 0.36, G_q1);

  G_q1->add_scaled(15, G_q2, -5, G_q1);
  solver_->fieldSolve(G_q1, f);
  
  // Iterations 6-9
  for(int i=6; i<10; i++) {
    linear_->rhs(G_q1, f, GRhs);

    if(nonlinear_ != NULL) {
      nonlinear_->nlps5d(G_q1, f, GStar);    
      GRhs->add_scaled(1., GRhs, 1., GStar);
    }

    G_q1->add_scaled(1., G_q1, dt_/6., GRhs);
    solver_->fieldSolve(G_q1, f);
  }

  // 10th and final stage
  linear_->rhs(G_q1, f, GRhs);

  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(G_q1, f, GStar);    
    GRhs->add_scaled(1., GRhs, 1., GStar);
  }

  G->add_scaled(1., G_q2, 0.6, G_q1, 0.1*dt_, GRhs);
  
  if (forcing_ != NULL) forcing_->stir(G);
  
  solver_->fieldSolve(G, f);
  *t += dt_;
  return 0;
}




