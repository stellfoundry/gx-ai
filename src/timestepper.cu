#include "timestepper.h"
#include "get_error.h"

// ======= SSPx2 =======
SSPx2::SSPx2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in)
{
  // new objects for temporaries
  GStar = new MomentsG(pars, grids);
  GRhs  = new MomentsG(pars, grids);
  G1    = new MomentsG(pars, grids);
  G2    = new MomentsG(pars, grids);
}

SSPx2::~SSPx2()
{
  delete GStar;
  delete GRhs;
  delete G1; 
  delete G2; 
}

// ======== SSPx2  ==============
void SSPx2::EulerStep(MomentsG* G1, MomentsG* G, MomentsG* GRhs, Fields* f, MomentsG* GStar, bool setdt)
{
  linear_->rhs(G, f, GRhs);

  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(G, f, GStar);
    GRhs->add_scaled(1., GRhs, 1., GStar);
    if (setdt) dt_ = nonlinear_->cfl(dt_max);
  }

  if (pars_->eqfix) G1->copyFrom(G);   
  G1->add_scaled(1., G, dt_/sqrt(2.), GRhs);

}

int SSPx2::advance(double *t, MomentsG* G, Fields* f)
{

  EulerStep (G1, G, GRhs, f, GStar, true); 
  solver_->fieldSolve(G1, f);

  EulerStep (G2, G1, GRhs, f, GStar, false);

  G->add_scaled(2.-sqrt(2.), G, sqrt(2.)-2., G1, 1., G2);
  
  if (forcing_ != NULL) forcing_->stir(G);  

  solver_->fieldSolve(G, f);

  *t += dt_;
  return 0;
}

// ======= SSPx3 =======
SSPx3::SSPx3(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	     Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in)
{
  
  /*
  printf("a  = %f \n",adt);
  printf("w1 = %f \n",w1);
  printf("w2 = %f \n",w2);
  printf("w3 = %f \n",w3);
  */
  
  // new objects for temporaries
  GStar = new MomentsG(pars, grids);
  GRhs  = new MomentsG(pars, grids);
  G1    = new MomentsG(pars, grids);
  G2    = new MomentsG(pars, grids);
  G3    = new MomentsG(pars, grids);
}

SSPx3::~SSPx3()
{
  delete GStar;
  delete GRhs;
  delete G1; 
  delete G2; 
  delete G3; 
}

// ======== SSPx3  ==============
void SSPx3::EulerStep(MomentsG* G1, MomentsG* G, MomentsG* GRhs, Fields* f, MomentsG* GStar, bool setdt)
{
  linear_->rhs(G, f, GRhs);

  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(G, f, GStar);
    GRhs->add_scaled(1., GRhs, 1., GStar);
    if (setdt) dt_ = nonlinear_->cfl(dt_max);
  }

  if (pars_->eqfix) G1->copyFrom(G);   
  G1->add_scaled(1., G, adt*dt_, GRhs);
}

int SSPx3::advance(double *t, MomentsG* G, Fields* f)
{

  EulerStep (G1, G, GRhs, f, GStar, true); 
  solver_->fieldSolve(G1, f);

  EulerStep (G2, G1, GRhs, f, GStar, false);

  G2->add_scaled((1.-w1), G, (w1-1.), G1, 1., G2);
  solver_->fieldSolve(G2, f);

  EulerStep (G3, G2, GRhs, f, GStar, false);

  G->add_scaled((1.-w2-w3), G, w3, G1, (w2-1.), G2, 1., G3);
  
  if (forcing_ != NULL) forcing_->stir(G);  

  solver_->fieldSolve(G, f);

  *t += dt_;
  return 0;
}

// ======= RK2 =======
RungeKutta2::RungeKutta2(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in)
{
  // new objects for temporaries
  GStar = new MomentsG(pars, grids);
  GRhs  = new MomentsG(pars, grids);
  G1    = new MomentsG(pars, grids);
}

RungeKutta2::~RungeKutta2()
{
  delete GStar;
  delete GRhs;
  delete G1;
}

// ======== rk2  ==============
void RungeKutta2::EulerStep(MomentsG* G1, MomentsG* G0, MomentsG* G, MomentsG* GRhs,
			    Fields* f, MomentsG* GStar, double adt, bool setdt)
{
  linear_->rhs(G0, f, GRhs); 
  
  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(G0, f, GStar);
    GRhs->add_scaled(1., GRhs, 1., GStar);

    if (setdt) dt_ = nonlinear_->cfl(dt_max);
  }

  if (pars_->eqfix) G1->copyFrom(G);   
  G1->add_scaled(1., G, adt*dt_, GRhs); 

}

int RungeKutta2::advance(double *t, MomentsG* G, Fields* f)
{
  EulerStep (G1, G, G, GRhs, f, GStar, 0.5, true);    solver_->fieldSolve(G1, f);
  EulerStep (G, G1, G, GRhs, f, GStar, 1.0, false);   

  if (forcing_ != NULL) forcing_->stir(G);  
   
  solver_->fieldSolve(G, f);

  *t += dt_;
  return 0;
}

// ============= RK4 =============
// why is dt double precision? 
RungeKutta4::RungeKutta4(Linear *linear, Nonlinear *nonlinear, Solver *solver,
			 Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
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

// ======== rk4  ==============

int RungeKutta4::advance(double *t, MomentsG* G, Fields* f)
{
  //////////////////////////////////////////
  /// In: G                             ////
  /// Out: GRhs1, GRhs3                 ////
  //////////////////////////////////////////

  //  EulerStep (GRhs3, G, G, GRhs1, f, 
  
  // GRhs1 = RHS(t)
  linear_->rhs(G, f, GRhs1);

  if (nonlinear_ != NULL) {  
    nonlinear_->nlps5d(G, f, GRhs3);         // GRhs3 is a tmp array here
    GRhs1->add_scaled(1., GRhs1, 1., GRhs3); // GRhs3 is a tmp array here

      // choose the timestep
    dt_ = nonlinear_->cfl(dt_max);
  }
  
  // GRhs3 = G(t) + RHS(t) * dt/2 = G(t+dt/2) (first approximation)
  if (pars_->eqfix) GRhs3->copyFrom(G);

  GRhs3->add_scaled(1., G, dt_/2., GRhs1);  
  solver_->fieldSolve(GRhs3, f);            

  //////////////////////////////////////////
  /// In: G, GRhs3                      ////
  /// Out: GStar_(GRhs2), GRhs4_(GStar) ////
  //////////////////////////////////////////

  // Gstar = RHS(t+dt/2)  (using first approx from t+dt/2)
  linear_->rhs(GRhs3, f, GStar);            

  if (nonlinear_ != NULL) {  
    nonlinear_->nlps5d(GRhs3, f, GRhs4);    // GRhs4 is tmp array here
    GStar->add_scaled(1., GStar, 1., GRhs4);// GRhs4 is tmp array here
  }
  
  // GRhs4 = G + linear(t+dt/2) * dt/2      (this is second approx for t+dt/2)
  if (pars_->eqfix) GRhs4->copyFrom(G);

  GRhs4->add_scaled(1., G, dt_/2., GStar);  
  solver_->fieldSolve(GRhs4, f);            

  /////////////////////////////////////////////////
  /// First: GRhs1 == dt/6 GRhs1 + dt/3 Gstar  ////
  /////////////////////////////////////////////////

  // Do a partial accumulation of final update to save memory
  GRhs1->add_scaled(dt_/6., GRhs1, dt_/3., GStar);
  // GStar is now available as a tmp array
  
  // GRhs3 = linear(t+dt/2)  (using second approx from t+dt/2)
  linear_->rhs(GRhs4, f, GRhs3);            

  if (nonlinear_ != NULL) {  
    nonlinear_->nlps5d(GRhs4, f, GStar);    // GRhs4 is a tmp array here
    GRhs3->add_scaled(1., GRhs3, 1., GStar);
  }

  if (pars_->eqfix) GStar->copyFrom(G);

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
  linear_(linear), nonlinear_(nonlinear), solver_(solver), grids_(grids), pars_(pars),
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

  if (pars_->eqfix) GRhs1->copyFrom(G);

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

  if (pars_->eqfix) GRhs2->copyFrom(GRhs1);

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
  if (pars_->eqfix) GRhs1->copyFrom(GRhs2);
  GRhs1->add_scaled(1., GRhs2, dt_, GRhs1);
  
  G->add_scaled(1., G, 1./6., GRhs1);

  if (forcing_ != NULL) forcing_->stir(G);
  
  solver_->fieldSolve(G, f);
  *t += dt_;
  return 0;
}
/*
// ============= SDC, forward Euler ========
SDCe::SDCe(Linear *linear, Nonlinear *nonlinear, Solver *solver,
	    Parameters *pars, Grids *grids, Forcing *forcing, double dt_in) :
  linear_(linear), nonlinear_(nonlinear), solver_(solver), pars_(pars), grids_(grids),
  forcing_(forcing), dt_max(dt_in), dt_(dt_in)
{
  // set up storage 
  // set up nodes and spectral integration matrix
  // initialization should provide M (# of timepoints, incl ends) and K (# of iterations)
  
  double nodes_3  = {-1.,  0.,                 1.};
  double nodes_4  = {-1., -0.447213595499958,  0.447213595499958,  1.};
  double nodes_5  = {-1., -0.654653670707977,  0.,                 0.654653670707977,  1.};
  double nodes_6  = {-1., -0.765055323929465, -0.285231516480645,  0.285231516480645,  0.765055323929465, 1.};
  double nodes_7  = {-1., -0.830223896278567, -0.468848793470714,  0.,                 0.468848793470714,
		           0.830223896278567,  1.};
  double nodes_8  = {-1., -0.871740148509607, -0.591700181433142, -0.209299217902479,  0.209299217902479,
              		   0.591700181433142,  0.871740148509607,  1.};
  double nodes_9  = {-1., -0.89975799541146,  -0.677186279510738, -0.363117463826178,  0.,
		           0.363117463826178,  0.677186279510738,  0.89975799541146,   1.};
  double nodes_10 = {-1., -0.919533908166459, -0.738773865105505, -0.477924949810444, -0.165278957666387,
		           0.165278957666387,  0.477924949810444,  0.738773865105505,  0.919533908166459, 1.};
  double nodes_11 = {-1., -0.934001430408059, -0.784483473663144, -0.565235326996205, -0.295758135586939, 0.,
		           0.295758135586939,  0.565235326996205,  0.784483473663144,  0.934001430408059, 1.};
  double nodes_12 = {-1., -0.944899272222882, -0.819279321644007, -0.632876153031861, -0.399530940965349,
		          -0.136552932854928,  0.136552932854928,  0.399530940965349,  0.632876153031861,
		           0.819279321644007,  0.944899272222882,  1.};
  double nodes_13 = {-1., -0.953309846642164, -0.846347564651872, -0.686188469081757, -0.482909821091336,
		          -0.24928693010624,   0.,	           0.24928693010624,   0.482909821091336,
		           0.686188469081757,  0.846347564651872,  0.953309846642164,  1.};
  double nodes_14 = {-1., -0.959935045267261, -0.867801053830347, -0.728868599091326, -0.550639402928647,
		          -0.342724013342713, -0.116331868883704,  0.116331868883704,  0.342724013342713,
		           0.550639402928647,  0.728868599091326,  0.867801053830347,  0.959935045267261, 1.};
  double nodes_15 = {-1., -0.965245926503839, -0.885082044222976, -0.763519689951815, -0.606253205469846,
		          -0.420638054713672, -0.215353955363794,  0.,		       0.215353955363794,
		           0.420638054713672,  0.606253205469846,  0.763519689951815,  0.885082044222976,
		           0.965245926503839,  1.};

  double s4[12] = { 0.110300566479165,     0.189699433520835,    -0.0339073642291439,   0.0103005664791649,
		   -0.0372677996249965,    0.260874597374975,     0.260874597374975,   -0.0372677996249965,
		    0.0103005664791649,   -0.0339073642291439,    0.189699433520835,    0.110300566479165};

  double s5[20] = { 0.0677284321861569,    0.119744769343412,    -0.0217357218665581,    0.0106358242254155,
		   -0.00370013924241453,  -0.0271034321861569,    0.183439413979631,     0.199513499644336,
		   -0.0415977853262361,	   0.0130751392424145,    0.0130751392424145,   -0.0415977853262361,
		    0.199513499644336,     0.183439413979631,    -0.0271034321861569,   -0.00370013924241453,
		    0.0106358242254155,   -0.0217357218665581,    0.119744769343412,     0.0677284321861569};

  double s6[30] = { 0.045679805133755,     0.0818678170089706,   -0.01487460578909,      0.00762767611825106,
		   -0.00447178044057369,   0.00164342600395453,  -0.0197714197458752,    0.131972991623855,
		    0.148835341439951,    -0.0316317508514058,    0.0162794768182334,   -0.00577273556034828,
		    0.0115542575018473,   -0.0364110268615619,    0.167472527600037,     0.167472527600037,
		   -0.0364110268615619,    0.0115542575018473,   -0.00577273556034828,   0.0162794768182334,
		   -0.0316317508514058,    0.148835341439951,     0.131972991623855,    -0.0197714197458752,
		    0.00164342600395453,  -0.00447178044057369,   0.00762767611825106,  -0.01487460578909,
		    0.0818678170089706,    0.045679805133755};
  double s7[42] = { 0.0328462643282927,    0.0593228940275514,   -0.0107685944511896,    0.00559759178056978,
		   -0.00348892997080757,   0.00221709658891453,  -0.000838270442615104, -0.0148440411264775,
		    0.0983782366141377,    0.113123406498051,    -0.0240758510540288,    0.0130665100715488,
		   -0.00789896115513893,   0.00293825155583389,   0.00952753870294674,  -0.0299128750818516,
		    0.135130840674784,     0.140383021178221,    -0.0311905422174555,    0.0163066326871699,
		   -0.00582021920845688,  -0.00582021920845688,   0.0163066326871699,   -0.0311905422174555,
		    0.140383021178221,     0.135130840674784,    -0.0299128750818516,    0.00952753870294674,
		    0.00293825155583389,  -0.00789896115513893,   0.0130665100715488,   -0.0240758510540288,
		    0.113123406498051,     0.0983782366141377,   -0.0148440411264775,   -0.000838270442615104,
		    0.00221709658891453,  -0.00348892997080757,   0.00559759178056978,  -0.0107685944511896,
		    0.0593228940275514,    0.0328462643282927};
  double s9[72] = { 0.0192938382010432,    0.0351255209776218,   -0.00636410241870477,   0.00333377719700016,
		   -0.00213684701760824,   0.00149429916272894,  -0.00107406069938176,   0.000733772666539296,
		   -0.000285195774966302, -0.00910975779281556,   0.0599609874717403,    0.0702960980471429,
		   -0.01491950853084,      0.00828678391816004,  -0.00547512494991058,   0.00382938478880073,
		   -0.00258127780592289,   0.000998272804007648,  0.00638528943534415,  -0.0199929913531551,
		    0.0889500653964454,    0.095671210022756,    -0.0212806100728863,    0.0119814293575819,
		   -0.00785164995116065,   0.00513712970942321,  -0.0019654647020686,   -0.00457935248246069,
		    0.0127763166268562,   -0.0241998387183004,    0.105672601627005,     0.108010491766439,
		   -0.0245444283997944,    0.0136834598052393,   -0.00841177751269997,   0.00315125920080503,
		    0.00315125920080503,  -0.00841177751269997,   0.0136834598052393,   -0.0245444283997944,
		    0.108010491766439,     0.105672601627005,    -0.0241998387183004,    0.0127763166268562,
		   -0.00457935248246069,  -0.0019654647020686,    0.00513712970942321,  -0.00785164995116065,
		    0.0119814293575819,   -0.0212806100728863,    0.095671210022756,     0.0889500653964454,
		   -0.0199929913531551,    0.00638528943534415,   0.000998272804007648, -0.00258127780592289,
		    0.00382938478880073,  -0.00547512494991058,   0.00828678391816004,  -0.01491950853084,
		    0.0702960980471429,    0.0599609874717403,   -0.00910975779281556,  -0.000285195774966302,
		    0.000733772666539296, -0.00107406069938176,   0.00149429916272894,  -0.00213684701760824,
		    0.00333377719700016,  -0.00636410241870477,   0.0351255209776218,    0.0192938382010432};
  double s15[210]={ 0.00665750641965142,   0.0122092686702348,   -0.00220659832125655,   0.00115926243556674,
		   -0.000749969850354452,  0.000536887296767535, -0.000408190199877821,  0.00032265380974312,
		   -0.000261607939170038,  0.000215336552764969, -0.000178139028336527,  0.000146185115355756,
		   -0.000116256582726242,  8.4223060645853e-05,  -3.35224606978186e-05, -0.0032693624392877,
		    0.0213937664652264,    0.025514277702154,    -0.0053872216299896,    0.0030136281549915,
		   -0.00203338199369699,   0.00149953698113971,  -0.00116439239469809,   0.000933565387386992,
		   -0.000762767825584886,  0.000627849389309036, -0.000513479787075772,  0.000407446015096129,
		   -0.000294789915152038,  0.000117269362444164,  0.00246691171325054,  -0.00771203966730401,
		    0.0339200415973666,    0.0375379801213251,   -0.0082913478778648,    0.00474289451087654,
		   -0.00325670665791232,   0.00243363794122606,  -0.00190668022165637,   0.00153494720415949,
		   -0.00125107371664465,   0.00101646097370194,  -0.000803115821569198,  0.000579601505307152,
		   -0.000230333680863731, -0.00199101627472316,   0.00554418433214813,  -0.0104530935625624,
		    0.0447885278788096,    0.0477816413735958,   -0.0107796490383067,    0.00622908261217786,
		   -0.0043117784365252,    0.00323943124855487,  -0.00254221764857639,   0.00203841100341434,
		   -0.00163849470160566,   0.00128573077835788,  -0.000924201062368188,  0.00036668383450532,
		    0.00164160418085794,  -0.00436934732242561,   0.00706106122733339,  -0.0125395762513541,
		    0.05353864099293,      0.0557785390373004,   -0.0127420954945379,    0.00739908924297036,
		   -0.00513733143675415,   0.00386018757195966,  -0.00301468047932309,   0.00238322365441623,
		   -0.00185072145069962,   0.00132238914113777,  -0.000523407232423355, -0.00135710239006153,
		    0.0035316092096086,   -0.00534351953560879,   0.00802978724462842,  -0.0139764697679065,
		    0.0597729074728839,    0.0611590250818361,   -0.0140903956175119,    0.00819719638788731,
		   -0.0056878141361441,    0.00425206175372267,  -0.00327498173313826,   0.0025034702547636,
		   -0.00177302906518804,   0.000699304515184602,  0.0011121079718601,   -0.00285708839258616,
		    0.00417433570986587,  -0.00582614656246577,   0.00853187705512651,  -0.0147254230531162,
		    0.06320401640696,      0.0636732145419967,   -0.0147624437430509,    0.00858317203058133,
		   -0.00592760537586336,   0.00437432314302265,  -0.00326302284764722,   0.00228039955500578,
		   -0.000894738757792039, -0.000894738757792039,  0.00228039955500578,  -0.00326302284764722,
		    0.00437432314302265,  -0.00592760537586336,   0.00858317203058133,  -0.0147624437430509,
		    0.0636732145419967,    0.06320401640696,     -0.0147254230531162,    0.00853187705512651,
		   -0.00582614656246577,   0.00417433570986587,  -0.00285708839258616,   0.0011121079718601,
		    0.000699304515184602, -0.00177302906518804,   0.0025034702547636,   -0.00327498173313826,
		    0.00425206175372267,  -0.0056878141361441,    0.00819719638788731,  -0.0140903956175119,
		    0.0611590250818361,    0.0597729074728839,   -0.0139764697679065,    0.00802978724462842,
		   -0.00534351953560879,   0.0035316092096086,   -0.00135710239006153,  -0.000523407232423355,
		    0.00132238914113777,  -0.00185072145069962,   0.00238322365441623,  -0.00301468047932309,
		    0.00386018757195966,  -0.00513733143675415,   0.00739908924297036,  -0.0127420954945379,
		    0.0557785390373004,    0.05353864099293,     -0.0125395762513541,    0.00706106122733339,
		   -0.00436934732242561,   0.00164160418085794,   0.00036668383450532,  -0.000924201062368188,
		    0.00128573077835788,  -0.00163849470160566,   0.00203841100341434,  -0.00254221764857639,
		    0.00323943124855487,  -0.0043117784365252,    0.00622908261217786,  -0.0107796490383067,
		    0.0477816413735958,    0.0447885278788096,   -0.0104530935625624,    0.00554418433214813,
		   -0.00199101627472316,  -0.000230333680863731,  0.000579601505307152, -0.000803115821569198,
		    0.00101646097370194,  -0.00125107371664465,   0.00153494720415949,  -0.00190668022165637,
		    0.00243363794122606,  -0.00325670665791232,   0.00474289451087654,  -0.0082913478778648,
		    0.0375379801213251,    0.0339200415973666,   -0.00771203966730401,   0.00246691171325054,
		    0.000117269362444164, -0.000294789915152038,  0.000407446015096129, -0.000513479787075772,
		    0.000627849389309036, -0.000762767825584886,  0.000933565387386992, -0.00116439239469809,
		    0.00149953698113971,  -0.00203338199369699,   0.0030136281549915,   -0.0053872216299896,
		    0.025514277702154,     0.0213937664652264,   -0.0032693624392877,   -3.35224606978186e-05,
		    8.4223060645853e-05,  -0.000116256582726242,  0.000146185115355756, -0.000178139028336527,
		    0.000215336552764969, -0.000261607939170038,  0.00032265380974312,  -0.000408190199877821,
		    0.000536887296767535, -0.000749969850354452,  0.00115926243556674,  -0.00220659832125655,
		    0.0122092686702348,    0.00665750641965142};  
  
  // there will be an input value of M, the number of timepoints per timestep, including the endpoints.
  int M_input = 5;
  int M = M_input;
  
  if (M==4 || M==5 || M==6 || M==7 || M==9 || M==15) {
    continue;
  } else {
    //error
  }
  
  if (M==4) { starg = s4;    xtarg = nodes_4; }
  if (M==5) { starg = s5;    xtarg = nodes_5; }
  if (M==6) { starg = s6;    xtarg = nodes_6; }
  if (M==7) { starg = s7;    xtarg = nodes_7; }
  if (M==9) { starg = s9;    xtarg = nodes_9; }
  if (M==15){ starg = s15;   xtarg = nodes_15;}
  
  // could work with double precision but we are using single for now: 
  float x[M_input]; // should be defined at a higher level, available throughout SDCe
  //  s matrix needs to be declared 


  for (int m=0; m<M; m++) x[m] = (float) xtarg[m];

  for (int m=0; m<M-1; m++) {
    for (int n=0; n<M; n++) {
      s[m][n] = (float) starg[M*m + n];
      //      printf("s[%d][%d] = %.*e \n",m,n,14,s[m][n]); // for double precision
      printf("s[%d][%d] = %.*e \n", m, n, 6, s[m][n]);
    }
    printf("\n");
  }
  
}
SDCe::~SDCe()
{
  // delete storage
}
void SDCe::full_rhs(MomentsG* G_q1, MomentsG* GRhs, Fields* f, MomentsG* GStar)
{
  linear_->rhs(G_q1, f, GRhs);

  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(G_q1, f, GStar);
    GRhs->add_scaled(1., GRhs, 1., GStar);
  }

  //  if (setdt) dt_ = nonlinear_->cfl(dt_max);
  //  G_q1->add_scaled(1., G_q1, dt_/6., GRhs);

  //    printf("another G_q1 (K10):\n");
  //    G_q1->qvar(grids_->Nmoms*grids_->NxNycNz);
  
  //  solver_->fieldSolve(G_q1, f);    
}

int SDCe::advance(double *t, MomentsG* G, Fields* f)
{
}
*/
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

void Ketcheson10::EulerStep(MomentsG* G_q1, MomentsG* GRhs, Fields* f, MomentsG* GStar, bool setdt)
{
  linear_->rhs(G_q1, f, GRhs);

  if(nonlinear_ != NULL) {
    nonlinear_->nlps5d(G_q1, f, GStar);
    GRhs->add_scaled(1., GRhs, 1., GStar);
    if (setdt) dt_ = nonlinear_->cfl(dt_max);
  }

  G_q1->add_scaled(1., G_q1, dt_/6., GRhs);
  solver_->fieldSolve(G_q1, f);    
}

int Ketcheson10::advance(double *t, MomentsG* G, Fields* f)
{

  G_q1->copyFrom(G);
  G_q2->copyFrom(G);

  bool setdt;
  
  // First five iterations
  for(int i=1; i<6; i++) {
    if (i==1) {setdt = true; } else { setdt = false;}      
    EulerStep(G_q1, GRhs, f, GStar, setdt);
    //    if (setdt) printf("another G (k10):\n");
    //    if (setdt) GRhs->qvar(grids_->Nmoms*grids_->NxNycNz);
    //    if (setdt) f->print_phi();

  }

  // middle bit
  G_q2->add_scaled(0.04, G_q2, 0.36, G_q1);
  G_q1->add_scaled(15, G_q2, -5, G_q1);
  solver_->fieldSolve(G_q1, f);
  
  setdt = false;
  // Next four iterations
  for(int i=6; i<10; i++) EulerStep(G_q1, GRhs, f, GStar, setdt);

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
