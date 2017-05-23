#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "grad_parallel.h"
#include "moments.h"
#include "diagnostics.h"
#include "cuda_constants.h"

class TestGradParallel1D : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    pars->nx_in = 1;
    pars->ny_in = 1;
    pars->nz_in = 32;
    pars->nperiod = 2;
    pars->nspec_in = 1;
    pars->nhermite_in = 4;
    pars->nlaguerre_in = 2;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;

    grids = new Grids(pars);
    grad_par = new GradParallel(grids);
  }

  virtual void TearDown() {
    delete grids;
    delete grad_par;
    delete pars;
  }

  Parameters* pars;
  Grids *grids;
  GradParallel *grad_par;
};

class TestGradParallel3D : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    pars->nx_in = 32;
    pars->ny_in = 32;
    pars->nz_in = 32;
    pars->nperiod = 1;
    pars->nspec_in = 1;
    pars->nhermite_in = 4;
    pars->nlaguerre_in = 2;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;

    grids = new Grids(pars);
    grad_par = new GradParallel(grids);
  }

  virtual void TearDown() {
    delete grids;
    delete grad_par;
    delete pars;
  }

  Parameters* pars;
  Grids *grids;
  GradParallel *grad_par;
};

TEST_F(TestGradParallel1D, EvaluateDerivative) {
  Geometry* geo;
  geo = new S_alpha_geo(pars);
  
  Moments* momsInit, *momsRes;
  momsInit = new Moments(grids);
  momsRes = new Moments(grids);
  pars->init_single = false;
  pars->init = DENS;
  pars->init_amp = .01;
  pars->kpar_init = 2.;
  momsInit->initialConditions(pars, geo);

  float* init_check = (float*) malloc(sizeof(float)*grids->Nz);
  float* deriv_check = (float*) malloc(sizeof(float)*grids->Nz);

  for(int i=0; i<grids->Nz; i++) {
    init_check[i] = pars->init_amp*cos(2.*geo->z_h[i]);
    deriv_check[i] = -2.*pars->init_amp*sin(2.*geo->z_h[i]);
  }

  // check initial condition
  for(int i=0; i<grids->Nz; i++) {
    EXPECT_FLOAT_EQ_D(&momsInit->dens_ptr[0][i].x, init_check[i]);
  }

  // out-of-place, with only a single moment
  grad_par->eval(momsInit->dens_ptr[0], momsRes->dens_ptr[0]);
  
  for(int i=0; i<grids->Nz; i++) {
    EXPECT_FLOAT_EQ_D(&momsRes->dens_ptr[0][i].x, deriv_check[i], 1.e-7);
  }

  // in place, with entire HL moms array
  grad_par->eval(momsInit);
  for(int i=0; i<grids->Nz; i++) {
    EXPECT_FLOAT_EQ_D(&momsInit->dens_ptr[0][i].x, deriv_check[i], 1.e-7);
  }

  free(init_check);
  free(deriv_check);
  delete momsInit;
  delete momsRes;
  delete geo;
}

TEST_F(TestGradParallel3D, EvaluateDerivative) {
  Geometry* geo;
  geo = new S_alpha_geo(pars);
  //strncpy(pars->run_name, "test", strlen("test")-1);
  //Diagnostics* diagnostics;
  //diagnostics = new Diagnostics(pars, grids, geo);

  Moments *momsInit, *momsRes;
  momsInit = new Moments(grids);
  momsRes = new Moments(grids);
  pars->init_single = false;
  pars->init = UPAR;
  pars->init_amp = .01;
  pars->kpar_init = 2.;
  momsInit->initialConditions(pars, geo);

  float* init_check = (float*) malloc(sizeof(float)*grids->NxNycNz);
  float* deriv_check = (float*) malloc(sizeof(float)*grids->NxNycNz);

  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        init_check[index] = pars->init_amp*cos(2.*geo->z_h[k]);
        deriv_check[index] = -2.*pars->init_amp*sin(2.*geo->z_h[k]);
      }
    }
  }

  // check initial condition
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&momsInit->upar_ptr[0][index].x, init_check[index]);
      }
    }
  }

  // out-of-place, with only a single moment
  grad_par->eval(momsInit->upar_ptr[0], momsRes->upar_ptr[0]);
  
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&momsRes->upar_ptr[0][index].x, deriv_check[index], 1.e-7);
      }
    }
  }

  // in place, with entire HL moms array
  grad_par->eval(momsInit);
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&momsInit->upar_ptr[0][index].x, deriv_check[index], 1.e-7);
      }
    }
  }
  free(init_check);
  free(deriv_check);
  delete momsInit;
  delete momsRes;
  delete geo;
}
