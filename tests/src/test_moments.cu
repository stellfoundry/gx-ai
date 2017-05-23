#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "moments.h"
#include "device_funcs.h"
#include "diagnostics.h"
#include "cuda_constants.h"

class TestMoments : public ::testing::Test {

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
    moms = new Moments(grids);
    geo = new S_alpha_geo(pars);
  }

  virtual void TearDown() {
    delete grids;
    delete moms;
    delete pars;
    delete geo;
  }

  Parameters* pars;
  Grids *grids;
  Moments *moms;
  Geometry* geo;
};

TEST_F(TestMoments, InitConditions) {

  pars->init_single = false;
  pars->init_amp = .01;
  pars->kpar_init = 2.;

  float* init_check = (float*) malloc(sizeof(float)*grids->NxNycNz);

  srand(22);
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      float ra = (float) (pars->init_amp * (rand()-RAND_MAX/2) / RAND_MAX);
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        init_check[index] = ra*cos(2.*geo->z_h[k]);
      }
    }
  }

  // initialize upar
  pars->init = UPAR;
  moms->initialConditions(pars, geo);

  // check initial condition
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&moms->upar_ptr[0][index].x, init_check[index]);
      }
    }
  }

  // initialize dens
  pars->init = DENS;
  moms->initialConditions(pars, geo);

  // check initial condition
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&moms->dens_ptr[0][index].x, init_check[index]);
        EXPECT_FLOAT_EQ_D(&moms->upar_ptr[0][index].x, init_check[index]);
        EXPECT_FLOAT_EQ_D(&moms->tpar_ptr[0][index].x, 0.);
        EXPECT_FLOAT_EQ_D(&moms->gHL(3,0)[index].x, 0.);
      }
    }
  }
 
  free(init_check);
}

TEST_F(TestMoments, AddMoments) 
{
  pars->init_single = false;
  pars->init_amp = .01;
  pars->kpar_init = 2.;
  float* init_check = (float*) malloc(sizeof(float)*grids->NxNycNz);
  srand(22);
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      float ra = (float) (pars->init_amp * (rand()-RAND_MAX/2) / RAND_MAX);
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        init_check[index] = ra*cos(2.*geo->z_h[k]);
      }
    }
  }
  // initialize upar
  pars->init = UPAR;
  moms->initialConditions(pars, geo);
  // moms = init(z) * upar

  Moments* moms2;
  moms2 = new Moments(grids);
  pars->init = DENS;
  moms2->initialConditions(pars, geo);
  // moms2 = init(z) * dens

  moms->add_scaled(1., moms, 2., moms2);
  // moms = init(z) * ( 2*dens + upar )

  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&moms->dens_ptr[0][index].x, 2.*init_check[index]);
        EXPECT_FLOAT_EQ_D(&moms->upar_ptr[0][index].x, init_check[index]);
        EXPECT_FLOAT_EQ_D(&moms->tpar_ptr[0][index].x, 0.);
        EXPECT_FLOAT_EQ_D(&moms->gHL(3,0)[index].x, 0.);
      }
    }
  }
  
  pars->init = UPAR;
  moms2->initialConditions(pars, geo);
  // moms2 = init(z) * ( dens + upar )
  moms->add_scaled(1., moms, 2., moms2);
  // moms = init(z) * ( 4*dens + 3*upar )

  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&moms->dens_ptr[0][index].x, 4.*init_check[index]);
        EXPECT_FLOAT_EQ_D(&moms->upar_ptr[0][index].x, 3.*init_check[index]);
        EXPECT_FLOAT_EQ_D(&moms->tpar_ptr[0][index].x, 0.);
        EXPECT_FLOAT_EQ_D(&moms->gHL(3,1)[index].x, 0.);
      }
    }
  }

  /////////  single moment addition /////////

  // 1d thread blocks over xyz
  dim3 dimBlock = 512;
  dim3 dimGrid = grids->NxNycNz/dimBlock.x+1;
  add_scaled_singlemom_kernel<<<dimGrid,dimBlock>>>
      (moms->gHL(3,0), 1., moms->gHL(1,0), 1., moms->gHL(0,0));
  // qpar = 7*init(z)
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&moms->gHL(3,0)[index].x, 7.*init_check[index]);
      }
    }
  }
}
