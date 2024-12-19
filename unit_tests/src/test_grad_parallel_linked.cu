#include "gtest/gtest.h"
#include "gpu_defs.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "grad_parallel.h"
#include "moments.h"
#include "diagnostics.h"
#include "device_funcs.h"
#include "diagnostics.h"

void qvar (cuComplex* G, int N, Grids* grids_)
{
  cuComplex* G_h;
  int Nk = grids_->Nyc*grids_->Nx;
  G_h = (cuComplex*) malloc (sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {G_h[i].x = 0.; G_h[i].y = 0.;}
  CP_TO_CPU (G_h, G, N*sizeof(cuComplex));

  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d,%d) = (%e, %e)  \n", i%grids_->Nyc, i/grids_->Nyc%grids_->Nx, i/Nk, G_h[i].x, G_h[i].y);
  printf("\n");

  free (G_h);
}

void qvar (float* G, int N, Grids* grids_)
{
  float* G_h;
  int Nx = grids_->Ny*grids_->Nx;
  G_h = (float*) malloc (sizeof(float)*N);
  for (int i=0; i<N; i++) G_h[i] = 0.;
  CP_TO_CPU (G_h, G, N*sizeof(float));

  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d,%d) = %e \n", i%grids_->Ny, i/grids_->Ny%grids_->Nx, i/Nx, G_h[i]);
  printf("\n");

  free (G_h);
}

class TestGradParallelLinked3D_nLink1 : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    pars->get_nml_vars("inputs/cyc_nl");
    // overwrite some parameters
    pars->nx_in = 1;
    pars->ny_in = 4;
    pars->nz_in = 8;
    pars->nperiod = 1;
    pars->nspec_in = 1;
    pars->nm_in = 4;
    pars->nl_in = 2;
    pars->Zp = 1.;
    int jtwist = 5;
    pars->initf = inits::density;
    pars->init_amp = .01;
    pars->ikpar_init = 2;
    pars->linear = true;
    pars->random_seed = 22;
    pars->random_init = false;

    grids = new Grids(pars);
    grad_par = new GradParallelLinked(pars, grids);
  }

  virtual void TearDown() {
    delete grids;
    delete grad_par;
    delete pars;
  }

  Parameters* pars;
  Grids *grids;
  GradParallelLinked *grad_par;
};

class TestGradParallelLinked3D : public ::testing::Test {
protected:
  virtual void SetUp() {
    pars = new Parameters;
    pars->get_nml_vars("inputs/cyc_nl");
    // overwrite some parameters
    pars->nx_in = 16;
    pars->ny_in = 16;
    pars->nz_in = 8;
    pars->nperiod = 1;
    pars->nspec_in = 1;
    pars->nm_in = 4;
    pars->nl_in = 2;
    pars->Zp = 1.;
    int jtwist = 5;
    pars->initf = inits::density;
    pars->init_amp = .01;
    pars->ikpar_init = 2;
    pars->linear = true;
    pars->random_seed = 22;
    pars->random_init = false;

    grids = new Grids(pars);
    grad_par = new GradParallelLinked(pars, grids);
  }

  virtual void TearDown() {
    delete grids;
    delete grad_par;
    delete pars;
  }

  Parameters* pars;
  Grids *grids;
  GradParallelLinked *grad_par;
};

//TEST_F(TestGradParallelLinked3D_nLink1, linkPrint) {
//  grad_par->linkPrint();
//}
//TEST_F(TestGradParallelLinked3D, linkPrint) {
//  grad_par->linkPrint();
//}

TEST_F(TestGradParallelLinked3D_nLink1, identity) {
  Geometry* geo;
  geo = new S_alpha_geo(pars, grids);

  MomentsG *G, *G2;
  G = new MomentsG(pars, grids);
  G2 = new MomentsG(pars, grids);
  pars->initf = inits::all;
  G->initialConditions();
  G->reality();
  G2->copyFrom(G);

  grad_par->identity(G);
  printf("Checking...\n");
  for(int index=0; index<grids->Nx*grids->Nyc*grids->Nz*grids->Nmoms; index++) {
    EXPECT_FLOAT_EQ_D(&G->G()[index].x, &G2->G()[index].x, 1.e-7);
    EXPECT_FLOAT_EQ_D(&G->G()[index].y, &G2->G()[index].y, 1.e-7);
  }
  
}
TEST_F(TestGradParallelLinked3D, identity) {
  Geometry* geo;
  geo = new S_alpha_geo(pars, grids);

  MomentsG *G, *G2;
  G = new MomentsG(pars, grids);
  G2 = new MomentsG(pars, grids);
  pars->initf = inits::all;
  G->initialConditions();
  G->reality();
  G2->copyFrom(G);

  grad_par->identity(G);
  printf("Checking...\n");
  for(int index=0; index<grids->Nx*grids->Nyc*grids->Nz*grids->Nmoms; index++) {
    EXPECT_FLOAT_EQ_D(&G->G()[index].x, &G2->G()[index].x, 1.e-7);
    EXPECT_FLOAT_EQ_D(&G->G()[index].y, &G2->G()[index].y, 1.e-7);
  }
  
}

TEST_F(TestGradParallelLinked3D_nLink1, EvaluateDerivative) {
  Geometry* geo;
  geo = new S_alpha_geo(pars, grids);
  strncpy(pars->run_name, "gpar_test", strlen("gpar_test"));

  MomentsG *GInit, *GRes;
  GInit = new MomentsG(pars, grids);
  GRes = new MomentsG(pars, grids);
  GInit->initialConditions();
  //qvar(GInit->dens_ptr, grids->NxNycNz, grids);

  float* init_check = (float*) malloc(sizeof(float)*grids->NxNycNz);
  float* deriv_check = (float*) malloc(sizeof(float)*grids->NxNycNz);
  memset(init_check, 0., sizeof(float)*grids->NxNycNz);
  memset(deriv_check, 0., sizeof(float)*grids->NxNycNz);

  srand(pars->random_seed);
  int idx;
  for(int i=0; i < 1 + (grids->Nx - 1)/3; i++) {
    for(int j=1; j < 1 + (grids->Ny - 1)/3; j++) {
      float ra = (float) (pars->init_amp * (rand()-RAND_MAX/2) / RAND_MAX);
      float rb = (float) (pars->init_amp * (rand()-RAND_MAX/2) / RAND_MAX);
      for (int js=0; js < 2; js++) {
        if (i==0) {
          idx = i;
        } else {
          idx = (js==0) ? i : grids->Nx-i;
        }
        for(int k=0; k<grids->Nz; k++) {
          int index = j + grids->Nyc*idx + grids->NxNyc*k;
	  float amp;
	  if (js == 0) amp = ra;
	  else amp = rb;
          init_check[index] = amp*cos(2.*grids->z_h[k]);
          deriv_check[index] = -2.*amp*sin(2.*grids->z_h[k]);
        }
      }
    }
  }

  printf("Checking initial condition...\n");
  // check initial condition
  for(int i=0; i<grids->Ny/3+1; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GInit->dens_ptr[index].x, init_check[index], 1e-6);
      }
    }
  }

  // out-of-place, with only a single moment
  printf("Checking out-of-place with single moment...\n");
  grad_par->dz(GInit->dens_ptr, GRes->dens_ptr);

  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GRes->dens_ptr[index].x, deriv_check[index], 1.e-7);
      }
    }
  }

  // in place, with entire LH G array
  printf("Checking in-place with entire G array...\n");
  grad_par->dz(GInit, GInit);
  for(int i=0; i<grids->Nx; i++) {
    for(int j=0; j<grids->Nyc; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = j + grids->Nyc*i + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GInit->dens_ptr[index].x, deriv_check[index], 1.e-7);
      }
    }
  }

  free(init_check);
  free(deriv_check);
  delete GInit;
  delete GRes;
  delete geo;
}

TEST_F(TestGradParallelLinked3D, EvaluateDerivative) {
  Geometry* geo;
  geo = new S_alpha_geo(pars, grids);
  strncpy(pars->run_name, "gpar_test", strlen("gpar_test"));

  MomentsG *GInit, *GRes;
  GInit = new MomentsG(pars, grids);
  GRes = new MomentsG(pars, grids);
  pars->initf = inits::upar;
  GInit->initialConditions();

  float* init_check = (float*) malloc(sizeof(float)*grids->NxNycNz);
  float* deriv_check = (float*) malloc(sizeof(float)*grids->NxNycNz);

  memset(init_check, 0., sizeof(float)*grids->NxNycNz);
  memset(deriv_check, 0., sizeof(float)*grids->NxNycNz);

  srand(pars->random_seed);
  int idx;
  for(int i=0; i < 1 + (grids->Nx - 1)/3; i++) {
    for(int j=1; j < 1 + (grids->Ny - 1)/3; j++) {
      float ra = (float) (pars->init_amp * (rand()-RAND_MAX/2) / RAND_MAX);
      float rb = (float) (pars->init_amp * (rand()-RAND_MAX/2) / RAND_MAX);
      for (int js=0; js < 2; js++) {
        if (i==0) {
          idx = i;
        } else {
          idx = (js==0) ? i : grids->Nx-i;
        }
        for(int k=0; k<grids->Nz; k++) {
          int index = j + grids->Nyc*idx + grids->NxNyc*k;
	  float amp;
	  if (js == 0) amp = ra;
	  else amp = rb;
          init_check[index] = amp*cos(2.*grids->z_h[k]);
          deriv_check[index] = -2.*amp*sin(2.*grids->z_h[k]);
        }
      }
    }
  }

  // check initial condition
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[index].x, init_check[index]);
      }
    }
  }

  // out-of-place, with only a single moment
  grad_par->dz(GInit->upar_ptr, GRes->upar_ptr);

  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GRes->upar_ptr[index].x, deriv_check[index], 1.e-7);
      }
    }
  }

  // out-of-place, with entire LH G array, with accumulate
  grad_par->dz(GInit, GRes, true);
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GRes->upar_ptr[index].x, 2*deriv_check[index], 1.e-7);
      }
    }
  }

  // in place, with entire LH G array
  grad_par->dz(GInit, GInit);
  for(int i=0; i<grids->Nyc; i++) {
    for(int j=0; j<grids->Nx; j++) {
      for(int k=0; k<grids->Nz; k++) {
        int index = i + grids->Nyc*j + grids->NxNyc*k;
        EXPECT_FLOAT_EQ_D(&GInit->upar_ptr[index].x, deriv_check[index], 1.e-7);
      }
    }
  }

  free(init_check);
  free(deriv_check);
  delete GInit;
  delete GRes;
  delete geo;
}
