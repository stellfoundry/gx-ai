#include "gtest/gtest.h"
#include "parameters.h"
#include "grids.h"

class TestGrids : public ::testing::Test {
protected:
  virtual void SetUp() {
    Parameters* pars = new Parameters;
    pars->nx_in = 20;
    pars->ny_in = 48;
    pars->nz_in = 32;
    pars->nspec_in = 1;
    pars->nhermite_in = 4;
    pars->nlaguerre_in = 2;
    pars->Zp = 1.;
    pars->x0 = 10.;
    pars->y0 = 10.;

    grids = new Grids(pars);
  }

  virtual void TearDown() {
    delete grids;
  }

  Grids *grids;
};

TEST_F(TestGrids, Dimensions) {

  EXPECT_EQ(grids->Nx, 20);
  
}
