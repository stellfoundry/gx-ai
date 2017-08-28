#include "gtest/gtest.h"
#include "device_test.h"
#include "parameters.h"
#include "grids.h"
#include "diagnostics.h"
#include "run_gx.h"


TEST(TestParameters, CycloneLinearBeer) {

  Parameters* pars = new Parameters;
  char f[200];
  sprintf(f, "inputs/test_cyclone_linear_beer.in");
  pars->read_namelist(f);
  pars->iproc = 0;
  printf("Initializing geometry...\n");
  Geometry* geo = new S_alpha_geo(pars);
  printf("Initializing grids...\n");
  Grids* grids = new Grids(pars);
  printf("Grid dimensions: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n", 
     grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);
  printf("Initializing diagnostics...\n");
  Diagnostics* diagnostics = new Diagnostics(pars, grids, geo);
  
  run_gx(pars, grids, geo, diagnostics);

  cuDoubleComplex growth_rates_correct[11] = {
	0,		0,
	0.052151,	0.038353,
	0.154697,	0.100987,
	0.278418,	0.159390,
	0.437257,	0.191752,
	0.604513,	0.224344,
	0.775233,	0.222227,
	0.944013,	0.205547,
	1.105223,	0.161124,
	1.258342,	0.099509,
	1.398031,	0.021141
  };

  for(int i=1; i<11; i++) {
    EXPECT_NEAR(growth_rates_correct[i].x, diagnostics->growth_rates_h[i].x, 1e-4);
    EXPECT_NEAR(growth_rates_correct[i].y, diagnostics->growth_rates_h[i].y, 1e-4);
  }

  delete pars;
  delete grids;
  delete geo;
  delete diagnostics;
}
