#include <unity.h>
#include "global_vars.h"
#include "gryfx_lib.h"

/*int main(int argc, char ** argv) {*/
/*return 0;*/
/*}*/

int mpcom;
int argc_glob;
char** argv_glob;

void setUp(void)
{
  gryfx_main(argc_glob, argv_glob, mpcom);
}
 
void tearDown(void)
{
}
 
void test_secondary_growth_rate(void)
{
}

int main(int argc, char* argv[])
{
#ifdef GS2_zonal
        MPI_Init(&argc, &argv);
        mpcom = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif
  
  argc_glob = argc;
  argv_glob = argv;

  UnityBegin("tests/nonlinear_tests/secondary.c");

  RUN_TEST(test_secondary_growth_rate);
#ifdef GS2_zonal
        MPI_Finalize();
#endif
	return 0;

}
