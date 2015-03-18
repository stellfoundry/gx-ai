#include <unity.h>

/*int main(int argc, char ** argv) {*/
/*return 0;*/
/*}*/

void setUp(void)
{
}
 
void tearDown(void)
{
}
 
void test_secondary_growth_rate(void)
{
}

int main(void)
{
  UnityBegin("tests/nonlinear_tests/secondary.c");

  RUN_TEST(test_secondary_growth_rate);
	return 0;
}
