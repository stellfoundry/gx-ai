#include <stdio.h>
#include <stdlib.h>

#include "read_geo.cu"

int main(int argc, char ** argv){

	printf("We are running with ocelot!\n");
	int nz;

	read_geo_test(&nz);

	return 0;
}
