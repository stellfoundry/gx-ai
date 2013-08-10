
//#include "global_vars.h"

void writedat_set_run_name(char * input_file){

	/* If input file is "myrun.in", run_name is "myrun"*/
	run_name = (char*)malloc(sizeof(char)*(strlen(input_file)-3+1));
	strncpy(run_name, input_file, strlen(input_file)-3);
	run_name[strlen(input_file)-3] = '\0';
	printf("Run name is %s\n", run_name);


}

void writedat_beginning()
{  

	char * filename;
	filename = (char*)malloc(sizeof(char)*(strlen(run_name)+5));
	sprintf(filename, "%s.cdf", run_name);
  sdatio_createfile(&sdatfile, filename);
  
  sdatio_add_dimension(&sdatfile, "r", 2, "real and imag parts", "(none)");
  sdatio_add_dimension(&sdatfile, "x", Nx, "kx coordinate", "1/rho_i");
  sdatio_add_dimension(&sdatfile, "y", Ny/2+1, "ky coordinate", "1/rho_i");
  sdatio_add_dimension(&sdatfile, "z", Nz, "z coordinate", "a");
  sdatio_add_dimension(&sdatfile, "s", nSpecies, "species coordinate", "(none)");
  sdatio_add_dimension(&sdatfile, "t", SDATIO_UNLIMITED, "time coordinate","a/vt_i");

  sdatio_create_variable(&sdatfile, SDATIO_FLOAT, "kx", "x", "kx grid", "1/rho_i");
  
  sdatio_create_variable(&sdatfile, SDATIO_FLOAT, "phi", "szxyr", "potential", "Ti/e");
  sdatio_create_variable(&sdatfile, SDATIO_FLOAT, "phi2", "t", "phi**2 summed over all modes", "(Ti/e)**2");
  sdatio_create_variable(&sdatfile, SDATIO_FLOAT, "hflux_tot", "t", "total heat flux", "(figure it out)");
  
  sdatio_write_variable(&sdatfile, "kx", &kx_h[0]);
  
}


void writedat_each()
{
  sdatio_write_variable(&sdatfile, "phi", &phi_h[0]);
  sdatio_write_variable(&sdatfile, "phi2", &phi2);
  sdatio_write_variable(&sdatfile, "hflux_tot", &hflux_tot);
  sdatio_increment_start(&sdatfile, "t");
  
}

void writedat_end()
{
  sdatio_close(&sdatfile);
}
