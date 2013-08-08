/* These must be in the same order that they appear in
 * fluxes.fpp in trinity*/
struct gryfx_parameters_struct {
	 /* Name of gryfx/gryffin input file*/
	/*char input_file[1000];*/
	/*Base geometry parameters - not currently set by trinity 
	!See geometry.f90*/
	 int equilibrium_type;
	 /*char eqfile[800];*/
	 int irho;
	 double rhoc;
	 int bishop;
	 int nperiod;
	 int ntheta;

	/* Miller parameters*/
	 double rmaj;
	 double r_geo;
	 double akappa;
	 double akappri;
	 double tri;
	 double tripri;
	 double shift;
	 double qinp;
	 double shat;
	 double asym;
	 double asympri;

	 /* Other geometry parameters - Bishop/Greene & Chance*/
	 double beta_prime_input;
	 double s_hat_input;

	 /*Flow shear*/
	 double g_exb;

	 /* Species parameters... I think allowing 20 species should be enough!*/
	 int ntspec;
	 double dens[20];
	 double temp[20];
	 double fprim[20];
	 double tprim[20];
	 double nu[20];
};

struct gryfx_outputs_struct {
       double pflux[20];
       double qflux[20];
       double heat[20];
       double dvdrho;
       double grho;
};

extern "C"
void gryfx_get_default_parameters_(struct gryfx_parameters_struct *, char * namelistFile );
extern "C"
void gryfx_get_fluxes_(struct gryfx_parameters_struct *, struct gryfx_outputs_struct*);
