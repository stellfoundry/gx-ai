
//global host arrays from eik.out
EXTERN_SWITCH float *gbdrift_h, *grho_h, *z_h, *z_regular_h; 
EXTERN_SWITCH float *cvdrift_h, *gds2_h, *bmag_h, *bgrad_h;
EXTERN_SWITCH float *gds21_h, *gds22_h, *cvdrift0_h, *gbdrift0_h, *jacobian_h;
EXTERN_SWITCH float *Rplot_h, *Zplot_h, *aplot_h;
EXTERN_SWITCH float *Rprime_h, *Zprime_h, *aprime_h;
EXTERN_SWITCH float *Xplot_h, *Yplot_h, *deltaFL_h, *gradpar_h;


//global device arrays from eik.out
EXTERN_SWITCH float *gbdrift, *grho, *z, *cvdrift, *gds2, *bmag, *bgrad;
EXTERN_SWITCH float *gds21, *gds22, *cvdrift0, *gbdrift0;

EXTERN_SWITCH float *bmagInv;
EXTERN_SWITCH cuComplex *bmag_complex;
EXTERN_SWITCH float* jacobian;

typedef struct {
	float *gbdrift;
  float *grho;
  float *cvdrift;
  float *bmag;
  float *bgrad;
  float *gds2;
	float *gds21;
  float *gds22;
  float *cvdrift0;
  float *gbdrift0;
  float *jacobian;
	float * Rplot;
	float * Zplot;
	float * aplot;
	float * Rprime;
	float * Zprime;
	float * aprime;

	//float drhodpsi;
	float gradpar;

	float fluxDen;

	cuComplex * bmag_complex;
	float * bmagInv;
} geometry_coefficents_struct;

void read_geo(grids_struct * grids, geometry_coefficents_struct * geo, struct gryfx_parameters_struct * gryfxpars);
