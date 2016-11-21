//Globals... to be deleted!
#ifndef NO_GLOBALS
EXTERN_SWITCH cuComplex* phi_h;
EXTERN_SWITCH cuComplex* dens_h;
EXTERN_SWITCH cuComplex* upar_h;
EXTERN_SWITCH cuComplex* tpar_h;
EXTERN_SWITCH cuComplex* tprp_h;
EXTERN_SWITCH cuComplex* qpar_h;
EXTERN_SWITCH cuComplex* qprp_h;
#endif

typedef struct {
	// Whole fields and moments

	cuComplex * phi;
	
	/* On device only*/
	cuComplex * field;
	cuComplex * phi1;

        cuComplex * apar;
        cuComplex * apar1;

	cuComplex ** dens;
	cuComplex ** dens1;
	cuComplex ** upar;
	cuComplex ** upar1;
	cuComplex ** tpar;
	cuComplex ** tpar1;
	cuComplex ** tprp;
	cuComplex ** tprp1;
	cuComplex ** qpar;
	cuComplex ** qpar1;
	cuComplex ** qprp;
	cuComplex ** qprp1;
	
} fields_struct;
