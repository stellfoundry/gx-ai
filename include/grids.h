//Globals... to be deleted!!
EXTERN_SWITCH int Nx, Ny, Nz;
EXTERN_SWITCH float *kx, *ky, *kz, *kz_complex, *kx_abs;
EXTERN_SWITCH float *kx_h, *ky_h, *kz_h;

typedef struct {
	float * ky;
	float * kx;
	float * z;
	float * kz;
	int Nx;
	int Ny;
	int Nz;
	int Nspecies; /* set equal to pars.nspec */

	int NxNyc;
	int NxNy;
	int NxNycNz;
	int NxNz;
	int NycNz;

	/* The number of evolved modes, as opposed to the number used
	 * in dealiasing */
	int Naky;
	int Nakx;

	/* The number of evolved complex modes, which for real fields
	 * is approx half the number of real gridpoints*/
	int Ny_complex;
	int Nz_complex;

	/* A grid the size of kx, true if in the dealiased zone*/
	bool * kx_mask;

	/* Flow shear arrays*/
	float * kx_shift;
	int * jump;

	/* Stuff for z covering */
	int nClasses;
	/* Note: the arrays below are allocated in initialize_z_covering, not
	 * in the main allocation routines. Also note that the ** 
	 * pointers to pointers point to arrays of pointers on the
	 * host. The pointers themselves then may point to arrays 
	 * on either the device or host */
	int * nLinks;
	int * nChains;
	int ** kxCover;
	int ** kyCover;
	cuComplex ** g_covering;
	float ** kz_covering;

} grids_struct;
