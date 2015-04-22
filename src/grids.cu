#define EXTERN_SWITCH extern
#include "cufft.h"
#include "grids.h"

void set_grid_masks_and_unaliased_sizes(grids_struct * grids){
	grids->Naky =(grids->Ny-1)/3+1;
	grids->Ny_complex = grids->Ny/2+1;
	grids->Nakx = grids->Nx - (2*grids->Nx/3+1 - ((grids->Nx-1)/3+1));
	grids->NxNyc = grids->Nx * grids->Ny_complex;
	grids->NxNy = grids->Nx * grids->Ny;
	grids->NxNycNz = grids->Nx * grids->Ny_complex * grids->Nz;
	grids->NxNz = grids->Nx * grids->Nz;
	grids->NycNz = grids->Ny_complex * grids->Nz;
}
