#define NO_GLOBALS true
#include "cufft.h"
#include "simpledataio_cuda.h"
#include "everything_struct.h"
#include "write_data.h"
#include "profile.h"
//#include "global_vars.h"


int dims_are(struct sdatio_variable * svar, char * dimensions){
	return !(strcmp(svar->dimension_list, dimensions));
}

  
/* The purpose of this function is to write the outputs into
 * the netcdf file in exactly the same way that gs2 does, 
 * i.e. to tranpose the indices to the gs2 order and mask
 * the alias meshpoints */
void writedat_mask_trans_write_variable_2(grids_struct * grids, struct sdatio_file * sfile, char * variable_name, void * address, bool mask){
	struct sdatio_dimension * tdim;
	tdim = sdatio_find_dimension(sfile, "t");
	/* The current value of the t index in the output file*/
	int tstart = tdim->start;

  //Local copies for convenience
  int Nx = grids->Nx;
  int Ny = grids->Ny;
  int Nz = grids->Nz;

	unsigned char * address_char = (unsigned char *)address;

	struct sdatio_variable * svar;
	svar = sdatio_find_variable(sfile, variable_name);
	int sz = svar->type_size;

	if (dims_are(svar, "t") || 
			dims_are(svar, "z") || 
			dims_are(svar, "zr") || 
			dims_are(svar, "tz") || 
			dims_are(svar, "tzr") || 
			dims_are(svar, "Y") || 
			dims_are(svar, "Yr") || 
			dims_are(svar, "tY") || 
			dims_are(svar, "tYr") || 
			dims_are(svar, "s") || 
			dims_are(svar, "sr") || 
			dims_are(svar, "ts") || 
			dims_are(svar, "tsr") || 
			dims_are(svar, "tr")
		)
		sdatio_write_variable(sfile, variable_name, address);
	else if (dims_are(svar, "X") ||
			     dims_are(svar, "Xr") ||
			     dims_are(svar, "tX") ||
					 dims_are(svar, "tXr") )
	{
		int out_index[3];
		out_index[0] = tstart;
		out_index[1] = 0;
		out_index[2] = 0;

		int * out_ptr;
		if (dims_are(svar, "X") || dims_are(svar, "Xr")) out_ptr = &out_index[1];
		else out_ptr = &out_index[0];

		int ri_count;
		if (dims_are(svar, "Xr")|| dims_are(svar, "tXr")) ri_count = 2;
		else ri_count = 1;

  	for(int i=0; i<Nx; i++) {
			if (mask){if (i>=((Nx-1)/3+1) && i<(2*Nx/3+1)) continue; }
      else { if ( i>= grids->Nakx ) continue;}
			for (int j=0; j<ri_count; j++){
				out_index[2] = j;
				sdatio_write_variable_at_index_fast(sfile, svar, out_ptr, &address_char[(i*ri_count+j)*sz]);
			}
			out_index[1]++;
		}
	}
	else if (dims_are(svar, "YX") ||
			     dims_are(svar, "YXr") ||
			     dims_are(svar, "tYX") ||
					 dims_are(svar, "tYXr") )
	{
		int out_index[4];
		out_index[0] = tstart;
		out_index[1] = 0;
		out_index[2] = 0;
		out_index[3] = 0;

		int * out_ptr;
		if (dims_are(svar, "YX") || dims_are(svar, "YXr")) out_ptr = &out_index[1];
		else out_ptr = &out_index[0];

		int ri_count;
		if (dims_are(svar, "YXr")|| dims_are(svar, "tYXr")) ri_count = 2;
		else ri_count = 1;

  	for(int iy=0; iy<((Ny-1)/3+1); iy++) {
			out_index[2] = 0;
  		for(int i=0; i<Nx; i++) {
        if (mask){if (i>=((Nx-1)/3+1) && i<(2*Nx/3+1)) continue; }
        else { if ( i>= grids->Nakx ) continue;}
				for (int j=0; j<ri_count; j++){
					out_index[3] = j;
					sdatio_write_variable_at_index_fast(sfile, svar, out_ptr, &address_char[(i*(Ny/2+1)*ri_count+iy*ri_count+j)*sz]);
				}
				out_index[2]++;
			}
			out_index[1]++;
		}
	}
	else if (dims_are(svar, "YXz") ||
			     dims_are(svar, "YXzr") ||
			     dims_are(svar, "tYXz") ||
					 dims_are(svar, "tYXzr") )
	{
		int out_index[5];
		out_index[0] = tstart;
		out_index[1] = 0;
		out_index[2] = 0;
		out_index[3] = 0;
		out_index[4] = 0;

		int * out_ptr;
		if (dims_are(svar, "YXz") || dims_are(svar, "YXzr")) out_ptr = &out_index[1];
		else out_ptr = &out_index[0];

		int ri_count;
		if (dims_are(svar, "YXzr")|| dims_are(svar, "tYXzr")) ri_count = 2;
		else ri_count = 1;

  	for(int iy=0; iy<((Ny-1)/3+1); iy++) {
			out_index[2] = 0;
  		for(int ix=0; ix<Nx; ix++) {
				out_index[3] = 0;
				if (ix>=((Nx-1)/3+1) && ix<(2*Nx/3+1)) continue;
				for (int iz=0; iz<Nz; iz++){
					for (int j=0; j<ri_count; j++){
						out_index[4] = j;
						sdatio_write_variable_at_index_fast(sfile, svar, out_ptr, 
								&address_char[(iz*(Ny/2+1)*Nx*ri_count + ix*(Ny/2+1)*ri_count+iy*ri_count+j)*sz]);
					}
					out_index[3]++;
				}
				out_index[2]++;
			}
			out_index[1]++;
		}
	}
	else {
		printf("Can't handle these dimensions: %s in writedat_mask_trans_write_variable\n", svar->dimension_list);
		abort();
	}	

	sdatio_sync(sfile);

}

// Call without specifying mask
void writedat_mask_trans_write_variable(grids_struct * grids, struct sdatio_file * sfile, char * variable_name, void * address){
  writedat_mask_trans_write_variable_2(grids, sfile, variable_name, address, true);
}


void writedat_beginning(everything_struct * ev)
{  

	char * filename;
	struct sdatio_file * sdatfile = &(ev->outs.sdatfile);
	filename = (char*)malloc(sizeof(char)*(strlen(ev->info.run_name)+5));
	sprintf(filename, "%s.cdf", ev->info.run_name);
  sdatio_init(sdatfile, filename);
  sdatio_create_file(sdatfile);
  
  int Nx = ev->grids.Nx;
  int Ny = ev->grids.Ny;
  int Nz = ev->grids.Nz;

  sdatio_add_dimension(sdatfile, "r", 2, "real and imag parts", "(none)");
    //for(int i=(Nx-1)/3+1; i<2*Nx/3+1; i++) {
     // for(int j=((Ny-1)/3+1); j<(Ny/2+1); j++) {
  sdatio_add_dimension(sdatfile, "X", Nx - (2*Nx/3+1 - ((Nx-1)/3+1)), "kx coordinate", "1/rho_i");
  sdatio_add_dimension(sdatfile, "Y", ((Ny-1)/3+1), "ky coordinate", "1/rho_i");
  sdatio_add_dimension(sdatfile, "z", Nz, "z coordinate", "a");
  sdatio_add_dimension(sdatfile, "s", ev->pars.nspec, "species coordinate", "(none)");
  sdatio_add_dimension(sdatfile, "t", SDATIO_UNLIMITED, "time coordinate","a/vt_i");

  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "kx", "X", "kx grid", "1/rho_i");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "ky", "Y", "ky grid", "1/rho_i");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "theta", "z", "theta grid (parallel coordinate, referred to as z within gryfx)", "radians");
  sdatio_create_variable(sdatfile, SDATIO_DOUBLE, "t", "t", "Values of time", "a/vt_i");
  
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi", "YXzr", "Electric potential", "Ti/e");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi_t", "tYXzr", "Electric potential as a function of time.", "Ti/e");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi2", "t", "phi**2 summed over all modes", "(Ti/e)**2");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi2_by_ky", "tY", "phi^2 summed over all kx as a function of ky and time", "(Ti/e)**2");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi2_by_kx", "tX", "phi^2 summed over all ky as a function of kx and time", "(Ti/e)**2");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "omega", "tYXr", "Real part is frequency, imaginary part is growth rate, as a function of time, x and y", "v_ti/a");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "omega_average", "tYXr", "Average frequency and growth rate, as a function of time, x and y", "v_ti/a");

	/* Fluxes */
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "hflux_tot", "t", "total heat flux", "(figure it out)");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "es_heat_flux", "ts", " heat fluxi by species", "(figure it out)");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "es_mom_flux", "ts", "momentum flux by species", "(figure it out)");

	/* Plotting coefficients for making visualisations. */
	sdatio_create_variable(sdatfile, SDATIO_FLOAT, "Rplot", "z", "Geometric plotting coefficient", "a");
	sdatio_create_variable(sdatfile, SDATIO_FLOAT, "Zplot", "z", "Geometric plotting coefficient", "a");
	sdatio_create_variable(sdatfile, SDATIO_FLOAT, "aplot", "z", "Geometric plotting coefficient", "a");
	sdatio_create_variable(sdatfile, SDATIO_FLOAT, "Rprime", "z", "Geometric plotting coefficient", "a");
	sdatio_create_variable(sdatfile, SDATIO_FLOAT, "Zprime", "z", "Geometric plotting coefficient", "a");
	sdatio_create_variable(sdatfile, SDATIO_FLOAT, "aprime", "z", "Geometric plotting coefficient", "a");

  
  writedat_mask_trans_write_variable(&ev->grids, sdatfile, "kx", &(ev->grids.kx[0]));
  writedat_mask_trans_write_variable(&ev->grids, sdatfile, "ky", &(ev->grids.ky[0]));
  writedat_mask_trans_write_variable(&ev->grids, sdatfile, "theta", &(ev->grids.z[0]));

	geometry_coefficents_struct geo = ev->geo;
	writedat_mask_trans_write_variable(&ev->grids, sdatfile, "Rplot",  &(geo.Rplot[0]));
	writedat_mask_trans_write_variable(&ev->grids, sdatfile, "Zplot",  &(geo.Zplot[0]));
	writedat_mask_trans_write_variable(&ev->grids, sdatfile, "aplot",  &(geo.aplot[0]));
	writedat_mask_trans_write_variable(&ev->grids, sdatfile, "Rprime", &(geo.Rprime[0]));
	writedat_mask_trans_write_variable(&ev->grids, sdatfile, "Zprime", &(geo.Zprime[0]));
	writedat_mask_trans_write_variable(&ev->grids, sdatfile, "aprime", &(geo.aprime[0]));
	sdatio_print_variables(sdatfile);
  
}


/* No need to pass outs by reference as we are not modifiying anything*/
void writedat_each(grids_struct * grids, outputs_struct * outs, fields_struct * flds, time_struct * time)
{
	struct sdatio_file * sdatfile = &(outs->sdatfile);
#ifdef PROFILE
PUSH_RANGE("writedat: t",0);
#endif
  writedat_mask_trans_write_variable(grids, sdatfile, "t", &(time->runtime));
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: phi",1);
#endif
  writedat_mask_trans_write_variable(grids, sdatfile, "phi", &(flds->phi[0]));
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: phi_t",2);
#endif
  writedat_mask_trans_write_variable(grids, sdatfile, "phi_t", &(flds->phi[0]));
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: phi2",3);
#endif
  writedat_mask_trans_write_variable(grids, sdatfile, "phi2", &(outs->phi2));
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: phi2_by_ky",4);
#endif
  writedat_mask_trans_write_variable(grids, sdatfile, "phi2_by_ky", &(outs->phi2_by_ky[0]));
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: phi2_by_kx",5);
#endif
  writedat_mask_trans_write_variable(grids, sdatfile, "phi2_by_kx", &(outs->phi2_by_kx[0]));
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: omega",0);
#endif
  writedat_mask_trans_write_variable(grids, sdatfile, "omega", &(outs->omega[0]));
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: omega_average",1);
#endif
  writedat_mask_trans_write_variable_2(grids, sdatfile, "omega_average", &(outs->omega_out[0]), false);
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: hflux_tot",2);
#endif
  writedat_mask_trans_write_variable(grids, sdatfile, "hflux_tot", &(outs->hflux_tot));
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: es_heat_flux",3);
#endif
  writedat_mask_trans_write_variable(grids, sdatfile, "es_heat_flux", &(outs->hflux_by_species));
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("sdatio_increment_start",4);
#endif
  sdatio_increment_start(sdatfile, "t");
#ifdef PROFILE
POP_RANGE;
#endif
  
}

void writedat_end(outputs_struct outs)
{
	struct sdatio_file * sdatfile = &(outs.sdatfile);
  sdatio_close(sdatfile);
}
