#define EXTERN_SWITCH extern
#include "cufft.h"
#include "simpledataio_cuda.h"
#include "everything_struct.h"
#include "write_data.h"
//#include "global_vars.h"

/* Allocate and set run name */
/* If input file is "myrun.in", run_name is "myrun"*/
void writedat_set_run_name(char ** run_name_ptr, char * input_file){

	*run_name_ptr = (char*)malloc(sizeof(char)*(strlen(input_file)-3+1));
	printf("Allocated run name pointer\n");
	strncpy(*run_name_ptr, input_file, strlen(input_file)-3);
	printf("Copied run name\n");
	(*run_name_ptr)[strlen(input_file)-3] = '\0';
	printf("Run name is %s\n", *run_name_ptr);


}

int dims_are(struct sdatio_variable * svar, char * dimensions){
	return !(strcmp(svar->dimension_list, dimensions));
}

void writedat_mask_trans_write_variable(struct sdatio_file * sfile, char * variable_name, void * address){
	struct sdatio_dimension * tdim;
	tdim = sdatio_find_dimension(sfile, "t");
	/* The current value of the t index in the output file*/
	int tstart = tdim->start;

	unsigned char * address_char = (unsigned char *)address;

	struct sdatio_variable * svar;
	svar = sdatio_find_variable(sfile, variable_name);
	int sz = svar->type_size;

	if (dims_are(svar, "t") || 
			dims_are(svar, "z") || 
			dims_are(svar, "zr") || 
			dims_are(svar, "tz") || 
			dims_are(svar, "tzr") || 
			dims_are(svar, "y") || 
			dims_are(svar, "yr") || 
			dims_are(svar, "ty") || 
			dims_are(svar, "tyr") || 
			dims_are(svar, "s") || 
			dims_are(svar, "sr") || 
			dims_are(svar, "ts") || 
			dims_are(svar, "tsr") || 
			dims_are(svar, "tr")
		)
		sdatio_write_variable(sfile, variable_name, address);
	else if (dims_are(svar, "x") ||
			     dims_are(svar, "xr") ||
			     dims_are(svar, "tx") ||
					 dims_are(svar, "txr") )
	{
		int out_index[3];
		out_index[0] = tstart;
		out_index[1] = 0;
		out_index[2] = 0;

		int * out_ptr;
		if (dims_are(svar, "x") || dims_are(svar, "xr")) out_ptr = &out_index[1];
		else out_ptr = &out_index[0];

		int ri_count;
		if (dims_are(svar, "xr")|| dims_are(svar, "txr")) ri_count = 2;
		else ri_count = 1;

  	for(int i=0; i<Nx; i++) {
			if (i>=((Nx-1)/3+1) && i<(2*Nx/3+1)) continue;
			for (int j=0; j<ri_count; j++){
				out_index[2] = j;
				sdatio_write_variable_at_index_fast(sfile, svar, out_ptr, &address_char[(i*ri_count+j)*sz]);
			}
			out_index[1]++;
		}
	}
	else if (dims_are(svar, "yx") ||
			     dims_are(svar, "yxr") ||
			     dims_are(svar, "tyx") ||
					 dims_are(svar, "tyxr") )
	{
		int out_index[4];
		out_index[0] = tstart;
		out_index[1] = 0;
		out_index[2] = 0;
		out_index[3] = 0;

		int * out_ptr;
		if (dims_are(svar, "yx") || dims_are(svar, "yxr")) out_ptr = &out_index[1];
		else out_ptr = &out_index[0];

		int ri_count;
		if (dims_are(svar, "yxr")|| dims_are(svar, "tyxr")) ri_count = 2;
		else ri_count = 1;

  	for(int iy=0; iy<((Ny-1)/3+1); iy++) {
			out_index[2] = 0;
  		for(int i=0; i<Nx; i++) {
				if (i>=((Nx-1)/3+1) && i<(2*Nx/3+1)) continue;
				for (int j=0; j<ri_count; j++){
					out_index[3] = j;
					sdatio_write_variable_at_index_fast(sfile, svar, out_ptr, &address_char[(i*(Ny/2+1)*ri_count+iy*ri_count+j)*sz]);
				}
				out_index[2]++;
			}
			out_index[1]++;
		}
	}
	else if (dims_are(svar, "yxz") ||
			     dims_are(svar, "yxzr") ||
			     dims_are(svar, "tyxz") ||
					 dims_are(svar, "tyxzr") )
	{
		int out_index[5];
		out_index[0] = tstart;
		out_index[1] = 0;
		out_index[2] = 0;
		out_index[3] = 0;
		out_index[4] = 0;

		int * out_ptr;
		if (dims_are(svar, "yxz") || dims_are(svar, "yxzr")) out_ptr = &out_index[1];
		else out_ptr = &out_index[0];

		int ri_count;
		if (dims_are(svar, "yxzr")|| dims_are(svar, "tyxzr")) ri_count = 2;
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

void writedat_beginning(everything_struct * ev)
{  

	char * filename;
	struct sdatio_file * sdatfile = &(ev->outs.sdatfile);
	filename = (char*)malloc(sizeof(char)*(strlen(ev->info.run_name)+5));
	sprintf(filename, "%s.cdf", ev->info.run_name);
  sdatio_init(sdatfile, filename);
  sdatio_create_file(sdatfile);
  
  sdatio_add_dimension(sdatfile, "r", 2, "real and imag parts", "(none)");
    //for(int i=(Nx-1)/3+1; i<2*Nx/3+1; i++) {
     // for(int j=((Ny-1)/3+1); j<(Ny/2+1); j++) {
  sdatio_add_dimension(sdatfile, "x", Nx - (2*Nx/3+1 - ((Nx-1)/3+1)), "kx coordinate", "1/rho_i");
  sdatio_add_dimension(sdatfile, "y", ((Ny-1)/3+1), "ky coordinate", "1/rho_i");
  sdatio_add_dimension(sdatfile, "z", Nz, "z coordinate", "a");
  sdatio_add_dimension(sdatfile, "s", nSpecies, "species coordinate", "(none)");
  sdatio_add_dimension(sdatfile, "t", SDATIO_UNLIMITED, "time coordinate","a/vt_i");

  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "kx", "x", "kx grid", "1/rho_i");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "ky", "y", "ky grid", "1/rho_i");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "theta", "z", "theta grid (parallel coordinate, referred to as z within gryfx)", "radians");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "t", "t", "Values of time", "a/vt_i");
  
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi", "yxzr", "Electric potential", "Ti/e");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi_t", "tyxzr", "Electric potential as a function of time.", "Ti/e");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi2", "t", "phi**2 summed over all modes", "(Ti/e)**2");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi2_by_ky", "ty", "phi^2 summed over all kx as a function of ky and time", "(Ti/e)**2");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi2_by_kx", "tx", "phi^2 summed over all ky as a function of kx and time", "(Ti/e)**2");
  sdatio_create_variable(sdatfile, SDATIO_FLOAT, "omega", "tyxr", "Real part is frequency, imaginary part is growth rate, as a function of time, x and y", "v_ti/a");

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

  
  writedat_mask_trans_write_variable(sdatfile, "kx", &(ev->grids.kx[0]));
  writedat_mask_trans_write_variable(sdatfile, "ky", &(ev->grids.ky[0]));
  writedat_mask_trans_write_variable(sdatfile, "theta", &(ev->grids.z[0]));

	geometry_coefficents_struct geo = ev->geo;
	writedat_mask_trans_write_variable(sdatfile, "Rplot",  &(geo.Rplot[0]));
	writedat_mask_trans_write_variable(sdatfile, "Zplot",  &(geo.Zplot[0]));
	writedat_mask_trans_write_variable(sdatfile, "aplot",  &(geo.aplot[0]));
	writedat_mask_trans_write_variable(sdatfile, "Rprime", &(geo.Rprime[0]));
	writedat_mask_trans_write_variable(sdatfile, "Zprime", &(geo.Zprime[0]));
	writedat_mask_trans_write_variable(sdatfile, "aprime", &(geo.aprime[0]));
	sdatio_print_variables(sdatfile);
  
}


/* No need to pass outs by reference as we are not modifiying anything*/
void writedat_each(outputs_struct * outs, fields_struct * flds, time_struct * time)
{
	struct sdatio_file * sdatfile = &(outs->sdatfile);
  writedat_mask_trans_write_variable(sdatfile, "t", &(time->runtime));
  writedat_mask_trans_write_variable(sdatfile, "phi", &(flds->phi[0]));
  writedat_mask_trans_write_variable(sdatfile, "phi_t", &(flds->phi[0]));
  writedat_mask_trans_write_variable(sdatfile, "phi2", &(outs->phi2));
  writedat_mask_trans_write_variable(sdatfile, "phi2_by_ky", &(outs->phi2_by_ky[0]));
  writedat_mask_trans_write_variable(sdatfile, "phi2_by_kx", &(outs->phi2_by_kx[0]));
  writedat_mask_trans_write_variable(sdatfile, "omega", &(outs->omega[0]));
  writedat_mask_trans_write_variable(sdatfile, "hflux_tot", &(outs->hflux_tot));
  sdatio_increment_start(sdatfile, "t");
  
}

void writedat_end(outputs_struct outs)
{
	struct sdatio_file * sdatfile = &(outs.sdatfile);
  sdatio_close(sdatfile);
}
