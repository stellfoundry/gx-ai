#define NO_GLOBALS true
#include "cufft.h"
#include "simpledataio_cuda.h"
#include "everything_struct.h"
#include "write_data.h"
#include "profile.h"
//#include "global_vars.h"


static int dims_are(struct sdatio_variable * svar, char * dimensions){
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

  unsigned char * temp_store;

	struct sdatio_variable * svar;
	svar = sdatio_find_variable(sfile, variable_name);
	int sz = svar->type_size;

	if (dims_are(svar, "t") || 
			dims_are(svar, "theta") || 
			dims_are(svar, "theta,ri") || 
			dims_are(svar, "t,theta") || 
			dims_are(svar, "t,theta,ri") || 
			dims_are(svar, "ky") || 
			dims_are(svar, "ky,ri") || 
			dims_are(svar, "t,ky") || 
			dims_are(svar, "t,ky,ri") || 
			dims_are(svar, "species") || 
			dims_are(svar, "species,ri") || 
			dims_are(svar, "t,species") || 
			dims_are(svar, "t,species,ri") || 
			dims_are(svar, "t,ri")
		)
    temp_store = address_char;
		//sdatio_write_variable(sfile, variable_name, address);
	else if (dims_are(svar, "kx") ||
			     dims_are(svar, "kx,ri") ||
			     dims_are(svar, "t,kx") ||
					 dims_are(svar, "t,kx,ri") )
	{
    // Allocate enough memory to store the unmasked transposed array.
    temp_store = (unsigned char *)malloc(sizeof(unsigned char)*sz*grids->Nx*2);
    int counter = 0;
		int out_index[3];
		out_index[0] = tstart;
		out_index[1] = 0;
		out_index[2] = 0;

		int * out_ptr;
		if (dims_are(svar, "kx") || dims_are(svar, "kx,ri")) out_ptr = &out_index[1];
		else out_ptr = &out_index[0];

		int ri_count;
		if (dims_are(svar, "kx,ri")|| dims_are(svar, "t,kx,ri")) ri_count = 2;
		else ri_count = 1;

  	for(int i=0; i<Nx; i++) {
			if (mask){if (i>=((Nx-1)/3+1) && i<(2*Nx/3+1)) continue; }
      else { if ( i>= grids->Nakx ) continue;}
			//for (int j=0; j<ri_count; j++){
				//out_index[2] = j;
        //printf(" Value of xvar at counter %d, ix %d, is %f\n", counter, i, (float)address_char[i*sz]);
        memcpy(&temp_store[counter], &address_char[i*sz*ri_count], sz*ri_count);
        //printf(" Value of temp_store at counter %d, ix %d, is %f\n", counter, i, (float)temp_store[counter]);
        counter = counter + sz*ri_count;
				//sdatio_write_variable_at_index_fast(sfile, svar, out_ptr, &address_char[(i*ri_count+j)*sz]);
			//}
			out_index[1]++;
		}
	}
	else if (dims_are(svar, "ky,kx") ||
			     dims_are(svar, "ky,kx,ri") ||
			     dims_are(svar, "t,ky,kx") ||
					 dims_are(svar, "t,ky,kx,ri") )
	{
    // Allocate enough memory to store the unmasked transposed array.
    temp_store = (unsigned char *)malloc(sizeof(unsigned char)*sz*grids->Nx*grids->Ny*2);
    int counter = 0;
		int out_index[4];
		out_index[0] = tstart;
		out_index[1] = 0;
		out_index[2] = 0;
		out_index[3] = 0;

		int * out_ptr;
		if (dims_are(svar, "ky,kx") || dims_are(svar, "ky,kx,ri")) out_ptr = &out_index[1];
		else out_ptr = &out_index[0];

		int ri_count;
		if (dims_are(svar, "ky,kx,ri")|| dims_are(svar, "t,ky,kx,ri")) ri_count = 2;
		else ri_count = 1;

  	for(int iy=0; iy<((Ny-1)/3+1); iy++) {
			out_index[2] = 0;
  		for(int i=0; i<Nx; i++) {
        if (mask){if (i>=((Nx-1)/3+1) && i<(2*Nx/3+1)) continue; }
        else { if ( i>= grids->Nakx ) continue;}
				//for (int j=0; j<ri_count; j++){
					//out_index[3] = j;
					//sdatio_write_variable_at_index_fast(sfile, svar, out_ptr, &address_char[(i*(Ny/2+1)*ri_count+iy*ri_count+j)*sz]);
				  memcpy(&temp_store[counter], &address_char[(i*(Ny/2+1)+iy)*sz*ri_count ], sz*ri_count);
          counter = counter + sz*ri_count;
				//}
				out_index[2]++;
			}
			out_index[1]++;
		}
	}
	else if (dims_are(svar, "ky,kx,theta") ||
			     dims_are(svar, "ky,kx,theta,ri") ||
			     dims_are(svar, "t,ky,kx,theta") ||
					 dims_are(svar, "t,ky,kx,theta,ri") )
	{
    // Allocate enough memory to store the unmasked transposed array.
    temp_store = (unsigned char *)malloc(sizeof(unsigned char)*sz*grids->Nx*grids->Ny*grids->Nz*2);
    int counter = 0;
		int out_index[5];
		out_index[0] = tstart;
		out_index[1] = 0;
		out_index[2] = 0;
		out_index[3] = 0;
		out_index[4] = 0;

		int * out_ptr;
		if (dims_are(svar, "ky,kx,theta") || dims_are(svar, "ky,kx,theta,ri")) out_ptr = &out_index[1];
		else out_ptr = &out_index[0];

		int ri_count;
		if (dims_are(svar, "ky,kx,theta,ri")|| dims_are(svar, "t,ky,kx,theta,ri")) ri_count = 2;
		else ri_count = 1;

  	for(int iy=0; iy<((Ny-1)/3+1); iy++) {
			out_index[2] = 0;
  		for(int ix=0; ix<Nx; ix++) {
				out_index[3] = 0;
				if (ix>=((Nx-1)/3+1) && ix<(2*Nx/3+1)) continue;
				for (int iz=0; iz<Nz; iz++){
					//for (int j=0; j<ri_count; j++){
					//	out_index[4] = j;
						//sdatio_write_variable_at_index_fast(sfile, svar, out_ptr, 
						//		&address_char[(iz*(Ny/2+1)*Nx*ri_count + ix*(Ny/2+1)*ri_count+iy*ri_count+j)*sz]);
            memcpy(&temp_store[counter], &address_char[(iz*(Ny/2+1)*Nx + ix*(Ny/2+1)+iy)*sz*ri_count], sz*ri_count);
            counter = counter + sz*ri_count;
            //printf(" Value of temp_store, address_char for var %s at counter %d, ix %d, iy %d, iz %d, is %f, %f\n", variable_name, counter, ix, iy, iz, (float)temp_store[counter], (float)address_char);
					//}
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

  sdatio_write_variable(sfile, variable_name, &*temp_store);
  if (temp_store!=address_char) free(temp_store);

	//sdatio_sync(sfile);

}

// Call without specifying mask
void writedat_mask_trans_write_variable(grids_struct * grids, struct sdatio_file * sfile, char * variable_name, void * address){
  writedat_mask_trans_write_variable_2(grids, sfile, variable_name, address, true);
  //sdatio_write_variable(sfile, variable_name, address);
}
void writedat_mask_trans_read_variable(grids_struct * grids, struct sdatio_file * sfile, char * variable_name, void * address){
  //writedat_mask_trans_write_variable_2(grids, sfile, variable_name, address, true);
  //sdatio_write_variable(sfile, variable_name, address);
}

void writedat_create_write_read(diagnostics_file_struct * gnostics, grids_struct * grids, 
      int stype, char * variable_name, char * dims, char * description, char * units, void * address){
  if (gnostics->create) 
    sdatio_create_variable(&gnostics->sdatfile, stype, variable_name, dims, description, units);
  if (gnostics->write)
    if (gnostics->trans)
      writedat_mask_trans_write_variable(grids, &gnostics->sdatfile, variable_name, address); 
    else 
      sdatio_write_variable(&gnostics->sdatfile, variable_name, address);
  else if (gnostics->read)
    if (gnostics->trans)
      writedat_mask_trans_read_variable(grids, &gnostics->sdatfile, variable_name, address); 
    else 
      sdatio_read_variable(&gnostics->sdatfile, variable_name, address);
}

void writedat_beginning(everything_struct * ev)
{  

	char * filename;
	struct sdatio_file * sdatfile = &(ev->outs.gnostics.sdatfile);
  outputs_struct * outs = &ev->outs;
  grids_struct * grids = &ev->grids;
  diagnostics_file_struct * gnostics = &outs->gnostics;
  bool appending; 


	filename = (char*)malloc(sizeof(char)*(strlen(ev->info.run_name)+5));
	sprintf(filename, "%s.cdf", ev->info.run_name);
  sdatio_init(sdatfile, filename);

  appending = false;
  if (ev->pars.append_old){
    FILE * temp_file;
    // If an output file already exists we open it and
    // append to it. Otherwise we create a new one.
    temp_file = fopen(filename, "r");
    if (temp_file!=NULL){
      fclose(temp_file);
      appending = true;
      printf("Appending to file %s\n", filename);
    }
  }
  if (appending) {
    sdatio_open_file(sdatfile);
    struct sdatio_dimension * tdim;
    tdim = sdatio_find_dimension(sdatfile, "t");
    printf("Time index start is %d\n", tdim->start);
  }
  else sdatio_create_file(sdatfile);


  // Setup initial status for creation only.
  gnostics->create = !appending;
  gnostics->write = false;
  gnostics->read = false;
  gnostics->trans = true;


  
  int Nx = ev->grids.Nx;
  int Ny = ev->grids.Ny;
  int Nz = ev->grids.Nz;

  if (!appending) {
    sdatio_add_dimension(sdatfile, "ri", 2, "real and imag parts", "(none)");
      //for(int i=(Nx-1)/3+1; i<2*Nx/3+1; i++) {
       // for(int j=((Ny-1)/3+1); j<(Ny/2+1); j++) {
    sdatio_add_dimension(sdatfile, "kx", Nx - (2*Nx/3+1 - ((Nx-1)/3+1)), "kx coordinate", "1/rho_i");
    sdatio_add_dimension(sdatfile, "ky", ((Ny-1)/3+1), "ky coordinate", "1/rho_i");
    sdatio_add_dimension(sdatfile, "theta", Nz, "z coordinate", "a");
    sdatio_add_dimension(sdatfile, "species", ev->pars.nspec, "species coordinate", "(none)");
    sdatio_add_dimension(sdatfile, "t", SDATIO_UNLIMITED, "time coordinate","a/vt_i");
  }
  if (ev->time.trinity_timestep > 0 && !sdatio_dimension_exists(sdatfile, "ttrin")){
    sdatio_add_dimension(sdatfile, "trinstep", SDATIO_UNLIMITED, "trinity timestep","");
    sdatio_add_dimension(sdatfile, "triniter", SDATIO_UNLIMITED, "trinity iteration","");
    sdatio_add_dimension(sdatfile, "ttrin", SDATIO_UNLIMITED, "gryfx timesteps in this trin iteration","");
  }
  if (ev->time.trinity_timestep > 0){
    if (ev->time.trinity_conv_count==0) sdatio_set_dimension_start(sdatfile, "ttrin", 0);
    sdatio_set_dimension_start(sdatfile, "trinstep", ev->time.trinity_timestep - 1);
    sdatio_set_dimension_start(sdatfile, "triniter", ev->time.trinity_iteration - 1);
  }


  // Temporarily set write true for writing constant data
  gnostics->write = true;
  
  /*Create and write grids*/
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, 
      "kx", "kx", "kx grid", "1/rho_i", &*grids->kx);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, 
      "ky", "ky", "ky grid", "1/rho_i", &*grids->kx);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT,
      "theta", "theta", "theta grid (parallel coordinate, referred to as z within gryfx)", "radians", &*grids->z);

	/* Plotting coefficients for making visualisations. */
	geometry_coefficents_struct * geo = &ev->geo;
	writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "Rplot", "theta", 
      "Geometric plotting coefficient", "a", &*geo->Rplot);
	writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "Zplot", "theta", 
      "Geometric plotting coefficient", "a", &*geo->Zplot);
	writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "aplot", "theta", 
      "Geometric plotting coefficient", "a", &*geo->aplot);
	writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "Rprime", "theta", 
      "Geometric plotting coefficient", "a", &*geo->Rprime);
	writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "Zprime", "theta", 
      "Geometric plotting coefficient", "a", &*geo->Zprime);
	writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "aprime", "theta", 
      "Geometric plotting coefficient", "a", &*geo->aprime);
  gnostics->write = false;

  gnostics->write = false;
  
  //sdatio_create_variable(sdatfile, SDATIO_FLOAT, "phi_t", "t,ky,kx,z,ri", "Electric potential as a function of time.", "Ti/e");

	/* Fluxes */

	sdatio_print_variables(sdatfile);
  
}


/* No need to pass outs by reference as we are not modifiying anything*/
void writedat_each(grids_struct * grids, outputs_struct * outs, fields_struct * flds, time_struct * time)
{
	struct sdatio_file * sdatfile = &outs->gnostics.sdatfile;
  diagnostics_file_struct * gnostics = &outs->gnostics;
#ifdef PROFILE
PUSH_RANGE("writedat: t",0);
#endif
  writedat_create_write_read(gnostics, grids, SDATIO_DOUBLE, "t", "t", "Values of time", "a/vt_i", &time->runtime);
  writedat_create_write_read(gnostics, grids, SDATIO_DOUBLE, "dt", "t", "Values of the timestep", "a/vt_i", &time->dt);
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: phi",1);
#endif
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "phi", "ky,kx,theta,ri", "Electric potential", "Ti/e", &*flds->phi);
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: phi_t",2);
#endif
  //writedat_mask_trans_write_variable(grids, sdatfile, "phi_t", &(flds->phi[0]));
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: phi2",3);
#endif
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "phi2", "t", "phi**2 summed over all modes", "(Ti/e)**2", &outs->phi2);
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: phi2_by_ky",4);
#endif
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "phi2_by_ky", "t,ky", "phi^2 summed over all kx as a function of ky and time", "(Ti/e)**2", &*outs->phi2_by_ky);
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: phi2_by_kx",5);
#endif
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "phi2_by_kx", "t,kx", "phi^2 summed over all ky as a function of kx and time", "(Ti/e)**2", &*outs->phi2_by_kx);
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: omega",0);
#endif
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "omega", "t,ky,kx,ri", "Real part is frequency, imaginary part is growth rate, as a function of time, x and y", "v_ti/a", &*outs->omega);
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: omega_average",1);
#endif
  //writedat_mask_trans_write_variable_2(grids, sdatfile, "omega_average", &(outs->omega_out[0]), false);
  //sdatio_create_variable(sdatfile, SDATIO_FLOAT, "omega_average", "t,ky,kx,ri", "Average frequency and growth rate, as a function of time, x and y", "v_ti/a");
  //writedat_mask_trans_write_variable(grids, sdatfile, "omega_average", &(outs->omega_out[0]));
  // omega_average not masked... needs fixing
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: hflux_tot",2);
#endif

  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "hflux_tot", "t", "total heat flux", "(figure it out)", &outs->hflux_tot);
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("writedat: es_heat_flux",3);
#endif
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "es_heat_flux", "t,species", " heat fluxi by species", "(figure it out)", &*outs->hflux_by_species);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "es_heat_flux_avg", "t,species", " heat flux by species moving average", "(figure it out)", &*outs->hflux_by_species_movav);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "mu_avg", "t", "Factor in exponential moving average (roughly 1 - alpha) when navg is high", "1", &outs->mu_avg);
#ifdef PROFILE
POP_RANGE;
PUSH_RANGE("sdatio_increment_start",4);
#endif
  if (gnostics->write) sdatio_increment_start(sdatfile, "t");
#ifdef PROFILE
POP_RANGE;
#endif

if (time->trinity_timestep > 0){
  gnostics->trans=false;
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "hflux_tot_trin", "trinstep,triniter,ttrin", "total heat flux for this trinity iteration", "(figure it out)", &outs->hflux_tot);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "es_heat_flux_trin", "trinstep,triniter,ttrin,species", " heat fluxi by species for this trinity iteration", "(figure it out)", &*outs->hflux_by_species);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "es_heat_flux_avg_trin", "trinstep,triniter,ttrin,species", " heat flux by species moving average for this trinity iteration", "(figure it out)", &*outs->hflux_by_species_movav);
  writedat_create_write_read(gnostics, grids, SDATIO_DOUBLE, "ttrin", "trinstep,triniter,ttrin", "Values of time for this trinity iteration", "a/vt_i", &time->runtime);
  if (gnostics->write) sdatio_increment_start(sdatfile, "ttrin");
  gnostics->trans=true;
}
  
}

void writedat_create_write_read_gpu(diagnostics_file_struct * gnostics, grids_struct * grids, 
      int stype, char * variable_name, char * dims, char * description, char * units, 
      int mem_size, void * address){
  void * temp_address = malloc(mem_size);
  if (gnostics->write) cudaMemcpy(temp_address, address, mem_size, cudaMemcpyDeviceToHost);
  writedat_create_write_read(gnostics, grids, stype, variable_name, dims, description, units, temp_address);
  if (gnostics->read) cudaMemcpy(address, temp_address, mem_size, cudaMemcpyHostToDevice);
  free(temp_address);
}

void writedat_io_restart(diagnostics_file_struct * gnostics, everything_struct * ev_h, everything_struct * ev_hd)
{
  grids_struct * grids = &ev_h->grids;
  outputs_struct * outs = &ev_h->outs;
  //Local copies for convenience
  int Nx = grids->Nx;
  int Ny = grids->Ny;
  int Nz = grids->Nz;

  //Read/write data that is stored on the host
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "hflux_by_species_movav", "species",
      "Moving average of species heat flux", "(figure it out)", &*outs->hflux_by_species_movav);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "pflux_by_species_movav", "species",
      "Moving average of species particle flux", "(figure it out)", &*outs->pflux_by_species_movav);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "expectation_ky_movav", "",
      "Moving average of the ky expectation value (Int ky phi^2 / Int phi^2)", 
      "(figure it out)", &outs->expectation_ky_movav);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "expectation_kx_movav", "",
      "Moving average of the kx expectation value (Int kx phi^2 / Int phi^2)", 
      "(figure it out)", &outs->expectation_kx_movav);

  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "dtSum", "",
      "??", "(figure it out)", &ev_h->time.dtSum);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "Phi_zf_kx1_avg", "",
      "??", "(figure it out)", &ev_h->nlpm.Phi_zf_kx1_avg);
  writedat_create_write_read(gnostics, grids, SDATIO_INT, "counter", "",
      "Timestep counter", "1", &ev_h->time.counter);
  writedat_create_write_read(gnostics, grids, SDATIO_DOUBLE, "runtime", "",
      "Simulation time", "TBC", &ev_h->time.runtime);
  writedat_create_write_read(gnostics, grids, SDATIO_DOUBLE, "dt", "",
      "Current timestep", "TBC", &ev_h->time.dt);
  writedat_create_write_read(gnostics, grids, SDATIO_FLOAT, "timer", "",
      "Current wallclock time", "TBC", &ev_h->time.timer);


  //Now we read/write data that is stored on the gpu.
  writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "phi2_kxky_sum", "kx,ky",
      "Moving average of phi2 by mode", "(figure it out)", sizeof(float)*Nx*(Ny/2+1),
      &*ev_hd->outs.phi2_by_mode_movav);
  writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "phi", "theta,kx,ky,ri",
      "Current value of electrostatic potential phi", "(figure it out)", sizeof(cuComplex)*Nz*Nx*(Ny/2+1),
      &*ev_hd->fields.phi);
  writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "apar", "theta,kx,ky,ri",
      "Current value of the parallel component of the magnetic vector potential", 
      "(figure it out)", sizeof(cuComplex)*Nz*Nx*(Ny/2+1),
      &*ev_hd->fields.apar);
  writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "phi2_zonal_by_kx_movav", "kx",
      "Moving average of the zonal electrostatic field squared", 
      "(figure it out)", sizeof(float)*Nx,
      &*ev_hd->outs.phi2_zonal_by_kx_movav);
  writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "par_corr_kydz_movav", "theta,ky",
      "Moving average of the parallel correlation function", 
      "(figure it out)", sizeof(float)*Nz*(Ny/2+1),
      &*ev_hd->outs.par_corr_kydz_movav);

  for(int s=0; s<ev_h->pars.nspec; s++) 
  {
    if (gnostics->create){
      if (s>0) break;
    }
    else {
      int temp = 1;
      sdatio_set_dimension_start(&gnostics->sdatfile, "species", s);
      sdatio_set_count(&gnostics->sdatfile, "dens", "species", &temp);
      sdatio_set_count(&gnostics->sdatfile, "upar", "species", &temp);
      sdatio_set_count(&gnostics->sdatfile, "tpar", "species", &temp);
      sdatio_set_count(&gnostics->sdatfile, "tprp", "species", &temp);
      sdatio_set_count(&gnostics->sdatfile, "qpar", "species", &temp);
      sdatio_set_count(&gnostics->sdatfile, "qprp", "species", &temp);
    }
    writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "dens", "species,theta,kx,ky,ri",
        "Current value of the density", 
        "(figure it out)", sizeof(cuComplex)*Nz*Nx*(Ny/2+1),
        &*ev_hd->fields.dens[s]);
    writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "upar", "species,theta,kx,ky,ri",
        "Current value of the parallel flow", 
        "(figure it out)", sizeof(cuComplex)*Nz*Nx*(Ny/2+1),
        &*ev_hd->fields.upar[s]);
    writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "tpar", "species,theta,kx,ky,ri",
        "Current value of the parallel temperature", 
        "(figure it out)", sizeof(cuComplex)*Nz*Nx*(Ny/2+1),
        &*ev_hd->fields.tpar[s]);
    writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "tprp", "species,theta,kx,ky,ri",
        "Current value of the perpendicular temperature", 
        "(figure it out)", sizeof(cuComplex)*Nz*Nx*(Ny/2+1),
        &*ev_hd->fields.tprp[s]);
    writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "qpar", "species,theta,kx,ky,ri",
        "Current value of the parallel heat flow", 
        "(figure it out)", sizeof(cuComplex)*Nz*Nx*(Ny/2+1),
        &*ev_hd->fields.qpar[s]);
    writedat_create_write_read_gpu(gnostics, grids, SDATIO_FLOAT, "qprp", "species,theta,kx,ky,ri",
        "Current value of the perpendicular heat flow", 
        "(figure it out)", sizeof(cuComplex)*Nz*Nx*(Ny/2+1),
        &*ev_hd->fields.qprp[s]);
   }


}

void writedat_write_restart(everything_struct * ev_h, everything_struct * ev_hd)
{
  diagnostics_file_struct gnostics_restart;
	char * filename;
	struct sdatio_file * sdatfile = &gnostics_restart.sdatfile;
	filename = (char*)malloc(sizeof(char)*(strlen(ev_h->info.run_name)+8+5));
	sprintf(filename, "%s.restart.cdf", ev_h->info.run_name);
  sdatio_init(sdatfile, filename);
  sdatio_create_file(sdatfile);


  sdatio_add_dimension(sdatfile, "ri", 2, "real and imag parts", "(none)");
    //for(int i=(Nx-1)/3+1; i<2*Nx/3+1; i++) {
     // for(int j=((Ny-1)/3+1); j<(Ny/2+1); j++) {
  sdatio_add_dimension(sdatfile, "kx", ev_h->grids.Nx, "kx coordinate", "1/rho_i");
  sdatio_add_dimension(sdatfile, "ky", (ev_h->grids.Ny/2+1), "ky coordinate", "1/rho_i");
  sdatio_add_dimension(sdatfile, "theta", ev_h->grids.Nz, "z coordinate", "a");
  sdatio_add_dimension(sdatfile, "species", ev_h->pars.nspec, "species coordinate", "(none)");

  //No need to transpose variables in the restart file.
  gnostics_restart.trans = false;

  // Setup initial status for creation only.
  gnostics_restart.create = true;
  gnostics_restart.write = false;
  gnostics_restart.read = false;
  writedat_io_restart(&gnostics_restart, ev_h, ev_hd);

  //Now write variables
  gnostics_restart.create = false;
  gnostics_restart.write = true;
  writedat_io_restart(&gnostics_restart, ev_h, ev_hd);

  sdatio_close(sdatfile);
  sdatio_free(sdatfile);
  free(filename);

}

void writedat_read_restart(everything_struct * ev_h, everything_struct * ev_hd)
{
  diagnostics_file_struct gnostics_restart;

	char * filename;
	filename = (char*)malloc(sizeof(char)*(strlen(ev_h->info.run_name)+8+5));
	sprintf(filename, "%s.restart.cdf", ev_h->info.run_name);

  printf("Restarting from %s\n", filename);


	struct sdatio_file * sdatfile = &gnostics_restart.sdatfile;
  sdatio_init(sdatfile, filename);
  sdatio_open_file(sdatfile);

  //No need to transpose variables in the restart file.
  gnostics_restart.trans = false;

  // Setup status for reading
  gnostics_restart.create = false;
  gnostics_restart.write = false;
  gnostics_restart.read = true;
  printf("Runtime is %f\n", ev_h->time.runtime);
  writedat_io_restart(&gnostics_restart, ev_h, ev_hd);
  printf("Runtime after is %f\n", ev_h->time.runtime);

  sdatio_close(sdatfile);
  sdatio_free(sdatfile);

  free(filename);
}

void writedat_end(outputs_struct * outs)
{
	struct sdatio_file * sdatfile = &outs->gnostics.sdatfile;
  sdatio_close(sdatfile);
  sdatio_free(sdatfile);
}
