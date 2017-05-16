void set_cuda_constants();

//From run_gryfx_functions.cu
void initialize_run_control(run_control_struct * ctrl, grids_struct * grids);

void calculate_additional_geo_arrays(
    int Nz,
    float * kz,
    float * tmpZ,
		input_parameters_struct * pars, 
		cuda_dimensions_struct * cdims, 
		geometry_coefficents_struct * geo_d, 
		geometry_coefficents_struct * geo_h);

void initialize_grids(input_parameters_struct * pars, grids_struct * grids, grids_struct * grids_h, cuda_dimensions_struct * cdims);

void zero_moving_averages(grids_struct * grids_h, cuda_dimensions_struct * cdims_h, outputs_struct * outs_hd, outputs_struct * outs_h, time_struct * time);

void create_cufft_plans(grids_struct * grids, cuffts_struct * ffts);
void initialize_z_covering(int iproc, grids_struct * grids_hd, grids_struct * grids_h, grids_struct * grids_d, input_parameters_struct * pars, cuffts_struct * ffts_h, cuda_streams_struct * streams, cuda_dimensions_struct * cdims, cuda_events_struct * events);

void set_initial_conditions_no_restart(input_parameters_struct * pars_h, input_parameters_struct * pars_d, grids_struct * grids_h, grids_struct * grids_d, cuda_dimensions_struct * cdims, geometry_coefficents_struct * geo_d, fields_struct * fields_hd, temporary_arrays_struct * tmp);

void load_fixed_arrays_from_restart(
      int Nz,
      cuComplex * CtmpZ_h,
      input_parameters_struct * pars,
      secondary_fixed_arrays_struct * sfixed, 
      fields_struct * fields /* fields on device */
);

void create_cuda_events_and_streams(cuda_events_struct * events, cuda_streams_struct * streams, int nClasses);

void initialize_hybrid_arrays(int iproc,
  grids_struct * grids,
  hybrid_zonal_arrays_struct * hybrid_h,
  hybrid_zonal_arrays_struct * hybrid_d);

void copy_hybrid_arrays_from_host_to_device_async(
  grids_struct * grids,
  hybrid_zonal_arrays_struct * hybrid_h,
  hybrid_zonal_arrays_struct * hybrid_d,
  cuda_streams_struct * streams
);

void copy_hybrid_arrays_from_device_to_host_async(
  grids_struct * grids,
  hybrid_zonal_arrays_struct * hybrid_h,
  hybrid_zonal_arrays_struct * hybrid_d,
  cuda_streams_struct * streams
);
void replace_zonal_fields_with_hybrid(
  int first_call,
  cuda_dimensions_struct * cdims,
  fields_struct * fields_d,
  cuComplex * phi_d, //Needs to be passed in separately
  hybrid_zonal_arrays_struct * hybrid_d,
  cuComplex * field_h
);
void copy_fixed_modes_into_fields(
  cuda_dimensions_struct * cdims,
  fields_struct * fields_d,
  cuComplex * phi_d, //Need  phi separate cos sometimes need dens1 but not phi1 etc
  secondary_fixed_arrays_struct * sfixed,
  input_parameters_struct * pars
    );
void write_initial_fields(
  cuda_dimensions_struct * cdims,
  fields_struct * fields_d,
  temporary_arrays_struct * tmp_d,
  cuComplex * field_h,
  float * tmpX_h
);
void update_nlpm_coefficients(
    cuda_dimensions_struct * cdims,
    input_parameters_struct * pars,
    outputs_struct * outs,
    nlpm_struct * nlpm,
    nlpm_struct * nlpm_hd,
    nlpm_struct * nlpm_d,
    cuComplex * Phi,  
    temporary_arrays_struct * tmp_d,
    time_struct * tm
);
void initialize_nlpm_coefficients(
    cuda_dimensions_struct * cdims,
    nlpm_struct* nlpm_h, 
    nlpm_struct* nlpm_d,
    int Nz
);
void initialize_run_control(run_control_struct * ctrl, grids_struct * grids);
void initialize_averaging_parameters(outputs_struct * outs, int navg);
void initialize_phi_avg_denom(
    cuda_dimensions_struct * cdims,
    input_parameters_struct * pars_h,
    grids_struct * grids_d,
    geometry_coefficents_struct * geo_d,
    specie * species_d,
    float * tmpXZ
    );

//From gryfx_run_diagnostics.cu
void gryfx_run_diagnostics(
  everything_struct * ev_h,
  everything_struct * ev_hd
);


//From diagnostics.cu
void restartRead(everything_struct * ev_h, everything_struct * ev_hd);
void restartWrite(everything_struct * ev_h,
  everything_struct * ev_hd);
void fieldWrite(cuComplex* f_d, cuComplex* f_h, char* ext, char* filename);
void fieldWrite(cuComplex* f_d, cuComplex* f_h, char* ext, char* filename, int Nx, int Ny, int Nz);
void fieldWrite_nopad(cuComplex* f_nopad_d, cuComplex* f_nopad_h, char* ext, char* filename, int Nx, int Ny, int Nz, int ntheta0, int naky);
void fieldWrite_nopad_h(cuComplex* f_nopad_h, char* ext, char* filename, int Nx, int Ny, int Nz, int ntheta0, int naky);
void gryfx_finish_diagnostics(
  everything_struct * ev_h,
  everything_struct * ev_hd,
   bool end
);


//From timestep_gryfx.cu

void nonlinear_timestep(
  int is,
  int first_half_step,
  everything_struct * ev_h,
  everything_struct * ev_hd,
  everything_struct * ev_d) ;

void linear_timestep(
  int is,
  int first_half_step,
  everything_struct * ev_h,
  everything_struct * ev_hd,
  everything_struct * ev_d 
);

void linear_electron_timestep(
  int is,
  int first_half_step,
  everything_struct * ev_h,
  everything_struct * ev_hd,
  everything_struct * ev_d 
);

//From nlpm.cu
void filterNLPM(
  int is,
  fields_struct * fields_d, 
  temporary_arrays_struct * tmp_d,
  nlpm_struct * nlpm_d,
  nlpm_struct * nlpm_h,
  float dt_loc,
  specie s,
  float* Dnlpm_d
      );

void filterNLPMcomplex(
  int is,
  fields_struct * fields_d, 
  temporary_arrays_struct * tmp_d,
  nlpm_struct * nlpm_d,
  nlpm_struct * nlpm_h,
  float dt_loc,
  specie s,
  float* Dnlpm_d
);
void filterHyper_aniso(int is, fields_struct * fields_d,		 float* tmpXYZ, hyper_struct * hyper, float dt_loc);
void filterHyper_iso(int is, fields_struct * fields_d, 
		 float* tmpXYZ, float* shear_rate_nz, float dt_loc);


//From qneut.cu
void qneut(cuComplex* Phi, cuComplex* Apar, cuComplex** Dens, cuComplex** Tprp, cuComplex** Upar, cuComplex** Qprp, 
  cuComplex* PhiAvgNum_tmp, cuComplex* nbar_tmp, cuComplex* nbartot_field, cuComplex* ubar_tmp, cuComplex* ubartot_field, 
  specie* species, specie* species_d, input_parameters_struct* pars);// bool adiabatic, float fapar, float beta, bool snyder);
