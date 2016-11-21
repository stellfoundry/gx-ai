
void gryfx_run_gs2_only(char * namelistFile);
void gryfx_initialize_gs2(grids_struct * grids, struct gryfx_parameters_struct * gryfxpars, char * namelistFile, int mpcom);
void gryfx_finish_gs2();
void gryfx_advance_gs2(hybrid_zonal_arrays_struct * hybrid, time_struct* tm);
void gryfx_get_gs2_moments(hybrid_zonal_arrays_struct * hybrid_h);

double gs2_time();
double gs2_dt();
void  set_gs2_dt_cfl(double dt_cfl);

extern "C" void init_gs2_file_utils(int* strlength, char* namelistFile);//Needed for the geometry module
extern "C" void finish_gs2_file_utils();//Needed for the geometry module
