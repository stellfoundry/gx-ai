#define NO_GLOBALS true
#include "standard_headers.h"
#include "mpi.h"
#include "gs2.h"
#include "profile.h"

#ifdef GS2_zonal

extern "C" void init_gs2(int* strlength, char* namelistFile, int * mpcom, int * Nz, float * gryfx_theta, struct gryfx_parameters_struct * gryfxpars);
extern "C" void finish_gs2();
extern "C" void advance_gs2(int* gs2_counter, cuComplex* dens_ky0_h, cuComplex* upar_ky0_h, cuComplex* tpar_ky0_h, cuComplex* tprp_ky0_h, cuComplex* qpar_ky0_h, cuComplex* qprp_ky0_h, cuComplex* phi_ky0_h, int* first_half_flag);
extern "C" void getmoms_gryfx(cuComplex* dens, cuComplex* upar, cuComplex* tpar, cuComplex* tprp, cuComplex* qpar, cuComplex* qprp, cuComplex* phi);
extern "C" double gs2_time_mp_code_dt_;
extern "C" double gs2_time_mp_code_dt_cfl_;
extern "C" double gs2_time_mp_code_time_;
//#endif

extern "C" void gs2_main_mp_run_gs2_(char* namelistFile, int * strlength);

void gryfx_run_gs2_only(char * namelistFile){
  //int iproc;
  printf("Running GS2 simulation from within GryfX\n\n");
  int length = strlen(namelistFile);
  printf("Namelist is %s\n", namelistFile);
  gs2_main_mp_run_gs2_(namelistFile, &length);

  //iproc = *mp_mp_iproc_;

  exit(1);
}


void gryfx_initialize_gs2(grids_struct * grids, struct gryfx_parameters_struct * gryfxpars, char * namelistFile, int mpcom){
  int iproc;
  MPI_Barrier(mpcom);
  MPI_Comm_rank(mpcom, &iproc);

  // GS2 needs to know z_h so we have to allocate it on all 
  // procs
  MPI_Bcast(&grids->Nz, 1, MPI_INT, 0, mpcom);
  if(iproc!=0) grids->z = (float*)malloc(sizeof(float)*grids->Nz);
  //Local pointer for convenience
  float * z_h = grids->z;
  printf("z_h (1) is %f %f %f\n", z_h[0], z_h[1], z_h[2]);
  MPI_Bcast(&z_h[0], grids->Nz, MPI_FLOAT, 0, mpcom);

  printf("z_h is %f %f %f\n", z_h[0], z_h[1], z_h[2]);

  //gs2_main_mp_init_gs2_(namelistFile, &length);
  int length = strlen(namelistFile);
  //init_gs2(&length, namelistFile, &mpcom, gryfxpars);
  init_gs2(&length, namelistFile, &mpcom, &grids->Nz, z_h, gryfxpars);
  MPI_Barrier(mpcom);

  printf("z_h after is %f %f %f\n", z_h[0], z_h[1], z_h[2]);
}

void gryfx_finish_gs2(){
  finish_gs2();
}

void gryfx_advance_gs2(hybrid_zonal_arrays_struct * hybrid_h, time_struct* tm)
{
#ifdef PROFILE
PUSH_RANGE("advance_gs2",1);
//nvtxRangeId_t gs2_step = nvtxRangeStart("advance_gs2");
#endif
    advance_gs2(&tm->gs2_counter, hybrid_h->dens_h, hybrid_h->upar_h, hybrid_h->tpar_h, hybrid_h->tprp_h, hybrid_h->qpar_h, hybrid_h->qprp_h, hybrid_h->phi, &tm->first_half_flag);
    tm->gs2_counter++;
#ifdef PROFILE
POP_RANGE;
//nvtxRangeEnd(gs2_step);
#endif
}

void gryfx_get_gs2_moments(hybrid_zonal_arrays_struct * hybrid_h)
{
    getmoms_gryfx(hybrid_h->dens_h, hybrid_h->upar_h, hybrid_h->tpar_h, hybrid_h->tprp_h, hybrid_h->qpar_h, hybrid_h->qprp_h, hybrid_h->phi);
}
double gs2_time(){
  //This need to be fixed.
  double g2t = gs2_time_mp_code_time_;
  return g2t;
}
double gs2_dt(){
  //This need to be fixed.
  double gdt = gs2_time_mp_code_dt_ ;
  return gdt;
}

void  set_gs2_dt_cfl(double dt_cfl){
  //This need to be fixed.
  gs2_time_mp_code_dt_cfl_ = dt_cfl;
}

#endif
