#include "standard_headers.h"
#include "mpi.h"
#include "gs2.h"

#ifdef GS2_zonal

extern "C" void init_gs2(int* strlength, char* namelistFile, int * mpcom, int * Nz, float * gryfx_theta, struct gryfx_parameters_struct * gryfxpars);
extern "C" void finish_gs2();
#endif

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


void gryfx_initialize_gs2(everything_struct * ev, struct gryfx_parameters_struct * gryfxpars, char * namelistFile, int mpcom){
  int iproc;
  MPI_Barrier(mpcom);
  MPI_Comm_rank(mpcom, &iproc);

  // GS2 needs to know z_h so we have to allocate it on all 
  // procs
  MPI_Bcast(&Nz, 1, MPI_INT, 0, mpcom);
  if(iproc!=0) z_h = (float*)malloc(sizeof(float)*Nz);
  printf("z_h (1) is %f %f %f\n", z_h[0], z_h[1], z_h[2]);
  MPI_Bcast(&z_h[0], Nz, MPI_FLOAT, 0, mpcom);

  printf("z_h is %f %f %f\n", z_h[0], z_h[1], z_h[2]);

  //gs2_main_mp_init_gs2_(namelistFile, &length);
  int length = strlen(namelistFile);
  //init_gs2(&length, namelistFile, &mpcom, gryfxpars);
  init_gs2(&length, namelistFile, &mpcom, &Nz, z_h, gryfxpars);
  MPI_Barrier(mpcom);

  printf("z_h after is %f %f %f\n", z_h[0], z_h[1], z_h[2]);
}

void gryfx_finish_gs2(){
  finish_gs2();
}
