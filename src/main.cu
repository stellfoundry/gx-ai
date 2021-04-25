#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include "gx_lib.h"
#include "version.h"
#include "helper_cuda.h"
// #include "reservoir.h"
#include "reductions.h"

int main(int argc, char* argv[])
{

  MPI_Init(&argc, &argv);
  MPI_Comm mpcom = MPI_COMM_WORLD;
  int iproc;
  MPI_Comm_rank(mpcom, &iproc);
  
  int devid = 0; // This should be determined (optionally) on the command line
  checkCuda(cudaSetDevice(devid));
  cudaDeviceSynchronize();

  
  /*
  int N = 5000;
  int K = 3;
  
  float radius = 0.6;
  float *R;
  float *y;
  float *x;
  float *A_in;
  int *A_col;
  
  checkCuda(cudaMalloc((void**) &R,  sizeof(float)*N  ) ); 
  checkCuda(cudaMalloc((void**) &y,  sizeof(float)*N  ) ); 
  checkCuda(cudaMalloc((void**) &x,  sizeof(float)*N*K) ); 
  
  // define the number of elements in a typical row of A to be ResDensity*N
  int nnz = K * N;
  bool first = true;
  float *A_h;
  int * A_j;
  cudaMallocHost((void**) &A_h, sizeof(float) * nnz);
  cudaMallocHost((void**) &A_j, sizeof(int)   * nnz);
  
  std::random_device rd;         std::mt19937 gen(rd()); 
  std::uniform_real_distribution<> unif( 0., radius*2./((float) K));

  std::vector<int> col(N);     std::iota(col.begin(), col.end(), 0);
  std::vector<int> cin(K);
  
  for (int n=0; n<N; n++) {

    std::shuffle(col.begin(), col.end(), gen);
    for (int k=0; k<K; k++) cin[k] = col[k];
    std::sort(cin.begin(), cin.end());

    for (int k=0; k<K; k++) {
      A_j[k + K*n] = cin[k];
      A_h[k + K*n] = unif(gen);
    }    
  }
 
  checkCuda(cudaMalloc((void**) &A_in,  sizeof(float)*nnz) ); 
  checkCuda(cudaMalloc((void**) &A_col, sizeof(int)  *nnz) ); 

  CP_TO_GPU (A_in,  A_h, sizeof(float) * nnz);
  CP_TO_GPU (A_col, A_j, sizeof(int)   * nnz);

  cudaFreeHost(A_h);
  cudaFreeHost(A_j);

  Red *red;
  
  red = new Block_Reduce(N); cudaDeviceSynchronize(); 
  int nn0 = N;   int nt0 = min(N, 512);    int nb0 = 1 + (nn0-1)/nt0;
  int kn0 = N*K; int kt0 = min(N*K, 512);  int kb0 = 1 + (kn0-1)/kt0;
  float *x2norm;    cudaMalloc((void **) &x2norm, sizeof(float)   );
  float *y2norm;    cudaMalloc((void **) &y2norm, sizeof(float)   );
  float *xynorm;    cudaMalloc((void **) &xynorm, sizeof(float)   );
  float *x2;        cudaMalloc((void **) &x2,     sizeof(float)*N );
  float *y2;        cudaMalloc((void **) &y2,     sizeof(float)*N );
  float *xy;        cudaMalloc((void **) &xy,     sizeof(float)*N );

  setval <<< nb0, nt0 >>> (R, 1., N);
  setval <<< nb0, nt0 >>> (y, 1., N);
  setval <<< kb0, kt0 >>> (x, 1., nnz);
  
  float eval, eval_old, tol, ex, ey;
  eval=0.;  eval_old = 10.;  tol = 1.e-7;  ex = 0.;  ey = 0.;
  while (abs(eval-eval_old)/abs(eval_old) > tol) {    
    
    eval = eval_old;
    
    myPrep <<< kb0, kt0 >>> (x, R, A_col, nnz, first); first = false;    
    mySpMV <<< nb0, nt0 >>> (x2, xy, y2, y, x, A_in, R, K, N);
    red->Sum(y2, y2norm);    red->Sum(x2, x2norm);    red->Sum(xy, xynorm);
    
    inv_scale_kernel <<< nb0, nt0 >>> (R, y, y2norm, N); 
    CP_TO_CPU(&ex, x2norm, sizeof(float));
    CP_TO_CPU(&ey, xynorm, sizeof(float));
    eval_old  = ey/ex;
    
    //    printf("eval = %f \t",eval_old);
  }
  printf(ANSI_COLOR_GREEN);
  printf("spectral radius is %f \n",eval_old);
  printf(ANSI_COLOR_RESET);
  
  // print the residual
  myPrep <<< kb0, kt0 >>> (x, R, A_col, nnz, first);
  mySpMV <<< nb0, nt0 >>> (x2, xy, y2, y, x, A_in, R, K, N);  
  eig_residual <<< nb0, nt0 >>> (y, A_in, x, R, x2, eval_old, K, N);
  red->Sum(x2, x2norm);  CP_TO_CPU(&ex, x2norm, sizeof(float));
  printf(ANSI_COLOR_YELLOW);  printf("RMS residual = %f \n",sqrtf(ex));  printf(ANSI_COLOR_RESET);
  
  //  exit(1);
  */

  char *run_name;
  if ( argc < 1) {
    fprintf(stderr, "The correct usage is:\n gx <runname>\n");
    exit(1);
  } else {    
    run_name = argv[1];
    printf("Running %s \n",run_name);
  }
   
  printf("Version: %s \t Compiled: %s \n", build_git_sha, build_git_time);

  Parameters * pars         = nullptr;
  pars = new Parameters();
  pars->iproc = iproc;
  pars->get_nml_vars(run_name);
  
  Geometry    * geo         = nullptr;
  Grids       * grids       = nullptr;
  Diagnostics * diagnostics = nullptr;
  //  HermiteTransform* herm;
  
  DEBUGPRINT("Initializing grids...\n");
  grids = new Grids(pars);
  CUDA_DEBUG("Initializing grids: %s \n");

  DEBUGPRINT("Grid dimensions: Nx=%d, Ny=%d, Nz=%d, Nl=%d, Nm=%d, Nspecies=%d\n",
	     grids->Nx, grids->Ny, grids->Nz, grids->Nl, grids->Nm, grids->Nspecies);

  if(iproc==0) {
    int igeo = pars->igeo;
    DEBUGPRINT("Initializing geometry...\n");
    if(igeo==0) {
      geo = new S_alpha_geo(pars, grids);
      CUDA_DEBUG("Initializing geometry s_alpha: %s \n");
    }
    else if(igeo==1) {
      geo = new File_geo(pars, grids);
      printf("************************* \n \n \n");
      printf("Warning: assumed grho = 1 \n \n \n");
      printf("************************* \n");
      CUDA_DEBUG("Initializing geometry from file: %s \n");
    } 
    else if(igeo==2) {
      DEBUGPRINT("igeo = 2 not yet implemented!\n");
      exit(1);
      //geo = new Eik_geo();
    } 
    else if(igeo==3) {
      DEBUGPRINT("igeo = 3 not yet implemented!\n");
      exit(1);
      //geo = new Gs2_geo();
    }

    DEBUGPRINT("Initializing diagnostics...\n");
    diagnostics = new Diagnostics(pars, grids, geo);
    CUDA_DEBUG("Initializing diagnostics: %s \n");    

    //    DEBUGPRINT("Initializing Hermite transforms...\n");
    //    herm = new HermiteTransform(grids, 1); // batch size could ultimately be nspec
    //    CUDA_DEBUG("Initializing Hermite transforms: %s \n");    
  }

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  
  run_gx(pars, grids, geo, diagnostics);

  delete pars;
  delete grids;
  delete geo;
  delete diagnostics;

  MPI_Finalize();
  cudaDeviceReset();
}
