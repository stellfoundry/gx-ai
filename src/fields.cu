#include "fields.h"
#include "get_error.h"

Fields::Fields(Grids* grids) : size_(sizeof(cuComplex)*grids->NxNycNz),
			       N(grids->NxNycNz), grids_(grids)
{
  checkCuda(cudaMalloc((void**) &phi, size_));
  cudaMemset(phi, 0., size_);

  cudaMallocHost((void**) &phih, size_);
  fft = new GradPerp(grids_, grids_->Nz);
  
}

Fields::~Fields() {
  delete fft;
  cudaFree(phi);
  cudaFreeHost(phih);
}

void Fields::print_phi(void)
{
  CP_TO_CPU(phih, phi, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("phi(%d) = (%e, %e) \n",j, phih[j].x, phih[j].y);
  printf("\n");
}

void Fields::chk_fft()
{

  size_t fs = sizeof(float)*grids_->NxNyNz;
  
  printf("d Phi/dx in real space: \n");
  float* fr;
  checkCuda(cudaMalloc((void**) &fr, fs));
  cudaMemset(fr, 0., fs);
  
  float* fh;
  cudaMallocHost((void**) &fh, fs);
  
  fft->dxC2R(phi, fr);  CP_TO_CPU(fh, fr, fs);  cudaFree(fr);
  
  for (int ig=0; ig<grids_->NxNyNz; ig++) {
    printf("phi(%d) = %e \n", ig, fh[ig]);
    //    printf("phi(%d,%d) = %e \n", ig%4, ig/4, fh[ig]);
    //    if (ig%4==3) printf("\n");
  }
  
  cudaFreeHost(fh);
}
