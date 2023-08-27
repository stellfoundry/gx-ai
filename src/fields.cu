#include "fields.h"
#include "get_error.h"
#include "grad_perp.h"

Fields::Fields(Parameters* pars, Grids* grids) :
  size_(sizeof(cuComplex)*grids->NxNycNz), sizeReal_(sizeof(float)*grids->NxNyNz), N(grids->NxNycNz), pars_(pars), grids_(grids),
  phi(nullptr), phi_h(nullptr), apar(nullptr), apar_h(nullptr), bpar(nullptr), bpar_h(nullptr),
  apar_ext(nullptr), apar_ext_h(nullptr), apar_ext_realspace_h(nullptr), apar_ext_realspace(nullptr),
  ne(nullptr), ne_h(nullptr), ue(nullptr), ue_h(nullptr), Te(nullptr), Te_h(nullptr)
{
  checkCuda(cudaMalloc((void**) &phi, size_));

  int nn = grids->NxNycNz; int nt = min(nn, 512); int nb = 1 + (nn-1)/nt;  cuComplex zero = make_cuComplex(0.,0.);
  setval <<< nb, nt >>> (phi, zero, nn);

  //  cudaMemset(phi, 0., size_);
  bool debug = pars->debug;

  phi_h = (cuComplex*) malloc(size_);
  DEBUGPRINT("Allocated a field array of size %.2f MB\n", size_/1024./1024.);

  
  checkCuda(cudaMalloc((void**) &apar, size_));
  DEBUGPRINT("Allocated a field array of size %.2f MB\n", size_/1024./1024.);
  
  setval <<< nb, nt >>> (apar, zero, nn);
  
  checkCuda(cudaMalloc((void**) &apar_ext, size_));
  if(debug) printf("Allocated a field array of size %.2f MB\n", size_/1024./1024.);
  
  checkCuda(cudaMalloc((void**) &apar_ext_realspace, sizeReal_));
  
  setval <<< nb, nt >>> (apar_ext, zero, nn);
  
  apar_h = (cuComplex*) malloc(size_);
  apar_ext_realspace_h = (float*) malloc(sizeReal_);
  apar_ext_h = (cuComplex*) malloc(size_);

    
  checkCuda(cudaMalloc((void**) &bpar, size_));
  if(debug) printf("Allocated a field array of size %.2f MB\n", size_/1024./1024.);
  
  setval <<< nb, nt >>> (bpar, zero, nn);
  
  bpar_h = (cuComplex*) malloc(size_);
  
  //if (pars_->beta > 0. || pars_->krehm) {
  //  if (!pars_->krehm) {
  //    checkCuda(cudaMalloc((void**) &ne, size_));
  //    printf("Allocated ne array of size %.2f MB\n", size_/1024./1024.);
  //    
  //    cudaMemset(ne, 0., size_); setval <<< nb, nt >>> (ne, zero, nn);
  //    
  //    ne_h = (cuComplex*) malloc(size_);
  //    
  //    checkCuda(cudaMalloc((void**) &ue, size_));
  //    printf("Allocated ue array of size %.2f MB\n", size_/1024./1024.);
  //    
  //    cudaMemset(ue, 0., size_); setval <<< nb, nt >>> (ue, zero, nn);
  //    
  //    ue_h = (cuComplex*) malloc(size_);
  //    
  //    checkCuda(cudaMalloc((void**) &Te, size_));
  //    printf("Allocated Te array of size %.2f MB\n", size_/1024./1024.);
  //    
  //    cudaMemset(Te, 0., size_); setval <<< nb, nt >>> (Te, zero, nn);
  //    
  //    Te_h = (cuComplex*) malloc(size_);
  //  }
  //}

  if (pars_->harris_sheet) {

    assert((fapar > 0.) && "Harris sheet equilibrium requires setting fapar = 1.0");
    
    int nBatch = grids_->Nz;
    GradPerp * grad_perp = new GradPerp(grids_, nBatch, grids_->NxNycNz);
    
    //set up harris sheet in real space   
    for(int idz = 0; idz < grids_->Nz; idz++) {
      for(int idx = 0; idx < grids_->Nx; idx++) {
        for(int idy = 0; idy < grids_->Ny; idy++) {
           float x = grids_->x_h[idx];

	   int index = idy + idx * grids_->Ny + idz * grids_->NxNy;
	   float A0 = 1.29904; // this value makes B_ext_max = 1
	   apar_ext_realspace_h[index] = A0/pow(cosh((x-M_PI*pars_->x0)/pars_->x0),2)*((pow(tanh(x/pars_->x0),2)+(pow(tanh(x/pars_->x0-2*M_PI),2))-(pow(tanh(2*M_PI),2)))/(2*pow(tanh(M_PI),2)-pow(tanh(2*M_PI),2)));
	}
      }
    }

    //copy apar_ext to GPU and do Fourier transformation
    CP_TO_GPU(apar_ext_realspace, apar_ext_realspace_h, sizeof(float) * grids_->NxNyNz); 
    grad_perp->R2C(apar_ext_realspace, apar_ext, true);
    
    delete grad_perp;

    //debug part

    //grad_perp->qvar(apar_ext_realspace, grids_->NxNyNz); 
    //CP_TO_CPU(apar_ext_h, apar_ext, sizeof(cuComplex) * grids_->NxNycNz);
  //  for(int idz = 0; idz < grids_->Nz; idz++){
  //    for(int idx = 0; idx < grids_->Nx; idx++){
  //      for(int idy = 0; idy < grids_->Nyc; idy++){
  //         unsigned int idxyz = idy + grids_->Nyc *(idx + grids_->Nx*idz); 
  //         printf("idxyz%d:",idxyz);
  //         printf("%f\n",apar_ext_h[idxyz].x);
  //      }
  //    }
  //  }
    //grad_perp->qvar(apar_ext, grids_->NxNycNz); 
  }
}

Fields::~Fields() {
  if (phi)     cudaFree(phi);
  if (apar)    cudaFree(apar);
  if (bpar)    cudaFree(bpar);

  if (phi_h)   free(phi_h);
  if (apar_h)  free(apar_h);
  if (bpar_h)  free(bpar_h);

  if (ne)      cudaFree(ne);
  if (ue)      cudaFree(ue);
  if (Te)      cudaFree(Te);

  if (ne_h)    free(ne_h);
  if (ue_h)    free(ue_h);
  if (Te_h)    free(Te_h);
  
}

void Fields::print_phi(void)
{
  CP_TO_CPU(phi_h, phi, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("phi(%d) = (%e, %e) \n",j, phi_h[j].x, phi_h[j].y);
  printf("\n");
}

void Fields::print_apar(void)
{
  CP_TO_CPU(apar_h, apar, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("apar(%d) = (%e, %e) \n",j, apar_h[j].x, apar_h[j].y);
  printf("\n");
}

void Fields::print_bpar(void)
{
  CP_TO_CPU(bpar_h, bpar, size_);
  printf("\n");
  for (int j=0; j<N; j++) printf("bpar(%d) = (%e, %e) \n",j, bpar_h[j].x, bpar_h[j].y);
  printf("\n");
}

void Fields::rescale(float * phi_max) {
  int nn1 = grids_->NxNyc; int nt1 = min(nn1, 32); int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nz;    int nt2 = min(nn2, 32); int nb2 = 1 + (nn2-1)/nt2;
  dim3 dB, dG;
  dB = dim3(nt1, nt2, 1);
  dG = dim3(nb1, nb2, 1);
  rescale_kernel <<< dG, dB >>> (phi, phi_max, 1);
  if(pars_->fapar > 0.) rescale_kernel <<< dG, dB >>> (apar,  phi_max, 1);
}
