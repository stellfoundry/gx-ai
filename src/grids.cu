#include "grids.h"

Grids::Grids(Parameters* pars) :
  // copy from input parameters
  Nx       ( pars->nx_in       ),
  Ny       ( pars->ny_in       ),
  Nz       ( pars->nz_in       ),
  Nspecies ( pars->nspec_in    ),
  Nm       ( pars->nm_in       ),
  Nl       ( pars->nl_in       ),
  Nj       ( 3*pars->nl_in/2-1 ),

  Nyc      ( 1 + Ny/2          ),
  Naky     ( 1 + (Ny-1)/3      ),
  Nakx     ( 1 + 2*((Nx-1)/3)  ), 
  NxNyc    ( Nx * Nyc          ),
  NxNy     ( Nx * Ny           ),
  NxNycNz  ( Nx * Nyc * Nz     ),
  NxNyNz   ( Nx * Ny * Nz      ),
  NxNz     ( Nx * Nz           ),
  NycNz    ( Nyc * Nz          ),
  Nmoms    ( Nm * Nl           ),
  size_G( sizeof(cuComplex) * NxNycNz * Nmoms * Nspecies), 
  pars_(pars)
{
  ky              = nullptr;  kx              = nullptr;  kz              = nullptr;
  ky_h            = nullptr;  kx_h            = nullptr;  kz_h            = nullptr;
  kx_outh         = nullptr;//  theta0_h        = nullptr;
  kz_outh         = nullptr;
  y_h             = nullptr; 
  //  kx_mask         = nullptr;  kx_shift        = nullptr;  jump            = nullptr;
  //  nLinks          = nullptr;  nChains         = nullptr;
  //  kxCover         = nullptr;  kyCover         = nullptr;  kz_covering     = nullptr; 
  //  kxCover_d       = nullptr;  kyCover_d       = nullptr;  kz_covering_d   = nullptr;
  //  covering_scaler = nullptr;

  // kz is defined without the factor of gradpar
  
  checkCuda(cudaDeviceSynchronize());

  checkCuda(cudaMallocHost ( (void**) &kx_outh, sizeof(float) * Nakx )); 
  cudaMallocHost ( (void**) &kz_outh, sizeof(float) * Nz   );
  cudaMallocHost ( (void**) &kx_h,    sizeof(float) * Nx   ); 
  cudaMallocHost ( (void**) &ky_h,    sizeof(float) * Nyc  );
  cudaMallocHost ( (void**) &kz_h,    sizeof(float) * Nz   );
  cudaMalloc     ( (void**) &kx,      sizeof(float) * Nx   );
  cudaMalloc     ( (void**) &ky,      sizeof(float) * Nyc  );
  cudaMalloc     ( (void**) &kz,      sizeof(float) * Nz   );
  cudaMallocHost ( (void**) &y_h,     sizeof(float) * Ny   );
  checkCuda(cudaGetLastError());

  //  printf("In grids constructor. Nyc = %i \n",Nyc);
  
  setdev_constants(Nx, Ny, Nyc, Nz, Nspecies, Nm, Nl, Nj, pars_->Zp, pars_->ikx_fixed, pars_->iky_fixed);

  checkCuda(cudaDeviceSynchronize());

  // initialize k arrays
  int Nmax = max(max(Nx, Nyc), Nz);
  int nt = min(32, Nmax);
  int nb = 1 + (Nmax-1)/nt;

  kInit <<<nb, nt>>> (kx, ky, kz, pars_->x0, pars_->y0, pars_->Zp);

  CP_TO_CPU (kx_h, kx, sizeof(float)*Nx);
  CP_TO_CPU (ky_h, ky, sizeof(float)*Nyc);
  CP_TO_CPU (kz_h, kz, sizeof(float)*Nz);

  if (Nx<4) {
    //    printf("Nx, Nakx = %d, %d \n",Nx, Nakx);
    //    printf("kx_h = %f \n",kx_h[0]);
    for (int i=0; i<Nx; i++) kx_outh[i] = kx_h[i];
  } else {    
    kx_outh[0] = kx_h[2*Nx/3+1];
    for (int i = 1; i < 1 + (Nx-1)/3 ; i++) {
      kx_outh[i]         = kx_h[i + 2*Nx/3 + 1];    
      kx_outh[i + Nx/3 ] = kx_h[i];
    }
    /*
    for (int i=0; i<Nx; i++) {
      printf("kx_h[%d] = %f \n",i,kx_h[i]);
    }
    for (int i = 0; i < Nakx; i++) {
      printf("kx_outh[%d] = %f \n",i, kx_outh[i]);
    }
    */
  }
  if (Nz>1) {
    for (int i = 0; i < Nz ; i++) kz_outh[i] = kz_h[ (i + Nz/2 + 1) % Nz ];
  } else {
    for (int i = 0; i < Nz ; i++) kz_outh[i] = kz_h[ i ];
  }
  
  // define the y coordinate
  y_h[0] = 0.;
  for (int i = 1; i < Ny ; i++) y_h[i] = y_h[i-1] + (float) 2*M_PI*(pars_->y0)/Ny;
 
}

Grids::~Grids() {
  if (kx)              cudaFree(kx);
  if (ky)              cudaFree(ky);
  if (kz)              cudaFree(kz);
   
  if (kz_outh)         cudaFreeHost(kz_outh);
  if (kx_outh)         cudaFreeHost(kx_outh);
  if (kx_h)            cudaFreeHost(kx_h);
  if (ky_h)            cudaFreeHost(ky_h);
  if (kz_h)            cudaFreeHost(kz_h);
  if (y_h)             cudaFreeHost(y_h);
}


