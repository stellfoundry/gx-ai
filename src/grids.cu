#include "grids.h"
#include "laguerre_transform.h"

Grids::Grids(Parameters* pars) :
  // copy from input parameters
  Nx       ( pars->nx_in       ),
  Ny       ( pars->ny_in       ),
  Nz       ( pars->nz_in       ),
  Nz_glob  ( pars->nz_in       ),
  Nl       ( pars->nl_in       ),
  Nj       ( max(1, 3*pars->nl_in/2-1) ),

  Nyc      ( 1 + Ny/2          ),
  Naky     ( 1 + (Ny-1)/3      ),
  Nakx     ( 1 + 2*((Nx-1)/3)  ), 
  NxNyc    ( Nx * Nyc          ),
  NxNy     ( Nx * Ny           ),
  NxNycNz  ( Nx * Nyc * Nz     ),
  NxNyNz   ( Nx * Ny * Nz      ),
  NxNz     ( Nx * Nz           ),
  NycNz    ( Nyc * Nz          ),
  Zp(pars->Zp),
  iproc(pars->iproc),
  nprocs(pars->nprocs),
  pars_(pars)
{
  ky              = nullptr;  kx              = nullptr;  kz              = nullptr;
  ky_h            = nullptr;  kx_h            = nullptr;  kz_h            = nullptr;
  kx_outh         = nullptr;
  kz_outh         = nullptr;  kpar_outh       = nullptr;  kzp             = nullptr;
  y_h             = nullptr;  x_h             = nullptr;
  theta0_h        = nullptr;  th0             = nullptr;  z_h             = nullptr;
  m0_h            = nullptr;  phasefac_ntft   = nullptr;  phasefacminus_ntft = nullptr;
  iKx             = nullptr;  x               = nullptr;
  kxstar          = nullptr;  kxbar_ikx_new   = nullptr;  kxbar_ikx_old   = nullptr;
  phasefac_exb    = nullptr;  phasefacminus_exb = nullptr; 


  Nspecies = pars->nspec_in;
  Nspecies_glob = Nspecies;
  Nm = pars->nm_in;
  Nm_glob = pars->nm_in;
  is_lo = 0;
  is_up = Nspecies;
  m_lo = 0;
  m_up = Nm;
  z_lo = 0;
  z_up = Nz;
  m_ghost = 0;
  nprocs_s = 1;
  nprocs_m = 1;
  nprocs_z = 1;
  iproc_m = 0;
  iproc_s = 0;
  iproc_z = 0;
  mpcom_sm = MPI_COMM_WORLD;
  mpcom_z = MPI_COMM_WORLD;

  // compute parallel decomposition
  if(nprocs>1) {
    nprocs_z = pars->nproc_theta;
    assert((nprocs_z > 0) && "nproc_theta must be positive");
    assert((nprocs%nprocs_z == 0) && "nprocs must be an integer multiple of nproc_theta");
    assert((Nz_glob%nprocs_z == 0) && "ntheta must be an integer multiple of nproc_theta");
    int nprocs_sm = nprocs/nprocs_z;

    iproc_z = iproc/nprocs_sm;
    z_lo = iproc_z*(Nz_glob/nprocs_z);
    z_up = (iproc_z+1)*(Nz_glob/nprocs_z);
    Nz = Nz_glob/nprocs_z;
    NxNycNz = Nx * Nyc * Nz;
    NxNyNz = Nx * Ny * Nz;
    NxNz = Nx * Nz;
    NycNz = Nyc * Nz;

    // prioritize species decomp
    if(nprocs_sm<=Nspecies) {
      assert((Nspecies%nprocs_sm == 0) && "nprocs/nproc_theta <= nspecies, so nspecies must be an integer multiple of nprocs/nproc_theta\n");
      // this is now the local Nspecies on this proc
      Nspecies = Nspecies/nprocs_sm;
      nprocs_s = nprocs_sm;
      nprocs_m = 1;
      iproc_s = iproc%nprocs_sm;
      iproc_m = 0;
      is_lo = iproc_s*Nspecies;
      is_up = (iproc_s+1)*Nspecies;

      m_lo = 0;
      m_up = Nm;

      //printf("GPU %d: is_lo = %d, is_up = %d, m_lo = %d, m_up = %d\n", iproc, is_lo, is_up, m_lo, m_up);
    } else { // decomp in species and hermite
      assert((nprocs_sm%Nspecies == 0) && "nprocs/nproc_theta > nspecies, so nprocs/nproc_theta must be an integer multiple of nspecies\n");
      nprocs_s = Nspecies;
      nprocs_m = nprocs_sm/Nspecies;
      iproc_s = (iproc%nprocs_sm)/nprocs_m;
      iproc_m = (iproc%nprocs_sm)%nprocs_m;

      // this is now the local Nspecies on this proc
      Nspecies = 1;
      is_lo = iproc_s*Nspecies;
      is_up = (iproc_s+1)*Nspecies; // is_up is never used

      assert((Nm%nprocs_m == 0) && "Nm must be an integer multiple of nprocs_m=nprocs/nspecies\n");
      // this is now the local Nm on this proc
      Nm = Nm/nprocs_m;

      m_lo = (iproc_m    )*Nm;
      m_up = (iproc_m + 1)*Nm;

      // add ghosts in m
      if(pars->slab && Nm>1) {
        m_ghost = 1;
      } else {
        m_ghost = 2;
      }
    }
  }

  if(nprocs == 1) {
    nprocs_z = pars->nproc_theta;
    assert((nprocs_z == 1) && "nproc_theta must be 1 when running with one MPI process");
  }

  if(nprocs_z > 1) {
    assert(!pars->restart && "restart reads are not yet implemented with theta decomposition");
    if(pars->save_for_restart) {
      if(iproc == 0) printf("Warning: restart writes are not yet implemented with theta decomposition; disabling save_for_restart.\n");
      pars->save_for_restart = false;
    }
    assert(pars->use_NCCL && "theta decomposition currently requires use_NCCL = true");
    assert(!(pars->forcing_init && pars->forcing_kz != 0) && "forcing_init with nonzero forcing_kz is not yet implemented with theta decomposition");
  }

  if(nprocs > 1) {
    MPI_Comm_split(MPI_COMM_WORLD, iproc_z, iproc_m + nprocs_m*iproc_s, &mpcom_sm);
    MPI_Comm_split(MPI_COMM_WORLD, iproc_m + nprocs_m*iproc_s, iproc_z, &mpcom_z);
  }

  //
  // When solving Toby's collisional slab ETG model, use nhermite = 1, nlaguerre = 2
  // The zeroth moment will be (m=0, l=0) == Phi
  // The first moment will be (m=0, l=1) == delta T
  // These values are automatically set in parameters when this equation set is selected.

  // Should add an assert statement here? Something like "if we are solving cetg, assert Nm == 1)" and so forth
  
  Nmoms = Nm * Nl;
  size_G = sizeof(cuComplex) * NxNycNz * (Nm + 2*m_ghost) * Nl; // this includes ghosts on either end of m grid
  // kz is defined without the factor of gradpar
  
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());

  kx_outh   = (float*) malloc(sizeof(float) * Nakx       );
  cudaMalloc     ( (void**) &kzm,       sizeof(int)   * Nz       );
  cudaMalloc     ( (void**) &kzp,       sizeof(float) * Nz       );
  kz_outh   = (float*) malloc(sizeof(float) * Nz       );
  kpar_outh = (float*) malloc(sizeof(float) * Nz       );
  theta0_h  = (float*) malloc(sizeof(float) * Nx       ); 
  kx_h      = (float*) malloc(sizeof(float) * Nx       ); 
  ky_h      = (float*) malloc(sizeof(float) * Nyc      );
  kz_h      = (float*) malloc(sizeof(float) * Nz       );
  cudaMalloc     ( (void**) &kx,        sizeof(float) * Nx       );
  checkCuda(cudaMemset(kx, 0., sizeof(float)*Nx));
  cudaMalloc     ( (void**) &th0,       sizeof(float) * Nx       );
  cudaMalloc     ( (void**) &ky,        sizeof(float) * Nyc      );
  checkCuda(cudaMemset(ky, 0., sizeof(float)*Nyc));
  cudaMalloc     ( (void**) &kz,        sizeof(float) * Nz       );
  x_h      = (float*) malloc(sizeof(float) * Nx       ); 
  y_h      = (float*) malloc(sizeof(float) * Ny       );
  z_h      = (float*) malloc(sizeof(float) * Nz       );
  cudaMalloc     ( (void**) &x,        sizeof(float) * Nx       );
  if (pars->nonTwist) {
    m0_h   = (int*)   malloc(sizeof(int) * Nyc * Nz );
    if (!pars->linear) {      
      cudaMalloc     ( (void**) &phasefac_ntft,       sizeof(cuComplex) * Nx * Nyc * Nz);
      cudaMalloc     ( (void**) &phasefacminus_ntft,  sizeof(cuComplex) * Nx * Nyc * Nz);
      cudaMalloc     ( (void**) &iKx,                 sizeof(cuComplex) * Nx * Nyc * Nz);
    }
  }
  if (pars_->ExBshear) {
    cudaMalloc     ( (void**) &phasefac_exb,   sizeof(cuComplex) * Nx * Nyc);
    cudaMalloc     ( (void**) &phasefacminus_exb,   sizeof(cuComplex) * Nx * Nyc);
    cudaMalloc     ( (void**) &kxstar,         sizeof(double) * Nx * Nyc);
    cudaMalloc     ( (void**) &kxbar_ikx_new,  sizeof(int) * Nx * Nyc);
    cudaMalloc     ( (void**) &kxbar_ikx_old,  sizeof(int) * Nx * Nyc);
  }

  checkCuda(cudaGetLastError());

  //  printf("In grids constructor. Nyc = %i \n",Nyc);
  
  setdev_constants(Nx, Ny, Nyc, Nz, Nspecies, Nm, Nl, Nj, pars_->Zp, pars_->ikx_fixed, pars_->iky_fixed, is_lo, is_up, m_lo, m_up, m_ghost, pars_->nm_in);

  checkCuda(cudaDeviceSynchronize());

  DEBUGPRINT("Initializing NCCL comms...\n");
  //cudaStreamCreate(&ncclStream);
  if(iproc == 0) ncclGetUniqueId(&ncclId);
  if(nprocs > 1) {
    MPI_Bcast((void *)&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  // set up some additional ncclIds
  ncclId_m0.resize(nprocs_z);
  for(int i=0; i<nprocs_z; i++) {
    if(iproc == 0) ncclGetUniqueId(&ncclId_m0[i]);
    if(nprocs > 1) {
      MPI_Bcast((void *)&ncclId_m0[i], sizeof(ncclId_m0[i]), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
  }
  ncclId_s.resize(nprocs_s*nprocs_z);
  for(int i=0; i<nprocs_s*nprocs_z; i++) {
    if(iproc == 0) ncclGetUniqueId(&ncclId_s[i]);
    if(nprocs > 1) {
      MPI_Bcast((void *)&ncclId_s[i], sizeof(ncclId_s[i]), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
  }
  ncclId_m.resize(nprocs_m*nprocs_z);
  for(int i=0; i<nprocs_m*nprocs_z; i++) {
    if(iproc == 0) ncclGetUniqueId(&ncclId_m[i]);
    if(nprocs > 1) {
      MPI_Bcast((void *)&ncclId_m[i], sizeof(ncclId_m[i]), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
  }
  ncclId_z.resize(nprocs_m*nprocs_s);
  for(int i=0; i<nprocs_m*nprocs_s; i++) {
    if(iproc == 0) ncclGetUniqueId(&ncclId_z[i]);
    if(nprocs > 1) {
      MPI_Bcast((void *)&ncclId_z[i], sizeof(ncclId_z[i]), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
  }

  DEBUGPRINT("Got NCCL IDs\n");

  checkCuda(ncclCommInitRank(&ncclComm, nprocs, ncclId, iproc));
  // same species and theta slab; ranks differ only in Hermite block
  checkCuda(ncclCommInitRank(&ncclComm_s, nprocs_m, ncclId_s[iproc_s + nprocs_s*iproc_z], iproc_m));
  // same Hermite block and theta slab; ranks differ only in species
  checkCuda(ncclCommInitRank(&ncclComm_m, nprocs_s, ncclId_m[iproc_m + nprocs_m*iproc_z], iproc_s));
  // same Hermite/species block; ranks differ only in theta slab
  checkCuda(ncclCommInitRank(&ncclComm_z, nprocs_z, ncclId_z[iproc_m + nprocs_m*iproc_s], iproc_z));
  // set up NCCL communicator that involves only GPUs containing m=0 in this theta slab
  if(iproc_m == 0) {
    if(nprocs_m > 1)
      checkCuda(ncclCommInitRank(&ncclComm_m0, nprocs_s, ncclId_m0[iproc_z], iproc_s));
    else
      ncclComm_m0 = ncclComm_m;
  }
  DEBUGPRINT("Finished initializaing NCCL comms.\n");
}

Grids::~Grids() {
  if (kx)              cudaFree(kx);
  if (ky)              cudaFree(ky);
  if (kz)              cudaFree(kz);
  if (kzm)             cudaFree(kzm);
  if (kzp)             cudaFree(kzp);
  if (th0)             cudaFree(th0);
  if (phasefac_ntft)        cudaFree(phasefac_ntft);
  if (phasefacminus_ntft)   cudaFree(phasefacminus_ntft);
  if (phasefac_exb)         cudaFree(phasefac_exb);
  if (phasefacminus_exb)         cudaFree(phasefacminus_exb);
  if (iKx)	       cudaFree(iKx);
  if (x)               cudaFree(x);
  if (kxstar)          cudaFree(kxstar);
  if (kxbar_ikx_new)   cudaFree(kxbar_ikx_new);
  if (kxbar_ikx_old)   cudaFree(kxbar_ikx_old);
  
  if (kpar_outh)       free(kpar_outh);
  if (kz_outh)         free(kz_outh);
  if (kx_outh)         free(kx_outh);
  if (kx_h)            free(kx_h);
  if (ky_h)            free(ky_h);
  if (kz_h)            free(kz_h);
  if (x_h)             free(x_h);
  if (y_h)             free(y_h);
  if (z_h)             free(z_h);
  if (m0_h)            free(m0_h);
  if (theta0_h)        free(theta0_h); 
 
  ncclCommDestroy(ncclComm);
  ncclCommDestroy(ncclComm_s);
  ncclCommDestroy(ncclComm_m);
  ncclCommDestroy(ncclComm_z);
  if(nprocs_m > 1 && iproc_m == 0) ncclCommDestroy(ncclComm_m0);
  if(nprocs > 1) {
    MPI_Comm_free(&mpcom_sm);
    MPI_Comm_free(&mpcom_z);
  }
}

void Grids::init_ks_and_coords()
{
  // initialize k arrays
  int Nmax = max(max(Nx, Nyc), Nz);
  int nt = min(32, Nmax);
  int nb = 1 + (Nmax-1)/nt;

  kInit <<<nb, nt>>> (kx, ky, kz, kzm, kzp, pars_->x0, pars_->y0, pars_->Zp, pars_->dealias_kz);

  CP_TO_CPU (kx_h, kx, sizeof(float)*Nx);
  CP_TO_CPU (ky_h, ky, sizeof(float)*Nyc);
  CP_TO_CPU (kz_h, kz, sizeof(float)*Nz);

  // If this is a restarted run, should get kxstar from the restart file
  // otherwise:
  if (pars_->ExBshear) {
    int nn1, nt1, nb1, nn2, nt2, nb2, nt3, nb3;
    nn1 = Nyc;       nt1 = min(32, nn1);     nb1 = 1 + (nn1-1)/nt1;
    nn2 = Nx;        nt2 = min(16, nn2);     nb2 = 1 + (nn2-1)/nt2;
                     nt3 = 1;                nb3 = 1;
    dB = dim3(nt1, nt2, nt3);
    dG = dim3(nb1, nb2, nb3);
    init_kxstar_kxbar_phasefac <<< dG, dB >>> (kxstar, kxbar_ikx_new, kxbar_ikx_old, phasefac_exb, phasefacminus_exb, kx); // Do we really need th0 here? // JFP
    //CP_TO_CPU (theta0_h, th0, sizeof(float)*Nx);
  }
  
  if (Nx<4) {
    //    printf("Nx, Nakx = %d, %d \n",Nx, Nakx);
    //    printf("kx_h = %f \n",kx_h[0]);
    for (int i=0; i<Nx; i++) kx_outh[i] = kx_h[i];
  } else {    
    kx_outh[Nakx/2] = 0.;
    for (int i=1; i<Nakx/2+1; i++) {
      kx_outh[Nakx/2 + i] = kx_h[i];
      kx_outh[Nakx/2 - i] = kx_h[Nx-i];
    }
    /*
    for (int i=0; i<Nx; i++) {
      printf("kx_h[%d] = %f \t",i,kx_h[i]);
    }
    printf("\n");
    for (int i = 0; i < Nakx; i++) {
      printf("kx_outh[%d] = %f \t",i, kx_outh[i]);
    }
    */
  }

  if (Nz>1) {
    if(nprocs_z > 1) {
      for (int i = 0; i < Nz ; i++) {
        int kg_out = (z_lo + i + Nz_glob/2 + 1) % Nz_glob;
        kz_outh[i] = (kg_out < Nz_glob/2+1) ? (float) kg_out/pars_->Zp : (float) (kg_out - Nz_glob)/pars_->Zp;
      }
    } else {
      for (int i = 0; i < Nz ; i++) kz_outh[i] = kz_h[ (i + Nz/2 + 1) % Nz ];
    }
  } else {
    for (int i = 0; i < Nz ; i++) kz_outh[i] = kz_h[ i ];
  }
  
  // define the y coordinate
  y_h[0] = 0.;
  for (int i = 1; i < Ny ; i++) y_h[i] = y_h[i-1] + (float) 2*M_PI*(pars_->y0)/Ny;

  // Could define a variable that keeps track of y(t) when there is ExB shear but it can be derived from what is
  // already written
  
  // define the x coordinate
  x_h[0] = 0.;
  for (int i = 1; i < Nx ; i++) x_h[i] = x_h[i-1] + (float) 2*M_PI*(pars_->x0)/Nx;

  // define the z coordinate
  for(int k=0; k<Nz; k++) {
    int kg = z_lo + k;
    z_h[k] = 2.*M_PI *pars_->Zp *(kg-Nz_glob/2)/Nz_glob;
  }

  LaguerreTransform * laguerre = new LaguerreTransform(this, 1);
  // Estimate v_parallel_max conservatively
  vpar_max = 2.0 * sqrtf( Nm_glob );
  muB_max = laguerre->get_vmax();
  kx_max = kx_h[(Nx-1)/3];
  ky_max = ky_h[(Ny-1)/3];
  kz_max = (nprocs_z > 1) ? (float) (Nz_glob/2)/pars_->Zp : kz_h[Nz/2];
  kperp_min = min(kx_h[1], ky_h[1]);
  delete laguerre;
}
