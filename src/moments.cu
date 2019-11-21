#include "netcdf.h"
#include "moments.h"
#include "device_funcs.h"
#include "get_error.h"
#include "cuda_constants.h"

MomentsG::MomentsG(Parameters* pars, Grids* grids) : 
  grids_(grids), pars_(pars), 
  LHsize_(sizeof(cuComplex)*grids_->NxNycNz*grids_->Nmoms*grids_->Nspecies), 
  Momsize_(sizeof(cuComplex)*grids->NxNycNz)
{
  int Nm = grids_->Nm;
  int Nl = grids_->Nl;
  checkCuda(cudaMalloc((void**) &G_lm, LHsize_));
  dens_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  upar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  tpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  tprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  qpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);
  qprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*)*grids_->Nspecies);

  cudaMemset(G_lm, 0., LHsize_);

  printf("Allocated a G_lm array of size %.2f MB\n", LHsize_/1024./1024.);

  for(int s=0; s<grids->Nspecies; s++) {
    // set up pointers for named moments that point to parts of G_lm
    int l,m;
    l = 0, m = 0; // density
    if(l<Nl && m<Nm) dens_ptr[s] = G(l,m,s);
    
    l = 0, m = 1; // u_parallel
    if(l<Nl && m<Nm) upar_ptr[s] = G(l,m,s);
    
    l = 0, m = 2; // T_parallel / sqrt(2)
    if(l<Nl && m<Nm) tpar_ptr[s] = G(l,m,s);
    
    l = 0, m = 3; // q_parallel / sqrt(6)
    if(l<Nl && m<Nm) qpar_ptr[s] = G(l,m,s);

    l = 1, m = 0; // T_perp 
    if(l<Nl && m<Nm) tprp_ptr[s] = G(l,m,s);
    
    l = 1, m = 1; // q_perp
    if(l<Nl && m<Nm) qprp_ptr[s] = G(l,m,s);
  }

  dimBlock = dim3(32, min(4, Nl), min(4, Nm));
  dimGrid = dim3(grids_->NxNycNz/dimBlock.x+1, 1, 1);
}

MomentsG::~MomentsG() {
  free(dens_ptr);
  free(upar_ptr);
  free(tpar_ptr);
  free(tprp_ptr);
  free(qpar_ptr);
  free(qprp_ptr);
  cudaFree(G_lm);
}

int MomentsG::initialConditions(Geometry* geo, double* time) {
 
  cudaDeviceSynchronize(); // to make sure its safe to operate on host memory

  cuComplex* init_h = (cuComplex*) malloc(Momsize_);
  for (int idx=0; idx<grids_->NxNycNz; idx++) {
    init_h[idx].x = 0.;
    init_h[idx].y = 0.;
  }
  
  if(pars_->init_single) {
    //initialize single mode
    int iky = pars_->iky_single;
    int ikx = pars_->ikx_single;
    DEBUG_PRINT("ikx, iky: %d \t %d \n",ikx, iky);
    float fac;
    if(pars_->nlpm_test && iky==0) fac = .5;
    else fac = 1.;
    DEBUG_PRINT("fac = %f \n",fac);
    for(int iz=0; iz<grids_->Nz; iz++) {
      int index = iky + grids_->Nyc*ikx + grids_->NxNyc*iz;
      init_h[index].x = pars_->init_amp*fac;
      init_h[index].y = 0.; //init_amp;
    }
  } else {
    srand(22);
    float samp;
    for(int i=0; i<grids_->Nyc; i++) {
      for(int j=0; j<grids_->Nx; j++) {
	// Do not kick the kx=ky=0 mode
	if(i>0 || j>0) {
	  samp = pars_->init_amp;
	  
          float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
          //float rb = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
	  
          //loop over z *here*, to get rid of randomness in z in initial condition
          for(int k=0; k<grids_->Nz; k++) {
	    int index = i + grids_->Nyc*j + grids_->NxNyc*k;
	    init_h[index].x = ra*cos(pars_->kpar_init*geo->z_h[k]/pars_->Zp);
	    init_h[index].y = ra*cos(pars_->kpar_init*geo->z_h[k]/pars_->Zp);
          }
	}
      }
    }
  }
  
  // copy initial condition into device memory
  if(pars_->init == DENS) {         CP_TO_GPU(dens_ptr[0], init_h, Momsize_);
  }
  if(pars_->init == UPAR) {         CP_TO_GPU(upar_ptr[0], init_h, Momsize_);
  }
  if(pars_->init == TPAR) {         CP_TO_GPU(tpar_ptr[0], init_h, Momsize_);
  }
  if(pars_->init == TPRP) {         CP_TO_GPU(tprp_ptr[0], init_h, Momsize_);
  }
  free(init_h);

  // restart_read goes here, if restart == T
  // as in gs2, if restart_read is true, we want to *add* the restart values to anything
  // that has happened above, in this routine. 
  if(pars_->restart) {
    DEBUG_PRINT("reading restart file \n");
    this->restart_read(time);
  }
  
  this->reality();

  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  return cudaGetLastError();
}

int MomentsG::zero() {
  cudaMemset(G_lm, 0., LHsize_);
  return 0;
}

int MomentsG::zero(int l, int m, int s) {
  cudaMemset(G(l,m,s), 0., Momsize_);
  return 0;
}

int MomentsG::scale(double scalar) {
  scale_kernel<<<dimGrid,dimBlock>>>(G_lm, G_lm, scalar);
  return 0;
}

int MomentsG::scale(cuComplex scalar) {
  scale_kernel<<<dimGrid,dimBlock>>>(G_lm, G_lm, scalar);
  return 0;
}

int MomentsG::add_scaled(double c1, MomentsG* G1,
			 double c2, MomentsG* G2) {
  if(pars_->eqfix) {
    bool bdum = true;
    add_scaled_kernel <<<dimGrid,dimBlock>>> (G_lm,
					      c1, G1->G_lm,
					      c2, G2->G_lm, bdum);
  } else {
    add_scaled_kernel <<<dimGrid,dimBlock>>> (G_lm,
					      c1, G1->G_lm,
					      c2, G2->G_lm);
  }
  return 0;
}

int MomentsG::add_scaled(double c1, MomentsG* G1,
			 double c2, MomentsG* G2,
			 double c3, MomentsG* G3) {
  if(pars_->eqfix) {
    bool bdum = true;
    add_scaled_kernel <<<dimGrid,dimBlock>>> (G_lm,
					      c1, G1->G_lm,
					      c2, G2->G_lm,
					      c3, G3->G_lm, bdum);
  } else {
    add_scaled_kernel <<<dimGrid,dimBlock>>> (G_lm,
					      c1, G1->G_lm,
					      c2, G2->G_lm,
					      c3, G3->G_lm);
  }
  return 0;
}

int MomentsG::add_scaled(double c1, MomentsG* G1,
			 double c2, MomentsG* G2,
			 double c3, MomentsG* G3,
			 double c4, MomentsG* G4) {
  if(pars_->eqfix) {
    bool bdum = true;
    add_scaled_kernel <<<dimGrid,dimBlock>>> (G_lm,
					      c1, G1->G_lm,
					      c2, G2->G_lm,
					      c3, G3->G_lm,
					      c4, G4->G_lm, bdum);
  } else {
    add_scaled_kernel <<<dimGrid,dimBlock>>> (G_lm,
					      c1, G1->G_lm,
					      c2, G2->G_lm,
					      c3, G3->G_lm,
					      c4, G4->G_lm);
  }
  return 0;
}

int MomentsG::add_scaled(double c1, MomentsG* G1,
			 double c2, MomentsG* G2, 
			 double c3, MomentsG* G3,
			 double c4, MomentsG* G4,
			 double c5, MomentsG* G5)
{
  if(pars_->eqfix) {
    bool bdum = true;
    add_scaled_kernel<<<dimGrid,dimBlock>>>(G_lm,
					    c1, G1->G_lm,
					    c2, G2->G_lm,
					    c3, G3->G_lm,
					    c4, G4->G_lm,
					    c5, G5->G_lm, bdum);
  } else {
    add_scaled_kernel<<<dimGrid,dimBlock>>>(G_lm,
					    c1, G1->G_lm,
					    c2, G2->G_lm,
					    c3, G3->G_lm,
					    c4, G4->G_lm,
					    c5, G5->G_lm);
  }
  return 0;
  
}

int MomentsG::reality() 
{
  dim3 dB = dim3(32,min(grids_->Nz,32),1);
  dim3 dG = dim3(grids_->Nx/2/dB.x+1, grids_->Nz/dB.y+1, 1);
  reality_kernel<<<dG,dB>>>(G_lm);
  return 0;
}

int MomentsG::restart_write(double* time)
{
  float* G_out;
  cuComplex* G_h;

  int retval;
  int ncres;
  int moments_out[7];
  size_t start[7];
  size_t count[7];
  
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;
  int nspec = pars_->nspec;
  int Nm   = grids_->Nm;
  int Nl   = grids_->Nl;
  
  // handles
  int id_ri, id_nx, id_nz, id_Nkx, id_Nky;
  int id_Nyc, id_nh, id_nl, id_ns, id_mask;
  int id_G, id_time;
  int mask = !pars_->linear;

  char strb[512];
  strcpy(strb, pars_->restart_to_file);

  //  if(pars_->restart) {
  // ultimately, appending to an existing file
  // if appending, are the time values consistent?
  // inquire/define the variable names
  //  } else {
  int ri=2;
  if (retval = nc_create(strb, NC_CLOBBER, &ncres)) ERR(retval);

  if (retval = nc_def_dim(ncres, "ri",  ri,    &id_ri)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nx",  Nx,    &id_nx)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nz",  Nz,    &id_nz)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nkx", Nakx,  &id_Nkx)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nky", Naky,  &id_Nky)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nyc", Nyc,   &id_Nyc)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nl",  Nl,    &id_nl)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nm",  Nm,    &id_nh)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Ns",  nspec, &id_ns)) ERR(retval);

  moments_out[0] = id_ns;  count[0] = nspec;
  moments_out[1] = id_nh;  count[1] = Nm;
  moments_out[2] = id_nl;  count[2] = Nl;
  moments_out[3] = id_nz;  count[3] = Nz;  
  if (mask) {
    moments_out[4] = id_Nkx; count[4] = Nakx;
    moments_out[5] = id_Nky; count[5] = Naky;
  } else {
    moments_out[4] = id_nx;  count[4] = Nx;
    moments_out[5] = id_Nyc; count[5] = Nyc;
  }
  moments_out[6] = id_ri;   count[6] = ri;

  start[0] = 0; start[1] = 0; start[2] = 0; start[3] = 0; start[4] = 0; start[5] = 0; start[6] = 0;
  if (retval = nc_def_var(ncres, "G", NC_FLOAT, 7, moments_out, &id_G)) ERR(retval);
  if (retval = nc_def_var(ncres, "time", NC_DOUBLE, 0, 0, &id_time)) ERR(retval);
  if (retval = nc_def_var(ncres, "mask", NC_INT, 0, 0, &id_mask)) ERR(retval);
  if (retval = nc_enddef(ncres)) ERR(retval);

  if (retval = nc_put_var(ncres, id_time, time)) ERR(retval);
  
  int idum;
  idum = mask ? 1 : 0;
  if (retval = nc_put_var(ncres, id_mask, &idum)) ERR(retval);
  
  int itot;
  if (mask) {
    itot = Nakx * Naky * Nz * Nm * Nl * nspec;
    cudaMallocHost((void**) &G_h,   sizeof(cuComplex)* itot    );
    cudaMallocHost((void**) &G_out, sizeof(float)    * itot * 2);
  } else {
    itot = Nx * Nyc * Nz * Nm * Nl * nspec;
    cudaMallocHost((void**) &G_h, sizeof(cuComplex) * itot);
    cudaMallocHost((void**) &G_out, sizeof(float)   * itot * 2);
  }

  CP_TO_CPU(G_h, G_lm, sizeof(cuComplex)*itot);

  if (mask) {
    for (int is=0; is < nspec; is++) {
      for (int m=0; m < Nm; m++) {
	for (int l=0; l < Nl; l++) {
	  for (int k=0; k < Nz; k++) {
	    
	    for (int i=0; i < ((Nx-1)/3+1); i++) {
	      for (int j=0; j < Naky; j++) {
		int index     = j + Nyc*i  + Nyc*Nx*k    + Nyc*Nx*Nz*l    + Nyc*Nx*Nz*Nl*m    + Nyc*Nx*Nz*Nl*Nm*is;
		int index_out = j + Naky*i + Naky*Nakx*k + Naky*Nakx*Nz*l + Naky*Nakx*Nz*Nl*m + Naky*Nakx*Nz*Nl*Nm*is;
		G_out[2*index_out]   = G_h[index].x;
		G_out[2*index_out+1] = G_h[index].y;
	      }
	    }

	    for (int i=2*Nx/3+1; i < Nx; i++) {
	      for (int j=0; j < Naky; j++) {
		int index     = j + Nyc*i
		  + Nyc*Nx*k    + Nyc*Nx*Nz*l    + Nyc*Nx*Nz*Nl*m    + Nyc*Nx*Nz*Nl*Nm*is;

		int index_out = j + Naky*(i-2*Nx/3+(Nx-1)/3)
		  + Naky*Nakx*k + Naky*Nakx*Nz*l + Naky*Nakx*Nz*Nl*m + Naky*Nakx*Nz*Nl*Nm*is;

		G_out[2*index_out]   = G_h[index].x;
		G_out[2*index_out+1] = G_h[index].y;
	      }
	    }
	  }
	}
      }
    }
  } else {
    for (int is=0; is < nspec; is++) {
      for (int m=0; m < Nm; m++) {
	for (int l=0; l < Nl; l++) {
	  for (int k=0; k < Nz; k++) {	    
	    for (int i=0; i < Nx; i++) {
	      for (int j=0; j < Nyc; j++) {
		int index     = j + Nyc*i + Nyc*Nx*k + Nyc*Nx*Nz*l + Nyc*Nx*Nz*Nl*m + Nyc*Nx*Nz*Nl*Nm*is;
		G_out[2*index]   = G_h[index].x;
		G_out[2*index+1] = G_h[index].y;
	      }
	    }
	  }
	}
      }
    }
  }

  if (retval = nc_put_vara(ncres, id_G, start, count, G_out)) ERR(retval);
      
  cudaFreeHost(G_out);
  cudaFreeHost(G_h);

  if (retval = nc_close(ncres)) ERR(retval);
  return retval;
}

int MomentsG::restart_read(double* time)
{
  float scale;
  float* G_in;
  cuComplex* G_h;
  cuComplex* G_hold;

  int retval;
  int ncres;
  
  size_t ldum;
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;
  int nspec = pars_->nspec;
  int Nm   = grids_->Nm;
  int Nl   = grids_->Nl;
  
  // handles
  int id_nx, id_nz, id_Nkx, id_Nky;
  int id_Nyc, id_nh, id_nl, id_ns, id_mask;
  int id_G, id_time;
  int mask = !pars_->linear;
  int mask_file; 

  char stra[NC_MAX_NAME+1];
  char strb[512];
  strcpy(strb, pars_->restart_from_file);

  if (retval = nc_open(strb, 0, &ncres)) ERR(retval);
  
  if (retval = nc_inq_varid(ncres, "mask", &id_mask)) ERR(retval);
  if (retval = nc_get_var(ncres, id_mask, &mask_file)) ERR(retval);
  if (mask!=mask_file) {
    printf("Cannot restart because of mask mismatch: %d \t %d \n", mask, mask_file);
    exit (1);
  }

  if (mask) {
    if (retval = nc_inq_dimid(ncres, "Nkx",  &id_Nkx))  ERR(retval);
    if (retval = nc_inq_dimid(ncres, "Nky",  &id_Nky))  ERR(retval);    
  } else {
    if (retval = nc_inq_dimid(ncres, "Nx",   &id_nx))   ERR(retval);
    if (retval = nc_inq_dimid(ncres, "Nyc",  &id_Nyc))  ERR(retval);
  }
  
  if (retval = nc_inq_dimid(ncres, "Nz",   &id_nz))   ERR(retval);
  if (retval = nc_inq_dimid(ncres, "Nl",   &id_nl))   ERR(retval);
  if (retval = nc_inq_dimid(ncres, "Nm",   &id_nh))   ERR(retval);
  if (retval = nc_inq_dimid(ncres, "Ns",   &id_ns))   ERR(retval);
  if (retval = nc_inq_varid(ncres, "G",    &id_G))    ERR(retval);
  if (retval = nc_inq_varid(ncres, "time", &id_time)) ERR(retval);
  
  if (retval = nc_inq_dim(ncres, id_ns, stra, &ldum))  ERR(retval);
  if (nspec!=ldum) {
    printf("Cannot restart because of nspec mismatch: %d \t %zu \n", nspec, ldum);
    exit (1);
  }

  if (retval = nc_inq_dim(ncres, id_nh, stra, &ldum))  ERR(retval);
  if (Nm!=ldum) {
    printf("Cannot restart because of Nm mismatch: %d \t %zu \n", Nm, ldum);
    exit (1);
  }

  if (retval = nc_inq_dim(ncres, id_nl, stra, &ldum))  ERR(retval);
  if (Nl!=ldum) {
    printf("Cannot restart because of Nl mismatch: %d \t %zu \n", Nl, ldum);
    exit (1);
  }

  if (retval = nc_inq_dim(ncres, id_nz, stra, &ldum))  ERR(retval);
  if (Nz!=ldum) {
    printf("Cannot restart because of nz mismatch: %d \t %zu \n", Nz, ldum);
    exit (1);
  }

  if (mask) {
    if (retval = nc_inq_dim(ncres, id_Nkx, stra, &ldum))  ERR(retval);
    if (Nakx!=ldum) {
      printf("Cannot restart because of Nkx mismatch: %d \t %zu \n", Nakx, ldum);
      exit (1);
    }
    
    if (retval = nc_inq_dim(ncres, id_Nky, stra, &ldum))  ERR(retval);
    if (Naky!=ldum) {
      printf("Cannot restart because of Nky mismatch: %d \t %zu \n", Naky, ldum);
      exit (1);
    }
  } else {
    if (retval = nc_inq_dim(ncres, id_nx, stra, &ldum))  ERR(retval);
    if (Nx!=ldum) {
      printf("Cannot restart because of Nx mismatch: %d \t %zu \n", Nx, ldum);
      exit (1);
    }
    
    if (retval = nc_inq_dim(ncres, id_Nyc, stra, &ldum))  ERR(retval);
    if (Nyc!=ldum) {
      printf("Cannot restart because of Nyc mismatch: %d \t %zu \n", Nyc, ldum);
      exit (1);
    }    
  }

  int itot;
  if (mask) {
    itot = Nakx * Naky * Nz * Nm * Nl * nspec;
    cudaMallocHost((void**) &G_hold,  sizeof(cuComplex)* itot    );
    cudaMallocHost((void**) &G_h,     sizeof(cuComplex)* itot    );
    cudaMallocHost((void**) &G_in,    sizeof(float)    * itot * 2);
  } else {
    itot = Nx * Nyc * Nz * Nm * Nl * nspec;
    cudaMallocHost((void**) &G_hold,  sizeof(cuComplex) * itot);
    cudaMallocHost((void**) &G_h,     sizeof(cuComplex) * itot);
    cudaMallocHost((void**) &G_in,    sizeof(float)   * itot * 2);
  }

  CP_TO_CPU(G_hold, G_lm, sizeof(cuComplex)*itot);
  
  if (retval = nc_get_var(ncres, id_G, G_in)) ERR(retval);
  if (retval = nc_get_var(ncres, id_time, time)) ERR(retval);
  if (retval = nc_close(ncres)) ERR(retval);

  scale = pars_->scale;
  if (mask) {
    for (int is=0; is < nspec; is++) {
      for (int m=0; m < Nm; m++) {
	for (int l=0; l < Nl; l++) {
	  for (int k=0; k < Nz; k++) {
	    
	    for (int i=0; i < ((Nx-1)/3+1); i++) {
	      for (int j=0; j < Naky; j++) {
		int index    = j + Nyc*i  + Nyc*Nx*k    + Nyc*Nx*Nz*l    + Nyc*Nx*Nz*Nl*m    + Nyc*Nx*Nz*Nl*Nm*is;
		int index_in = j + Naky*i + Naky*Nakx*k + Naky*Nakx*Nz*l + Naky*Nakx*Nz*Nl*m + Naky*Nakx*Nz*Nl*Nm*is;
		G_h[index].x = scale * G_in[2*index_in]   + G_hold[index].x;
		G_h[index].y = scale * G_in[2*index_in+1] + G_hold[index].y;
	      }
	    }

	    for (int i=2*Nx/3+1; i < Nx; i++) {
	      for (int j=0; j < Naky; j++) {
		int index    = j + Nyc*i
		  + Nyc*Nx*k    + Nyc*Nx*Nz*l    + Nyc*Nx*Nz*Nl*m    + Nyc*Nx*Nz*Nl*Nm*is;

		int index_in = j + Naky*(i-2*Nx/3+(Nx-1)/3)
		  + Naky*Nakx*k + Naky*Nakx*Nz*l + Naky*Nakx*Nz*Nl*m + Naky*Nakx*Nz*Nl*Nm*is;

		G_h[index].x = scale * G_in[2*index_in]   + G_hold[index].x;
		G_h[index].y = scale * G_in[2*index_in+1] + G_hold[index].y;
	      }
	    }
	  }
	}
      }
    }
  } else {
    for (int is=0; is < nspec; is++) {
      for (int m=0; m < Nm; m++) {
	for (int l=0; l < Nl; l++) {
	  for (int k=0; k < Nz; k++) {	    
	    for (int i=0; i < Nx; i++) {
	      for (int j=0; j < Nyc; j++) {
		int index     = j + Nyc*i + Nyc*Nx*k + Nyc*Nx*Nz*l + Nyc*Nx*Nz*Nl*m + Nyc*Nx*Nz*Nl*Nm*is;
		G_h[index].x = scale * G_in[2*index]   + G_hold[index].x;
		G_h[index].y = scale * G_in[2*index+1] + G_hold[index].y;
	      }
	    }
	  }
	}
      }
    }
  }

  cudaFreeHost(G_in);
  cudaFreeHost(G_hold);

  CP_TO_GPU(G_lm, G_h, sizeof(cuComplex)*itot);
  
  cudaFreeHost(G_h);

  return retval;
}
