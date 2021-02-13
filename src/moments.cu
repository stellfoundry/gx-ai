#include "moments.h"

MomentsG::MomentsG(Parameters* pars, Grids* grids) : 
  grids_(grids), pars_(pars)
{
  G_lm       = nullptr;  dens_ptr   = nullptr;  upar_ptr   = nullptr;  tpar_ptr   = nullptr;
  tprp_ptr   = nullptr;  qpar_ptr   = nullptr;  qprp_ptr   = nullptr;

  size_t lhsize = grids_->size_G;
  
  checkCuda(cudaMalloc((void**) &G_lm, lhsize));
  cudaMemset(G_lm, 0., lhsize);
  
  dens_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);
  upar_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);
  tpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);
  tprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);
  qpar_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);
  qprp_ptr = (cuComplex**) malloc(sizeof(cuComplex*) * grids_->Nspecies);

  printf("Allocated a G_lm array of size %.2f MB\n", lhsize/1024./1024.);

  int Nm = grids_->Nm;
  int Nl = grids_->Nl;

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

  if (pars_->ks) {
    int nbx = 128;
    int ngx = (grids_->Nyc-1)/nbx + 1;
    
    dB_all = dim3(nbx, 1, 1);
    dG_all = dim3(ngx, 1, 1);

    dimBlock = dim3(nbx, 1, 1);
    dimGrid  = dim3(ngx, 1, 1);
    
  } else {

    dimBlock = dim3(32, min(4, Nl), min(4, Nm));
    dimGrid  = dim3((grids_->NxNycNz-1)/dimBlock.x+1, 1, 1);
    
    int nyx = grids_->Nyc*grids_->Nx;
    int nslm =  grids_->Nspecies*grids_->Nm*grids_->Nl;
    
    int nbx = 32;
    int ngx = (nyx-1)/nbx + 1;
    
    int nby = 32;
    int ngy = (grids_->Nz-1)/nby + 1;
    
    dB_all = dim3(nbx, nby, 1);
    dG_all = dim3(ngx, ngy, nslm);	 
  }
}

MomentsG::~MomentsG() {
  free (dens_ptr);
  free (upar_ptr);
  free (tpar_ptr);
  free (qpar_ptr);
  free (qprp_ptr);

  if ( G_lm     ) cudaFree ( G_lm );
}

void MomentsG::initialConditions(double *time) {

  size_t momsize = sizeof(cuComplex)*grids_->NxNycNz;
  cuComplex *init_h;  cudaMallocHost((void**) &init_h, momsize); 
  
  for (int idy = 0; idy<grids_->Nyc; idy++) {
    init_h[idy].x = 0.0;
    init_h[idy].y = 0.0;
  }

  init_h[1].x =  0.5;
  init_h[2].y = -0.25;
  
  CP_TO_GPU(G_lm, init_h, momsize);
  
  cudaFree(init_h);

  // restart_read goes here, if restart == T
  // as in gs2, if restart_read is true, we want to *add* the restart values to anything
  // that has happened above and also move the value of time up to the end of the previous run
  if(pars_->restart) {
    DEBUG_PRINT("reading restart file \n");
    this->restart_read(time);
  }

  cudaDeviceSynchronize();
  //  checkCuda(cudaGetLastError());

  //  return cudaGetLastError();
  
}

void MomentsG::initialConditions(float* z_h, double* time) {
 
  cudaDeviceSynchronize(); // to make sure its safe to operate on host memory

  size_t momsize = sizeof(cuComplex)*grids_->NxNycNz;
  cuComplex *init_h = nullptr;  cudaMallocHost((void**) &init_h, momsize); 
  for (int idx=0; idx<grids_->NxNycNz; idx++) {
    init_h[idx].x = 0.;
    init_h[idx].y = 0.;
  }
  
  if (pars_->ks) {
    init_h[1].x =  0.5;
    init_h[2].y = -0.25;
  } else {
  
    if(pars_->init_single) {
      //initialize single mode
      int iky = pars_->iky_single;
      int ikx = pars_->ikx_single;
      DEBUG_PRINT("ikx, iky: %d \t %d \n",ikx, iky);
      //    float fac;
      //    if(pars_->nlpm_test && iky==0) fac = .5;
      //    else fac = 1.;
      //    DEBUG_PRINT("fac = %f \n",fac);
      for(int iz=0; iz<grids_->Nz; iz++) {
	int index = iky + grids_->Nyc*ikx + grids_->NxNyc*iz;
	init_h[index].x = pars_->init_amp; //*fac;
	init_h[index].y = 0.; //init_amp;
      }
    } else {
      srand(22);
      float samp;
      int idx;
      for(int i=0; i < (grids_->Nx - 1)/3 + 1; i++) { 
	// do not kick the ky=0 modes
	for(int j=1; j< (grids_->Ny - 1)/3 + 1; j++) {
	  for (int js=0; js < 2; js++) {
	    if (i==0) {
	      idx = i;
	    } else {
	      idx = js*(grids_->Nx - 2*i) + i;
	    }
	    samp = pars_->init_amp;
	    float ra = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
	    float rb = (float) (samp * (rand()-RAND_MAX/2) / RAND_MAX);
	    //	  printf("ra = %e \t rb = %e \n",ra,rb);
	    //loop over z *here*, to get rid of randomness in z in initial condition
	    for(int k=0; k<grids_->Nz; k++) {
	      int index = j + grids_->Nyc*idx + grids_->NxNyc*k;
	      init_h[index].x = ra*cos(pars_->kpar_init*z_h[k]/pars_->Zp);
	      init_h[index].y = rb*cos(pars_->kpar_init*z_h[k]/pars_->Zp);
	      //	    printf("init_h[%d] = (%e, %e) \n",index,init_h[index].x,init_h[index].y);
	    }
	  }
	}
      }
    }
  }

  // copy initial condition into device memory
  for (int is=0; is<grids_->Nspecies; is++) {
    if(pars_->init == DENS) CP_TO_GPU(dens_ptr[is], init_h, momsize);
    if(pars_->init == UPAR) CP_TO_GPU(upar_ptr[is], init_h, momsize);
    if(pars_->init == TPAR) CP_TO_GPU(tpar_ptr[is], init_h, momsize);
    if(pars_->init == QPAR) CP_TO_GPU(qpar_ptr[is], init_h, momsize);
    if(pars_->init == TPRP) CP_TO_GPU(tprp_ptr[is], init_h, momsize);
    if(pars_->init == QPRP) CP_TO_GPU(qprp_ptr[is], init_h, momsize);
  }
  cudaFree(init_h);

  // restart_read goes here, if restart == T
  // as in gs2, if restart_read is true, we want to *add* the restart values to anything
  // that has happened above and also move the value of time up to the end of the previous run
  if(pars_->restart) {
    DEBUG_PRINT("reading restart file \n");
    this->restart_read(time);
  }

  cudaDeviceSynchronize();
  //  checkCuda(cudaGetLastError());

  //  return cudaGetLastError();
}

void MomentsG::scale(double scalar) {
  scale_kernel <<< dG_all, dB_all >>>(G_lm, scalar);
}

void MomentsG::scale(cuComplex scalar) {
  scale_kernel <<< dG_all, dB_all >>>(G_lm, scalar);
}

void MomentsG::add_scaled(double c1, MomentsG* G1,
			  double c2, MomentsG* G2) {
  bool neqfix = !pars_->eqfix;
  add_scaled_kernel <<< dG_all, dB_all >>> (G_lm, c1, G1->G_lm, c2, G2->G_lm, neqfix);
}

void MomentsG::add_scaled(double c1, MomentsG* G1,
			  double c2, MomentsG* G2,
			  double c3, MomentsG* G3) {
  bool neqfix = !pars_->eqfix;
  add_scaled_kernel <<< dG_all, dB_all >>> (G_lm, c1, G1->G_lm, c2, G2->G_lm, c3, G3->G_lm, neqfix);
}

void MomentsG::add_scaled(double c1, MomentsG* G1,
			  double c2, MomentsG* G2,
			  double c3, MomentsG* G3,
			  double c4, MomentsG* G4) {
  bool neqfix = !pars_->eqfix;
  add_scaled_kernel <<< dG_all, dB_all >>> (G_lm, c1, G1->G_lm, c2, G2->G_lm, c3, G3->G_lm, c4, G4->G_lm, neqfix);
}

void MomentsG::add_scaled(double c1, MomentsG* G1,
			  double c2, MomentsG* G2, 
			  double c3, MomentsG* G3,
			  double c4, MomentsG* G4,
			  double c5, MomentsG* G5)
{
  bool neqfix = !pars_->eqfix;
  add_scaled_kernel<<< dG_all, dB_all >>>(G_lm, c1, G1->G_lm, c2, G2->G_lm, c3, G3->G_lm, c4, G4->G_lm, c5, G5->G_lm, neqfix);
}

void MomentsG::reality(int ngz) 
{
  dim3 dB;
  dim3 dG;

  int ngx = (grids_->Nx-1)/3 + 1;
  
  dB.x = 32;
  dG.x = (ngx-1)/dB.x + 1;
  
  int ngy = grids_->Nz;

  dB.y = 8;
  dG.y = (ngy-1)/dB.y + 1;
  
  dB.z = 4;
  dG.z = (ngz-1)/dB.z + 1;

  reality_kernel <<< dG, dB >>> (G_lm, ngz);
}

void MomentsG::restart_write(double* time)
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
  int id_ri, id_nz, id_Nkx, id_Nky;
  int id_nh, id_nl, id_ns;
  int id_G, id_time;

  char strb[512];
  strcpy(strb, pars_->restart_to_file.c_str());

  //  if(pars_->restart) {
  // ultimately, appending to an existing file
  // if appending, are the time values consistent?
  // inquire/define the variable names
  //  } else {
  int ri=2;
  if (retval = nc_create(strb, NC_CLOBBER, &ncres)) ERR(retval);

  if (retval = nc_def_dim(ncres, "ri",  ri,    &id_ri)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nz",  Nz,    &id_nz)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nkx", Nakx,  &id_Nkx)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nky", Naky,  &id_Nky)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nl",  Nl,    &id_nl)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Nm",  Nm,    &id_nh)) ERR(retval);
  if (retval = nc_def_dim(ncres, "Ns",  nspec, &id_ns)) ERR(retval);

  moments_out[0] = id_ns;  count[0] = nspec;
  moments_out[1] = id_nh;  count[1] = Nm;
  moments_out[2] = id_nl;  count[2] = Nl;
  moments_out[3] = id_nz;  count[3] = Nz;  
  moments_out[4] = id_Nkx; count[4] = Nakx;
  moments_out[5] = id_Nky; count[5] = Naky;
  moments_out[6] = id_ri;  count[6] = ri;

  start[0] = 0; start[1] = 0; start[2] = 0; start[3] = 0; start[4] = 0; start[5] = 0; start[6] = 0;
  if (retval = nc_def_var(ncres, "G", NC_FLOAT, 7, moments_out, &id_G)) ERR(retval);
  if (retval = nc_def_var(ncres, "time", NC_DOUBLE, 0, 0, &id_time)) ERR(retval);
  if (retval = nc_enddef(ncres)) ERR(retval);

  if (retval = nc_put_var(ncres, id_time, time)) ERR(retval);
  
  unsigned int itot, jtot;
  jtot = Nx   * Nyc  * Nz * Nm * Nl * nspec;
  itot = Nakx * Naky * Nz * Nm * Nl * nspec;
  cudaMallocHost((void**) &G_h,   sizeof(cuComplex) * jtot); 
  cudaMallocHost((void**) &G_out, sizeof(float) * itot * 2);

  for (unsigned int index=0; index <   jtot; index++) {G_h[index].x = 0.; G_h[index].y = 0.;}
  for (unsigned int index=0; index < 2*itot; index++) G_out[index] = 0.;
  
  CP_TO_CPU(G_h, G_lm, sizeof(cuComplex)*jtot);
  
  for (int is=0; is < nspec; is++) {
    for (int m=0; m < Nm; m++) {
      for (int l=0; l < Nl; l++) {
	for (int k=0; k < Nz; k++) {
	  
	  for (int i=0; i < (Nx-1)/3+1; i++) {
	    for (int j=0; j < Naky; j++) {
	      unsigned int index     = j + Nyc*i  + Nyc*Nx*k    + Nyc*Nx*Nz*l    + Nyc*Nx*Nz*Nl*m    + Nyc*Nx*Nz*Nl*Nm*is;
	      unsigned int index_out = j + Naky*i + Naky*Nakx*k + Naky*Nakx*Nz*l + Naky*Nakx*Nz*Nl*m + Naky*Nakx*Nz*Nl*Nm*is;
	      G_out[2*index_out]   = G_h[index].x; 
	      G_out[2*index_out+1] = G_h[index].y;
	    }
	  }
	  
	  for (int i=2*Nx/3+1; i < Nx; i++) {
	    for (int j=0; j < Naky; j++) {
	      unsigned int index     = j + Nyc*i  + Nyc*Nx*k    + Nyc*Nx*Nz*l    + Nyc*Nx*Nz*Nl*m    + Nyc*Nx*Nz*Nl*Nm*is;
	      
	      unsigned int index_out = j + Naky*(i-2*Nx/3+(Nx-1)/3)
		+ Naky*Nakx*k + Naky*Nakx*Nz*l + Naky*Nakx*Nz*Nl*m + Naky*Nakx*Nz*Nl*Nm*is;
	      
	      G_out[2*index_out]   = G_h[index].x;
	      G_out[2*index_out+1] = G_h[index].y;
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
}

void MomentsG::restart_read(double* time)
{
  float scale;
  float* G_in;
  cuComplex* G_h;
  cuComplex* G_hold;

  int retval;
  int ncres;
  
  size_t lhsize = grids_->size_G;
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
  int id_nz, id_Nkx, id_Nky;
  int id_nh, id_nl, id_ns;
  int id_G, id_time;

  char stra[NC_MAX_NAME+1];
  char strb[512];
  strcpy(strb, pars_->restart_from_file.c_str());

  if (retval = nc_open(strb, NC_NOWRITE, &ncres)) { printf("file: %s \n",strb); ERR(retval);}
  
  if (retval = nc_inq_dimid(ncres, "Nkx",  &id_Nkx))  ERR(retval);
  if (retval = nc_inq_dimid(ncres, "Nky",  &id_Nky))  ERR(retval);    
  
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

  unsigned int itot;
  itot = Nakx * Naky * Nz * Nm * Nl * nspec;
  cudaMallocHost((void**) &G_hold, lhsize);
  cudaMallocHost((void**) &G_h,    lhsize);
  cudaMallocHost((void**) &G_in,   sizeof(float) * itot * 2);
  
  for (unsigned int index=0; index < itot; index++) {G_hold[index].x = 0.; G_hold[index].y = 0.;}
  for (unsigned int index=0; index < itot; index++) {G_h[index].x = 0.; G_h[index].y = 0.;}
  for (unsigned int index=0; index<2*itot; index++) {G_in[index] = 0.;}
  CP_TO_CPU(G_hold, G_lm, sizeof(cuComplex)*itot);
  
  if (retval = nc_get_var(ncres, id_G, G_in)) ERR(retval);
  if (retval = nc_get_var(ncres, id_time, time)) ERR(retval);
  if (retval = nc_close(ncres)) ERR(retval);

  scale = pars_->scale;
  for (int is=0; is < nspec; is++) {
    for (int m=0; m < Nm; m++) {
      for (int l=0; l < Nl; l++) {
	for (int k=0; k < Nz; k++) {
	  
	  for (int i=0; i < ((Nx-1)/3+1); i++) {
	    for (int j=0; j < Naky; j++) {
	      unsigned int index    = j + Nyc*i  + Nyc*Nx*k    + Nyc*Nx*Nz*l    + Nyc*Nx*Nz*Nl*m    + Nyc*Nx*Nz*Nl*Nm*is;
	      unsigned int index_in = j + Naky*i + Naky*Nakx*k + Naky*Nakx*Nz*l + Naky*Nakx*Nz*Nl*m + Naky*Nakx*Nz*Nl*Nm*is;
	      G_h[index].x = scale * G_in[2*index_in]   + G_hold[index].x;
	      G_h[index].y = scale * G_in[2*index_in+1] + G_hold[index].y;
	    }
	  }
	  
	  for (int i=2*Nx/3+1; i < Nx; i++) {
	    for (int j=0; j < Naky; j++) {
	      unsigned int index    = j + Nyc*i
		+ Nyc*Nx*k    + Nyc*Nx*Nz*l    + Nyc*Nx*Nz*Nl*m    + Nyc*Nx*Nz*Nl*Nm*is;
	      
	      unsigned int index_in = j + Naky*(i-2*Nx/3+(Nx-1)/3)
		+ Naky*Nakx*k + Naky*Nakx*Nz*l + Naky*Nakx*Nz*Nl*m + Naky*Nakx*Nz*Nl*Nm*is;
	      
	      G_h[index].x = scale * G_in[2*index_in]   + G_hold[index].x;
	      G_h[index].y = scale * G_in[2*index_in+1] + G_hold[index].y;
	    }
	  }
	}
      }
    }
  }

  cudaFreeHost(G_in);
  cudaFreeHost(G_hold);

  unsigned int jtot = Nx * Nyc * Nz * Nm * Nl * nspec;
  CP_TO_GPU(G_lm, G_h, sizeof(cuComplex)*jtot);
  
  cudaFreeHost(G_h);
}
void MomentsG::qvar(int N)
{
  cuComplex* G_h;
  int Nk = grids_->Nyc;
  //  int Nk = grids_->NxNycNz;
  G_h = (cuComplex*) malloc (sizeof(cuComplex)*N);
  for (int i=0; i<N; i++) {G_h[i].x = 0.; G_h[i].y = 0.;}
  CP_TO_CPU (G_h, G_lm, N*sizeof(cuComplex));
  printf("\n");
  for (int i=0; i<N; i++) printf("var(%d,%d) = (%e, %e) \n", i%Nk, i/Nk, G_h[i].x, G_h[i].y);
  // for (int i=0; i<5; i++) printf("m var(%d,%d) = (%e, %e) \n", i%Nk, i/Nk, G_h[i].x, G_h[i].y);
  printf("\n");

  free (G_h);
}

