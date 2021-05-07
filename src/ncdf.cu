#include "netcdf.h"
#include "ncdf.h"
#define loop_R  <<< dGr,  dBr  >>>
#define loop_xy <<< dgxy, dbxy >>> 
#define loop_x  <<< dgx,  dbx  >>> 
#define loop_y  <<< dgp,  dbp  >>> 

NetCDF_ids::NetCDF_ids(Grids* grids, Parameters* pars, Geometry* geo) :
  grids_(grids), pars_(pars), geo_(geo),
  red(nullptr), pot(nullptr), ph2(nullptr), all_red(nullptr), grad_phi(nullptr), grad_perp(nullptr)
{

  primary     = nullptr;  secondary   = nullptr;  tertiary    = nullptr;   vEk = nullptr;

  if (pars_->diagnosing_spectra || pars_->diagnosing_kzspec) {
    float dum = 1.0;
    red = new          All_Reduce(grids_, pars_->wspectra); CUDA_DEBUG("Reductions: %s \n"); // G**2
    pot = new Grid_Species_Reduce(grids_, pars_->aspectra); CUDA_DEBUG("Reductions: %s \n"); // (1-G0) Phi**2 keeping track of species
    ph2 = new         Grid_Reduce(grids_, pars_->aspectra); CUDA_DEBUG("Reductions: %s \n"); // Phi**2
  }

  int nS  = grids_->Nspecies;
  int nM  = grids_->Nm;
  int nL  = grids_->Nl;
  int nY  = grids_->Nyc;
  int nYk = grids_->Naky;
  int nX  = grids_->Nx;
  int nXk = grids_->Nakx;
  int nZ  = grids_->Nz;
  int nR  = nX * nY * nZ;
  int nK  = nXk * nYk * nZ;
  int nG  = nR * grids_->Nmoms * nS;

  theta_extended = nullptr;
  char strb[263];
  strcpy(strb, pars_->run_name); 
  strcat(strb, ".nc");

  int retval, idum;

  int nbx = min(grids_->NxNyNz, 1024);
  int ngx = 1 + (grids_->NxNyNz-1)/nbx;

  dBr = dim3(nbx, 1, 1);
  dGr = dim3(ngx, 1, 1);
  
  nbx = min(grids_->Nx, 512);
  ngx = 1 + (grids_->Nx-1)/nbx;

  dbx = dim3(nbx, 1, 1);
  dgx = dim3(ngx, 1, 1);
  
  nbx = min(grids_->Ny, 512);
  ngx = 1 + (grids_->Ny-1)/nbx;

  dbp = dim3(nbx, 1, 1);
  dgp = dim3(ngx, 1, 1);

  nbx = min(32, grids_->Ny);
  ngx = 1 + (grids_->Ny-1)/nbx;

  int nby = min(32, grids_->Nx);
  int ngy = 1 + (grids_->Nx-1)/nbx;
  
  dbxy = dim3(nbx, nby, 1);
  dgxy = dim3(ngx, ngy, 1);

  if (pars_->write_kmom || pars_->write_xymom) {
    int nbatch = grids_->Nz;
    grad_phi = new GradPerp(grids_, nbatch, grids_->NxNycNz);

    cudaMalloc     (&vEk,  sizeof(cuComplex)*grids_->NxNycNz);
  } 
  
  file = pars_->ncid;
  if (retval = nc_redef(file));
  
  int ri;
  // Get handles for the dimensions
  if (retval = nc_inq_dimid(file, "ri", &ri))  ERR(retval);
  
  if (retval = nc_def_dim(file, "kz",        grids_->Nz,    &nkz))       ERR(retval);  
  if (retval = nc_def_dim(file, "ky",        grids_->Naky,  &ky_dim))    ERR(retval);
  if (retval = nc_def_dim(file, "kx",        grids_->Nakx,  &kx_dim))    ERR(retval);
  if (retval = nc_def_dim(file, "theta",     grids_->Nz,    &nz))        ERR(retval);  
  
  if (retval = nc_inq_dimid(file, "x",       &x_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "y",       &y_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "m",       &m_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "l",       &l_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "s",       &s_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "time",    &time_dim)) ERR(retval);

  if (retval = nc_def_var(file, "periodic",       NC_INT, 0, 0, &periodic))        ERR(retval);
  if (retval = nc_def_var(file, "local_limit",    NC_INT, 0, 0, &local_limit))     ERR(retval);

  v_ky[0] = ky_dim;
  if (retval = nc_def_var(file, "ky",       NC_FLOAT, 1, v_ky, &ky))              ERR(retval);

  v_kx[0] = kx_dim;
  if (retval = nc_def_var(file, "kx",       NC_FLOAT, 1, v_kx, &kx))              ERR(retval);

  v_kz[0] = nkz;
  if (retval = nc_def_var(file, "kz",       NC_FLOAT, 1, v_kz, &kz))              ERR(retval);

  v_ky[0] = y_dim;
  if (retval = nc_def_var(file, "y",        NC_FLOAT, 1, v_ky, &y))               ERR(retval);  

  v_kx[0] = x_dim;
  if (retval = nc_def_var(file, "x",        NC_FLOAT, 1, v_kx, &x))               ERR(retval);  

  //  v_z[0] = nz;
  //  if (retval = nc_def_var(file, "z",        NC_FLOAT, 1, v_z, &z_h))              ERR(retval);
  
  // z_h needs to be defined.
  // Z0 would typically be q R
  // and then z_h would run from - (2 pi q R)/2 : + (2 pi q R)/2
  // but there are complications to get right:
  // normalization of R?
  // Allow for Z0 to be specified directly
  // Allow nperiod > 1 
  
  geo_v_theta[0] = nz;
  if (retval = nc_def_var(file, "theta",    NC_FLOAT, 1, geo_v_theta, &theta))    ERR(retval);
  if (retval = nc_def_var(file, "bmag",     NC_FLOAT, 1, geo_v_theta, &bmag))     ERR(retval);
  if (retval = nc_def_var(file, "bgrad",    NC_FLOAT, 1, geo_v_theta, &bgrad))    ERR(retval);
  if (retval = nc_def_var(file, "gbdrift",  NC_FLOAT, 1, geo_v_theta, &gbdrift))  ERR(retval);
  if (retval = nc_def_var(file, "gbdrift0", NC_FLOAT, 1, geo_v_theta, &gbdrift0)) ERR(retval);
  if (retval = nc_def_var(file, "cvdrift",  NC_FLOAT, 1, geo_v_theta, &cvdrift))  ERR(retval);
  if (retval = nc_def_var(file, "cvdrift0", NC_FLOAT, 1, geo_v_theta, &cvdrift0)) ERR(retval);
  if (retval = nc_def_var(file, "gds2",     NC_FLOAT, 1, geo_v_theta, &gds2))     ERR(retval);
  if (retval = nc_def_var(file, "gds21",    NC_FLOAT, 1, geo_v_theta, &gds21))    ERR(retval);
  if (retval = nc_def_var(file, "gds22",    NC_FLOAT, 1, geo_v_theta, &gds22))    ERR(retval);
  if (retval = nc_def_var(file, "grho",     NC_FLOAT, 1, geo_v_theta, &grho))     ERR(retval);
  if (retval = nc_def_var(file, "jacobian", NC_FLOAT, 1, geo_v_theta, &jacobian)) ERR(retval);
  
  ////////////////////////////
  //                        //
  //       TIME             //
  //                        //
  ////////////////////////////

  time = new nca(0); 
  time -> write_v_time = true;

  time -> time_dims[0] = time_dim;
  if (retval = nc_def_var(file, "time",     NC_DOUBLE, 1, time -> time_dims, &time -> time))    ERR(retval);
  
  ////////////////////////////
  //                        //
  //       DENSITY          //
  //                        //
  ////////////////////////////

  den = new nca(0);
  den -> write = pars_->write_moms;

  if (den -> write) {
    den -> dims[0] = s_dim;
    den -> dims[1] = nz;
    den -> dims[2] = kx_dim;
    den -> dims[3] = ky_dim;
    den -> dims[4] = ri;

    if (retval = nc_def_var(file, "density",  NC_FLOAT, 5, den->dims, &den->idx )) ERR(retval);
    
    den -> start[0] = 0;
    den -> start[1] = 0;
    den -> start[2] = 0;
    den -> start[3] = 0; 
    den -> start[4] = 0; 
    
    den -> count[0] = grids_->Nspecies;
    den -> count[1] = grids_->Nz;
    den -> count[2] = grids_->Nakx;
    den -> count[3] = grids_->Naky;
    den -> count[4] = 2;

    den -> ns = grids_->Nspecies;
  }
  
  ////////////////////////////
  //                        //
  //       DENSITY(t=0)     //
  //                        //
  ////////////////////////////

  den0 = new nca(0); 
  den0 -> write = pars_->write_moms;

  if (den0 -> write) {
    den0 -> dims[0] = s_dim;
    den0 -> dims[1] = nz;
    den0 -> dims[2] = kx_dim;
    den0 -> dims[3] = ky_dim;
    den0 -> dims[4] = ri;

    if (retval = nc_def_var(file, "density0",  NC_FLOAT, 5, den0 -> dims, &den0 -> idx )) ERR(retval);
    
    den0 -> start[0] = 0;
    den0 -> start[1] = 0;
    den0 -> start[2] = 0;
    den0 -> start[3] = 0; 
    den0 -> start[4] = 0; 
    
    den0 -> count[0] = grids_->Nspecies;
    den0 -> count[1] = grids_->Nz;
    den0 -> count[2] = grids_->Nakx;
    den0 -> count[3] = grids_->Naky;
    den0 -> count[4] = 2;

    den0 -> ns = grids_->Nspecies;
  }
  
  ////////////////////////////
  //                        //
  //       Phi              //
  //                        //
  ////////////////////////////

  wphi = new nca(0); 
  wphi -> write = pars_->write_phi;

  if (wphi -> write) {
    wphi -> dims[0] = nz;
    wphi -> dims[1] = kx_dim;
    wphi -> dims[2] = ky_dim;
    wphi -> dims[3] = ri;
    
    if (retval = nc_def_var(file, "phi",      NC_FLOAT, 4, wphi -> dims, &wphi -> idx ))      ERR(retval);

    wphi -> start[0] = 0;
    wphi -> start[1] = 0;
    wphi -> start[2] = 0;
    wphi -> start[3] = 0; 
    
    wphi -> count[0] = grids_->Nz;
    wphi -> count[1] = grids_->Nakx;
    wphi -> count[2] = grids_->Naky;
    wphi -> count[3] = 2;

    wphi -> ns = 1; 
  }

  ////////////////////////////
  //                        //
  //       Phi(t=0)         //
  //                        //
  ////////////////////////////

  wphi0 = new nca(0); 
  wphi0 -> write = pars_->write_phi;

  if (wphi0 -> write) {
    wphi0 -> dims[0] = nz;
    wphi0 -> dims[1] = kx_dim;
    wphi0 -> dims[2] = ky_dim;
    wphi0 -> dims[3] = ri;
    
    if (retval = nc_def_var(file, "phi0",      NC_FLOAT, 4, wphi0 -> dims, &wphi0 -> idx ))      ERR(retval);

    wphi0 -> start[0] = 0;
    wphi0 -> start[1] = 0;
    wphi0 -> start[2] = 0;
    wphi0 -> start[3] = 0; 
    
    wphi0 -> count[0] = grids_->Nz;
    wphi0 -> count[1] = grids_->Nakx;
    wphi0 -> count[2] = grids_->Naky;
    wphi0 -> count[3] = 2;

    wphi0 -> ns = 1;
  }

  ////////////////////////////
  //                        //
  //   DENSITY(kpar)        //
  //                        //
  ////////////////////////////

  denk = new nca(0); 
  denk -> write = (pars_->write_phi_kpar and pars_->write_moms);

  if (denk -> write) {
    denk -> dims[0] = s_dim;
    denk -> dims[1] = nkz;
    denk -> dims[2] = kx_dim;
    denk -> dims[3] = ky_dim;
    denk -> dims[4] = ri;

    if (retval = nc_def_var(file, "density_kpar", NC_FLOAT, 5, denk -> dims, &denk -> idx)) ERR(retval);    

    denk -> start[0] = 0;
    denk -> start[1] = 0;
    denk -> start[2] = 0;
    denk -> start[3] = 0; 
    denk -> start[4] = 0; 
    
    denk -> count[0] = grids_->Nspecies;
    denk -> count[1] = grids_->Nz;
    denk -> count[2] = grids_->Nakx;
    denk -> count[3] = grids_->Naky;
    denk -> count[4] = 2;

    denk -> ns = 1;
  }

  ////////////////////////////
  //                        //
  //   Phi(kpar)            //
  //                        //
  ////////////////////////////

  wphik = new nca(0); 
  wphik -> write = pars_->write_phi_kpar;

  if (wphik -> write) {
    wphik -> dims[0] = nkz;
    wphik -> dims[1] = kx_dim;
    wphik -> dims[2] = ky_dim;
    wphik -> dims[3] = ri;
    
    if (retval = nc_def_var(file, "phi2_kz",    NC_FLOAT, 4, wphik -> dims, &wphik -> idx))     ERR(retval);

    wphik -> start[0] = 0;
    wphik -> start[1] = 0;
    wphik -> start[2] = 0;
    wphik -> start[3] = 0; 
    
    wphik -> count[0] = grids_->Nz;
    wphik -> count[1] = grids_->Nakx;
    wphik -> count[2] = grids_->Naky;
    wphik -> count[3] = 2;

    wphik -> ns = 1;
  }

  ////////////////////////////
  //                        //
  //   Frequencies          //
  //                        //
  ////////////////////////////

  omg = new nca(-nX * nY, 2 * nXk * nYk);
  omg -> write_v_time = pars_->write_omega;
  
  if (omg -> write_v_time) {
    omg -> time_dims[0] = time_dim; 
    omg -> time_dims[1] = ky_dim; 
    omg -> time_dims[2] = kx_dim;
    omg -> time_dims[3] = ri;
    
    if (retval = nc_def_var(file, "omega_v_time", NC_FLOAT, 4, omg -> time_dims, &omg -> time)) ERR(retval);

    omg -> time_start[0] = 1;
    
    omg -> time_count[1] = grids_->Naky;
    omg -> time_count[2] = grids_->Nakx;
    omg -> time_count[3] = 2;

    for (int i=0; i < nXk * nYk * 2; i++) omg->cpu[i] = 0.;
  }

  ////////////////////////////
  //                        //
  // Rosenbluth-Hinton      //
  //                        //
  ////////////////////////////

  rh = new nca(0); 
  rh -> write = pars_->write_rh;

  if (rh -> write) {
    rh -> time_dims[0] = time_dim;
    rh -> time_dims[1] = ri;
    
    if (retval = nc_def_var(file, "phi_rh", NC_FLOAT, 2, rh -> time_dims, &rh -> time)) ERR(retval);

    rh -> time_count[1] = 2;
  }

  ////////////////////////////
  //                        //
  //     PZT estimates      //
  //                        //
  ////////////////////////////

  Pzt = new nca(0);
  pZt = new nca(0);
  pzT = new nca(0);
  Pzt -> write_v_time = pars_->write_pzt;
  
  if (Pzt -> write_v_time) {

    Pzt -> time_dims[0] = time_dim;
    pZt -> time_dims[0] = time_dim;
    pzT -> time_dims[0] = time_dim;

    if (retval = nc_def_var(file, "prim", NC_FLOAT, 1, Pzt -> time_dims, &Pzt -> idx)) ERR(retval);
    if (retval = nc_def_var(file, "sec",  NC_FLOAT, 1, pZt -> time_dims, &pZt -> idx)) ERR(retval);
    if (retval = nc_def_var(file, "tert", NC_FLOAT, 1, pzT -> time_dims, &pzT -> idx)) ERR(retval);

    cudaMallocHost (&primary,   sizeof(float));    primary[0] = 0.;  
    cudaMallocHost (&secondary, sizeof(float));    secondary[0] = 0.;
    cudaMallocHost (&tertiary,  sizeof(float));    tertiary[0] = 0.;
    cudaMalloc     (&t_bar,     sizeof(cuComplex) * nR * nS);
  }

  ////////////////////////////
  //                        //
  // (1-G0)phi**2 (species)  //
  //                        //
  ////////////////////////////

  Ps = new nca(nS); 
  if (pars_->pspectra[PSPECTRA_species] > 0) Ps -> write_v_time = true;

  if (Ps -> write_v_time) {
    Ps -> time_dims[0] = time_dim;
    Ps -> time_dims[1] = s_dim;
    
    if (retval = nc_def_var(file, "Pst", NC_FLOAT, 2, Ps -> time_dims, &Ps -> time))  ERR(retval);
    
    Ps -> time_count[1] = grids_->Nspecies;
  }
  
  ////////////////////////////
  //                        //
  //   P (kx, species)      //
  //                        //
  ////////////////////////////

  Pkx = new nca(nX*nS, nXk*nS); 
  if (pars_->pspectra[PSPECTRA_kx] > 0) Pkx -> write_v_time = true;

  if (Pkx -> write_v_time) {
    Pkx -> time_dims[0] = time_dim;
    Pkx -> time_dims[1] = s_dim;
    Pkx -> time_dims[2] = kx_dim;
    
    if (retval = nc_def_var(file, "Pkxst", NC_FLOAT, 3, Pkx -> time_dims, &Pkx -> time))  ERR(retval);

    Pkx -> time_count[1] = grids_->Nspecies;
    Pkx -> time_count[2] = grids_->Nakx;      
  }
  
  ////////////////////////////
  //                        //
  //   P (ky, species)      //
  //                        //
  ////////////////////////////

  Pky = new nca(nY*nS, nYk*nS);
  if (pars_->pspectra[PSPECTRA_ky] > 0) Pky -> write_v_time = true;

  if (Pky -> write_v_time) {
    Pky -> time_dims[0] = time_dim;
    Pky -> time_dims[1] = s_dim;
    Pky -> time_dims[2] = ky_dim;
    
    if (retval = nc_def_var(file, "Pkyst", NC_FLOAT, 3, Pky->time_dims, &Pky->time))  ERR(retval);

    Pky -> time_count[1] = grids_->Nspecies;
    Pky -> time_count[2] = grids_->Naky;
    
  }
  
  ////////////////////////////
  //                        //
  //   P (kz, species)      //
  //                        //
  ////////////////////////////

  Pkz = new nca(nZ*nS, nZ*nS); 
  if (pars_->pspectra[PSPECTRA_kz] > 0) Pkz -> write_v_time = true;

  if (Pkz -> write_v_time) {
    Pkz -> time_dims[0] = time_dim;
    Pkz -> time_dims[1] = s_dim;
    Pkz -> time_dims[2] = nkz;
    
    if (retval = nc_def_var(file, "Pkzst", NC_FLOAT, 3, Pkz -> time_dims, &Pkz -> time))  ERR(retval);

    Pkz -> time_count[1] = grids_->Nspecies;
    Pkz -> time_count[2] = grids_->Nz;
  }
  
  ////////////////////////////
  //                        //
  //   P (z, species)       //
  //                        //
  ////////////////////////////

  Pz = new nca(nZ*nS); 
  if (pars_->pspectra[PSPECTRA_z] > 0) Pz -> write_v_time = true;

  if (Pz -> write_v_time) {
    Pz -> time_dims[0] = time_dim;
    Pz -> time_dims[1] = s_dim;
    Pz -> time_dims[2] = nz;
    
    if (retval = nc_def_var(file, "Pzst", NC_FLOAT, 3, Pz -> time_dims, &Pz -> time))  ERR(retval);
    
    Pz -> time_count[1] = grids_->Nspecies;
    Pz -> time_count[2] = grids_->Nz;
  }
  
  ////////////////////////////
  //                        //
  //   P (kx,ky,  species)  //
  //                        //
  ////////////////////////////

  Pkxky = new nca(nX * nY * nS, nXk * nYk * nS); 

  if (pars_->pspectra[PSPECTRA_kxky] > 0) Pkxky -> write_v_time = true;

  if (Pkxky -> write_v_time) {
    Pkxky -> time_dims[0] = time_dim;
    Pkxky -> time_dims[1] = s_dim;
    Pkxky -> time_dims[2] = ky_dim;
    Pkxky -> time_dims[3] = kx_dim;
    
    if (retval = nc_def_var(file, "Pkxkyst", NC_FLOAT, 4, Pkxky -> time_dims, &Pkxky -> time))  ERR(retval);
    
    Pkxky -> time_count[1] = grids_->Nspecies;
    Pkxky -> time_count[2] = grids_->Naky;
    Pkxky -> time_count[3] = grids_->Nakx;
  }   

  ////////////////////////////
  //                        //
  //   W (species)          //
  //                        //
  ////////////////////////////

  Ws = new nca(nS); 
  if (pars_->wspectra[WSPECTRA_species] > 0) Ws -> write_v_time = true;

  if (Ws -> write_v_time) {
    Ws -> time_dims[0] = time_dim;
    Ws -> time_dims[1] = s_dim;
    
    if (retval = nc_def_var(file, "Wst", NC_FLOAT, 2, Ws -> time_dims, &Ws -> time))  ERR(retval);
    
    Ws -> time_count[1] = grids_->Nspecies;
  }

  ////////////////////////////
  //                        //
  //   W (kx, species)      //
  //                        //
  ////////////////////////////

  Wkx = new nca(nX*nS, nXk*nS); 
  if (pars_->wspectra[WSPECTRA_kx] > 0) Wkx -> write_v_time = true;

  if (Wkx -> write_v_time) {
    Wkx -> time_dims[0] = time_dim;
    Wkx -> time_dims[1] = s_dim;
    Wkx -> time_dims[2] = kx_dim;
    
    if (retval = nc_def_var(file, "Wkxst", NC_FLOAT, 3, Wkx -> time_dims, &Wkx -> time))  ERR(retval);
    
    Wkx -> time_count[1] = grids_->Nspecies;
    Wkx -> time_count[2] = grids_->Nakx;      
  }
  
  ////////////////////////////
  //                        //
  //   W (ky, species)      //
  //                        //
  ////////////////////////////

  Wky = new nca(nY*nS, nYk*nS); 
  if (pars_->wspectra[WSPECTRA_ky] > 0) Wky -> write_v_time = true;

  if (Wky -> write_v_time) {
    Wky -> time_dims[0] = time_dim;
    Wky -> time_dims[1] = s_dim;
    Wky -> time_dims[2] = ky_dim;
    
    if (retval = nc_def_var(file, "Wkyst", NC_FLOAT, 3, Wky -> time_dims, &Wky -> time))  ERR(retval);
    
    Wky -> time_count[1] = grids_->Nspecies;
    Wky -> time_count[2] = grids_->Naky;      
  }   
  
  ////////////////////////////
  //                        //
  //   W (kz, species)      //
  //                        //
  ////////////////////////////

  Wkz = new nca(nZ * nS, nZ * nS); 
  if (pars_->wspectra[WSPECTRA_kz] > 0) Wkz -> write_v_time = true;

  if (Wkz -> write_v_time) {
    Wkz -> time_dims[0] = time_dim;
    Wkz -> time_dims[1] = s_dim;
    Wkz -> time_dims[2] = nkz;
    
    if (retval = nc_def_var(file, "Wkzst", NC_FLOAT, 3, Wkz -> time_dims, &Wkz -> time))  ERR(retval);
    
    Wkz -> time_count[1] = grids_->Nspecies;
    Wkz -> time_count[2] = grids_->Nz;
  }   

  ////////////////////////////
  //                        //
  //   W (z, species)       //
  //                        //
  ////////////////////////////

  Wz = new nca(nZ*nS); 
  if (pars_->wspectra[WSPECTRA_z] > 0) Wz -> write_v_time = true;

  if (Wz -> write_v_time) {
    Wz -> time_dims[0] = time_dim;
    Wz -> time_dims[1] = s_dim;
    Wz -> time_dims[2] = nz;
    
    if (retval = nc_def_var(file, "Wzst", NC_FLOAT, 3, Wz -> time_dims, &Wz -> time))  ERR(retval);
    
    Wz -> time_count[1] = grids_->Nspecies;
    Wz -> time_count[2] = grids_->Nz;
  }   
  
  ////////////////////////////
  //                        //
  //   W (kx,ky,  species)  //
  //                        //
  ////////////////////////////

  Wkxky = new nca(nX * nY * nS, nXk * nYk * nS); 
  if (pars_->wspectra[WSPECTRA_kxky] > 0) Wkxky -> write_v_time = true;

  if (Wkxky -> write_v_time) {
    Wkxky -> time_dims[0] = time_dim;
    Wkxky -> time_dims[1] = s_dim;
    Wkxky -> time_dims[2] = ky_dim;
    Wkxky -> time_dims[3] = kx_dim;
    
    if (retval = nc_def_var(file, "Wkxkyst", NC_FLOAT, 4, Wkxky -> time_dims, &Wkxky -> time))  ERR(retval);
    
    Wkxky -> time_count[1] = grids_->Nspecies;
    Wkxky -> time_count[2] = grids_->Naky;      
    Wkxky -> time_count[3] = grids_->Nakx;
  }   

  ////////////////////////////
  //                        //
  // W (adiabatic species)  //
  //                        //
  ////////////////////////////

  As = new nca(1); 
  As -> write_v_time = (pars_->aspectra[ASPECTRA_species] > 0);

  if (As -> write_v_time) {
    As -> time_dims[0] = time_dim;
    
    if (retval = nc_def_var(file, "At", NC_FLOAT, 1, As -> time_dims, &As -> time))  ERR(retval);

  }  

  ////////////////////////////
  //                        //
  //   W (kx) adiabatic     //
  //                        //
  ////////////////////////////

  Akx = new nca(nX, nXk); 
  if (pars_->aspectra[ASPECTRA_kx] > 0) Akx -> write_v_time = true;

  if (Akx -> write_v_time) {
    Akx -> time_dims[0] = time_dim;
    Akx -> time_dims[1] = kx_dim;
    
    if (retval = nc_def_var(file, "Akxst", NC_FLOAT, 2, Akx -> time_dims, &Akx -> time))  ERR(retval);
    
    Akx -> time_count[1] = grids_->Nakx;      
  }   
  
  ////////////////////////////
  //                        //
  //   W (ky) adiabatic     //
  //                        //
  ////////////////////////////

  Aky = new nca(nY, nYk); 
  if (pars_->aspectra[ASPECTRA_ky] > 0) Aky -> write_v_time = true;

  if (Aky -> write_v_time) {
    Aky -> time_dims[0] = time_dim;
    Aky -> time_dims[1] = ky_dim;
    
    if (retval = nc_def_var(file, "Akyst", NC_FLOAT, 2, Aky -> time_dims, &Aky -> time))  ERR(retval);

    Aky -> time_count[1] = grids_->Naky;      
  }   
  
  ////////////////////////////
  //                        //
  //   A (kz)  adiabatic    //
  //                        //
  ////////////////////////////

  Akz = new nca(nZ, nZ); 
  if (pars_->aspectra[ASPECTRA_kz] > 0) Akz -> write_v_time = true;

  if (Akz -> write_v_time) {
    Akz -> time_dims[0] = time_dim;
    Akz -> time_dims[1] = nkz;
    
    if (retval = nc_def_var(file, "Akzst", NC_FLOAT, 2, Akz -> time_dims, &Akz -> time))  ERR(retval);
    
    Akz -> time_count[1] = grids_->Nz;      
  }   
  
  ////////////////////////////
  //                        //
  //   A (z)  adiabatic     //
  //                        //
  ////////////////////////////

  Az = new nca(nZ); 
  if (pars_->aspectra[ASPECTRA_z] > 0) Az -> write_v_time = true;

  if (Az -> write_v_time) {
    Az -> time_dims[0] = time_dim;
    Az -> time_dims[1] = nz;
    
    if (retval = nc_def_var(file, "Azst", NC_FLOAT, 2, Az -> time_dims, &Az -> time))  ERR(retval);
    
    Az -> time_count[1] = grids_->Nz;      
  }   
  
  ////////////////////////////
  //                        //
  //   W (kx,ky) adiabatic  //
  //                        //
  ////////////////////////////

  Akxky = new nca(nX * nY, nXk * nYk); 
  if (pars_->aspectra[ASPECTRA_kxky] > 0) Akxky -> write_v_time = true;

  if (Akxky -> write_v_time) {
    Akxky -> time_dims[0] = time_dim;
    Akxky -> time_dims[1] = ky_dim;
    Akxky -> time_dims[2] = kx_dim;
    
    if (retval = nc_def_var(file, "Akxkyst", NC_FLOAT, 3, Akxky -> time_dims, &Akxky -> time))  ERR(retval);
    
    Akxky -> time_count[1] = grids_->Naky;      
    Akxky -> time_count[2] = grids_->Nakx;
  }   
  
  ////////////////////////////
  //                        //
  // Lag-Herm spectrum      //
  //                        //
  ////////////////////////////

  Wlm = new nca(nL*nM*nS); 
  if (pars_->wspectra[WSPECTRA_lm] > 0) Wlm -> write_v_time = true;
  
  if (Wlm -> write_v_time) {
    Wlm -> time_dims[0] = time_dim;
    Wlm -> time_dims[1] = s_dim;
    Wlm -> time_dims[2] = m_dim;
    Wlm -> time_dims[3] = l_dim;
    
    if (retval = nc_def_var(file, "Wlmst", NC_FLOAT, 4, Wlm -> time_dims, &Wlm -> time))  ERR(retval);
    
    Wlm -> time_count[1] = grids_->Nspecies;
    Wlm -> time_count[2] = grids_->Nm;
    Wlm -> time_count[3] = grids_->Nl;      
  }
    
  ////////////////////////////
  //                        //
  // Laguerre spectrum      //
  //                        //
  ////////////////////////////

  Wl = new nca(nL*nS); 
  if (pars_->wspectra[WSPECTRA_l] > 0) Wl -> write_v_time = true;

  if (Wl -> write_v_time) {
    Wl -> time_dims[0] = time_dim;
    Wl -> time_dims[1] = s_dim;
    Wl -> time_dims[2] = l_dim;
    
    if (retval = nc_def_var(file, "Wlst", NC_FLOAT, 3, Wl -> time_dims, &Wl -> time))  ERR(retval);
    
    Wl -> time_count[1] = grids_->Nspecies;
    Wl -> time_count[2] = grids_->Nl;
  }
  
  ////////////////////////////
  //                        //
  //  Hermite spectrum      //
  //                        //
  ////////////////////////////

  Wm = new nca(nM*nS); 
  if (pars_->wspectra[WSPECTRA_m] > 0) Wm -> write_v_time = true;
  
  if (Wm -> write_v_time) {
    Wm -> time_dims[0] = time_dim;
    Wm -> time_dims[1] = s_dim;
    Wm -> time_dims[2] = m_dim;
    
    if (retval = nc_def_var(file, "Wmst", NC_FLOAT, 3, Wm -> time_dims, &Wm -> time))  ERR(retval);
    
    Wm -> time_count[1] = grids_->Nspecies;    
    Wm -> time_count[2] = grids_->Nm;          
  }

  bool linked = (not pars_->local_limit && not pars_->boundary_option_periodic);

  /*
  if (linked && false) {
    zkxky[0] = nz;
    zkxky[1] = kx_dim; 
    zkxky[2] = ky_dim;
    
    if (retval = nc_def_var(file, "theta_x",  NC_FLOAT, 3, zkxky, &theta_x))  ERR(retval);
  }
  */
  
  ////////////////////////////
  //                        //
  //  <v_ExB>_y,z (x)       // 
  //                        //
  ////////////////////////////

  vE = new nca(grids_->NxNyNz, grids_->Nx);
  if (pars_->write_vE) vE -> write_v_time = true;

  if (vE -> write_v_time) {
    vE -> time_dims[0] = time_dim;
    vE -> time_dims[1] = x_dim;
    
    if (retval = nc_def_var(file, "vE_xt", NC_FLOAT, 2, vE->time_dims, &vE->time))  ERR(retval);
    
    vE -> time_count[1] = grids_->Nx;
  }

  vE2 = new nca(grids_->NxNyNz, grids_->Nx); 
  if (pars_->write_vE2) vE2 -> write_v_time = true;

  if (vE2 -> write_v_time) {
    vE2 -> time_dims[0] = time_dim;
    
    if (retval = nc_def_var(file, "vE2_t", NC_FLOAT, 1, vE2->time_dims, &vE2->time))  ERR(retval);
  }

  ////////////////////////////
  //                        //
  //  <d/dx v_ExB>_y,z (x)  // 
  //                        //
  ////////////////////////////

  kvE = new nca(grids_->NxNyNz, grids_->Nx);
  if (pars_->write_kvE) kvE -> write_v_time = true;

  if (kvE -> write_v_time) {
    kvE -> time_dims[0] = time_dim;
    kvE -> time_dims[1] = x_dim;
    
    if (retval = nc_def_var(file, "kvE_xt", NC_FLOAT, 2, kvE -> time_dims, &kvE -> time))  ERR(retval);
    
    kvE -> time_count[1] = grids_->Nx;      
  }
  
  ////////////////////////////
  //                        //
  // <d/dx denh>_y,z (x)    // 
  //                        //
  ////////////////////////////

  kden = new nca(grids_->NxNyNz, grids_->Nx);
  if (pars_->write_kden) kden -> write_v_time = true;

  if (kden -> write_v_time) {
    kden -> time_dims[0] = time_dim;
    kden -> time_dims[1] = x_dim;
    
    if (retval = nc_def_var(file, "kden_xt", NC_FLOAT, 2, kden->time_dims, &kden->time))  ERR(retval);
    
    kden -> time_count[1] = grids_->Nx;      
  }

  ////////////////////////////
  //                        //
  // <d/dx uparh>_y,z (x)   // 
  //                        //
  ////////////////////////////

  kUpar = new nca(grids_->NxNyNz, grids_->Nx);
  if (pars_->write_kUpar) kUpar -> write_v_time = true;

  if (kUpar -> write_v_time) {
    kUpar -> time_dims[0] = time_dim;
    kUpar -> time_dims[1] = x_dim;
    
    if (retval = nc_def_var(file, "kUpar_xt", NC_FLOAT, 2, kUpar->time_dims, &kUpar->time))  ERR(retval);
    
    kUpar->time_count[1] = grids_->Nx;      
  }
  
  ////////////////////////////
  //                        //
  // <d/dx Tparh>_y,z (x)   // 
  //                        //
  ////////////////////////////

  kTpar = new nca(grids_->NxNyNz, grids_->Nx);
  if (pars_->write_kTpar) kTpar->write_v_time = true;

  if (kTpar -> write_v_time) {
    kTpar -> time_dims[0] = time_dim;
    kTpar -> time_dims[1] = x_dim;
    
    if (retval = nc_def_var(file, "kTpar_xt", NC_FLOAT, 2, kTpar->time_dims, &kTpar->time))  ERR(retval);
    
    kTpar -> time_count[1] = grids_->Nx;      
  }
  
  ////////////////////////////
  //                        //
  // <d/dx Tperph>_y,z (x)  // 
  //                        //
  ////////////////////////////

  kTperp = new nca(grids_->NxNyNz, grids_->Nx);
  if (pars_->write_kTperp) kTperp -> write_v_time = true;

  if (kTperp -> write_v_time) {
    kTperp -> time_dims[0] = time_dim;
    kTperp -> time_dims[1] = x_dim;
    
    if (retval = nc_def_var(file, "kTperp_xt", NC_FLOAT, 2, kTperp->time_dims, &kTperp->time))  ERR(retval);
    
    kTperp -> time_count[1] = grids_->Nx;      
  }
  
  ////////////////////////////
  //                        //
  // <d/dx qparh>_y,z (x)   // 
  //                        //
  ////////////////////////////

  kqpar = new nca(grids_->NxNyNz, grids_->Nx);
  if (pars_->write_kqpar) kqpar -> write_v_time = true;

  if (kqpar -> write_v_time) {
    kqpar -> time_dims[0] = time_dim;
    kqpar -> time_dims[1] = x_dim;
    
    if (retval = nc_def_var(file, "kqpar_xt", NC_FLOAT, 2, kqpar -> time_dims, &kqpar->time))  ERR(retval);
    
    kqpar -> time_count[1] = grids_->Nx;      
  }

  ////////////////////////////
  //                        //
  // <v_ExB>_z (x, y)       // 
  //                        //
  ////////////////////////////

  xyvE = new nca(grids_->NxNyNz, grids_->NxNy);
  if (pars_->write_xyvE) xyvE->write_v_time = true;
  
  if (xyvE -> write_v_time) {
    xyvE -> time_dims[0] = time_dim;
    xyvE -> time_dims[1] = y_dim;  // Transpose to accommodate ncview
    xyvE -> time_dims[2] = x_dim;
    
    if (retval = nc_def_var(file, "vE_xyt", NC_FLOAT, 3, xyvE -> time_dims, &xyvE->time)) ERR(retval);
    
    xyvE -> time_count[1] = grids_->Ny;      
    xyvE -> time_count[2] = grids_->Nx;          

    xyvE -> xydata = true;
  }
  
  ////////////////////////////
  //                        //
  // <d/dx v_ExB>_z (x, y)  // 
  //                        //
  ////////////////////////////

  xykvE = new nca(grids_->NxNyNz, grids_->NxNy);
  if (pars_ -> write_xykvE) xykvE -> write_v_time = true;
  
  if (xykvE -> write_v_time) {
    xykvE -> time_dims[0] = time_dim;
    xykvE -> time_dims[1] = y_dim;  // Transpose to accommodate ncview
    xykvE -> time_dims[2] = x_dim;
    
    if (retval = nc_def_var(file, "kvE_xyt", NC_FLOAT, 3, xykvE -> time_dims, &xykvE->time)) ERR(retval);
    
    xykvE -> time_count[1] = grids_->Ny;      
    xykvE -> time_count[2] = grids_->Nx;

    xykvE -> xydata = true;
  }
  
  ////////////////////////////
  //                        //
  // <denh>_z (x, y)        // 
  //                        //
  ////////////////////////////

  xyden = new nca(grids_->NxNyNz, grids_->NxNy);
  if (pars_->write_xyden) xyden->write_v_time = true;
  
  if (xyden -> write_v_time) {
    xyden -> time_dims[0] = time_dim;
    xyden -> time_dims[1] = y_dim;  // Transpose to accommodate ncview
    xyden -> time_dims[2] = x_dim;
    
    if (retval = nc_def_var(file, "den_xyt", NC_FLOAT, 3, xyden -> time_dims, &xyden->time)) ERR(retval);
    
    xyden -> time_count[1] = grids_->Ny;      
    xyden -> time_count[2] = grids_->Nx;

    xyden -> xydata = true;
  }
  
  ////////////////////////////
  //                        //
  // <Uparh>_z (x, y)       // 
  //                        //
  ////////////////////////////

  xyUpar = new nca(grids_->NxNyNz, grids_->NxNy);
  if (pars_->write_xyUpar) xyUpar->write_v_time = true;
  
  if (xyUpar -> write_v_time) {
    xyUpar -> time_dims[0] = time_dim;
    xyUpar -> time_dims[1] = y_dim;  // Transpose to accommodate ncview
    xyUpar -> time_dims[2] = x_dim;
    
    if (retval = nc_def_var(file, "upar_xyt", NC_FLOAT, 3, xyUpar -> time_dims, &xyUpar->time)) ERR(retval);
    
    xyUpar -> time_count[1] = grids_->Ny;      
    xyUpar -> time_count[2] = grids_->Nx;

    xyUpar -> xydata = true;
  }  
  
  ////////////////////////////
  //                        //
  // <Tparh>_z (x, y)       // 
  //                        //
  ////////////////////////////

  xyTpar = new nca(grids_->NxNyNz, grids_->NxNy);
  if (pars_->write_xyTpar) xyTpar->write_v_time = true;
  
  if (xyTpar -> write_v_time) {
    xyTpar -> time_dims[0] = time_dim;
    xyTpar -> time_dims[1] = y_dim;  // Transpose to accommodate ncview
    xyTpar -> time_dims[2] = x_dim;
    
    if (retval = nc_def_var(file, "Tpar_xyt", NC_FLOAT, 3, xyTpar -> time_dims, &xyTpar->time)) ERR(retval);
    
    xyTpar -> time_count[1] = grids_->Ny;      
    xyTpar -> time_count[2] = grids_->Nx;

    xyTpar -> xydata = true;
  }
  
  ////////////////////////////
  //                        //
  // <Tperph>_z (x, y)      // 
  //                        //
  ////////////////////////////

  xyTperp = new nca(grids_->NxNyNz, grids_->NxNy);
  if (pars_->write_xyTperp) xyTperp -> write_v_time = true;

  if (xyTperp -> write_v_time) {
    xyTperp -> time_dims[0] = time_dim;
    xyTperp -> time_dims[1] = y_dim;  // Transpose to accommodate ncview
    xyTperp -> time_dims[2] = x_dim;
    
    if (retval = nc_def_var(file, "Tperp_xyt", NC_FLOAT, 3, xyTperp -> time_dims, &xyTperp->time))  ERR(retval);
    
    xyTperp -> time_count[1] = grids_->Ny;      
    xyTperp -> time_count[2] = grids_->Nx;

    xyTperp -> xydata = true;
  }

  ////////////////////////////
  //                        //
  // <qparh>_z (x, y)       // 
  //                        //
  ////////////////////////////

  xyqpar = new nca(grids_->NxNyNz, grids_->NxNy);
  if (pars_->write_xyqpar) xyqpar->write_v_time = true;
  
  if (xyqpar -> write_v_time) {
    xyqpar -> time_dims[0] = time_dim;
    xyqpar -> time_dims[1] = y_dim;  // Transpose to accommodate ncview
    xyqpar -> time_dims[2] = x_dim;
    
    if (retval = nc_def_var(file, "qpar_xyt", NC_FLOAT, 3, xyqpar -> time_dims, &xyqpar->time)) ERR(retval);
    
    xyqpar -> time_count[1] = grids_->Ny;      
    xyqpar -> time_count[2] = grids_->Nx;

    xyqpar -> xydata = true;
  }

  ////////////////////////////
  //                        //
  //   g(y) for K-S eqn     // 
  //                        //
  ////////////////////////////

  g_y = new nca(grids_->Ny); 
  if (pars_->ks && pars_->write_ks)  g_y -> write_v_time = true;

  if (g_y -> write_v_time) {
    g_y -> time_dims[0] = time_dim;
    g_y -> time_dims[1] = y_dim;
    
    if (retval = nc_def_var(file, "g_yt", NC_FLOAT, 2, g_y -> time_dims, &g_y -> time))  ERR(retval);

    g_y -> time_count[1] = grids_->Ny;

    int nbatch = 1;
    grad_perp = new GradPerp(grids_, nbatch, grids_->Nyc);    
  }   
  
  ////////////////////////////
  //                        //
  //   Free energy          //
  //                        //
  ////////////////////////////

  Wtot = new nca(0); 
  Wtot -> write_v_time = pars_->write_free_energy;
  
  if (Wtot -> write_v_time) {
    Wtot -> time_dims[0] = time_dim;

    if (retval = nc_def_var(file, "W", NC_FLOAT, 1, Wtot -> time_dims, &Wtot -> time)) ERR(retval);

    totW = 0.;
  }

  ////////////////////////////
  //                        //
  //    Heat fluxes         //
  //                        //
  ////////////////////////////
  qs = new nca(nS); 
  qs -> write_v_time = pars_->write_fluxes;
  
  if (qs -> write_v_time) {
    qs -> time_dims[0] = time_dim;
    qs -> time_dims[1] = s_dim;
    
    if (retval = nc_def_var(file, "qflux", NC_FLOAT, 2, qs -> time_dims, &qs -> time)) ERR(retval);

    qs -> time_count[1] = grids_->Nspecies;

    all_red = new Species_Reduce(nR, nS);  cudaDeviceSynchronize();  CUDA_DEBUG("Reductions: %s \n");
  }

  DEBUGPRINT("ncdf:  ending definition mode for NetCDF \n");
  
  if (retval = nc_enddef(file)) ERR(retval);
  
  ///////////////////////////////////
  //                               //
  //        x                      //
  //                               //
  ///////////////////////////////////
  x_start[0] = 0;
  x_count[0] = grids_->Nx;

  if (retval = nc_put_vara(file, x, x_start, x_count, grids_->x_h))         ERR(retval);

  ///////////////////////////////////
  //                               //
  //        y                      //
  //                               //
  ///////////////////////////////////
  y_start[0] = 0;
  y_count[0] = grids_->Ny;

  if (retval = nc_put_vara(file, y, y_start, y_count, grids_->y_h))         ERR(retval);

  ///////////////////////////////////
  //                               //
  //         z                     //
  //                               //
  ///////////////////////////////////
  z_start[0] = 0;
  z_count[0] = grids_->Nz;

  //  if (retval = nc_put_vara(file, z, z_start, z_count, z_h))      ERR(retval);

  ///////////////////////////////////
  //                               //
  //        kz                     //
  //                               //
  ///////////////////////////////////
  kz_start[0] = 0;
  kz_count[0] = grids_->Nz;

  for (int i=0; i<grids_->Nz; i++) grids_->kpar_outh[i] = geo_->gradpar*grids_->kz_outh[i];
  if (retval = nc_put_vara(file, kz, kz_start, kz_count, grids_->kpar_outh))      ERR(retval);

  ///////////////////////////////////
  //                               //
  //        ky                     //
  //                               //
  ///////////////////////////////////
  ky_start[0] = 0;
  ky_count[0] = grids_->Naky;

  if (retval = nc_put_vara(file, ky, ky_start, ky_count, grids_->ky_h))         ERR(retval);
  
  ///////////////////////////////////
  //                               //
  //        kx                     //
  //                               //
  ///////////////////////////////////
  kx_start[0] = 0;
  kx_count[0] = grids_->Nakx;

  if (retval = nc_put_vara(file, kx, kx_start, kx_count, grids_->kx_outh))      ERR(retval);

  ///////////////////////////////////
  //                               //
  //  geometric information        //
  //                               //
  ///////////////////////////////////
  geo_start[0] = 0;
  geo_count[0] = grids_->Nz;
  
  if (retval = nc_put_vara(file, theta,    geo_start, geo_count, geo_->z_h))         ERR(retval);

  if (linked && false) {
    
    int Nx = grids_->Nx;
    int Ny = grids_->Ny;
    int Nz = grids_->Nz;
    
    int Naky = grids_->Naky;
    
    zkxky_count[0] = Nz;
    zkxky_count[1] = 1;
    zkxky_count[2] = 1;
    
    size_t size = sizeof(float)*Nz;
    cudaMallocHost((void**) &theta_extended, size);
    float th0;
    for (int i=0; i<(Nx-1)/3+1; i++) {
      for (int j=0; j<(Ny-1)/3+1; j++) {
	if (j==0) {th0 = 0.;} else {th0 = grids_->kx_h[i]/(grids_->ky_h[j]*pars_->shat);}
	for (int k=0; k<Nz; k++) {
	  theta_extended[k] = geo_->z_h[k] - th0;
	}
	zkxky_start[0] = 0;
	zkxky_start[1] = i;
	zkxky_start[2] = j;
	if (retval = nc_put_vara(file, theta_x, zkxky_start, zkxky_count, theta_extended)) ERR(retval);
      }
    }
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<Naky; j++) {
	if (j==0) {th0 = 0.;} else {th0 = grids_->kx_h[i]/(grids_->ky_h[j]*pars_->shat);}
	for (int k=0; k<Nz; k++) {
	  theta_extended[k] = geo_->z_h[k] - th0;
	}
	zkxky_start[0] = 0;
	zkxky_start[1] = i-2*Nx/3+(Nx-1)/3;
	zkxky_start[2] = j;
	if (retval = nc_put_vara(file, theta_x, zkxky_start, zkxky_count, theta_extended)) ERR(retval);
      }
    }
    if (theta_extended) cudaFreeHost(theta_extended);
  }

  //  if (retval = nc_put_vara(file, theta,    geo_start, geo_count, geo_->z_h))         ERR(retval);
  if (retval = nc_put_vara(file, bmag,     geo_start, geo_count, geo_->bmag_h))      ERR(retval);
  if (retval = nc_put_vara(file, bgrad,    geo_start, geo_count, geo_->bgrad_h))     ERR(retval);
  if (retval = nc_put_vara(file, gbdrift,  geo_start, geo_count, geo_->gbdrift_h))   ERR(retval);
  if (retval = nc_put_vara(file, gbdrift0, geo_start, geo_count, geo_->gbdrift0_h))  ERR(retval);
  if (retval = nc_put_vara(file, cvdrift,  geo_start, geo_count, geo_->cvdrift_h))   ERR(retval);
  if (retval = nc_put_vara(file, cvdrift0, geo_start, geo_count, geo_->cvdrift0_h))  ERR(retval);
  if (retval = nc_put_vara(file, gds2,     geo_start, geo_count, geo_->gds2_h))      ERR(retval);
  if (retval = nc_put_vara(file, gds21,    geo_start, geo_count, geo_->gds21_h))     ERR(retval);  
  if (retval = nc_put_vara(file, gds22,    geo_start, geo_count, geo_->gds22_h))     ERR(retval);
  if (retval = nc_put_vara(file, grho,     geo_start, geo_count, geo_->grho_h))      ERR(retval);
  if (retval = nc_put_vara(file, jacobian, geo_start, geo_count, geo_->jacobian_h))  ERR(retval);

  idum = pars_->boundary_option_periodic ? 1 : 0;
  if (retval = nc_put_var(file, periodic,      &idum))     ERR(retval);

  idum = pars_->local_limit ? 1 : 0;
  if (retval = nc_put_var(file, local_limit,   &idum))     ERR(retval);
}

NetCDF_ids::~NetCDF_ids() {

  if (primary)      cudaFreeHost ( primary   );
  if (secondary)    cudaFreeHost ( secondary );
  if (tertiary)     cudaFreeHost ( tertiary  );

  if (red)          delete red;
  if (pot)          delete pot;
  if (ph2)          delete ph2;
  if (all_red)      delete all_red;
}

void NetCDF_ids::write_zonal_nc(nca *D, bool endrun) {
  int retval;

  if (D->write && endrun) {if (retval=nc_put_vara(file, D->idx,  D->start,      D->count,      &D->zonal)) ERR(retval);} 
  if (D->write_v_time)    {if (retval=nc_put_vara(file, D->time, D->time_start, D->time_count, &D->zonal)) ERR(retval);}
  D->increment_ts(); 
}

void NetCDF_ids::write_nc(nca *D, bool endrun) {
  int retval;

  if (D->write && endrun) {if (retval=nc_put_vara(file, D->idx,  D->start,      D->count,      D->cpu)) ERR(retval);} 
  if (D->write_v_time)    {if (retval=nc_put_vara(file, D->time, D->time_start, D->time_count, D->cpu)) ERR(retval);}
  D->increment_ts(); 
}

void NetCDF_ids::write_nc(nca *D, double data, bool endrun) {
  int retval;

  if (D->write && endrun) {if (retval=nc_put_vara(file, D->idx,  D->start,      D->count,      &data)) ERR(retval);} 
  if (D->write_v_time)    {if (retval=nc_put_vara(file, D->time, D->time_start, D->time_count, &data)) ERR(retval);}
  D->increment_ts(); 
}

void NetCDF_ids::write_nc(nca *D, float data, bool endrun) {
  int retval;

  if (D->write && endrun) {if (retval=nc_put_vara(file, D->idx,  D->start,      D->count,      &data)) ERR(retval);} 
  if (D->write_v_time)    {if (retval=nc_put_vara(file, D->time, D->time_start, D->time_count, &data)) ERR(retval);}
  D->increment_ts(); 
}

/*
void NetCDF_ids::pzt(MomentsG* G, Fields* f)
{
  int threads=256;
  int blocks=(grids_->NxNycNz+threads-1)/threads;
  
  primary[0]=0.; secondary[0]=0.; tertiary[0]=0.;
  
  Tbar <<<blocks, threads>>> (t_bar, G->G(), f->phi, geo_->kperp2);
  get_pzt <<<blocks, threads>>> (&primary[0], &secondary[0], &tertiary[0], f->phi, t_bar);
}
*/

void NetCDF_ids::write_Pky(float* P2, bool endrun)
{
  if (Pky -> write_v_time || (Pky -> write && endrun)) {
    int i = grids_->Nyc*grids_->Nspecies;

    pot->Sum(P2, Pky->data, PSPECTRA_ky);               CP_TO_CPU(Pky->tmp, Pky->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      for (int ik = 0; ik < grids_->Naky; ik++) {
	Pky->cpu[ik + is*grids_->Naky] = Pky->tmp[ik + is*grids_->Nyc];
      }
    }
    write_nc(Pky, endrun);      
  }
}

void NetCDF_ids::write_Pkx(float* P2, bool endrun)
{
  if (Pkx -> write_v_time || (Pkx -> write && endrun)) {
    int i = grids_->Nx*grids_->Nspecies;
    int NK = grids_->Nakx/2;
    int NX = grids_->Nx;
    
    pot->Sum(P2, Pkx->data, PSPECTRA_kx);               CP_TO_CPU(Pkx->tmp, Pkx->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it  = 0;
      int itp = it + NK;
      Pkx->cpu[itp + is*grids_->Nakx] = Pkx->tmp[it  + is*grids_->Nx];
      
      for (int it = 1; it < NK+1; it++) {
	int itp = NK + it;
	int itn = NK - it;
	int itm = NX - it;
	Pkx->cpu[itp + is*grids_->Nakx] = Pkx->tmp[it  + is*grids_->Nx];
	Pkx->cpu[itn + is*grids_->Nakx] = Pkx->tmp[itm + is*grids_->Nx];	
      }
    }  
    write_nc(Pkx, endrun);     
  }
}

void NetCDF_ids::write_Pz(float* P2, bool endrun)
{
  if (Pz -> write_v_time || (Pz -> write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;
    
    pot->Sum(P2, Pz->data, PSPECTRA_z);         CP_TO_CPU(Pz->cpu, Pz->data, sizeof(float)*i);
    write_nc(Pz, endrun);        
  }
}

void NetCDF_ids::write_Pkz(float* P2, bool endrun)
{
  if (Pkz -> write_v_time || (Pkz -> write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;     int Nz = grids_->Nz;
    
    pot->Sum(P2, Pkz->data, PSPECTRA_kz);       CP_TO_CPU(Pkz->tmp, Pkz->data, sizeof(float)*i);

    for (int is = 0; is < grids_->Nspecies; is++) {
      if (Nz>1) {
	for (int i = 0; i < Nz; i++) Pkz->cpu[i + is*Nz] = Pkz->tmp[ (i + Nz/2 + 1) % Nz + is*Nz ];
      } else {
	for (int i = 0; i < Nz; i++) Pkz->cpu[i + is*Nz] = Pkz->tmp[ i + is*Nz ];
      }
    }
    
    write_nc(Pkz, endrun);      
  }
}

void NetCDF_ids::write_Pkxky(float* P2, bool endrun)
{
  if (Pkxky -> write_v_time || (Pkxky -> write && endrun)) {

    int i = grids_->Nyc*grids_->Nx*grids_->Nspecies;

    int NK = grids_->Nakx/2;
    int NX = grids_->Nx; 
    
    pot->Sum(P2, Pkxky->data, PSPECTRA_kxky);
    CP_TO_CPU(Pkxky->tmp, Pkxky->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it = 0;
      int itp = it + NK;
      for (int ik = 0; ik < grids_->Naky; ik++) {
	int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
	Pkxky->cpu[Qp] = Pkxky->tmp[Rp];
      }	
      for (int it = 1; it < NK+1; it++) {
	int itp = NK + it;
	int itn = NK - it;
	int itm = NX - it;
	
	for (int ik = 0; ik < grids_->Naky; ik++) {

	  int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc * NX;

	  int Qn = itn + ik *grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rm = ik  + itm*grids_->Nyc  + is*grids_->Nyc * NX;

	  Pkxky->cpu[Qp] = Pkxky->tmp[Rp];
	  Pkxky->cpu[Qn] = Pkxky->tmp[Rm];
	}
      }
    }
    write_nc(Pkxky, endrun);     
  }
}

void NetCDF_ids::write_Wz(float *G2, bool endrun)
{
  if (Wz -> write_v_time || (Wz -> write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;
    
    red->Sum(G2, Wz->data, WSPECTRA_z);          CP_TO_CPU(Wz->cpu, Wz->data, sizeof(float)*i);
    write_nc(Wz, endrun);        
  }
}

void NetCDF_ids::write_Wkz(float *G2, bool endrun)
{
  if (Wkz -> write_v_time || (Wkz -> write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;     int Nz = grids_->Nz;
    
    red->Sum(G2, Wkz->data, WSPECTRA_kz);        CP_TO_CPU(Wkz->tmp, Wkz->data, sizeof(float)*i);

    for (int is = 0; is < grids_->Nspecies; is++) {
      if (Nz>1) {
	for (int i = 0; i < Nz; i++) Wkz->cpu[i+is*Nz] = Wkz->tmp[ (i + Nz/2 + 1) % Nz + is*Nz ];
      } else {
	for (int i = 0; i < Nz; i++) Wkz->cpu[i+is*Nz] = Wkz->tmp[ i + is*Nz ];
      }
    }
    
    write_nc (Wkz, endrun);      
  }
}

void NetCDF_ids::write_Ws(float* G2, bool endrun)
{
  if (Ws -> write_v_time) {
    red->Sum(G2, Ws->data, WSPECTRA_species);    CP_TO_CPU(Ws->cpu, Ws->data, sizeof(float)*grids_->Nspecies);
    write_nc(Ws, endrun);        

    if (Wtot -> write_v_time) {
      for (int is=0; is < grids_->Nspecies; is++) totW += Ws->cpu[is];
    }
  }
}

void NetCDF_ids::write_Wky(float* G2, bool endrun)
{
  if (Wky -> write_v_time || (Wky -> write && endrun)) {
    int i = grids_->Nyc*grids_->Nspecies;
    
    red->Sum(G2, Wky->data, WSPECTRA_ky);                CP_TO_CPU(Wky->tmp, Wky->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      for (int ik = 0; ik < grids_->Naky; ik++) {
	Wky->cpu[ik + is*grids_->Naky] = Wky->tmp[ik + is*grids_->Nyc];
      }
    }
    write_nc(Wky, endrun);      
  }
}

void NetCDF_ids::write_Wkx(float* G2, bool endrun)
{  
  if (Wkx -> write_v_time || (Wkx -> write && endrun)) {
    int i = grids_->Nx*grids_->Nspecies; 
    int NX = grids_->Nx;
    int NK = grids_->Nakx/2;
        
    red->Sum(G2, Wkx->data, WSPECTRA_kx);                CP_TO_CPU(Wkx->tmp, Wkx->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it = 0;
      int itp = it + NK;
      Wkx->cpu[itp + is*grids_->Nakx] = Wkx->tmp[it  + is*grids_->Nx];
      
      for (int it = 1; it < NK+1; it++) {
	int itp = NK + it;
	int itn = NK - it;
	int itm = NX - it;

	Wkx->cpu[itp + is*grids_->Nakx] = Wkx->tmp[it  + is*grids_->Nx];
	Wkx->cpu[itn + is*grids_->Nakx] = Wkx->tmp[itm + is*grids_->Nx];
      }
    }
    write_nc(Wkx, endrun);      
  }
}

void NetCDF_ids::write_Wkxky(float* G2, bool endrun)
{
  if (Wkxky -> write_v_time || (Wkxky -> write && endrun)) {
    int i = grids_->Nyc*grids_->Nx*grids_->Nspecies;  // int NK = (grids_->Nx-1)/3+1;

    int NK = grids_->Nakx/2;
    int NX = grids_->Nx; 
    
    red->Sum(G2, Wkxky->data, WSPECTRA_kxky);            CP_TO_CPU(Wkxky->tmp, Wkxky->data, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it = 0;
      int itp = it + NK;
      for (int ik = 0; ik < grids_->Naky; ik++) {     
	int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
	Wkxky->cpu[Qp] = Wkxky->tmp[Rp];
      }
      
      for (int it = 1; it < NK+1; it++) {
	int itp = NK + it; 
	int itn = NK - it; 
	int itm = NX - it; 
	for (int ik = 0; ik < grids_->Naky; ik++) {     

	  int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc * NX;

	  int Qn = itn + ik *grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rm = ik  + itm*grids_->Nyc  + is*grids_->Nyc * NX;

	  Wkxky->cpu[Qp] = Wkxky->tmp[Rp];
	  Wkxky->cpu[Qn] = Wkxky->tmp[Rm];
	}
      }
    }
    write_nc(Wkxky, endrun);  
  }
}

void NetCDF_ids::write_Wm(float* G2, bool endrun)
{
  if (Wm -> write_v_time || (Wm -> write && endrun)) {  
    int i = grids_->Nm*grids_->Nspecies;

    red-> Sum(G2, Wm -> data, WSPECTRA_m);         CP_TO_CPU(Wm->cpu, Wm->data, sizeof(float)*i);
    write_nc(Wm, endrun);       
  }
}

void NetCDF_ids::write_Wlm(float* G2, bool endrun)
{
  if (Wlm -> write_v_time || (Wlm -> write && endrun)) {  
    int i = grids_->Nmoms*grids_->Nspecies;
    
    red->Sum(G2, Wlm->data, WSPECTRA_lm);        CP_TO_CPU(Wlm->cpu, Wlm->data, sizeof(float)*i);
    write_nc(Wlm, endrun);      
  }
}

void NetCDF_ids::write_Wl(float* G2, bool endrun)
{
  if (Wl -> write_v_time || (Wl -> write && endrun)) {  
    int i = grids_->Nl*grids_->Nspecies;
    
    red->Sum(G2, Wl->data, WSPECTRA_l);          CP_TO_CPU(Wl->cpu, Wl->data, sizeof(float)*i);
    write_nc(Wl, endrun);        
  }
}

void NetCDF_ids::write_Ps(float* P2, bool endrun)
{
  if (Ps -> write_v_time) {
    
    pot->Sum(P2, Ps->data, PSPECTRA_species);     CP_TO_CPU(Ps->cpu, Ps->data, sizeof(float)*grids_->Nspecies);
    write_nc(Ps, endrun);          

    if (Wtot -> write_v_time) {
      totW = 0.;
      for (int is=0; is < grids_->Nspecies; is++) totW += Ps->cpu[is];
    }
  }
    
}

void NetCDF_ids::write_Aky(float* P2, bool endrun)
{
  if (Aky -> write_v_time || (Aky -> write && endrun)) {
    int i = grids_->Naky;
    
    ph2->Sum(P2, Aky->data, ASPECTRA_ky);       CP_TO_CPU(Aky->cpu, Aky->data, sizeof(float)*i);
    write_nc(Aky, endrun);     
  }
}

void NetCDF_ids::write_Az(float* P2, bool endrun)
{
  if (Az -> write_v_time || (Az -> write && endrun)) {
    int i = grids_->Nz;
    
    ph2->Sum(P2, Az->data, ASPECTRA_z);         CP_TO_CPU(Az->cpu, Az->data, sizeof(float)*i);
    write_nc(Az, endrun);       
  }
}

void NetCDF_ids::write_Akz(float* P2, bool endrun)
{
  if (Akz -> write_v_time || (Akz -> write && endrun)) {
    int Nz = grids_->Nz;
    
    ph2->Sum(P2, Akz->data, ASPECTRA_kz);       CP_TO_CPU(Akz->tmp, Akz->data, sizeof(float)*Nz);

    if (Nz>1) {
      for (int i = 0; i < Nz; i++) Akz->cpu[i] = Akz->tmp[ (i + Nz/2 + 1) % Nz ];
    } else {
      for (int i = 0; i < Nz; i++) Akz->cpu[i] = Akz->tmp[ i ];
    }    
    write_nc(Akz, endrun);      
  }
}

void NetCDF_ids::write_Akx(float* P2, bool endrun)
{
  if (Akx -> write_v_time || (Akx -> write && endrun)) {
    int NX = grids_->Nx;
    int NK = grids_->Nakx/2;
    
    ph2->Sum(P2, Akx->data, ASPECTRA_kx);               CP_TO_CPU(Akx->tmp, Akx->data, sizeof(float)*NX);
    
    int it = 0;
    int itp = it + NK;
    Akx->cpu[itp] = Akx->tmp[it ];;
    
    for (int it = 1; it < NK+1; it++) {
      int itp = NK + it;
      int itn = NK - it;
      int itm = NX - it;
      
      Akx->cpu[itp] = Akx->tmp[it ];;
      Akx->cpu[itn] = Akx->tmp[itm];;
    }
    write_nc(Akx, endrun);      
  }
}

void NetCDF_ids::write_Akxky(float* P2, bool endrun)
{
  if (Akxky -> write_v_time || (Akxky -> write && endrun)) {
    int i = grids_->Nyc*grids_->Nx; int NK = grids_->Nakx/2;  int NX = grids_->Nx;
    
    ph2->Sum(P2, Akxky->data, ASPECTRA_kxky);    CP_TO_CPU(Akxky->tmp, Akxky->data, sizeof(float)*i);
    
    int it = 0;
    int itp = it + NK;
    for (int ik = 0; ik < grids_->Naky; ik++) {
      int Qp = itp + ik*grids_->Nakx ;
      int Rp = ik  + it*grids_->Nyc  ;
      Akxky->cpu[Qp] = Akxky->tmp[Rp];
    }
    
    for (int it = 1; it < NK+1; it++) {
      int itp = NK + it;
      int itn = NK - it;
      int itm = NX - it;

      for (int ik = 0; ik < grids_->Naky; ik++) {

	int Qp = itp + ik*grids_->Nakx ;
	int Rp = ik  + it*grids_->Nyc  ;
	
	int Qn = itn + ik *grids_->Nakx ;
	int Rm = ik  + itm*grids_->Nyc  ;

	Akxky->cpu[Qp] = Akxky->tmp[Rp];
	Akxky->cpu[Qn] = Akxky->tmp[Rm];
      }
    }
    write_nc(Akxky, endrun);  
  }
}

void NetCDF_ids::write_As(float *P2, bool endrun)
{
  if (As -> write_v_time) {  
    ph2->Sum(P2, As->data, ASPECTRA_species);    CP_TO_CPU (As->cpu, As->data, sizeof(float));
    write_nc(As, endrun);         

    if (Wtot -> write_v_time) totW += *As->cpu;
  }
}

void NetCDF_ids::write_Q (float* Q, bool endrun)
{
  if (qs -> write_v_time) {
    all_red->Sum(Q, qs->data);                   CP_TO_CPU (qs->cpu, qs->data, sizeof(float)*grids_->Nspecies);
    write_nc(qs, endrun);       

    for (int is=0; is<grids_->Nspecies; is++) printf ("%e \t ",qs->cpu[is]);
    printf("\n");
  }
}

void NetCDF_ids::write_omg(cuComplex *W, bool endrun)
{
  CP_TO_CPU (omg->z_tmp, W, sizeof(cuComplex)*grids_->NxNyc);

  reduce2k(omg->cpu, omg->z_tmp);
  write_nc(omg, endrun);
}

void NetCDF_ids::write_Wtot()
{
  if (Wtot -> write_v_time) {  write_nc(Wtot, totW);        totW = 0.;}
}

void NetCDF_ids::close_nc_file() {
  int retval;
  if (retval = nc_close(file)) ERR(retval);
}

void NetCDF_ids::write_zonal(nca *D, cuComplex* f, bool shear, float adj) {
  if (!D->write_v_time) return;

  if (shear) {
    ddx loop_y (vEk, f, grids_->kx); // this also depends on Nyc and Nz
  } else {
    CP_ON_GPU (vEk, f, sizeof(cuComplex)*grids_->NxNycNz); // dependence on NxNycNz is unfortunate
  }
  setval loop_R (D->data, 0., D->N_);
  grad_phi -> dxC2R(vEk, D->data);
  if (D->xydata) {
    zavg loop_xy (D->data, D->tmp, adj);
  } else {
    yavg loop_x (D->data, D->tmp, adj);
  }
  CP_TO_CPU(D->cpu, D->tmp, sizeof(float)*D->Nwrite_);

  D->zonal = 0;
  for (int idx = 0; idx<grids_->Nx; idx++) {
    D->zonal += D->cpu[idx] * D->cpu[idx];
  }
  write_zonal_nc(D);
  
}

void NetCDF_ids::write_moment(nca *D, cuComplex *f, bool shear, float adj) {

  if (!D->write_v_time) return;

  if (shear) {
    ddx loop_y (vEk, f, grids_->kx); // this also depends on Nyc and Nz
  } else {
    CP_ON_GPU (vEk, f, sizeof(cuComplex)*grids_->NxNycNz); // dependence on NxNycNz is unfortunate
  }
  setval loop_R (D->data, 0., D->N_);
  grad_phi -> dxC2R(vEk, D->data);
  if (D->xydata) {
    zavg loop_xy (D->data, D->tmp, adj);
  } else {
    yavg loop_x (D->data, D->tmp, adj);
  }
  CP_TO_CPU(D->cpu, D->tmp, sizeof(float)*D->Nwrite_);

  write_nc(D);
}

void NetCDF_ids::write_ks_data(nca *D, cuComplex *G) {
  if (!D->write_v_time) return;

  grad_perp->C2R(G, D->data);
  CP_TO_CPU (D->cpu, D->data, sizeof(float)*D->N_);
  write_nc(D);
}

void NetCDF_ids::write_ks_data(nca *D, float *G) {
  if (!D->write_v_time) return;

  CP_TO_CPU (D->cpu, G, sizeof(float)*D->N_);
  write_nc(D);
}

// condense a (ky,kx) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
void NetCDF_ids::reduce2k(float *fk, cuComplex* f) {
  
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  int NK = grids_->Nakx/2;
  
  int it = 0;
  int itp = it + NK;
  for (int ik=0; ik<Naky; ik++) {
    int Qp = itp + ik*Nakx;
    int Rp = ik  + it*Nyc;
    fk[2*Qp  ] = f[Rp].x;
    fk[2*Qp+1] = f[Rp].y;
  }

  for (int it = 1; it < NK+1; it++) {
    int itp = NK + it;
    int itn = NK - it;
    int itm = Nx - it;
    for (int ik=0; ik<Naky; ik++) {
      int Qp = itp + ik*Nakx;
      int Rp = ik  + it*Nyc;

      int Qn = itn + ik*Nakx;
      int Rm = ik  + itm*Nyc;
      fk[2*Qp  ] = f[Rp].x;
      fk[2*Qp+1] = f[Rp].y;

      fk[2*Qn  ] = f[Rm].x;
      fk[2*Qn+1] = f[Rm].y;
    }
  }  
}



