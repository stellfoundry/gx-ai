#include "netcdf.h"
#include "ncdf.h"

NetCDF_ids::NetCDF_ids(Grids* grids, Parameters* pars, Geometry* geo) :
  grids_(grids), pars_(pars), geo_(geo),
  red(nullptr), pot(nullptr), ph2(nullptr), all_red(nullptr)
{

  Ws_d        = nullptr;  Ws_h        = nullptr;    Wz_d        = nullptr;  Wz_h        = nullptr;  
  Ps_d        = nullptr;  Ps_h        = nullptr;    Pz_d        = nullptr;  Pz_h        = nullptr;
  As_d        = nullptr;  As_h        = nullptr;    Az_d        = nullptr;  Az_h        = nullptr;

  Wm_d        = nullptr;  Wm_h        = nullptr;
  Wl_d        = nullptr;  Wl_h        = nullptr;
  Wlm_d       = nullptr;  Wlm_h       = nullptr;

  Pkz_d       = nullptr;  Pkz_h       = nullptr;  tmp_Pkz_h   = nullptr;
  Pky_d       = nullptr;  Pky_h       = nullptr;  tmp_Pky_h   = nullptr; 
  Pkx_d       = nullptr;  Pkx_h       = nullptr;  tmp_Pkx_h   = nullptr;
  Pkxky_d     = nullptr;  Pkxky_h     = nullptr;  tmp_Pkxky_h = nullptr;
  
  Wkz_d       = nullptr;  Wkz_h       = nullptr;  tmp_Wkz_h   = nullptr;
  Wky_d       = nullptr;  Wky_h       = nullptr;  tmp_Wky_h   = nullptr;
  Wkx_d       = nullptr;  Wkx_h       = nullptr;  tmp_Wkx_h   = nullptr;
  Wkxky_d     = nullptr;  Wkxky_h     = nullptr;  tmp_Wkxky_h = nullptr;

  Akz_d       = nullptr;  Akz_h       = nullptr;  tmp_Akz_h   = nullptr;
  Aky_d       = nullptr;  Aky_h       = nullptr;  tmp_Aky_h   = nullptr;
  Akx_d       = nullptr;  Akx_h       = nullptr;  tmp_Akx_h   = nullptr;
  Akxky_d     = nullptr;  Akxky_h     = nullptr;  tmp_Akxky_h = nullptr;

  primary     = nullptr;  secondary   = nullptr;  tertiary    = nullptr;

  qs_h        = nullptr;  qs_d        = nullptr;  
  omg_h       = nullptr;  tmp_omg_h   = nullptr;  
  
  if (pars_->diagnosing_spectra || pars_->diagnosing_kzspec) {
    float dum = 1.0;
    red = new Red(grids_, pars_->wspectra);       cudaDeviceSynchronize(); CUDA_DEBUG("Reductions: %s \n"); // G**2
    pot = new Red(grids_, pars_->pspectra, true); cudaDeviceSynchronize(); CUDA_DEBUG("Reductions: %s \n"); // (1-G0) Phi**2
    ph2 = new Red(grids_, pars_->aspectra,  dum); cudaDeviceSynchronize(); CUDA_DEBUG("Reductions: %s \n"); // Phi**2
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

  char strb[263];
  strcpy(strb, pars_->run_name); 
  strcat(strb, ".nc");

  int retval, idum;

  theta_extended = nullptr;
  
  //  if (retval = nc_open(strb, NC_NETCDF4|NC_WRITE, &file)) ERR(retval);
  file = pars_->ncid;
  if (retval = nc_redef(file));
  
  int ri;
  // Get handles for the dimensions
  if (retval = nc_inq_dimid(file, "ri", &ri))  ERR(retval);
  
  if (retval = nc_def_dim(file, "kz",        grids_->Nz,    &nkz))       ERR(retval);  
  if (retval = nc_def_dim(file, "ky",        grids_->Naky,  &ky_dim))    ERR(retval);
  if (retval = nc_def_dim(file, "kx",        grids_->Nakx,  &kx_dim))    ERR(retval);
  if (retval = nc_def_dim(file, "z",         grids_->Nz,    &nz))        ERR(retval);  

  if (retval = nc_inq_dimid(file, "m",       &m_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "l",       &l_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "s",       &s_dim))    ERR(retval);
  if (retval = nc_inq_dimid(file, "time",    &time_dim)) ERR(retval);

  //  if (retval = nc_def_var(file, "time",       NC_DOUBLE, 1, &time_dim, &time))    ERR(retval);
  if (retval = nc_def_var(file, "periodic",       NC_INT, 0, 0, &periodic))        ERR(retval);
  if (retval = nc_def_var(file, "local_limit",    NC_INT, 0, 0, &local_limit))     ERR(retval);

  v_ky[0] = ky_dim;
  if (retval = nc_def_var(file, "ky",       NC_FLOAT, 1, v_ky, &ky))              ERR(retval);

  v_kx[0] = kx_dim;
  if (retval = nc_def_var(file, "kx",       NC_FLOAT, 1, v_kx, &kx))              ERR(retval);

  v_kz[0] = nkz;
  if (retval = nc_def_var(file, "kz",       NC_FLOAT, 1, v_kz, &kz))              ERR(retval);
  
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
  
  //  time.dims[0] = 
  time.time_start[0] = 0;
  time.time_count[0] = 1;
  time.time_dims[0] = time_dim;
  if (retval = nc_def_var(file, "time",     NC_DOUBLE, 1, time.time_dims, &time.time))    ERR(retval);

  ////////////////////////////
  //                        //
  //       DENSITY          //
  //                        //
  ////////////////////////////

  den.write = pars_->write_moms;

  if (den.write) {
    den.dims[0] = s_dim;
    den.dims[1] = nz;
    den.dims[2] = kx_dim;
    den.dims[3] = ky_dim;
    den.dims[4] = ri;

    if (retval = nc_def_var(file, "density",  NC_FLOAT, 5, den.dims, &den.idx )) ERR(retval);
    
    den.start[0] = 0;
    den.start[1] = 0;
    den.start[2] = 0;
    den.start[3] = 0; 
    den.start[4] = 0; 
    
    den.count[0] = grids_->Nspecies;
    den.count[1] = grids_->Nz;
    den.count[2] = grids_->Nakx;
    den.count[3] = grids_->Naky;
    den.count[4] = 2;

    den.ns = grids_->Nspecies;
  }
  
  ////////////////////////////
  //                        //
  //       DENSITY(t=0)     //
  //                        //
  ////////////////////////////

  den0.write = pars_->write_moms;

  if (den0.write) {
    den0.dims[0] = s_dim;
    den0.dims[1] = nz;
    den0.dims[2] = kx_dim;
    den0.dims[3] = ky_dim;
    den0.dims[4] = ri;

    if (retval = nc_def_var(file, "density0",  NC_FLOAT, 5, den0.dims, &den0.idx )) ERR(retval);
    
    den0.start[0] = 0;
    den0.start[1] = 0;
    den0.start[2] = 0;
    den0.start[3] = 0; 
    den0.start[4] = 0; 
    
    den0.count[0] = grids_->Nspecies;
    den0.count[1] = grids_->Nz;
    den0.count[2] = grids_->Nakx;
    den0.count[3] = grids_->Naky;
    den0.count[4] = 2;

    den0.ns = grids_->Nspecies;
  }
  
  ////////////////////////////
  //                        //
  //       Phi              //
  //                        //
  ////////////////////////////

  wphi.write = pars_->write_phi;

  if (wphi.write) {
    wphi.dims[0] = nz;
    wphi.dims[1] = kx_dim;
    wphi.dims[2] = ky_dim;
    wphi.dims[3] = ri;
    
    if (retval = nc_def_var(file, "phi",      NC_FLOAT, 4, wphi.dims, &wphi.idx ))      ERR(retval);

    wphi.start[0] = 0;
    wphi.start[1] = 0;
    wphi.start[2] = 0;
    wphi.start[3] = 0; 
    
    wphi.count[0] = grids_->Nz;
    wphi.count[1] = grids_->Nakx;
    wphi.count[2] = grids_->Naky;
    wphi.count[3] = 2;

    wphi.ns = 1; 
  }

  ////////////////////////////
  //                        //
  //       Phi(t=0)         //
  //                        //
  ////////////////////////////

  wphi0.write = pars_->write_phi;

  if (wphi0.write) {
    wphi0.dims[0] = nz;
    wphi0.dims[1] = kx_dim;
    wphi0.dims[2] = ky_dim;
    wphi0.dims[3] = ri;
    
    if (retval = nc_def_var(file, "phi0",      NC_FLOAT, 4, wphi0.dims, &wphi0.idx ))      ERR(retval);

    wphi0.start[0] = 0;
    wphi0.start[1] = 0;
    wphi0.start[2] = 0;
    wphi0.start[3] = 0; 
    
    wphi0.count[0] = grids_->Nz;
    wphi0.count[1] = grids_->Nakx;
    wphi0.count[2] = grids_->Naky;
    wphi0.count[3] = 2;

    wphi0.ns = 1;
  }

  ////////////////////////////
  //                        //
  //   DENSITY(kpar)        //
  //                        //
  ////////////////////////////

  denk.write = (pars_->write_phi_kpar and pars_->write_moms);

  if (denk.write) {
    denk.dims[0] = s_dim;
    denk.dims[1] = nkz;
    denk.dims[2] = kx_dim;
    denk.dims[3] = ky_dim;
    denk.dims[4] = ri;

    if (retval = nc_def_var(file, "density_kpar", NC_FLOAT, 5, denk.dims, &denk.idx)) ERR(retval);    

    denk.start[0] = 0;
    denk.start[1] = 0;
    denk.start[2] = 0;
    denk.start[3] = 0; 
    denk.start[4] = 0; 
    
    denk.count[0] = grids_->Nspecies;
    denk.count[1] = grids_->Nz;
    denk.count[2] = grids_->Nakx;
    denk.count[3] = grids_->Naky;
    denk.count[4] = 2;

    denk.ns = 1;
  }

  ////////////////////////////
  //                        //
  //   Phi(kpar)            //
  //                        //
  ////////////////////////////

  wphik.write = pars_->write_phi_kpar;

  if (wphik.write) {
    wphik.dims[0] = nkz;
    wphik.dims[1] = kx_dim;
    wphik.dims[2] = ky_dim;
    wphik.dims[3] = ri;
    
    if (retval = nc_def_var(file, "phi2_kz",    NC_FLOAT, 4, wphik.dims, &wphik.idx))     ERR(retval);

    wphik.start[0] = 0;
    wphik.start[1] = 0;
    wphik.start[2] = 0;
    wphik.start[3] = 0; 
    
    wphik.count[0] = grids_->Nz;
    wphik.count[1] = grids_->Nakx;
    wphik.count[2] = grids_->Naky;
    wphik.count[3] = 2;

    wphi.ns = 1;
  }

  ////////////////////////////
  //                        //
  //   Frequencies          //
  //                        //
  ////////////////////////////

  omg.write_v_time = pars_->write_omega;
  
  if (omg.write_v_time) {
    omg.time_dims[0] = time_dim; 
    omg.time_dims[1] = ky_dim; 
    omg.time_dims[2] = kx_dim;
    omg.time_dims[3] = ri;
    
    if (retval = nc_def_var(file, "omega_v_time", NC_FLOAT, 4, omg.time_dims, &omg.time)) ERR(retval);

    omg.time_start[0] = 1;
    omg.time_start[1] = 0;
    omg.time_start[2] = 0;
    omg.time_start[3] = 0;
    
    omg.time_count[0] = 1;
    omg.time_count[1] = grids_->Naky;
    omg.time_count[2] = grids_->Nakx;
    omg.time_count[3] = 2;

    cudaMallocHost (&tmp_omg_h,   sizeof(cuComplex) * nX * nY); 
    cudaMallocHost (    &omg_h,   sizeof(cuComplex) * nXk * nYk);

    for (int i=0; i < nXk * nYk * 2; i++) omg_h[i] = 0.;    
  }

  ////////////////////////////
  //                        //
  // Rosenbluth-Hinton      //
  //                        //
  ////////////////////////////

  rh.write = pars_->write_rh;

  if (rh.write) {
    rh.time_dims[0] = time_dim;
    rh.time_dims[1] = ri;
    
    if (retval = nc_def_var(file, "phi_rh", NC_FLOAT, 2, rh.time_dims, &rh.time)) ERR(retval);

    rh.time_start[0] = 0;
    rh.time_start[1] = 0;

    rh.time_count[0] = 1;
    rh.time_count[1] = 2;
  }

  ////////////////////////////
  //                        //
  //     PZT estimates      //
  //                        //
  ////////////////////////////

  Pzt.write_v_time = pars_->write_pzt;
  
  if (Pzt.write_v_time) {

    Pzt.time_dims[0] = time_dim;
    pZt.time_dims[0] = time_dim;
    pzT.time_dims[0] = time_dim;

    if (retval = nc_def_var(file, "prim", NC_FLOAT, 1, Pzt.time_dims, &Pzt.idx)) ERR(retval);
    if (retval = nc_def_var(file, "sec",  NC_FLOAT, 1, pZt.time_dims, &pZt.idx)) ERR(retval);
    if (retval = nc_def_var(file, "tert", NC_FLOAT, 1, pzT.time_dims, &pzT.idx)) ERR(retval);

    Pzt.time_start[0] = 0;
    Pzt.time_count[0] = 1; 

    pZt.time_start[0] = 0;
    pZt.time_count[0] = 1; 

    pzT.time_start[0] = 0;
    pzT.time_count[0] = 1; 
  
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

  if (pars_->pspectra[PSPECTRA_species] > 0) Ps.write_v_time = true;

  if (Ps.write_v_time) {
    Ps.time_dims[0] = time_dim;
    Ps.time_dims[1] = s_dim;
    
    if (retval = nc_def_var(file, "Pst", NC_FLOAT, 2, Ps.time_dims, &Ps.time))  ERR(retval);
    Ps.time_start[0] = 0;
    Ps.time_start[1] = 0;
    
    Ps.time_count[0] = 1;
    Ps.time_count[1] = grids_->Nspecies;

    cudaMalloc     (&Ps_d,        sizeof(float) * nS);
    cudaMallocHost (&Ps_h,        sizeof(float) * nS);
  }
  
  ////////////////////////////
  //                        //
  //   P (ky, species)      //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_ky] > 0) Pky.write_v_time = true;

  if (Pky.write_v_time) {
    Pky.time_dims[0] = time_dim;
    Pky.time_dims[1] = s_dim;
    Pky.time_dims[2] = ky_dim;
    
    if (retval = nc_def_var(file, "Pkyst", NC_FLOAT, 3, Pky.time_dims, &Pky.time))  ERR(retval);
    Pky.time_start[0] = 0;
    Pky.time_start[1] = 0;
    Pky.time_start[2] = 0;
    
    Pky.time_count[0] = 1;
    Pky.time_count[1] = grids_->Nspecies;
    Pky.time_count[2] = grids_->Naky;
    
    cudaMalloc     (&Pky_d,       sizeof(float) * nY * nS);
    cudaMallocHost (&tmp_Pky_h,   sizeof(float) * nY * nS);
    cudaMallocHost (&Pky_h,       sizeof(float) * nYk * nS);    
  }
  
  ////////////////////////////
  //                        //
  //   P (kx, species)      //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_kx] > 0) Pkx.write_v_time = true;

  if (Pkx.write_v_time) {
    Pkx.time_dims[0] = time_dim;
    Pkx.time_dims[1] = s_dim;
    Pkx.time_dims[2] = kx_dim;
    
    if (retval = nc_def_var(file, "Pkxst", NC_FLOAT, 3, Pkx.time_dims, &Pkx.time))  ERR(retval);
    Pkx.time_start[0] = 0;
    Pkx.time_start[1] = 0;
    Pkx.time_start[2] = 0;
    
    Pkx.time_count[0] = 1;
    Pkx.time_count[1] = grids_->Nspecies;
    Pkx.time_count[2] = grids_->Nakx;      

    cudaMalloc     (&Pkx_d,       sizeof(float) * nX * nS); 
    cudaMallocHost (&tmp_Pkx_h,   sizeof(float) * nX * nS); 
    cudaMallocHost (&Pkx_h,       sizeof(float) * nXk * nS);     
  }
  
  ////////////////////////////
  //                        //
  //   P (kz, species)      //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_kz] > 0) Pkz.write_v_time = true;

  if (Pkz.write_v_time) {
    Pkz.time_dims[0] = time_dim;
    Pkz.time_dims[1] = s_dim;
    Pkz.time_dims[2] = nkz;
    
    if (retval = nc_def_var(file, "Pkzst", NC_FLOAT, 3, Pkz.time_dims, &Pkz.time))  ERR(retval);
    Pkz.time_start[0] = 0;
    Pkz.time_start[1] = 0;
    Pkz.time_start[2] = 0;
    
    Pkz.time_count[0] = 1;
    Pkz.time_count[1] = grids_->Nspecies;
    Pkz.time_count[2] = grids_->Nz;
  
    cudaMalloc     (&Pkz_d,        sizeof(float) * nZ * nS); 
    cudaMallocHost (&Pkz_h,        sizeof(float) * nZ * nS);     
    cudaMallocHost (&tmp_Pkz_h,    sizeof(float) * nZ * nS);     
  }
  
  ////////////////////////////
  //                        //
  //   P (z, species)       //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_z] > 0) Pz.write_v_time = true;

  if (Pz.write_v_time) {
    Pz.time_dims[0] = time_dim;
    Pz.time_dims[1] = s_dim;
    Pz.time_dims[2] = nz;
    
    if (retval = nc_def_var(file, "Pzst", NC_FLOAT, 3, Pz.time_dims, &Pz.time))  ERR(retval);
    Pz.time_start[0] = 0;
    Pz.time_start[1] = 0;
    Pz.time_start[2] = 0;
    
    Pz.time_count[0] = 1;
    Pz.time_count[1] = grids_->Nspecies;
    Pz.time_count[2] = grids_->Nz;
  
    cudaMalloc     (&Pz_d,        sizeof(float) * nZ * nS); 
    cudaMallocHost (&Pz_h,        sizeof(float) * nZ * nS);     
  }
  
  ////////////////////////////
  //                        //
  //   P (kx,ky,  species)  //
  //                        //
  ////////////////////////////

  if (pars_->pspectra[PSPECTRA_kxky] > 0) Pkxky.write_v_time = true;

  if (Pkxky.write_v_time) {
    Pkxky.time_dims[0] = time_dim;
    Pkxky.time_dims[1] = s_dim;
    Pkxky.time_dims[2] = ky_dim;
    Pkxky.time_dims[3] = kx_dim;
    
    if (retval = nc_def_var(file, "Pkxkyst", NC_FLOAT, 4, Pkxky.time_dims, &Pkxky.time))  ERR(retval);
    Pkxky.time_start[0] = 0;
    Pkxky.time_start[1] = 0;
    Pkxky.time_start[2] = 0;
    Pkxky.time_start[3] = 0;
    
    Pkxky.time_count[0] = 1;
    Pkxky.time_count[1] = grids_->Nspecies;
    Pkxky.time_count[2] = grids_->Naky;
    Pkxky.time_count[3] = grids_->Nakx;

    cudaMalloc     (&Pkxky_d,     sizeof(float) * nX * nY * nS);
    cudaMallocHost (&tmp_Pkxky_h, sizeof(float) * nX * nY * nS);
    cudaMallocHost (&Pkxky_h,     sizeof(float) * nXk * nYk * nS);    
  }   

  ////////////////////////////
  //                        //
  //   W (species)          //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_species] > 0) Ws.write_v_time = true;

  if (Ws.write_v_time) {
    Ws.time_dims[0] = time_dim;
    Ws.time_dims[1] = s_dim;
    
    if (retval = nc_def_var(file, "Wst", NC_FLOAT, 2, Ws.time_dims, &Ws.time))  ERR(retval);
    Ws.time_start[0] = 0;
    Ws.time_start[1] = 0;
    
    Ws.time_count[0] = 1;
    Ws.time_count[1] = grids_->Nspecies;

    cudaMalloc     (&Ws_d,        sizeof(float) * nS);
    cudaMallocHost (&Ws_h,        sizeof(float) * nS);    
  }
  
  ////////////////////////////
  //                        //
  // W (adiabatic species)  //
  //                        //
  ////////////////////////////

 
  As.write_v_time = (pars_->aspectra[ASPECTRA_species] > 0);

  if (As.write_v_time) {
    As.time_dims[0] = time_dim;
    
    if (retval = nc_def_var(file, "W_adiabatic", NC_FLOAT, 1, As.time_dims, &As.time))  ERR(retval);
    As.time_start[0] = 0;
    As.time_count[0] = 1;

    cudaMalloc     (&As_d,        sizeof(float)*2); // bd bug chase back to as.write
    cudaMallocHost (&As_h,        sizeof(float));      
  }
  
  ////////////////////////////
  //                        //
  //   W (ky, species)      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_ky] > 0) Wky.write_v_time = true;

  if (Wky.write_v_time) {
    Wky.time_dims[0] = time_dim;
    Wky.time_dims[1] = s_dim;
    Wky.time_dims[2] = ky_dim;
    
    if (retval = nc_def_var(file, "Wkyst", NC_FLOAT, 3, Wky.time_dims, &Wky.time))  ERR(retval);
    Wky.time_start[0] = 0;
    Wky.time_start[1] = 0;
    Wky.time_start[2] = 0;
    
    Wky.time_count[0] = 1;
    Wky.time_count[1] = grids_->Nspecies;
    Wky.time_count[2] = grids_->Naky;      

    cudaMalloc     (&Wky_d,       sizeof(float) * nY * nS); 
    cudaMallocHost (&tmp_Wky_h,   sizeof(float) * nY * nS); 
    cudaMallocHost (&Wky_h,       sizeof(float) * nYk * nS);
    
  }   

  ////////////////////////////
  //                        //
  //   W (ky) adiabatic     //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_ky] > 0) Aky.write_v_time = true;

  if (Aky.write_v_time) {
    Aky.time_dims[0] = time_dim;
    Aky.time_dims[1] = ky_dim;
    
    if (retval = nc_def_var(file, "Akyst", NC_FLOAT, 2, Aky.time_dims, &Aky.time))  ERR(retval);
    Aky.time_start[0] = 0;
    Aky.time_start[1] = 0;
    
    Aky.time_count[0] = 1;
    Aky.time_count[1] = grids_->Naky;      

    cudaMalloc     (&Aky_d,       sizeof(float) * nY); 
    cudaMallocHost (&tmp_Aky_h,   sizeof(float) * nY); 
    cudaMallocHost (&Aky_h,       sizeof(float) * nYk);
  }   
  
  ////////////////////////////
  //                        //
  //   W (kx, species)      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_kx] > 0) Wkx.write_v_time = true;

  if (Wkx.write_v_time) {
    Wkx.time_dims[0] = time_dim;
    Wkx.time_dims[1] = s_dim;
    Wkx.time_dims[2] = kx_dim;
    
    if (retval = nc_def_var(file, "Wkxst", NC_FLOAT, 3, Wkx.time_dims, &Wkx.time))  ERR(retval);
    Wkx.time_start[0] = 0;
    Wkx.time_start[1] = 0;
    Wkx.time_start[2] = 0;
    
    Wkx.time_count[0] = 1;
    Wkx.time_count[1] = grids_->Nspecies;
    Wkx.time_count[2] = grids_->Nakx;      

    cudaMalloc     (&Wkx_d,       sizeof(float) * nX * nS);
    cudaMallocHost (&tmp_Wkx_h,   sizeof(float) * nX * nS);
    cudaMallocHost (&Wkx_h,       sizeof(float) * nXk * nS);    
  }
  
  ////////////////////////////
  //                        //
  //   W (kx) adiabatic     //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_kx] > 0) Akx.write_v_time = true;

  if (Akx.write_v_time) {
    Akx.time_dims[0] = time_dim;
    Akx.time_dims[1] = kx_dim;
    
    if (retval = nc_def_var(file, "Akxst", NC_FLOAT, 2, Akx.time_dims, &Akx.time))  ERR(retval);
    Akx.time_start[0] = 0;
    Akx.time_start[1] = 0;
    
    Akx.time_count[0] = 1;
    Akx.time_count[1] = grids_->Nakx;      

    cudaMalloc     (&Akx_d,       sizeof(float) * nX);
    cudaMallocHost (&tmp_Akx_h,   sizeof(float) * nX);
    cudaMallocHost (&Akx_h,       sizeof(float) * nXk);
  }   
  
  ////////////////////////////
  //                        //
  //   W (kx,ky,  species)  //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_kxky] > 0) Wkxky.write_v_time = true;

  if (Wkxky.write_v_time) {
    Wkxky.time_dims[0] = time_dim;
    Wkxky.time_dims[1] = s_dim;
    Wkxky.time_dims[2] = ky_dim;
    Wkxky.time_dims[3] = kx_dim;
    
    if (retval = nc_def_var(file, "Wkxkyst", NC_FLOAT, 4, Wkxky.time_dims, &Wkxky.time))  ERR(retval);
    Wkxky.time_start[0] = 0;
    Wkxky.time_start[1] = 0;
    Wkxky.time_start[2] = 0;
    Wkxky.time_start[3] = 0;
    
    Wkxky.time_count[0] = 1;
    Wkxky.time_count[1] = grids_->Nspecies;
    Wkxky.time_count[2] = grids_->Naky;      
    Wkxky.time_count[3] = grids_->Nakx;

    cudaMalloc     (&Wkxky_d,     sizeof(float) * nX * nY * nS);
    cudaMallocHost (&tmp_Wkxky_h, sizeof(float) * nX * nY * nS);
    cudaMallocHost (&Wkxky_h,     sizeof(float) * nXk * nYk * nS);    
  }   

  ////////////////////////////
  //                        //
  //   W (kx,ky) adiabatic  //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_kxky] > 0) Akxky.write_v_time = true;

  if (Akxky.write_v_time) {
    Akxky.time_dims[0] = time_dim;
    Akxky.time_dims[1] = ky_dim;
    Akxky.time_dims[2] = kx_dim;
    
    if (retval = nc_def_var(file, "Akxkyst", NC_FLOAT, 3, Akxky.time_dims, &Akxky.time))  ERR(retval);
    Akxky.time_start[0] = 0;
    Akxky.time_start[1] = 0;
    Akxky.time_start[2] = 0;
    
    Akxky.time_count[0] = 1;
    Akxky.time_count[1] = grids_->Naky;      
    Akxky.time_count[2] = grids_->Nakx;

    cudaMalloc     (&Akxky_d,     sizeof(float) * nX * nY);
    cudaMallocHost (&tmp_Akxky_h, sizeof(float) * nX * nY);
    cudaMallocHost (&Akxky_h,     sizeof(float) * nXk * nYk);
  }   
  
  ////////////////////////////
  //                        //
  //   W (kz, species)      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_kz] > 0) Wkz.write_v_time = true;

  if (Wkz.write_v_time) {
    Wkz.time_dims[0] = time_dim;
    Wkz.time_dims[1] = s_dim;
    Wkz.time_dims[2] = nkz;
    
    if (retval = nc_def_var(file, "Wkzst", NC_FLOAT, 3, Wkz.time_dims, &Wkz.time))  ERR(retval);
    Wkz.time_start[0] = 0;
    Wkz.time_start[1] = 0;
    Wkz.time_start[2] = 0;
    
    Wkz.time_count[0] = 1;
    Wkz.time_count[1] = grids_->Nspecies;
    Wkz.time_count[2] = grids_->Nz;
    
    cudaMalloc     (&Wkz_d,        sizeof(float) * nZ * nS);
    cudaMallocHost (&Wkz_h,        sizeof(float) * nZ * nS);    
    cudaMallocHost (&tmp_Wkz_h,    sizeof(float) * nZ * nS);     
  }   

  ////////////////////////////
  //                        //
  //   W (z, species)       //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_z] > 0) Wz.write_v_time = true;

  if (Wz.write_v_time) {
    Wz.time_dims[0] = time_dim;
    Wz.time_dims[1] = s_dim;
    Wz.time_dims[2] = nz;
    
    if (retval = nc_def_var(file, "Wzst", NC_FLOAT, 3, Wz.time_dims, &Wz.time))  ERR(retval);
    Wz.time_start[0] = 0;
    Wz.time_start[1] = 0;
    Wz.time_start[2] = 0;
    
    Wz.time_count[0] = 1;
    Wz.time_count[1] = grids_->Nspecies;
    Wz.time_count[2] = grids_->Nz;
    
    cudaMalloc     (&Wz_d,        sizeof(float) * nZ * nS);
    cudaMallocHost (&Wz_h,        sizeof(float) * nZ * nS);    
  }   
  
  ////////////////////////////
  //                        //
  //   A (kz)  adiabatic    //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_kz] > 0) Akz.write_v_time = true;

  if (Akz.write_v_time) {
    Akz.time_dims[0] = time_dim;
    Akz.time_dims[1] = nkz;
    
    if (retval = nc_def_var(file, "Akzst", NC_FLOAT, 2, Akz.time_dims, &Akz.time))  ERR(retval);
    Akz.time_start[0] = 0;
    Akz.time_start[1] = 0;
    
    Akz.time_count[0] = 1;
    Akz.time_count[1] = grids_->Nz;      

    cudaMalloc     (&Akz_d,        sizeof(float) * nZ);
    cudaMallocHost (&Akz_h,        sizeof(float) * nZ);
    cudaMallocHost (&tmp_Akz_h,    sizeof(float) * nZ * nS);     
  }   
  
  ////////////////////////////
  //                        //
  //   A (z)  adiabatic     //
  //                        //
  ////////////////////////////

  if (pars_->aspectra[ASPECTRA_z] > 0) Az.write_v_time = true;

  if (Az.write_v_time) {
    Az.time_dims[0] = time_dim;
    Az.time_dims[1] = nz;
    
    if (retval = nc_def_var(file, "Azst", NC_FLOAT, 2, Az.time_dims, &Az.time))  ERR(retval);
    Az.time_start[0] = 0;
    Az.time_start[1] = 0;
    
    Az.time_count[0] = 1;
    Az.time_count[1] = grids_->Nz;      

    cudaMalloc     (&Az_d,        sizeof(float) * nZ);
    cudaMallocHost (&Az_h,        sizeof(float) * nZ);
  }   
  
  ////////////////////////////
  //                        //
  // Lag-Herm spectrum      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_lm] > 0) Wlm.write_v_time = true;
  
  if (Wlm.write_v_time) {
    Wlm.time_dims[0] = time_dim;
    Wlm.time_dims[1] = s_dim;
    Wlm.time_dims[2] = m_dim;
    Wlm.time_dims[3] = l_dim;
    
    if (retval = nc_def_var(file, "Wlmst", NC_FLOAT, 4, Wlm.time_dims, &Wlm.time))  ERR(retval);
    Wlm.time_start[0] = 0;
    Wlm.time_start[1] = 0;
    Wlm.time_start[2] = 0;
    Wlm.time_start[3] = 0;
    
    Wlm.time_count[0] = 1;
    Wlm.time_count[1] = grids_->Nspecies;
    Wlm.time_count[2] = grids_->Nm;
    Wlm.time_count[3] = grids_->Nl;      

    cudaMalloc     (&Wlm_d,       sizeof(float) * nL * nM * nS);
    cudaMallocHost (&Wlm_h,       sizeof(float) * nL * nM * nS);    
  }
    
  ////////////////////////////
  //                        //
  // Laguerre spectrum      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_l] > 0) Wl.write_v_time = true;

  if (Wl.write_v_time) {
    Wl.time_dims[0] = time_dim;
    Wl.time_dims[1] = s_dim;
    Wl.time_dims[2] = l_dim;
    
    if (retval = nc_def_var(file, "Wlst", NC_FLOAT, 3, Wl.time_dims, &Wl.time))  ERR(retval);
    Wl.time_start[0] = 0;
    Wl.time_start[1] = 0;
    Wl.time_start[2] = 0;
    
    Wl.time_count[0] = 1;
    Wl.time_count[1] = grids_->Nspecies;
    Wl.time_count[2] = grids_->Nl;

    cudaMalloc     (&Wl_d,        sizeof(float) * nL * nS); 
    cudaMallocHost (&Wl_h,        sizeof(float) * nL * nS);     
  }
  
  ////////////////////////////
  //                        //
  //  Hermite spectrum      //
  //                        //
  ////////////////////////////

  if (pars_->wspectra[WSPECTRA_m] > 0) Wm.write_v_time = true;
  
  if (Wm.write_v_time) {
    Wm.time_dims[0] = time_dim;
    Wm.time_dims[1] = s_dim;
    Wm.time_dims[2] = m_dim;
    
    if (retval = nc_def_var(file, "Wmst", NC_FLOAT, 3, Wm.time_dims, &Wm.time))  ERR(retval);
    Wm.time_start[0] = 0;
    Wm.time_start[1] = 0;
    Wm.time_start[2] = 0;
    
    Wm.time_count[0] = 1;    
    Wm.time_count[1] = grids_->Nspecies;    
    Wm.time_count[2] = grids_->Nm;          

    cudaMalloc     (&Wm_d,        sizeof(float) * nM * nS);
    cudaMallocHost (&Wm_h,        sizeof(float) * nM * nS);
  }

  bool linked = (not pars_->local_limit && not pars_->boundary_option_periodic);

  if (linked && false) {
    zkxky[0] = nz;
    zkxky[1] = kx_dim; 
    zkxky[2] = ky_dim;
    
    if (retval = nc_def_var(file, "theta_x",  NC_FLOAT, 3, zkxky, &theta_x))  ERR(retval);
  }

  ////////////////////////////
  //                        //
  //   Free energy          //
  //                        //
  ////////////////////////////

  Wtot.write_v_time = pars_->write_free_energy;
  
  if (Wtot.write_v_time) {
    Wtot.time_dims[0] = time_dim;

    if (retval = nc_def_var(file, "W", NC_FLOAT, 1, Wtot.time_dims, &Wtot.time)) ERR(retval);

    Wtot.time_start[0] = 0;
    Wtot.time_count[0] = 1;
    totW = 0.;
  }

  ////////////////////////////
  //                        //
  //    Heat fluxes         //
  //                        //
  ////////////////////////////

  qs.write_v_time = pars_->write_fluxes;
  
  if (qs.write_v_time) {
    qs.time_dims[0] = time_dim;
    qs.time_dims[1] = s_dim;
    
    if (retval = nc_def_var(file, "qflux", NC_FLOAT, 2, qs.time_dims, &qs.time)) ERR(retval);
    qs.time_start[0] = 0;
    qs.time_start[1] = 0;

    qs.time_count[0] = 1;
    qs.time_count[1] = grids_->Nspecies;

    //    cudaMallocHost(&pflux,  sizeof(float) * nS);
    cudaMallocHost(&qs_h,   sizeof(float) * nS);
    cudaMalloc    (&qs_d,   sizeof(float) * nS);
    all_red = new Red(nR, nS);  cudaDeviceSynchronize();  CUDA_DEBUG("Reductions: %s \n");
  }

  DEBUGPRINT("ncdf:  ending definition mode for NetCDF \n");
  
  if (retval = nc_enddef(file)) ERR(retval);
  
  ///////////////////////////////////
  //                               //
  //        kz                     //
  //                               //
  ///////////////////////////////////
  kz_start[0] = 0;
  kz_count[0] = grids_->Nz;

  if (retval = nc_put_vara(file, kz, kz_start, kz_count, grids_->kz_outh))      ERR(retval);

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

  if (retval = nc_put_vara(file, theta,    geo_start, geo_count, geo_->z_h))         ERR(retval);
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
  if (retval = nc_put_var(file, periodic,      &idum))                   ERR(retval);

  idum = pars_->local_limit ? 1 : 0;
  if (retval = nc_put_var(file, local_limit,   &pars_->local_limit))     ERR(retval);
}

NetCDF_ids::~NetCDF_ids() {

  if (Wm_d)         cudaFree     ( Wm_d     );
  if (Wm_h)         cudaFreeHost ( Wm_h     );

  if (Wl_d)         cudaFree     ( Wl_d     );
  if (Wl_h)         cudaFreeHost ( Wl_h     );

  if (Wlm_d)        cudaFree     ( Wlm_d     );
  if (Wlm_h)        cudaFreeHost ( Wlm_h     );

  if (Ws_d)         cudaFree     ( Ws_d      );
  if (Ws_h)         cudaFreeHost ( Ws_h      );
  if (Wz_d)         cudaFree     ( Wz_d      );
  if (Wz_h)         cudaFreeHost ( Wz_h      );

  if (Ps_d)         cudaFree     ( Ps_d      );
  if (Ps_h)         cudaFreeHost ( Ps_h      );
  if (Pz_d)         cudaFree     ( Pz_d      );
  if (Pz_h)         cudaFreeHost ( Pz_h      );

  if (As_d)         cudaFree     ( As_d      );
  if (As_h)         cudaFreeHost ( As_h      );
  if (Az_d)         cudaFree     ( Az_d      );
  if (Az_h)         cudaFreeHost ( Az_h      );

  if (Wkz_d)        cudaFree     ( Wkz_d     );
  if (tmp_Wkz_h)    cudaFreeHost ( tmp_Wkz_h );
  if (Wkz_h)        cudaFreeHost ( Wkz_h     );

  if (Wky_d)        cudaFree     ( Wky_d     );
  if (tmp_Wky_h)    cudaFreeHost ( tmp_Wky_h );
  if (Wky_h)        cudaFreeHost ( Wky_h     );

  if (Wkx_d)        cudaFree     ( Wkx_d     );
  if (tmp_Wkx_h)    cudaFreeHost ( tmp_Wkx_h );
  if (Wkx_h)        cudaFreeHost ( Wkx_h     );

  if (Pkz_d)        cudaFree     ( Pkz_d     );
  if (tmp_Pkz_h)    cudaFreeHost ( tmp_Pkz_h );
  if (Pkz_h)        cudaFreeHost ( Pkz_h     );

  if (Pky_d)        cudaFree     ( Pky_d     );
  if (tmp_Pky_h)    cudaFreeHost ( tmp_Pky_h );
  if (Pky_h)        cudaFreeHost ( Pky_h     );

  if (Pkx_d)        cudaFree     ( Pkx_d     );
  if (tmp_Pkx_h)    cudaFreeHost ( tmp_Pkx_h );
  if (Pkx_h)        cudaFreeHost ( Pkx_h     );

  if (Akz_d)        cudaFree     ( Akz_d     );
  if (tmp_Akz_h)    cudaFreeHost ( tmp_Akz_h );
  if (Akz_h)        cudaFreeHost ( Akz_h     );

  if (Aky_d)        cudaFree     ( Aky_d     );
  if (tmp_Aky_h)    cudaFreeHost ( tmp_Aky_h );
  if (Aky_h)        cudaFreeHost ( Aky_h     );

  if (Akx_d)        cudaFree     ( Akx_d     );
  if (tmp_Akx_h)    cudaFreeHost ( tmp_Akx_h );
  if (Akx_h)        cudaFreeHost ( Akx_h     );

  if (Wkxky_d)      cudaFree     ( Wkxky_d     );
  if (Wkxky_h)      cudaFreeHost ( Wkxky_h     );
  if (tmp_Wkxky_h)  cudaFreeHost ( tmp_Wkxky_h );

  if (Pkxky_d)      cudaFree     ( Pkxky_d     );
  if (tmp_Pkxky_h)  cudaFreeHost ( tmp_Pkxky_h );
  if (Pkxky_h)      cudaFreeHost ( Pkxky_h     );

  if (Akxky_d)      cudaFree     ( Akxky_d     );
  if (tmp_Akxky_h)  cudaFreeHost ( tmp_Akxky_h );
  if (Akxky_h)      cudaFreeHost ( Akxky_h     );

  if (primary)      cudaFreeHost ( primary   );
  if (secondary)    cudaFreeHost ( secondary );
  if (tertiary)     cudaFreeHost ( tertiary  );

  if (qs_d)         cudaFree     ( qs_d      );
  if (qs_h)         cudaFreeHost ( qs_h      );
  
  if (tmp_omg_h)    cudaFreeHost ( tmp_omg_h );
  if (omg_h)        cudaFreeHost ( omg_h     );

  if (red)          delete red;
  if (pot)          delete pot;
  if (ph2)          delete ph2;
  if (all_red)      delete all_red;
}

void NetCDF_ids::write_nc(int ncid, nca D, const float *data, bool endrun) {
  int retval;

  if (endrun) {
    if (retval=nc_put_vara(ncid, D.idx, D.start, D.count, data)) ERR(retval);
  } else {   
    if (retval=nc_put_vara(ncid, D.time, D.time_start, D.time_count, data)) ERR(retval);
  }
}

void NetCDF_ids::write_nc(int ncid, nca D, const double data, bool endrun) {
  int retval;

  if (endrun) {
    if (retval=nc_put_vara(ncid, D.idx, D.start, D.count, &data)) ERR(retval);
  } else {   
    if (retval=nc_put_vara(ncid, D.time, D.time_start, D.time_count, &data)) ERR(retval);
  }
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
  if (Pky.write_v_time || (Pky.write && endrun)) {
    int i = grids_->Nyc*grids_->Nspecies;

    pot->pSum(P2, Pky_d, PSPECTRA_ky);               CP_TO_CPU(tmp_Pky_h, Pky_d, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      for (int ik = 0; ik < grids_->Naky; ik++) {
	Pky_h[ik + is*grids_->Naky] = tmp_Pky_h[ik + is*grids_->Nyc];
      }
    }
    write_nc(file, Pky, Pky_h, endrun);      Pky.increment_ts();  
  }
}

void NetCDF_ids::write_Pkx(float* P2, bool endrun)
{
  if (Pkx.write_v_time || (Pkx.write && endrun)) {
    int i = grids_->Nx*grids_->Nspecies;             int NK = (grids_->Nx-1)/3+1;
    
    pot->pSum(P2, Pkx_d, PSPECTRA_kx);               CP_TO_CPU(tmp_Pkx_h, Pkx_d, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it = 0;
      int itp = it +   grids_->Nx/3;
      Pkx_h[itp + is*grids_->Nakx] = tmp_Pkx_h[it  + is*grids_->Nx];
      
      for (int it = 1; it < NK; it++) {
	int itp = it +   grids_->Nx/3;
	int itm = it + 2*grids_->Nx/3;
	Pkx_h[itp  + is*grids_->Nakx] = tmp_Pkx_h[it  + is*grids_->Nx];
	Pkx_h[it-1 + is*grids_->Nakx] = tmp_Pkx_h[itm + is*grids_->Nx];	
      }
    }  
    write_nc(file, Pkx, Pkx_h, endrun);      Pkx.increment_ts();  
  }
}

void NetCDF_ids::write_Pz(float* P2, bool endrun)
{
  if (Pz.write_v_time || (Pz.write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;
    
    pot->pSum(P2, Pz_d, PSPECTRA_z);         CP_TO_CPU(Pz_h, Pz_d, sizeof(float)*i);
    write_nc(file, Pz, Pz_h, endrun);        Pz.increment_ts();
  }
}

void NetCDF_ids::write_Pkz(float* P2, bool endrun)
{
  if (Pkz.write_v_time || (Pkz.write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;     int Nz = grids_->Nz;
    
    pot->pSum(P2, Pkz_d, PSPECTRA_kz);       CP_TO_CPU(tmp_Pkz_h, Pkz_d, sizeof(float)*i);

    for (int is = 0; is < grids_->Nspecies; is++) {
      if (Nz>1) {
	for (int i = 0; i < Nz; i++) Pkz_h[i + is*Nz] = tmp_Pkz_h[ (i + Nz/2 + 1) % Nz + is*Nz ];
      } else {
	for (int i = 0; i < Nz; i++) Pkz_h[i + is*Nz] = tmp_Pkz_h[ i + is*Nz ];
      }
    }
    
    write_nc(file, Pkz, Pkz_h, endrun);      Pkz.increment_ts();
  }
}

void NetCDF_ids::write_Pkxky(float* P2, bool endrun)
{
  if (Pkxky.write_v_time || (Pkxky.write && endrun)) {

    int i = grids_->Nyc*grids_->Nx*grids_->Nspecies; int NK = (grids_->Nx-1)/3+1;
    pot->pSum(P2, Pkxky_d, PSPECTRA_kxky);
    CP_TO_CPU(tmp_Pkxky_h, Pkxky_d, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it = 0;
      int itp = it +   grids_->Nx/3;
      for (int ik = 0; ik < grids_->Naky; ik++) {
	int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
	Pkxky_h[Qp] = tmp_Pkxky_h[Rp];
      }	
      for (int it = 1; it < NK; it++) {
	int itp = it +   grids_->Nx/3;
	int itm = it + 2*grids_->Nx/3;
	
	for (int ik = 0; ik < grids_->Naky; ik++) {

	  int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;

	  int Qm = it-1 + ik *grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rm = ik   + itm*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;

	  Pkxky_h[Qp] = tmp_Pkxky_h[Rp];
	  Pkxky_h[Qm] = tmp_Pkxky_h[Rm];
	}
      }
    }
    write_nc(file, Pkxky, Pkxky_h, endrun);     Pkxky.increment_ts();  
  }
}

void NetCDF_ids::write_Wz(float *G2, bool endrun)
{
  if (Wz.write_v_time || (Wz.write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;
    
    red->Sum(G2, Wz_d, WSPECTRA_z);          CP_TO_CPU(Wz_h, Wz_d, sizeof(float)*i);
    write_nc(file, Wz, Wz_h, endrun);        Wz.increment_ts();
  }
}

void NetCDF_ids::write_Wkz(float *G2, bool endrun)
{
  if (Wkz.write_v_time || (Wkz.write && endrun)) {
    int i = grids_->Nz*grids_->Nspecies;     int Nz = grids_->Nz;
    
    red->Sum(G2, Wkz_d, WSPECTRA_kz);        CP_TO_CPU(tmp_Wkz_h, Wkz_d, sizeof(float)*i);

    for (int is = 0; is < grids_->Nspecies; is++) {
      if (Nz>1) {
	for (int i = 0; i < Nz; i++) Wkz_h[i+is*Nz] = tmp_Wkz_h[ (i + Nz/2 + 1) % Nz + is*Nz ];
      } else {
	for (int i = 0; i < Nz; i++) Wkz_h[i+is*Nz] = tmp_Wkz_h[ i + is*Nz ];
      }
    }
    
    write_nc(file, Wkz, Wkz_h, endrun);      Wkz.increment_ts();
  }
}

void NetCDF_ids::write_Ws(float* G2, bool endrun)
{
  if (Ws.write_v_time) {
    red->Sum(G2, Ws_d, WSPECTRA_species);    CP_TO_CPU(Ws_h, Ws_d, sizeof(float)*grids_->Nspecies);
    write_nc(file, Ws, Ws_h, endrun);        Ws.increment_ts();

    if (Wtot.write_v_time) {
      for (int is=0; is < grids_->Nspecies; is++) totW += Ws_h[is];
    }
  }
}

void NetCDF_ids::write_Wky(float* G2, bool endrun)
{
  if (Wky.write_v_time || (Wky.write && endrun)) {
    int i = grids_->Nyc*grids_->Nspecies;
    
    red->Sum(G2, Wky_d, WSPECTRA_ky);                CP_TO_CPU(tmp_Wky_h, Wky_d, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      for (int ik = 0; ik < grids_->Naky; ik++) {
	Wky_h[ik + is*grids_->Naky] = tmp_Wky_h[ik + is*grids_->Nyc];
      }
    }
    write_nc(file, Wky, Wky_h, endrun);      Wky.increment_ts();
  }
}

void NetCDF_ids::write_Wkx(float* G2, bool endrun)
{  
  if (Wkx.write_v_time || (Wkx.write && endrun)) {
    int i = grids_->Nx*grids_->Nspecies;             int NK = (grids_->Nx-1)/3+1;
    
    red->Sum(G2, Wkx_d, WSPECTRA_kx);                CP_TO_CPU(tmp_Wkx_h, Wkx_d, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it = 0;
      int itp = it +   grids_->Nx/3;
      Wkx_h[itp + is*grids_->Nakx] = tmp_Wkx_h[it  + is*grids_->Nx];
      
      for (int it = 1; it < NK; it++) {
	int itp = it +   grids_->Nx/3;
	int itm = it + 2*grids_->Nx/3;
	Wkx_h[itp  + is*grids_->Nakx] = tmp_Wkx_h[it  + is*grids_->Nx];
	Wkx_h[it-1 + is*grids_->Nakx] = tmp_Wkx_h[itm + is*grids_->Nx];
      }
    }
    write_nc(file, Wkx, Wkx_h, endrun);      Wkx.increment_ts();  
  }
}

void NetCDF_ids::write_Wkxky(float* G2, bool endrun)
{
  if (Wkxky.write_v_time || (Wkxky.write && endrun)) {
    int i = grids_->Nyc*grids_->Nx*grids_->Nspecies; int NK = (grids_->Nx-1)/3+1;
    
    red->Sum(G2, Wkxky_d, WSPECTRA_kxky);            CP_TO_CPU(tmp_Wkxky_h, Wkxky_d, sizeof(float)*i);
    
    for (int is = 0; is < grids_->Nspecies; is++) {
      int it = 0;
      int itp = it +   grids_->Nx/3;
      for (int ik = 0; ik < grids_->Naky; ik++) {     
	int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
	Wkxky_h[Qp] = tmp_Wkxky_h[Rp];
      }
      
      for (int it = 1; it < NK; it++) {
	int itp = it +   grids_->Nx/3;
	int itm = it + 2*grids_->Nx/3;
	for (int ik = 0; ik < grids_->Naky; ik++) {     

	  int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;

	  int Qm = it-1  + ik *grids_->Nakx + is*grids_->Naky*grids_->Nakx;
	  int Rm = ik    + itm*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;

	  Wkxky_h[Qp] = tmp_Wkxky_h[Rp];
	  Wkxky_h[Qm] = tmp_Wkxky_h[Rm];
	}
      }
    }
    write_nc(file, Wkxky, Wkxky_h, endrun);  Wkxky.increment_ts();  
  }
}

void NetCDF_ids::write_Wm(float* G2, bool endrun)
{
  if (Wm.write_v_time || (Wm.write && endrun)) {  
    int i = grids_->Nm*grids_->Nspecies;

    red-> Sum(G2, Wm_d, WSPECTRA_m);         CP_TO_CPU(Wm_h, Wm_d, sizeof(float)*i);
    write_nc(file, Wm,  Wm_h, endrun);       Wm.increment_ts();
  }
}

void NetCDF_ids::write_Wlm(float* G2, bool endrun)
{
  if (Wlm.write_v_time || (Wlm.write && endrun)) {  
    int i = grids_->Nmoms*grids_->Nspecies;
    
    red->Sum(G2, Wlm_d, WSPECTRA_lm);        CP_TO_CPU(Wlm_h, Wlm_d, sizeof(float)*i);
    write_nc(file, Wlm, Wlm_h, endrun);      Wlm.increment_ts();
  }
}

void NetCDF_ids::write_Wl(float* G2, bool endrun)
{
  if (Wl.write_v_time || (Wl.write && endrun)) {  
    int i = grids_->Nl*grids_->Nspecies;
    
    red->Sum(G2, Wl_d, WSPECTRA_l);          CP_TO_CPU(Wl_h, Wl_d, sizeof(float)*i);
    write_nc(file, Wl, Wl_h, endrun);        Wl.increment_ts();
  }
}

void NetCDF_ids::write_Ps(float* P2, bool endrun)
{
  if (Ps.write_v_time) {
    
    pot->pSum(P2, Ps_d, PSPECTRA_species);     CP_TO_CPU(Ps_h, Ps_d, sizeof(float)*grids_->Nspecies);
    write_nc(file, Ps, Ps_h, endrun);          Ps.increment_ts();

    if (Wtot.write_v_time) {
      totW = 0.;
      for (int is=0; is < grids_->Nspecies; is++) totW += Ps_h[is];
    }
  }
    
}

void NetCDF_ids::write_Aky(float* P2, bool endrun)
{
  if (Aky.write_v_time || (Aky.write && endrun)) {
    int i = grids_->Naky;
    
    ph2->iSum(P2, Aky_d, ASPECTRA_ky);       CP_TO_CPU(Aky_h, Aky_d, sizeof(float)*i);
    write_nc(file, Aky, Aky_h, endrun);      Aky.increment_ts();  
  }
}

void NetCDF_ids::write_Az(float* P2, bool endrun)
{
  if (Az.write_v_time || (Az.write && endrun)) {
    int i = grids_->Nz;
    
    ph2->iSum(P2, Az_d, ASPECTRA_z);         CP_TO_CPU(Az_h, Az_d, sizeof(float)*i);
    write_nc(file, Az, Az_h, endrun);        Az.increment_ts();
  }
}

void NetCDF_ids::write_Akz(float* P2, bool endrun)
{
  if (Akz.write_v_time || (Akz.write && endrun)) {
    int Nz = grids_->Nz;
    
    ph2->iSum(P2, Akz_d, ASPECTRA_kz);       CP_TO_CPU(tmp_Akz_h, Akz_d, sizeof(float)*Nz);

    if (Nz>1) {
      for (int i = 0; i < Nz; i++) Akz_h[i] = tmp_Akz_h[ (i + Nz/2 + 1) % Nz ];
    } else {
      for (int i = 0; i < Nz; i++) Akz_h[i] = tmp_Akz_h[ i ];
    }    
    write_nc(file, Akz, Akz_h, endrun);      Akz.increment_ts();
  }
}

void NetCDF_ids::write_Akx(float* P2, bool endrun)
{
  if (Akx.write_v_time || (Akx.write && endrun)) {
    int i = grids_->Nx;                              int NK = (grids_->Nx-1)/3+1;
    
    ph2->iSum(P2, Akx_d, ASPECTRA_kx);               CP_TO_CPU(tmp_Akx_h, Akx_d, sizeof(float)*i);
    
    int it = 0;
    int itp = it +   grids_->Nx/3;
    Akx_h[itp] = tmp_Akx_h[it ];;
    
    for (int it = 1; it < NK; it++) {
      int itp = it +   grids_->Nx/3;
      int itm = it + 2*grids_->Nx/3;
      
      Akx_h[itp] = tmp_Akx_h[it ];;
      Akx_h[it-1] = tmp_Akx_h[itm];;
    }
    write_nc(file, Akx, Akx_h, endrun);      Akx.increment_ts();  
  }
}

void NetCDF_ids::write_Akxky(float* P2, bool endrun)
{
  if (Akxky.write_v_time || (Akxky.write && endrun)) {
    int i = grids_->Nyc*grids_->Nx; int NK = (grids_->Nx-1)/3+1;
    
    ph2->iSum(P2, Akxky_d, ASPECTRA_kxky);    CP_TO_CPU(tmp_Akxky_h, Akxky_d, sizeof(float)*i);
    
    int it = 0;
    int itp = it +   grids_->Nx/3;
    for (int ik = 0; ik < grids_->Naky; ik++) {
      int Qp = itp + ik*grids_->Nakx ;
      int Rp = ik  + it*grids_->Nyc  ;
      Akxky_h[Qp] = tmp_Akxky_h[Rp];
    }
    
    for (int it = 1; it < NK; it++) {
      int itp = it +   grids_->Nx/3;
      int itm = it + 2*grids_->Nx/3;

      for (int ik = 0; ik < grids_->Naky; ik++) {

	int Qp = itp + ik*grids_->Nakx ;
	int Rp = ik  + it*grids_->Nyc  ;
	
	int Qm = it-1 + ik *grids_->Nakx ;
	int Rm = ik   + itm*grids_->Nyc  ;

	Akxky_h[Qp] = tmp_Akxky_h[Rp];
	Akxky_h[Qm] = tmp_Akxky_h[Rm];
      }
    }
    write_nc(file, Akxky, Akxky_h, endrun);  Akxky.increment_ts();  
  }
}

void NetCDF_ids::write_As(float *P2, bool endrun)
{
  if (As.write_v_time) {  
    ph2->iSum(P2, As_d, ASPECTRA_species);    CP_TO_CPU (As_h, As_d, sizeof(float));
    write_nc(file, As, As_h, endrun);         As.increment_ts();

    if (Wtot.write_v_time) totW += *As_h;
  }
}

void NetCDF_ids::write_Q (float* Q, bool endrun)
{
  if (qs.write_v_time) {
    all_red->sSum(Q, qs_d);                   CP_TO_CPU (qs_h, qs_d, sizeof(float)*grids_->Nspecies);
    write_nc(file, qs, qs_h, endrun);         qs.increment_ts();

    for (int is=0; is<grids_->Nspecies; is++) printf ("%e \t ",qs_h[is]);
    printf("\n");
  }
}

void NetCDF_ids::write_omg(cuComplex *W, bool endrun)
{
  CP_TO_CPU (tmp_omg_h, W, sizeof(cuComplex)*grids_->NxNyc);

  reduce2k(omg_h, tmp_omg_h);
  write_nc(file, omg, omg_h, endrun);
  omg.increment_ts();
}

void NetCDF_ids::write_Wtot()
{
  if (Wtot.write_v_time) {  write_nc(file, Wtot, &totW);        Wtot.increment_ts();     totW = 0.;}
}

void NetCDF_ids::close_nc_file() {
  int retval;
  if (retval = nc_close(file)) ERR(retval);
}

// condense a (ky,kx) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
void NetCDF_ids::reduce2k(float *fk, cuComplex* f) {
  
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  for(int i=0; i < 1+(Nx-1)/3 ; i++) {
    for(int j=0; j<Naky; j++) {
      int index     = j + Nyc * i; 
      int index_out = i+Nx/3 + Nakx * j;
      fk[2*index_out]   = f[index].x;
      fk[2*index_out+1] = f[index].y;
    }
  }
  
  for(int i = 1+2*Nx/3; i < Nx; i++) {
    for(int j=0; j<Naky; j++) {
      int index = j + Nyc *i;
      int index_out = i - 2*Nx/3 - 1 + Nakx * j;
      fk[2*index_out]   = f[index].x;
      fk[2*index_out+1] = f[index].y;
    }
  }	  
}



