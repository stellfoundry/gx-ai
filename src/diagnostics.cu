#include "diagnostics.h"
#include "get_error.h"
#include "netcdf.h"
#include <sys/stat.h>
#define GSPEC <<< dG_spectra, dB_spectra >>>
#define GALL <<< dG_all, dB_all >>>
#define GFLA <<< 1 + (grids_->Nx*grids_->Nyc - 1)/grids_->Nakx, grids_->Nakx >>>	  
#define loop_y <<< dgp, dbp >>> 

Diagnostics::Diagnostics(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo),
  fields_old(nullptr), id(nullptr), grad_par(nullptr), amom_d(nullptr), grad_perp(nullptr)
{
  printf(ANSI_COLOR_BLUE);
  
  int nL  = grids_->Nl;
  int nM  = grids_->Nm;
  int nS  = grids_->Nspecies;
  int nX  = grids_->Nx;
  int nXk = grids_->Nakx;
  int nY  = grids_->Nyc;
  int nYk = grids_->Naky;
  int nZ  = grids_->Nz;
  int nR  = nX  * nY  * nZ;
  int nK  = nXk * nYk * nZ;
  int nG  = nR * grids_->Nmoms * nS;

  favg        = nullptr;  df          = nullptr;  val         = nullptr;  
  G2          = nullptr;  P2s         = nullptr;  Phi2        = nullptr;
  omg_d       = nullptr;  tmp_omg_h   = nullptr;  t_bar       = nullptr;  
  gy_d        = nullptr;  gy_h        = nullptr;
  
  id         = new NetCDF_ids(grids_, pars_, geo_); cudaDeviceSynchronize(); CUDA_DEBUG("NetCDF_ids: %s \n");
  fields_old = new      Fields(pars_, grids_);      cudaDeviceSynchronize(); CUDA_DEBUG("Fields: %s \n");

  if (pars_->Reservoir) rc = new Reservoir(pars_, grids_->NxNyNz*grids->Nmoms);  
  
  volDenom = 0. ;  cudaMallocHost (&vol_fac, sizeof(float) * nZ);
  for (int i=0; i < nZ; i++) volDenom   += geo_->jacobian_h[i]; 
  for (int i=0; i < nZ; i++) vol_fac[i]  = geo_->jacobian_h[i] / volDenom;

  fluxDenom = 0.;  cudaMallocHost (&flux_fac, sizeof(float) * nZ);
  for (int i=0; i<grids_->Nz; i++) fluxDenom   += geo_->jacobian_h[i]*geo_->grho_h[i];
  for (int i=0; i<grids_->Nz; i++) flux_fac[i]  = geo_->jacobian_h[i]*geo_->grho_h[i] / fluxDenom;

  if (pars_->diagnosing_spectra || pars_->diagnosing_kzspec) cudaMalloc (&G2, sizeof(float) * nG); 

  if (pars_->diagnosing_kzspec) {
    cudaMallocHost (&kvol_fac, sizeof(float) * nZ);
    for (int i=0; i < nZ; i++) kvol_fac[i] = 1.0;

    cudaMalloc (&amom_d, sizeof(cuComplex) * nR * nS); 
    if (pars_->local_limit) {
      // nothing, this is not defined, or could be defined as an identity.
    }
    else if(pars_->boundary_option_periodic) {
      grad_par = new GradParallelPeriodic(grids_);
    }
    else {
      grad_par = new GradParallelLinked(grids_, pars_->jtwist);
    }
  }
  // need if (pars_->write_flux || "diagnosing potential) {
  cudaMalloc (&P2s, sizeof(float) * nR * nS);

  if (id->g_y.write_v_time) {
    int nbatch = 1;
    grad_perp = new GradPerp(grids_, nbatch, grids_->Nyc);
  }
      
  if (id->rh.write) cudaMallocHost(&val, sizeof(float)*2);

  if (!pars_->all_kinetic) {

    if (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
      cudaMalloc((void**) &favg, sizeof(cuComplex) * nX);  
      cudaMalloc((void**)   &df, sizeof(cuComplex) * nR);
    }

    cudaMalloc (&Phi2, sizeof(float) * nR);  
  }
  
  if (id->omg.write_v_time) {
    cudaMalloc     (    &omg_d,   sizeof(cuComplex) * nX * nY);//     cudaMemset (omg_d, 0., sizeof(cuComplex) * nX * nY);
    cudaMallocHost (&tmp_omg_h,   sizeof(cuComplex) * nX * nY);
    int nn = nX*nY; int nt = min(nn, 512); int nb = 1 + (nn-1)/nt;  cuComplex zero = make_cuComplex(0.,0.);
    setval <<< nb, nt >>> (omg_d, zero, nn);
  }  
  if (id->g_y.write_v_time) {
    cudaMalloc     (&gy_d,       sizeof(float) * grids_->Ny);
    cudaMallocHost (&gy_h,       sizeof(float) * grids_->Ny);
  }
  
  /*
  if (pars_->diagnosing_pzt) {
    cudaMallocHost (&primary,   sizeof(float));    primary[0] = 0.;  
    cudaMallocHost (&secondary, sizeof(float));    secondary[0] = 0.;
    cudaMallocHost (&tertiary,  sizeof(float));    tertiary[0] = 0.;
    cudaMalloc     (&t_bar,     sizeof(cuComplex) * nR * nS);
  }
  */
    
  // Remember that delta theta is a constant in this formalism!   

  // set up stop file
  sprintf(stopfilename_, "%s.stop", pars_->run_name);

  //  dB_scale = min(512, nR);
  //  dG_scale = 1 + (nR-1)/dB_scale.x;

  dB_spectra = dim3(16, 8, 8);
  dG_spectra = dim3(1 + (nY-1)/dB_spectra.x, 1 + (nX-1)/dB_spectra.y, 1 + (nZ-1)/dB_spectra.z);  

  int nyx =  nY * nX;
  int nslm = nL * nM * nS;

  int nbx = 32;
  int ngx = 1 + (nyx-1)/nbx;

  int nby = 32;
  int ngy = 1 + (grids_->Nz-1)/nby;
  
  dB_all = dim3(nbx, nby, 1);
  dG_all = dim3(ngx, ngy, nslm);

  if (grids_->Nakx > 1024) {printf("Need to redefine GFLA in diagnostics \n"); exit(1);}

  nbx = min(grids_->Ny, 512);
  ngx = 1 + (grids_->Ny-1)/nbx;

  dbp = dim3(nbx, 1, 1);
  dgp = dim3(ngx, 1, 1);
  
  printf(ANSI_COLOR_RESET);
  ndiag = 1;
  Dks = 0.;
}

Diagnostics::~Diagnostics()
{
  if (fields_old) delete fields_old;
  if (id)         delete id;

  if (G2)         cudaFree      ( G2        );
  if (P2s)        cudaFree      ( P2s       );
  if (Phi2)       cudaFree      ( Phi2      );
  if (t_bar)      cudaFree      ( t_bar     );
  if (omg_d)      cudaFree      ( omg_d     );
  if (gy_d)       cudaFree      ( gy_d      );
  if (amom_d)     cudaFree      ( amom_d    );
  
  if (vol_fac)    cudaFreeHost  ( vol_fac   );
  if (flux_fac)   cudaFreeHost  ( flux_fac  );
  if (kvol_fac)   cudaFreeHost  ( kvol_fac  );
  if (val)        cudaFreeHost  ( val       );
  if (tmp_omg_h)  cudaFreeHost  ( tmp_omg_h );
  if (gy_h)       cudaFreeHost  ( gy_h      );
  
}

bool Diagnostics::loop(MomentsG* G, Fields* fields, double dt, int counter, double time) 
{
  int retval;
  bool stop = false;
  int nw;

  nw = pars_->nwrite;

  if ((counter % nw == nw-1) && id->omg.write_v_time) fields_old->copyPhiFrom(fields);

    
  if(counter%nw == 0) {

    fflush(NULL);  id->write_nc(id->file, id->time, time);    id->time.increment_ts(); // save current time
  
    if(id->omg.write_v_time && counter > 0) {                                                  // complex frequencies

      int nt = min(512, grids_->NxNyc) ;
      growthRates <<< 1 + (grids_->NxNyc-1)/nt, nt >>> (fields->phi, fields_old->phi, dt, omg_d);
      
      print_omg(omg_d);  id->write_omg(omg_d);
    }


    if ( id->qs.write_v_time) printf("Step %d: Time = %f \t Flux = ", counter, time);          // To screen
    if (!id->qs.write_v_time) printf("Step %d: Time = %f\n",          counter, time);
  
    if (id->qs.write_v_time) {                                                                 // heat flux
      
      for(int is=0; is<grids_->Nspecies; is++) {
	float rho2s = pars_->species_h[is].rho2;
	heat_flux_summand GSPEC (P2(is), fields->phi, G->G(0,0,is),
				 grids_->ky, flux_fac, geo_->kperp2, rho2s);
      }
      id->write_Q(P2s); 
    }      

    if (pars_->diagnosing_kzspec) {
      grad_par->zft(G); // get G = G(kz)
      W_summand GALL (G2, G->G(), kvol_fac, G->nt());
      //      W_summand GALL (G2, G->G(), kvol_fac, pars_->species); // get G2 = |G(kz)**2|
      grad_par->zft_inverse(G); // restore G

      grad_par->zft(fields->phi, amom_d); // get amom_d = phi(kz)
      
      for (int is=0; is < grids_->Nspecies; is++) {             // P2(s) = (1-G0(s)) |phi**2| for each kinetic species
	float rho2s = pars_->species_h[is].rho2;
	Wphi_summand GSPEC (P2(is), amom_d, kvol_fac, geo_->kperp2, rho2s);
	float qfac =  pars_->species_h[is].qneut;
	Wphi_scale GSPEC   (P2(is), qfac);
      }

      if (pars_->add_Boltzmann_species) {
	if (pars_->Boltzmann_opt == BOLTZMANN_IONS)  Wphi2_summand GSPEC (Phi2, amom_d, kvol_fac);
	
	if (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {	  
	  fieldlineaverage GFLA (favg, df, fields->phi, kvol_fac); // favg is a dummy variable
	  grad_par->zft(df, amom_d); // get df = df(kz)
	  Wphi2_summand GSPEC (Phi2, amom_d, kvol_fac); 	
	}

	float fac = 1./pars_->tau_fac;
	Wphi_scale GSPEC (Phi2, fac);
      }
      
      id->write_Wkz   (G2);    id->write_Pkz   (P2());    id->write_Akz   (Phi2);      
    }
    
    if (pars_->diagnosing_spectra) {                                        // Various spectra
      W_summand GALL (G2, G->G(), vol_fac, G->nt());
      //      W_summand GALL (G2, G->G(), vol_fac, pars_->species);
      
      if (pars_->gx) {
	for (int is=0; is < grids_->Nspecies; is++) {             // P2(s) = (1-G0(s)) |phi**2| for each kinetic species
	  float rho2s = pars_->species_h[is].rho2;
	  Wphi_summand GSPEC (P2(is), fields->phi, vol_fac, geo_->kperp2, rho2s);
	  float qnfac = pars_->species_h[is].qneut;
	  Wphi_scale GSPEC   (P2(is), qnfac);
	}

	if (pars_->add_Boltzmann_species) {
	  if (pars_->Boltzmann_opt == BOLTZMANN_IONS)  Wphi2_summand GSPEC (Phi2, fields->phi, vol_fac);
	  
	  if (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {	  
	    fieldlineaverage GFLA (favg, df, fields->phi, vol_fac); // favg is a dummy variable
	    Wphi2_summand GSPEC (Phi2, df, vol_fac); 	
	  }
	  
	  float fac = 1./pars_->tau_fac;
	  Wphi_scale GSPEC (Phi2, fac);
	}
      }

      if (pars_->ks) {
	cuComplex * g_h;
	cudaMallocHost (&g_h, sizeof(cuComplex)*grids_->Nyc);
	CP_TO_CPU(g_h, G->G(), sizeof(cuComplex)*grids_->Nyc);
	float Dtmp = 0.;
	for (int i=0; i<grids_->Naky; i++) {
	  Dtmp += (g_h[i].x*g_h[i].x + g_h[i].y*g_h[i].y)*grids_->ky_h[i]*grids_->ky_h[i];
	}
	Dks += Dtmp;
	printf("<D> = %f \t",Dks/((float) ndiag));
	ndiag += 1;
	cudaFreeHost  (g_h);
      }
      
      id->write_Wm    (G2   );    id->write_Wl    (G2   );    id->write_Wlm   (G2   );    
      id->write_Wz    (G2   );    id->write_Wky   (G2   );    id->write_Wkx   (G2   );    id->write_Wkxky (G2  );    
      id->write_Pz    (P2() );    id->write_Pky   (P2() );    id->write_Pkx   (P2() );    id->write_Pkxky (P2());    
      id->write_Az    (Phi2 );    id->write_Aky   (Phi2 );    id->write_Akx   (Phi2 );    id->write_Akxky (Phi2);
      /*
      float *dum;
      cudaMallocHost (&dum, sizeof(float)*grids_->NxNycNz);
      CP_TO_CPU (dum, Phi2, sizeof(float)*grids_->NxNycNz);
      //      for (int iz=0; iz<grids_->Nz; iz++) {
      int iz = 0;
      for (int ix=0; ix<grids_->Nx; ix++) {
	  for (int iy=0; iy<grids_->Naky; iy++) {
	    printf("dum(%d, %d, %d) = %e \n",iy, ix, iz, dum[iy+ix*grids_->Nyc+iz*grids_->Nyc*grids_->Nx]);
	  }
	}
	//      }
      cudaFreeHost(dum);
      */
      
      // Do not change the order of these four calls because totW is accumulated in order when it is requested:
      id->write_Ps(P2s);    id->write_Ws(G2);    id->write_As(Phi2);    id->write_Wtot();
    }
    
    // Rosenbluth-Hinton diagnostic
    if(id->rh.write) {
      get_rh(fields);   id->write_nc (id->file, id->rh, val);    id->rh.increment_ts();
    }
    /*
      if( counter%nw == 0 && id->Pzt.write_v_time) {
      pzt(G, fields);  // calculate each of P, Z, and T (very very rough diagnostic)
      cudaDeviceSynchronize();
      
      write_nc (id->file, id->Pzt, primary);
      id->Pzt.increment_ts();
      
      write_nc (id->file, id->pZt, secondary);
      id->pZt.increment_ts();
      
      write_nc (id->file, id->pzT, tertiary);
      id->pZt.increment_ts();
      }
    */

    // calculate g(y) for this timestep if appropriate
    if (id->g_y.write_v_time) {
      //      if (pars_->gx) {
      //	grad_perp->C2R(G->tprp_ptr[0], gy_d);
      //      } else {
	grad_perp->C2R(G->G(), gy_d);
	//      }
      CP_TO_CPU(gy_h, gy_d, sizeof(float)*grids_->Ny);
      
      id->write_gy(gy_h);
    }
    nc_sync(id->file);
  }
  if (pars_->Reservoir && counter%pars_->ResTrainingDelta == 0) {
    grad_perp->C2R(G->G(), gy_d);    rc->add_data(gy_d); 
  }
  
  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

void Diagnostics::finish(MomentsG* G, Fields* fields, double time) 
{
  if (pars_->Reservoir && rc->predicting()) {
    grad_perp -> C2R (G->G(), gy_d);
    double *gy_double;
    cudaMalloc(&gy_double, sizeof(double)*grids_->Ny);
    promote loop_y (gy_double, gy_d, grids_->Ny);
    
    for (int i=0; i<pars_->ResPredict_Steps; i++) {
      rc->predict(gy_double);
      time += pars_->dt * pars_->ResTrainingDelta;
      demote loop_y (gy_d, gy_double, grids_->Ny); 
      if (id->g_y.write_v_time) {      
	id->write_nc(id->file, id->time, time);
	id->time.increment_ts();
	CP_TO_CPU(gy_h, gy_d, sizeof(float)*grids_->Ny);
	id->write_gy(gy_h);
      }
      
    }
  }
  //
  //
  // could put some real space figure data here, or some other one-point-in-time diagnostics
  //
  //
  
  /*
  if (pars_->diagnosing_moments)  {

    if (id->den.write)   writeMomOrField (G->dens_ptr[0], id->den);
    if (id->wphi.write)  writeMomOrField (fields->phi, id->wphi);
    
    if (id->wphik.write) {
      grad_parallel->fft_only(fields->phi, amom_d, CUFFT_FORWARD);
      writeMomOrField (amom_d, id->wphik);
    }
    
    if (id->denk.write) {
      grad_parallel->fft_only(G->dens_ptr[0], amom_d, CUFFT_FORWARD);
      writeMomOrField (amom_d, id->denk); 
    }
  }
  */
  id->close_nc_file();  fflush(NULL);

}
/*
void Diagnostics::write_init(MomentsG* G, Fields* fields) {
  if (id->den.write)  writeMomOrField (G->dens_ptr[0], id->den0); 
  if (id->wphi.write) writeMomOrField (fields->phi,    id->wphi0);  
}
*/

/*
void Diagnostics::writeMomOrField(cuComplex* m, nca lid) { 
  int retval;
  int i = grids_->NxNycNz;
  int j = i * grids_->Nmoms;
  int k = 2 * grids_->Nakx*grids_->Naky*grids_->Nz*grids_->Nmoms;
  
  for (int is=0; is<lid.ns; is++) {
    CP_TO_CPU(tmp_amom_h, &m[is*j], sizeof(cuComplex)*i);
    reduce2z(&amom_h[is*k], tmp_amom_h);
  }

  if (retval = nc_put_vara(id->file, lid.idx, lid.start, lid.count, amom_h)) ERR(retval);
}
*/

void Diagnostics::print_omg(cuComplex *W)
{
  CP_TO_CPU (tmp_omg_h, W, sizeof(cuComplex)*grids_->NxNyc);
  print_growth_rates_to_screen(tmp_omg_h);
}




  // For each kx, z, l and m, sum the moments of G**2 + Phi**2 (1-Gamma_0) with weights:
  //
  // The full sum is 
  //
  // sum_s sum_l,m sum_ky sum_kx sum_z J(z) == SUM
  //
  // and there is a denominator for the z-integration, to normalize out the volume,
  // SUM => SUM / sum J(z)
  // 
  // W = SUM n_s tau_s G**2 / 2 + SUM n_s Z_s**2 / (2 tau_s) (1 - Gamma_0) Phi**2
  //
  // Since we store only half of the spatial Fourier coefficients for all but the
  // ky = 0 modes
  //
  // (for which we have the full set of kx modes, kx > 0 and kx < 0, and everything is zero for kx = ky = 0)  
  //
  // we multiply all the ky != 0 modes by a factor of two. This is easiest to see in device_functions/vol_summand.
  //
  // To make later manipulations easiest, all partial summations will be carried such that a simple sum of
  // whatever is leftover will give W.
  //

void Diagnostics::get_rh(Fields* f)
{
    ikx_local = 1; iky_local = 0; iz_local=grids_->Nz/2; // correct values for usual RH tests

    int idx = iky_local + grids_->Nyc*ikx_local + grids_->NxNyc*iz_local;
  
    CP_TO_CPU(&valphi, &f->phi[idx], sizeof(cuComplex));
    val[0] = valphi.x;
    val[1] = valphi.y;
}



bool Diagnostics::checkstop() 
{
  struct stat buffer;   
  bool stop = (stat (stopfilename_, &buffer) == 0);
  if (stop) remove(stopfilename_);
  return stop;
}


// condense a (ky,kx,z) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
/*
void Diagnostics::reduce2z(float *fz, cuComplex* f) {

  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;

  for (int k=0; k<Nz; k++) {
    for (int i=0; i<(Nx-1)/3+1; i++) {
      for (int j=0; j<Naky; j++) {
	int index     = j + Nyc *i + Nyc*Nx*k; 
	int index_out = j + Naky*i + Naky*Nakx*k; 
	fz[2*index_out]   = f[index].x;
	fz[2*index_out+1] = f[index].y;
      }
    }
  }
  
  for (int k=0; k<Nz; k++) {
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<Naky; j++) {
	int index     = j + Nyc *i + Nyc*Nx*k;
	int index_out = j + Naky*( i - 2*Nx/3 + (Nx-1)/3 ) + Naky*Nakx*k;
	fz[2*index_out]   = f[index].x;
	fz[2*index_out+1] = f[index].y;
      }
    }	
  }	
}
*/

void Diagnostics::print_growth_rates_to_screen(cuComplex* w)
{
  int Nx = grids_->Nx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  printf("ky\tkx\t\tomega\t\tgamma\n");

  for(int j=0; j<Naky; j++) {
    for(int i= 1 + 2*Nx/3; i<Nx; i++) {
      int index = j + Nyc*i;
      printf("%.4f\t%.4f\t\t%.6f\t%.6f",  grids_->ky_h[j], grids_->kx_outh[i], w[index].x, w[index].y);
      printf("\n");
    }
    for(int i=0; i < 1 + (Nx-1)/3; i++) {
      int index = j + Nyc*i;
      if(index!=0) {
	printf("%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_outh[i], w[index].x, w[index].y);
	printf("\n");
      } else {
	printf("%.4f\t%.4f\n", grids_->ky_h[j], grids_->kx_outh[i]);
      }
    }
    if (Nx>1) printf("\n");
  }
}


