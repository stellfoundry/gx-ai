#include "diagnostics.h"
#include "device_funcs.h"
#include "cuda_constants.h"
#include "get_error.h"
#include "gx_lib.h"
#include "netcdf.h"
#include <sys/stat.h>

__global__ void growthRates(cuComplex *phi, cuComplex *phiOld, float dt, cuComplex *omega)
{
  unsigned int idxy = get_id1();
  cuComplex i_dt = make_cuComplex(0., 1./dt);
  int J = nx*nyc;
  int IG = (int) .5*nz ;
  
  if ( idxy<J && idxy > 0) {
    if (abs(phi[idxy+J*IG].x)!=0 && abs(phi[idxy+J*IG].y)!=0) {
      cuComplex ratio = phi[ idxy + J*IG ] / phiOld[ idxy + J*IG ];
      
      cuComplex logr;
      logr.x = (float) log(cuCabsf(ratio));
      logr.y = (float) atan2(ratio.y,ratio.x);
      omega[idxy] = logr*i_dt;
    } else {
      omega[idxy].x = 0.;
      omega[idxy].y = 0.;
    }
  }
}


Diagnostics::Diagnostics(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{  
  fields_old = new Fields(grids_);
  cudaDeviceSynchronize();
  CUDA_DEBUG("Fields: %s \n");

  id = new NetCDF_ids(grids_, pars_, geo_);
  cudaDeviceSynchronize(); 
  CUDA_DEBUG("NetCDF_ids: %s \n");
  
  grad_parallel = new GradParallelPeriodic(grids_);
  cudaDeviceSynchronize();
  CUDA_DEBUG("Grad parallel periodic: %s \n");

  if (pars_->write_omega) {
    cudaMalloc    ((void**) &growth_rates,   sizeof(cuComplex)*grids_->NxNyc);
    cudaMallocHost((void**) &growth_rates_h, sizeof(cuComplex)*grids_->NxNyc);

    cudaMemset(growth_rates, 0., sizeof(cuComplex)*grids_->NxNyc);
  }
  
  cudaMallocHost((void**) &amom_h, sizeof(cuComplex)*grids_->NxNycNz);
  cudaMalloc((void**) &amom, sizeof(cuComplex)*grids_->NxNycNz);

  if (pars_->write_fluxes) {
    cudaMallocHost((void**) &pflux,  sizeof(float)*grids_->Nspecies);
    cudaMallocHost((void**) &qflux,  sizeof(float)*grids_->Nspecies);
  }
  cudaMallocHost((void**) &val,    sizeof(float)*2);
    
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  maxThreadsPerBlock_ = prop.maxThreadsPerBlock;

  // for volume averaging
  fluxDenom = 0.;
  for(int i=0; i<grids_->Nz; i++) {
    fluxDenom += geo_->jacobian_h[i]*geo_->grho_h[i];
  }
  
  // set up stop file
  sprintf(stopfilename_, "%s.stop", pars_->run_name);

}

Diagnostics::~Diagnostics()
{
  delete fields_old;
  cudaFree(growth_rates);
  cudaFreeHost(growth_rates_h); 
  cudaFreeHost(pflux);
  cudaFreeHost(qflux);
  cudaFree(val);
}

bool Diagnostics::loop_diagnostics(MomentsG* G, Fields* fields, float dt, int counter, double time) 
{
  int retval;
  bool stop = false;
  int nw;
  nw = pars_->nwrite;
  if(counter%nw == 0) {
    printf("Step %d: Time = %f\n", counter, time);
    fflush(NULL);
    if (retval = nc_put_vara(id->file, id->time, id->time_start, id->time_count,  &time))    ERR(retval);
    id->time_start[0] += 1; 
  }
  // write instantaneous growth rates
  if(pars_->write_omega) {
    if (counter%nw == nw-1 || counter%nw == 0) {
      growthRates<<<grids_->NxNyc/maxThreadsPerBlock_+1, maxThreadsPerBlock_>>>
	(fields->phi, fields_old->phi, dt, growth_rates);
      // save fields for next time
      fields_old->copyFrom(fields);
    }

    if(counter > 0 && counter%nw == 0) {
      CP_TO_CPU (growth_rates_h, growth_rates, sizeof(cuComplex)*grids_->NxNyc);
      print_growth_rates_to_screen();
      writeGrowthRates();
    }
  }

  if (counter%nw == 0 && pars_->write_spec_v_time) {
    if (pars_->write_h_spectrum)
                       writeHspectrum (G, false, pars_->ikx_single, pars_->iky_single);
    if (pars_->write_l_spectrum)
                       writeLspectrum (G, false, pars_->ikx_single, pars_->iky_single);
    if (pars_->write_lh_spectrum)
                       writeLHspectrum(G, false, pars_->ikx_single, pars_->iky_single);
  }

  // write time history of phi
  if(counter%nw == 0 && pars_->write_rh) {
    ikx_local = 1; iky_local = 0; iz_local=grids_->Nz/2; // correct values for usual RH tests
    //    ikx_local = 0; iky_local = 2; iz_local=grids_->Nz/2; // test values

    CP_TO_CPU(&valphi, &fields->phi[iky_local + grids_->Nyc*ikx_local + grids_->NxNyc*iz_local], sizeof(cuComplex));
    val[0] = valphi.x; val[1] = valphi.y;
    
    if (retval = nc_put_vara(id->file, id->phi_rh, id->rh_start, id->rh_count,  val))    ERR(retval);
    id->rh_start[0] += 1;
  }

  if( counter%nw == 0 && pars_->write_fluxes) {
    // calculate fluxes
    fluxes(G, fields);
    if (retval = nc_put_vara(id->file, id->qflux, id->flux_start, id->flux_count,  &qflux[0]))    ERR(retval);
    id->flux_start[0] += 1; 
  }

  // sync the netcdf data 
  if (counter%nw == 0) {
    if (retval = nc_sync(id->file)) ERR(retval);
  }

  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

void Diagnostics::final_diagnostics(MomentsG* G, Fields* fields) 
{
  // print final moments and fields

  if (pars_->write_moms) {
    writeMomOrField (G->dens_ptr[0], id->density);
    writeMomOrField (G->upar_ptr[0], id->upar);
    writeMomOrField (fields->phi,    id->phi);
  }
  
  if (pars_->write_phi_kpar) {

    grad_parallel->fft_only(G->dens_ptr[0], amom, CUFFT_FORWARD);
    writeMomOrField (amom, id->density_kpar);

    grad_parallel->fft_only(fields->phi, amom, CUFFT_FORWARD);
    writeMomOrField (amom, id->phi_kpar);
  }
  
  if (pars_->write_h_spectrum)
             writeHspectrum (G, true, pars_->ikx_single, pars_->iky_single);
  if (pars_->write_l_spectrum)
             writeLspectrum (G, true, pars_->ikx_single, pars_->iky_single);
  if (pars_->write_lh_spectrum)
             writeLHspectrum(G, true, pars_->ikx_single, pars_->iky_single);

  id->close_nc_file();
  fflush(NULL);
}


//void Diagnostics::write_something(NetCDF_ids* id)
void Diagnostics::write_something()
{
  printf("The current value of the nx handle is %i \n",id->nx);
}

void Diagnostics::writeGrowthRates()
{
  int retval;
  int Nx = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  float *omt_out;
  
  if (id->mask) {
    cudaMallocHost((void**) &omt_out, sizeof(float)*Nakx*Naky*2);
  } else {
    cudaMallocHost((void**) &omt_out, sizeof(float)*Nx*Nyc*2);
  }

  reduce2k(omt_out, growth_rates_h);

  if (retval = nc_put_vara(id->file, id->omega_t, id->omt_start, id->omt_count,  omt_out))    ERR(retval);
  id->omt_start[0] += 1;

  cudaFreeHost(omt_out); 
}

void Diagnostics::write_init(MomentsG* G, Fields* fields) {
  writeMomOrField (G->dens_ptr[0], id->density0);
  writeMomOrField (fields->phi,    id->phi0);  
}

void Diagnostics::writeMomOrField(cuComplex* m, int handle) {

  int retval;
  int Nx  = grids_->Nx;
  int Nyc = grids_->Nyc;
  int Nz  = grids_->Nz;

  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  float *mom_out;
  
  CP_TO_CPU(amom_h, m, sizeof(cuComplex)*Nx*Nyc*Nz);

  if (id->mask) {
    cudaMallocHost((void**) &mom_out, sizeof(float)*Nakx*Naky*Nz*2);
  } else {
    cudaMallocHost((void**) &mom_out, sizeof(float)*Nx*Nyc*Nz*2);
  }

  reduce2z(mom_out, amom_h);

  if (retval = nc_put_vara(id->file, handle, id->mom_start, id->mom_count, mom_out)) ERR(retval);

  cudaFreeHost(mom_out); 
}

void Diagnostics::writeLHspectrum(MomentsG* G, bool endrun, int ikx, int iky)
{
  int retval;
  float *lhspectrum, *lhspectrum_h; 
  cudaMalloc    ((void**) &lhspectrum,     sizeof(float)*grids_->Nmoms);
  cudaMallocHost((void**) &lhspectrum_h,   sizeof(float)*grids_->Nmoms);

  // calculate spectrum  
  LHspectrum(G, lhspectrum, ikx, iky);

  CP_TO_CPU(lhspectrum_h, lhspectrum, sizeof(float)*grids_->Nmoms);
  cudaFree(lhspectrum);

  if (endrun) {
    if (retval=nc_put_vara(id->file, id->lhspec, id->lh_start, id->lh_count, lhspectrum_h)) ERR(retval);
  } else {   
    if (retval=nc_put_vara(id->file, id->lhspec_t, id->lht_start, id->lht_count, lhspectrum_h)) ERR(retval);
    id->lht_start[0] += 1;
  }
  cudaFreeHost(lhspectrum_h);
}

void Diagnostics::writeLspectrum(MomentsG* G, bool endrun, int ikx, int iky)
{
  int retval;
  float *lspectrum, *lspectrum_h; 
  cudaMalloc    ((void**) &lspectrum,     sizeof(float)*grids_->Nl);
  cudaMallocHost((void**) &lspectrum_h,   sizeof(float)*grids_->Nl);

  // calculate spectrum  
  Lspectrum(G, lspectrum, ikx, iky);

  CP_TO_CPU(lspectrum_h, lspectrum, sizeof(float)*grids_->Nl);
  cudaFree(lspectrum);
  
  if (endrun) {
    if (retval=nc_put_vara(id->file, id->lspec, id->l_start, id->l_count, lspectrum_h)) ERR(retval);
  } else {
    if (retval=nc_put_vara(id->file, id->lspec_t, id->lt_start, id->lt_count, lspectrum_h)) ERR(retval);
    id->lt_start[0] += 1;
  }
  cudaFreeHost(lspectrum_h);
}

void Diagnostics::writeHspectrum(MomentsG* G, bool endrun, int ikx, int iky)
{
  int retval;
  float *hspectrum, *hspectrum_h; 
  cudaMalloc    ((void**) &hspectrum,     sizeof(float)*grids_->Nm);
  cudaMallocHost((void**) &hspectrum_h,   sizeof(float)*grids_->Nm);

  // calculate spectrum  
  Hspectrum(G, hspectrum, ikx, iky);

  CP_TO_CPU(hspectrum_h, hspectrum, sizeof(float)*grids_->Nm);
  cudaFree(hspectrum);
  
  if (endrun) {
    if (retval=nc_put_vara(id->file, id->hspec, id->m_start, id->m_count, hspectrum_h)) ERR(retval);
  } else {
    if (retval=nc_put_vara(id->file, id->hspec_t, id->mt_start, id->mt_count, hspectrum_h)) ERR(retval);
    id->mt_start[0] += 1;
  }    
  cudaFreeHost(hspectrum_h);
}

// sum over ky, kz, z with flux surface average
__global__ void volume_average(float* res, cuComplex* f, cuComplex* g, float* jacobian, float fluxDenomInv, int ikx=-1, int iky=-1) {
  // reduction code follows https://github.com/parallel-forall/code-samples/blob/master/posts/parallel_reduction_with_shfl/device_reduce_atomic.h
  // device_reduce_atomic_kernel
  float sum = 0.;
  cuComplex fg;
  for(int idxyz=blockIdx.x*blockDim.x+threadIdx.x;idxyz<nx*nyc*nz;idxyz+=blockDim.x*gridDim.x) {
    unsigned int idy = idxyz % (nx*nyc) % nyc; 
    unsigned int idx = idxyz % (nx*nyc) / nyc; 
    unsigned int idz = idxyz / (nx*nyc);
    float fac;
    if(idy==0) fac = 1.0;
    else fac = 2.;
    if(ikx<0 && iky<0) { // default: sum over all k's
      if(idy>0 || idx>0) {
        fg = cuConjf(f[idxyz])*g[idxyz]*jacobian[idz]*fac*fluxDenomInv;
        sum += fg.x;
      }
    } else {
      if(idy==iky && idx==ikx) {
        fg = cuConjf(f[idxyz])*g[idxyz]*jacobian[idz]*fac*fluxDenomInv;
        sum += fg.x;
      }
    }
  }
  atomicAdd(res,sum);
}

void Diagnostics::LHspectrum(MomentsG* G, float* lhspectrum, int ikx, int iky)
{
  int threads=256;
  int blocks=min((grids_->NxNycNz+threads-1)/threads,1024); // could be 1024?

  cudaMemset(lhspectrum, 0., sizeof(float)*grids_->Nmoms);

  for(int m=0; m<grids_->Nm; m++) {
    for(int l=0; l<grids_->Nl; l++) {
      volume_average <<<blocks,threads>>> (&lhspectrum[l+grids_->Nl*m], 
	 G->G(l,m), G->G(l,m), geo_->jacobian, 1./fluxDenom, ikx, iky);
    }
  }
}

void Diagnostics::Lspectrum(MomentsG* G, float* lspectrum, int ikx, int iky)
{
  int threads=256;
  int blocks=min((grids_->NxNycNz+threads-1)/threads,2048); // could be 1024?

  cudaMemset(lspectrum, 0., sizeof(float)*grids_->Nl); // use zero here?

  for(int m=0; m<grids_->Nm; m++) {
    for(int l=0; l<grids_->Nl; l++) {
      volume_average <<<blocks,threads>>> (&lspectrum[l], 
	 G->G(l,m), G->G(l,m), geo_->jacobian, 1./fluxDenom, ikx, iky);
    }
  }
}

void Diagnostics::Hspectrum(MomentsG* G, float* hspectrum, int ikx, int iky)
{
  int threads=256;
  int blocks=min((grids_->NxNycNz+threads-1)/threads,128); 

  cudaMemset(hspectrum, 0., sizeof(float)*grids_->Nm);

  for(int m=0; m<grids_->Nm; m++) {
    for(int l=0; l<grids_->Nl; l++) {
      //      printf("l, m = %d \t %d \t, ikx, iky = %d \t %d \n",l,m,ikx,iky);
      volume_average <<<blocks,threads>>> (&hspectrum[m], 
	 G->G(l,m), G->G(l,m), geo_->jacobian, 1./fluxDenom, ikx, iky);
    }
  }
}

# define G_(XYZ, L, M) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M)]
__global__ void heat_flux(float* qflux, cuComplex* phi, cuComplex* g, float* ky, 
                          float* jacobian, float fluxDenomInv, float *kperp2, float rho2_s, 
                          int ikx=-1, int iky=-1)
{
  float sum = 0.;
  cuComplex fg;
  for(int idxyz=blockIdx.x*blockDim.x+threadIdx.x;idxyz<nx*nyc*nz;idxyz+=blockDim.x*gridDim.x) {
    unsigned int idy = idxyz % (nx*nyc) % nyc; 
    unsigned int idx = idxyz % (nx*nyc) / nyc; 
    unsigned int idz = idxyz / (nx*nyc);
    cuComplex vE_r = make_cuComplex(0., ky[idy]) * phi[idxyz];

    float b_s = kperp2[idxyz]*rho2_s;

    // sum over l
    cuComplex p_bar = make_cuComplex(0.,0.);
    for(int l=0; l<nl; l++) {
      // G_(...) is defined by macro above
      p_bar = p_bar + 1./sqrtf(2.)*Jflr(l,b_s)*G_(idxyz, l, 2)
      	+ ( l*Jflr(l-1,b_s) + (2.*l+1.5)*Jflr(l,b_s) + (l+1)*Jflr(l+1,b_s) )*G_(idxyz, l, 0);
    }
  
    float fac;
    if(idy==0) fac = 0.5;
    else fac = 1.;
    if(ikx<0 && iky<0) { // default: sum over all k's
      if(idy>0 || idx>0) {
        fg = cuConjf(vE_r)*p_bar*jacobian[idz]*fac*fluxDenomInv;
        sum += fg.x;
      }
    } else { // single mode specified by ikx, iky
      if(idy==iky && idx==ikx) {
        fg = cuConjf(vE_r)*p_bar*jacobian[idz]*fac*fluxDenomInv;
        sum += fg.x;
      }
    }
  }
  atomicAdd(qflux,sum);
}


void Diagnostics::fluxes(MomentsG* G, Fields* f)
{
  int threads=256;
  int blocks=min((grids_->NxNycNz+threads-1)/threads,1024);

  for(int is=0; is<grids_->Nspecies; is++) {
    heat_flux <<<blocks,threads>>>
        (&qflux[is], f->phi, G->G(0,0,is), grids_->ky, geo_->jacobian, 1./fluxDenom, geo_->kperp2, pars_->species_h[is].rho2);
  }
}

bool Diagnostics::checkstop() 
{
  struct stat buffer;   
  // check if stopfile exists
  bool stop = (stat (stopfilename_, &buffer) == 0);
  // remove it if it does exist
  if(stop) remove(stopfilename_);
  return stop;
}

// condense a (ky,kx) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
void Diagnostics::reduce2k(float *fk, cuComplex* f) {

  int Nx = grids_->Nx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  if(id->mask) {
    for(int i=0; i<((Nx-1)/3+1); i++) {
      for(int j=0; j<Naky; j++) {
	int index     = j + Nyc *i; 
	int index_out = j + Naky*i; 
	fk[2*index_out]   = f[index].x;
	fk[2*index_out+1] = f[index].y;
      }
    }

    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<Naky; j++) {
	int index = j + Nyc *i;
	int index_out = j + Naky*( i - 2*Nx/3 + (Nx-1)/3 );
	fk[2*index_out]   = f[index].x;
	fk[2*index_out+1] = f[index].y;
      }
    }	
  } else {
    for(int i=0; i<Nx; i++) {
      for(int j=0; j<Nyc; j++) {
	int index = j + Nyc*i;
	fk[2*index]   = f[index].x;
	fk[2*index+1] = f[index].y;
      }
    }
  }

}

// condense a (ky,kx,z) object for netcdf output, taking into account the mask
// and changing the type from cuComplex to float
void Diagnostics::reduce2z(float *fz, cuComplex* f) {

  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;
  int Nz   = grids_->Nz;
  
  if(id->mask) {
    for (int k=0; k<Nz; k++) {
      for (int i=0; i<((Nx-1)/3+1); i++) {
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
  } else {
    for (int k=0; k<Nz; k++) {
      for(int i=0; i<Nx; i++) {
	for(int j=0; j<Nyc; j++) {
	  int index = j + Nyc*i + Nyc*Nx*k;
	  fz[2*index]   = f[index].x;
	  fz[2*index+1] = f[index].y;
	}
      }
    }
  }
}

void Diagnostics::print_growth_rates_to_screen()
{
  int Nx = grids_->Nx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  if(id->mask) {
    printf("ky\tkx\t\tomega\t\tgamma\n");

    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<Naky; j++) {
	int index = j + Nyc*i;
	printf("%.4f\t%.4f\t\t%.6f\t%.6f",
	       grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
	printf("\n");
      }
      printf("\n");
    }
    for(int i=0; i<((Nx-1)/3+1); i++) {
      for(int j=0; j<Naky; j++) {
	int index = j + Nyc*i;
	if(index!=0) {
	  printf("%.4f\t%.4f\t\t%.6f\t%.6f",
		 grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
	  printf("\n");
	}
      }
      printf("\n");
    }
  } else {
    printf("ky\tkx\t\tomega\t\tgamma\n");
    for(int i=0; i<Nx; i++) {
      for(int j=0; j<Nyc; j++) {
	int index = j + Nyc*i;
	if(index!=0) {
	  printf("%.4f\t%.4f\t\t%.6f\t%.6f",
		 grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
	  printf("\n");
	}
      }
      printf("\n");
    }    
  }	
}


