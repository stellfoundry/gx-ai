#include "diagnostics.h"
#include "device_funcs.h"
#include "cuda_constants.h"
#include "get_error.h"
#include <sys/stat.h>

__global__ void growthRates(cuComplex *phi, cuComplex *phiOld, float dt, cuDoubleComplex *omega)
{
  unsigned int idxy = get_id1();
  
  //i_dt = i/dt
  cuDoubleComplex i_dt = make_cuDoubleComplex(0., (double) 1./dt);
  
  if(idxy<nx*nyc)
  {
    cuDoubleComplex ratio = cuComplexFloatToDouble(phi[idxy+nx*nyc*((int)(.5*nz))]) / cuComplexFloatToDouble(phiOld[idxy+nx*nyc*((int)(.5*nz))]);
    
    cuDoubleComplex logr;
    logr.x = log(cuCabs(ratio));
    logr.y = atan2(ratio.y,ratio.x);
    omega[idxy] = logr*i_dt;
  }
}

__global__ void computeHermiteEnergySpectrum(cuComplex *m, float *hermite_energy_spectrum, float *hermite_energy_spectrum_avg, int count, int num_writes, int cutoff) {
  int i = threadIdx.x;
  int id = nx*nyc*nz*nl*i;

  // calculate |C_m| for each moment
  float value = m[id+1].x*m[id+1].x + m[id+1].y*m[id+1].y;

  hermite_energy_spectrum[i] = value;

  // hermite_spectrum_avg_cutoff in gx_knobs determines the cutoff number of counts to use in the energy spectrum time averaging
  // (starting from the end, i.e. a cutoff of 10 takes a time average of the last 10 counts/nwrite)
  if (count >= num_writes - cutoff) {
      int index = count - num_writes + cutoff;
      hermite_energy_spectrum_avg[i] = (index*hermite_energy_spectrum_avg[i] + value)/(index + 1);
  }
}

Diagnostics::Diagnostics(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo)
{
  fields_old = new Fields(grids_);
  cudaDeviceSynchronize();
  grad_parallel = new GradParallelPeriodic(grids_);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  cudaMalloc((void**) &growth_rates, sizeof(cuDoubleComplex)*grids_->NxNyc);
  cudaMallocHost((void**) &lhspectrum_h, sizeof(float)*grids_->Nmoms);
  cudaMalloc((void**) &lhspectrum, sizeof(float)*grids_->Nmoms);
  cudaMallocHost((void**) &growth_rates_h, sizeof(cuDoubleComplex)*grids_->NxNyc);

  cudaMallocHost((void**) &amom_h, sizeof(cuComplex)*grids_->NxNycNz);
  cudaMalloc((void**) &amom, sizeof(cuComplex)*grids_->NxNycNz);

  cudaMallocHost((void**) &pflux, sizeof(float)*grids_->Nspecies);
  cudaMallocHost((void**) &qflux, sizeof(float)*grids_->Nspecies);

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

  if(pars_->source_option == PHIEXT) {
    char ofilename[2000];
    sprintf(ofilename, "%s.%s.time", pars_->run_name, "phi");
    // open new file (i.e. overwrite an existing one)
    timefile = fopen(ofilename,"w");
  }

  // set up file for time history of flux(es)
  if(!pars_->linear) {
    char ofilename[2000];
    sprintf(ofilename, "%s.%s.time", pars_->run_name, "flux");
    fluxfile = fopen(ofilename,"w"); 
  }

  // skip over masked elements in x and y?
  mask = !pars_->linear;
  
  // for hermite energy spectrum diagnostic
  if (pars_->write_hermite_energy_spectrum) {
    cudaMalloc((void **) &hermite_energy_spectrum, sizeof(float)*pars_->nm_in);
    cudaMalloc((void **) &hermite_energy_spectrum_avg, sizeof(float)*pars_->nm_in);
    cudaMallocHost((void **) &hermite_energy_spectrum_h, sizeof(float)*pars_->nm_in);

    cudaMemset(hermite_energy_spectrum, 0, pars_->nm_in*sizeof(float));
    cudaMemset(hermite_energy_spectrum_avg, 0, pars_->nm_in*sizeof(float));

    sprintf(history_buffer, "%s.HermiteEnergySpectrumHistory.out", pars_->run_name);
    history_spectrum_file = fopen(history_buffer,"w+");
  }
}

Diagnostics::~Diagnostics()
{
  delete fields_old;
  cudaFree(growth_rates);
  cudaFree(lhspectrum);
  cudaFreeHost(lhspectrum_h);
  cudaFreeHost(growth_rates_h); 
  cudaFreeHost(pflux);
  cudaFreeHost(qflux);
  if(pars_->source_option == PHIEXT) {
    fclose(timefile);
  }

  if (pars_->write_hermite_energy_spectrum) {
    cudaFree(hermite_energy_spectrum);
    cudaFree(hermite_energy_spectrum_avg);
    cudaFreeHost(hermite_energy_spectrum_h);
    fclose(history_spectrum_file);
  }
}

// NOTE: needs to be called every step when calculating growth rates
// does not write to file every step
bool Diagnostics::loop_diagnostics(MomentsG* G, Fields* fields, float dt, int counter, float time) 
{
  bool stop = false;
  if(counter%pars_->nwrite==0) {
    printf("Step %d: Time = %f\n", counter, time);
    fflush(NULL);
  }
  // write instantaneous growth rates
  if(pars_->write_omega) {
    growthRates<<<grids_->NxNyc/maxThreadsPerBlock_+1, maxThreadsPerBlock_>>>
             (fields->phi, fields_old->phi, dt, growth_rates);
    // save fields for next time
    fields_old->copyFrom(fields);

    if(counter%pars_->nwrite==0) {
      cudaMemcpyAsync(growth_rates_h, growth_rates, sizeof(cuDoubleComplex)*grids_->NxNyc, cudaMemcpyDeviceToHost);
      print_growth_rates_to_screen();
    }
  }

  // compute and write to file hermite energy spectrum
  if(counter%pars_->nwrite == 0 && pars_->write_hermite_energy_spectrum) {
      HermiteEnergySpectrum(G, counter);
      writeHermiteEnergySpectrumHistory();
  }

  // write time history of phi
  if(pars_->source_option==PHIEXT) {
    writeTimeHistory(fields->phi, time, 1, 0, grids_->Nz/2, timefile);
  }

  if(!pars_->linear) {
    // calculate fluxes
    fluxes(G, fields);
    fprintf(fluxfile, "%f\t%e\n", time, qflux[0]);
    printf("\tqflux = %e\n", qflux[0]);
  }
  // check to see if we should stop simulation
  stop = checkstop();
  return stop;
}

void Diagnostics::final_diagnostics(MomentsG* G, Fields* fields) 
{
  // print final moments and fields
  writeMomOrField(G->dens_ptr[0], "dens");
  writeMomOrField(G->upar_ptr[0], "upar");
  writeMomOrField(fields->phi, "phi");

  writeMomOrFieldKpar(G->dens_ptr[0], "dens");
  writeMomOrFieldKpar(G->upar_ptr[0], "upar");
  writeMomOrFieldKpar(fields->phi, "phi");

  // write Hermite-Laguerre spectrum |G|**2(l,m)
  if(pars_->source_option==PHIEXT) {
    writeLHspectrum(G, 1, 0);
  } else {
    writeLHspectrum(G);
  }

  // write geometry coefficient arrays
  writeGeo();

  if(pars_->write_omega) writeGrowthRates();

  if (pars_->write_hermite_energy_spectrum) {
      writeHermiteEnergySpectrum();
  }

  fflush(NULL);
}


void Diagnostics::print_growth_rates_to_screen()
{
  int Nx = grids_->Nx;
  int Ny = grids_->Ny;
  int Nyc = grids_->Nyc;

     if(mask) {
  	printf("ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<((Nx-1)/3+1); i++) {
  	  for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      printf("%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      printf("\n");
  	    }
  	  }
  	  printf("\n");
  	}
  	//for(int i=2*Nx/3+1; i<2*Nx/3+1; i++) {
        for(int i=2*Nx/3+1; i<Nx; i++) {
            for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	      printf("%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      printf("\n");
  	  }
  	  printf("\n");
  	}
    } else {
  	printf("ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<Nx; i++) {
  	  for(int j=0; j<Nyc; j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      printf("%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      printf("\n");
  	    }
  	  }
  	  printf("\n");
  	}

    }	
}

void Diagnostics::writeGrowthRates()
{
  int Nx = grids_->Nx;
  int Ny = grids_->Ny;

  char ofilename[2000];
  sprintf(ofilename, "%s.omega.kykx", pars_->run_name);
  FILE* out = fopen(ofilename,"w+");

     if(mask) {
  	fprintf(out, "ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<((Nx-1)/3+1); i++) {
  	  for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      fprintf(out, "%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      fprintf(out, "\n");
  	    }
  	  }
  	  fprintf(out, "\n");
  	}
  	//for(int i=2*Nx/3+1; i<2*Nx/3+1; i++) {
        for(int i=2*Nx/3+1; i<Nx; i++) {
            for(int j=0; j<((Ny-1)/3+1); j++) {
  	    int index = j + (Ny/2+1)*i;
  	      fprintf(out, "%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      fprintf(out, "\n");
  	  }
  	  fprintf(out, "\n");
  	}	
    } else {
  	fprintf(out, "ky\tkx\t\tomega\t\tgamma\t\tconverged?\n");
  	//for(int i=0; i<1; i++) {
        for(int i=0; i<Nx; i++) {
  	  for(int j=0; j<Ny/2+1; j++) {
  	    int index = j + (Ny/2+1)*i;
  	    if(index!=0) {
  	      fprintf(out, "%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], growth_rates_h[index].x, growth_rates_h[index].y);
  	      fprintf(out, "\n");
  	    }
  	  }
  	  fprintf(out, "\n");
  	}
    }
  fclose(out);
}

void Diagnostics::writeMomOrFieldKpar(cuComplex* mom, const char* name) {
  int Nx = grids_->Nx;
  int Ny = grids_->Ny;
  int Nz = grids_->Nz;

  char ofilename[2000];
  sprintf(ofilename, "%s.%s.kpar_field", pars_->run_name, name);
  FILE* out = fopen(ofilename,"w+");
  grad_parallel->fft_only(mom, amom, CUFFT_FORWARD);
  cudaMemcpy(amom_h,amom,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToHost);
  fprintf(out, "#\tkz (1)\t\t\tky (2)\t\t\tkx (3)\t\t\tRe (4)\t\t\tIm (5)\t\t\t");  
  fprintf(out, "\n");
  int blockid = 0;
  if(mask) {
    for(int i=0; i<(Nx-1)/3+1; i++) {
      for(int j=0; j<(Ny-1)/3+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;      
        for(int k=0; k<Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
          //if(index!=0){
            fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", grids_->kz_h[k], grids_->ky_h[j], grids_->kx_h[i], amom_h[index].x, amom_h[index].y);    	  
          //}
        }     
      }
    }
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<(Ny-1)/3+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;
        for(int k=0; k<Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
          fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", grids_->kz_h[k], grids_->ky_h[j], grids_->kx_h[i], amom_h[index].x, amom_h[index].y);    	  
        }    
      }
    }
  } else {
    for(int i=0; i<Nx; i++) {
      for(int j=0; j<Ny/2+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;      
        for(int k=0; k<Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
          //if(index!=0){
            fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", grids_->kz_h[k], grids_->ky_h[j], grids_->kx_h[i], amom_h[index].x, amom_h[index].y);    	  
          //}
        }     
      }
    }

  }
  fclose(out);

}

void Diagnostics::writeMomOrField(cuComplex* m, const char* name) {
  int Nx = grids_->Nx;
  int Ny = grids_->Ny;
  int Nz = grids_->Nz;

  char ofilename[2000];
  sprintf(ofilename, "%s.%s.field", pars_->run_name, name);
  FILE* out = fopen(ofilename,"w+");
  cudaMemcpy(amom_h,m,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToHost);
  fprintf(out, "#\tz (1)\t\t\tky (2)\t\t\tkx (3)\t\t\tRe (4)\t\t\tIm (5)\t\t\t");  
  fprintf(out, "\n");
  int blockid = 0;
  if(mask) {
    for(int i=0; i<(Nx-1)/3+1; i++) {
      for(int j=0; j<(Ny-1)/3+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;      
        for(int k=0; k<=Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
  	//if(index!=0){
  	  if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -geo_->z_h[0], grids_->ky_h[j], grids_->kx_h[i], amom_h[j+(Ny/2+1)*i].x, amom_h[j+(Ny/2+1)*i].y); 
  	  else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", geo_->z_h[k], grids_->ky_h[j], grids_->kx_h[i], amom_h[index].x, amom_h[index].y);    	  
          //}
        }     
      }
    }
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<(Ny-1)/3+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;
        for(int k=0; k<=Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
  	if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -geo_->z_h[0], grids_->ky_h[j], grids_->kx_h[i], amom_h[j+(Ny/2+1)*i].x, amom_h[j+(Ny/2+1)*i].y); 
  	else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", geo_->z_h[k], grids_->ky_h[j], grids_->kx_h[i], amom_h[index].x, amom_h[index].y);    	  
        }    
      }
    }
  } else {

    for(int i=0; i<Nx; i++) {
      for(int j=0; j<Ny/2+1; j++) {
        fprintf(out, "\n#%d\n\n", blockid);
        blockid++;      
        for(int k=0; k<=Nz; k++) {
          int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
  	//if(index!=0){
  	  if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -geo_->z_h[0], grids_->ky_h[j], grids_->kx_h[i], amom_h[j+(Ny/2+1)*i].x, amom_h[j+(Ny/2+1)*i].y); 
  	  else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", geo_->z_h[k], grids_->ky_h[j], grids_->kx_h[i], amom_h[index].x, amom_h[index].y);    	  
          //}
        }     
      }
    }

  }
  fclose(out);
}

// MFM
void Diagnostics::writeGridFile(const char* name) {

  char ofilename[2000], c;
  sprintf(ofilename, "%s.%s", pars_->run_name, name);
  FILE* ifile = fopen(pars_->geofilename,"r");
  FILE* ofile = fopen(ofilename,"w+");

  c = fgetc(ifile);
  while (c != EOF) {
    fputc(c, ofile);
    c = fgetc(ifile);
  }
  fclose(ifile);
  fclose(ofile);
}


void Diagnostics::writeLHspectrum(MomentsG* G, int ikx, int iky)
{
  // calculate spectrum  
  LHspectrum(G, ikx, iky);
  //cudaDeviceSynchronize();

  char ofilename[2000];
  sprintf(ofilename, "%s.lhspectrum.out", pars_->run_name);
  FILE* out = fopen(ofilename,"w+");

  cudaMemcpy(lhspectrum_h, lhspectrum, sizeof(float)*grids_->Nmoms, cudaMemcpyDeviceToHost);
  
  for(int m=0; m<grids_->Nm; m++) {
    for(int l=0; l<grids_->Nl; l++) {
      fprintf(out, "%d\t%d\t%e\n", l, m, lhspectrum_h[l+grids_->Nl*m]);
    }
    fprintf(out, "\n");
  }
  fclose(out); 
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
    if(idy==0) fac = 0.5;
    else fac = 1.;
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

void Diagnostics::LHspectrum(MomentsG* G, int ikx, int iky)
{
  int threads=256;
  int blocks=min((grids_->NxNycNz+threads-1)/threads,2048);
  cudaMemset(lhspectrum, 0., sizeof(float)*grids_->Nmoms);
  for(int m=0; m<grids_->Nm; m++) {
    for(int l=0; l<grids_->Nl; l++) {
      volume_average<<<blocks,threads>>>
	(&lhspectrum[l+grids_->Nl*m], 
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
  int blocks=min((grids_->NxNycNz+threads-1)/threads,2048);

  for(int is=0; is<grids_->Nspecies; is++) {
    heat_flux<<<blocks,threads>>>
        (&qflux[is], f->phi, G->G(0,0,is), grids_->ky, geo_->jacobian, 1./fluxDenom, geo_->kperp2, pars_->species_h[is].rho2);
  }
}

void Diagnostics::writeGeo() 
{
  char ofilename[2000];
  sprintf(ofilename, "%s.geo.z", pars_->run_name);
  FILE* out = fopen(ofilename,"w+");

  fprintf(out, "#\tz:1\t\tbmag:2\t\tbgrad:3\t\tgbd:4\t\tgbd0:5\t\tcvd:6\t\tcvd0:7\t\tgds2:8\t\tgds21:9\t\tgds22:10\tgrho:11\t\tjacobian:12\n");
  for(int i=0; i<grids_->Nz; i++) {
    fprintf(out, "\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", geo_->z_h[i], geo_->bmag_h[i], geo_->bgrad_h[i], geo_->gbdrift_h[i], geo_->gbdrift0_h[i], geo_->cvdrift_h[i], geo_->cvdrift0_h[i], geo_->gds2_h[i], geo_->gds21_h[i], geo_->gds22_h[i], geo_->grho_h[i], geo_->jacobian_h[i]);
  }
  //periodic point
  int i=0;
  fprintf(out, "\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", -geo_->z_h[i], geo_->bmag_h[i], geo_->bgrad_h[i], geo_->gbdrift_h[i], geo_->gbdrift0_h[i], geo_->cvdrift_h[i], geo_->cvdrift0_h[i], geo_->gds2_h[i], geo_->gds21_h[i], geo_->gds22_h[i], geo_->grho_h[i], geo_->jacobian_h[i]);
}

void Diagnostics::writeTimeHistory(cuComplex* f, float time, int ikx, int iky, int iz, FILE* out) 
{
  cuComplex val;
  cudaMemcpy(&val, &f[iky + grids_->Nyc*ikx + grids_->NxNyc*iz], sizeof(cuComplex), cudaMemcpyDeviceToHost);
  fprintf(out, "%.4f\t\t%.4f\t\t%.4f\n", time, val.x, val.y);
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

// set write_hermite_energy_spectrum to "on" in gx_knobs to calculate and write to file the Hermite energy spectrum
void Diagnostics::HermiteEnergySpectrum(MomentsG *m, int count) {
    computeHermiteEnergySpectrum<<<1, pars_->nm_in>>>(m->G(0,0,0), hermite_energy_spectrum, hermite_energy_spectrum_avg, count/pars_->nwrite, pars_->nstep/pars_->nwrite, pars_->hermite_spectrum_avg_cutoff);
}

void Diagnostics::writeHermiteEnergySpectrum() {
  char ofilename[2000];
  sprintf(ofilename, "%s.HermiteEnergySpectrum.out", pars_->run_name);
  FILE *out = fopen(ofilename,"w+");

  cudaMemcpy(hermite_energy_spectrum_h, hermite_energy_spectrum_avg, sizeof(float)*pars_->nm_in, cudaMemcpyDeviceToHost);

  fprintf(out, "#\tm (1)\t\t\tC_m,k (2)\t\t\t\n");  

  for (int i = 0; i < pars_->nm_in; i++) {
    fprintf(out, "\t%d\t%e\n", i, hermite_energy_spectrum_h[i]);
  }

  fclose(out);
}

void Diagnostics::writeHermiteEnergySpectrumHistory() {
  cudaMemcpy(hermite_energy_spectrum_h, hermite_energy_spectrum, sizeof(float)*pars_->nm_in, cudaMemcpyDeviceToHost);

  fprintf(history_spectrum_file, "#\tm (1)\t\t\tC_m,k (2)\t\t\t\n");  

  for (int i = 0; i < pars_->nm_in; i++) {
    fprintf(history_spectrum_file, "\t%d\t%e\n", i, hermite_energy_spectrum_h[i]);
  }
  fprintf(history_spectrum_file, "\n");
}

