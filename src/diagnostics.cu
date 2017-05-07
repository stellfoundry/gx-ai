#include "diagnostics.h"
#include "device_funcs.h"

__global__ void growthRates(cuComplex *phi, cuComplex *phiOld, float dt, cuComplex *omega)
{
  unsigned int idxy = get_id1();
  
  //i_dt = i/dt
  cuComplex i_dt = {0., 1./dt};
  
  if(idxy<nx*nyc)
  {
    cuComplex ratio = phi[idxy+nx*nyc*((int)(.5*nz))] / phiOld[idxy+nx*nyc*((int)(.5*nz))];
    
    cuComplex log;
    log.x = logf(cuCabsf(ratio));
    log.y = atan2f(ratio.y,ratio.x);
    omega[idxy] = log*i_dt;
  }
}


Diagnostics::Diagnostics(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids)
{
  fields_old = new Fields(grids_);

  cudaMalloc((void**) &growth_rates, sizeof(cuComplex)*grids_->NxNyc);
  cudaMallocHost((void**) &growth_rates_h, sizeof(cuComplex)*grids_->NxNyc);

  cudaMallocHost((void**) &m_h, sizeof(cuComplex)*grids_->NxNycNz);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  maxThreadsPerBlock_ = prop.maxThreadsPerBlock;
}

Diagnostics::~Diagnostics()
{
  delete fields_old;
  cudaFree(growth_rates);
  cudaFreeHost(growth_rates_h);
}

// NOTE: needs to be called every step 
void Diagnostics::loop_diagnostics(Moments* moms, Fields* fields, float dt, int counter, float time) 
{
  if(pars_->write_omega) {
    growthRates<<<grids_->NxNyc/maxThreadsPerBlock_+1, maxThreadsPerBlock_>>>
             (fields->phi, fields_old->phi, dt, growth_rates);
    // save fields for next time
    fields_old->copyFrom(fields);

    if(counter%pars_->nwrite==0) {
        printf("Step %d: Time = %f\n", counter, time);
      cudaMemcpyAsync(growth_rates_h, growth_rates, sizeof(cuComplex)*grids_->NxNyc, cudaMemcpyDeviceToHost);
      print_growth_rates_to_screen();
    }
  } 
}

void Diagnostics::print_growth_rates_to_screen()
{
  int Nx = grids_->Nx;
  int Ny = grids_->Ny;

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
}

void Diagnostics::printMomOrField(cuComplex* m, char* filename) {
  int Nx = grids_->Nx;
  int Ny = grids_->Ny;
  int Nz = grids_->Nz;

  FILE* out = fopen(filename,"w+");
  cudaMemcpy(m_h,m,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToHost);
  fprintf(out, "#\tz (1)\t\t\tky (2)\t\t\tkx (3)\t\t\tRe (4)\t\t\tIm (5)\t\t\t");  
  fprintf(out, "\n");
  int blockid = 0;
  for(int i=0; i<(Nx-1)/3+1; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      fprintf(out, "\n#%d\n\n", blockid);
      blockid++;      
      for(int k=0; k<=Nz; k++) {
        int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
	//if(index!=0){
	  if(k==Nz) fprintf(out, "\t%d\t\t%f\t\t%f\t\t%e\t\t%e\t\n", k, grids_->ky_h[j], grids_->kx_h[i], m_h[j+(Ny/2+1)*i].x, m_h[j+(Ny/2+1)*i].y); 
	  else fprintf(out, "\t%d\t\t%f\t\t%f\t\t%e\t\t%e\t\n", k, grids_->ky_h[j], grids_->kx_h[i], m_h[index].x, m_h[index].y);    	  
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
	if(k==Nz) fprintf(out, "\t%d\t\t%f\t\t%f\t\t%e\t\t%e\t\n", k, grids_->ky_h[j], grids_->kx_h[i], m_h[j+(Ny/2+1)*i].x, m_h[j+(Ny/2+1)*i].y); 
	else fprintf(out, "\t%d\t\t%f\t\t%f\t\t%e\t\t%e\t\n", k, grids_->ky_h[j], grids_->kx_h[i], m_h[index].x, m_h[index].y);    	  
      }    
    }
  }
  fclose(out);
}

