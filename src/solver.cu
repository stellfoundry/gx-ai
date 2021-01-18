#include "solver.h"
#include "device_funcs.h"
#include "get_error.h"

Solver::Solver(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo),
  tmp(nullptr), nbar(nullptr), phiavgdenom(nullptr)
{

  if (pars_->ks) {
    // nothing
  } else {
    
    size_t cgrid = sizeof(cuComplex)*grids_->NxNycNz;
    cudaMalloc((void**) &nbar, cgrid); cudaMemset(nbar, 0., cgrid);
    cudaMalloc((void**) &tmp,  cgrid); cudaMemset(tmp,  0., cgrid);
    
    // set up phiavgdenom, which is stored for quasineutrality calculation
    cudaMalloc((void**) &phiavgdenom, sizeof(float)*grids_->Nx);
    cudaMemset(phiavgdenom, 0., sizeof(float)*grids_->Nx);
    
    int threads, blocks;
    threads = 128;
    blocks = 1 + (grids_->Nx-1)/threads;
    
    if (!pars_->all_kinetic && (pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS)) {     
      calc_phiavgdenom <<<blocks, threads>>> (phiavgdenom,
					      geo_->kperp2,
					      geo_->jacobian,
					      pars_->species,
					      pars_->tau_fac);
    }
  
    // cuda dims for qneut calculation
    dB = dim3(32, 4, 4);
    dG = dim3((grids_->Nyc-1)/dB.x+1, (grids_->Nx-1)/dB.y+1, (grids_->Nz-1)/dB.z+1);  
    
    db = dim3(32, 32, 1);
    dg = dim3((grids_->Nyc-1)/db.x+1, (grids_->Nx-1)/db.y+1, 1);
  }
}

Solver::~Solver() 
{
  if (nbar)        cudaFree(nbar);
  if (tmp)         cudaFree(tmp);
  if (phiavgdenom) cudaFree(phiavgdenom);
}

int Solver::fieldSolve(MomentsG* G, Fields* fields)
{
  if (pars_->ks) return 0;
  
  bool em = pars_->beta > 0. ? true : false;
  
  if (pars_->all_kinetic) {

    qneut <<< dG, dB >>> (fields->phi, G->G(), geo_->kperp2, pars_->species);

    if (em) ampere <<< dG, dB >>> (fields->apar, G->G(0,1,0), geo_->kperp2, pars_->species, pars_->beta);

  } else {

    int nts = 512;
    nts = min(nts, grids_->Nyc*grids_->Nx);
    int nbs = (grids_->NxNycNz-1)/nts + 1;

    real_space_density <<< nbs, nts >>> (nbar, G->G(), geo_->kperp2, pars_->species);

    // In these routines there is inefficiency because multiple threads
    // calculate the same field line averages. It is correct but inefficient.

    if(pars_->Boltzmann_opt == BOLTZMANN_ELECTRONS) {
      qneutAdiab_part1 <<< dG, dB >>> (tmp, nbar,
				       geo_->kperp2,
				       geo_->jacobian,
				       pars_->species,
				       pars_->tau_fac);
      
      cudaMemset(fields->phi, 0., sizeof(cuComplex)*grids_->NxNycNz);
      
      qneutAdiab_part2 <<< dG, dB >>> (fields->phi, tmp, nbar,
				       phiavgdenom,
				       geo_->kperp2,							    
				       pars_->species,
				       pars_->tau_fac);
    } 
    
    if(pars_->Boltzmann_opt == BOLTZMANN_IONS) qneutAdiab <<< dG, dB >>> (fields->phi, nbar,
									  geo_->kperp2,
									  pars_->species,
									  pars_->tau_fac);
  }
  
  if(pars_->source_option==PHIEXT) add_source <<< dG, dB >>> (fields->phi, pars_->phi_ext);

  return 0;
}

void Solver::svar (cuComplex* f, int N)
{
  cuComplex* f_h;  cudaMallocHost((void**) &f_h, sizeof(cuComplex)*N);

  for (int i=0; i<N; i++) { f_h[i].x=0.; f_h[i].y=0.; }

  CP_TO_CPU (f_h, f, N*sizeof(cuComplex));

  for (int i=0; i<N; i++) printf("solver: var(%d) = (%e, %e) \n", i, f_h[i].x, f_h[i].y);
  printf("\n");

  cudaFreeHost (f_h);
}

void Solver::svar (float* f, int N)
{
  float* f_h;
  cudaMallocHost((void**) &f_h, sizeof(float)*N);

  CP_TO_CPU (f_h, f, N*sizeof(float));

  for (int i=0; i<N; i++) printf("solver: var(%d) = %e \n", i, f_h[i]);
  printf("\n");
  
  cudaFreeHost (f_h);
}
