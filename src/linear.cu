#include "linear.h"

Linear::Linear(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo),
  GRhs_par(nullptr), closures(nullptr), grad_par(nullptr)
{
  ks = false;
  upar_bar  = nullptr;           uperp_bar = nullptr;            t_bar     = nullptr;
  
  GRhs_par = new MomentsG(pars_, grids_);

  // set up parallel ffts
  if(pars_->local_limit) {
    DEBUGPRINT("Using local limit for grad parallel.\n");
    grad_par = new GradParallelLocal(grids_);
  }
  else if(pars_->boundary_option_periodic) {
    DEBUGPRINT("Using periodic for grad parallel.\n");
    grad_par = new GradParallelPeriodic(grids_);
  }
  else {
    DEBUGPRINT("Using twist-and-shift for grad parallel.\n");
    grad_par = new GradParallelLinked(grids_, pars_->jtwist);
  }
 
  if(pars_->closure_model_opt > 0) {
    if(pars_->closure_model_opt == BEER42) {                                 DEBUGPRINT("Initializing Beer 4+2 closures\n");
      closures = new Beer42(pars_, grids_, geo_, grad_par);
    } else if (pars_->closure_model_opt == SMITHPERP) {                      DEBUGPRINT("Initializing Smith perpendicular toroidal closures\n");
      closures = new SmithPerp(pars_, grids_, geo_);
    } else if (pars_->closure_model_opt == SMITHPAR) {                       DEBUGPRINT("Initializing Smith parallel closures\n");
      closures = new SmithPar(pars_, grids_, geo_, grad_par);
    }
  }

  // allocate conservation terms for collision operator
  size_t size = sizeof(cuComplex)*grids_->NxNycNz*grids_->Nspecies;
  cudaMalloc((void**) &upar_bar, size);
  cudaMalloc((void**) &uperp_bar, size);
  cudaMalloc((void**) &t_bar, size);
  cudaMemset(upar_bar, 0., size);
  cudaMemset(uperp_bar, 0., size);
  cudaMemset(t_bar, 0., size);

  // set up CUDA grids for main linear kernel.  
  // NOTE: dimBlock.x = sharedSize.x = 32 gives best performance, but using 8 is only 5% worse.
  // this allows use of 4x more LH resolution without changing shared memory layouts.
  // dimBlock = dim3(8, min(4, grids_->Nl), min(4, grids_->Nm));

  dimBlock = dim3(pars_->i_share, min(4, grids_->Nl), min(4, grids_->Nm));
  dimGrid  = dim3((grids_->NxNycNz-1)/dimBlock.x+1, 1, 1);
  sharedSize = dimBlock.x*(grids_->Nl+2)*(grids_->Nm+4)*sizeof(cuComplex);
  DEBUGPRINT("For linear RHS: size of shared memory block = %f KB\n", sharedSize/1024.);
  if(sharedSize/1024.>96.) {
    printf("Error: currently cannot support this velocity resolution due to shared memory constraints.\n");
    printf("size of shared memory block must be less than 96 KB, so make sure (nm+4)*(nlaguerre+2)<%d.\n", 96*1024/8/dimBlock.x);
    exit(1);
  }
}

Linear::Linear(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids)
{
  ks = true;
  
  dB = dim3(min(128, grids_->Naky), 1, 1);
  dG = dim3((grids_->Naky-1)/dB.x + 1, 1, 1);
}

Linear::~Linear()
{
  if (closures) delete closures;
  if (grad_par) delete grad_par;
  if (GRhs_par) delete GRhs_par;

  if (upar_bar)   cudaFree(upar_bar);
  if (uperp_bar)  cudaFree(uperp_bar);
  if (t_bar)      cudaFree(t_bar);
}


void Linear::rhs(MomentsG* G, Fields* f, MomentsG* GRhs) {

  if (ks) {
    rhs_ks <<< dG, dB >>> (G->G(), GRhs->G(), grids_->ky);
    return;
  }

  // calculate conservation terms for collision operator
  if (pars_->collisions)  conservation_terms <<<(grids_->NxNycNz-1)/256+1, 256>>>
      (upar_bar, uperp_bar, t_bar, G->G(), f->phi, geo_->kperp2, pars_->species);
  
  // Moving streaming here. Get rhs_par, then d/dz of it. save as rhs.

  // change this to add to rhs that exists
  
  // calculate most of the RHS
  cudaFuncSetAttribute(rhs_linear, cudaFuncAttributeMaxDynamicSharedMemorySize, 12*1024*sizeof(cuComplex));    
  rhs_linear<<<dimGrid, dimBlock, sharedSize>>>
      	(G->G(), f->phi, upar_bar, uperp_bar, t_bar,
        geo_->kperp2, geo_->cv_d, geo_->gb_d, geo_->bgrad, 
       	grids_->ky, pars_->species, GRhs_par->G(), GRhs->G());

  // calculate streaming terms
  grad_par->dz(GRhs_par);

  // combine
  GRhs->add_scaled(1., GRhs, (float) geo_->gradpar, GRhs_par);

  // closures
  if(pars_->closure_model_opt>0) closures->apply_closures(G, GRhs);

  // hypercollisions
  if(pars_->hypercollisions) hypercollisions<<<dimGrid,dimBlock>>>(G->G(),
								   pars_->nu_hyper_l,
								   pars_->nu_hyper_m,
								   pars_->p_hyper_l,
								   pars_->p_hyper_m, GRhs->G());
}






