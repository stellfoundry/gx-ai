#include "linear.h"

Linear::Linear(Parameters* pars, Grids* grids, Geometry* geo) :
  pars_(pars), grids_(grids), geo_(geo),
  closures(nullptr), grad_par(nullptr)
{
  ks = false;
  upar_bar  = nullptr;           uperp_bar = nullptr;            t_bar     = nullptr;
  
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
 
  switch (pars_->closure_model_opt)
    {
    case Closure::none      :
      break;
    case Closure::beer42    :
      DEBUGPRINT("Initializing Beer 4+2 closures\n");
      closures = new Beer42(pars_, grids_, geo_, grad_par);
      break;
    case Closure::smithperp :
      DEBUGPRINT("Initializing Smith perpendicular toroidal closures\n");
      closures = new SmithPerp(pars_, grids_, geo_);
      break;
    case Closure::smithpar  :
      DEBUGPRINT("Initializing Smith parallel closures\n");
      closures = new SmithPar(pars_, grids_, geo_, grad_par);
      break;
    }
  
  // allocate conservation terms for collision operator
  size_t size = sizeof(cuComplex)*grids_->NxNycNz*grids_->Nspecies;
  cudaMalloc((void**) &upar_bar, size);
  cudaMalloc((void**) &uperp_bar, size);
  cudaMalloc((void**) &t_bar, size);
  cudaMemset(upar_bar, 0., size);
  cudaMemset(uperp_bar, 0., size);
  cudaMemset(t_bar, 0., size);

  int nn1 = grids_->Nyc;             int nt1 = min(nn1, 16);   int nb1 = 1 + (nn1-1)/nt1;
  int nn2 = grids_->Nx;              int nt2 = min(nn2,  4);   int nb2 = 1 + (nn2-1)/nt2;
  int nn3 = grids_->Nz*grids_->Nl;   int nt3 = min(nn3,  4);   int nb3 = 1 + (nn3-1)/nt3;
  
  dBs = dim3(nt1, nt2, nt3);
  dGs = dim3(nb1, nb2, nb3);

  // set up CUDA grids for main linear kernel.  
  // NOTE: nt1 = sharedSize = 32 gives best performance, but using 8 is only 5% worse.
  // this allows use of 4x more LH resolution without changing shared memory layouts
  // so i_share = 8 is used by default.

  nn1 = grids_->NxNycNz;         nt1 = pars_->i_share     ;   nb1 = 1 + (nn1-1)/nt1;
  nn2 = 1;                       nt2 = min(grids_->Nl, 4 );   nb2 = 1 + (nn2-1)/nt2;
  nn3 = 1;                       nt3 = min(grids_->Nm, 4 );   nb3 = 1 + (nn3-1)/nt3;

  dimBlock = dim3(nt1, nt2, nt3);
  dimGrid  = dim3(nb1, nb2, nb3);
  
  sharedSize = nt1 * (grids_->Nl+2) * (grids_->Nm+4) * sizeof(cuComplex);

  DEBUGPRINT("For linear RHS: size of shared memory block = %f KB\n", sharedSize/1024.);
  if(sharedSize/1024.>96.) {
    printf("Error: currently cannot support this velocity resolution due to shared memory constraints.\n");
    printf("If you wish to try to keep this velocity resolution, ");
    printf("you can try lowering i_share in your input file.\n");
    printf("You are using i_share = %d now, perhaps by default.\n", pars_->i_share);
    printf("The size of the shared memory block should be less than 96 KB ");
    printf("which means i_share*(nhermite+4)*(nlaguerre+2) < %d. \n", 12*1024);
    printf("Presently, you have set i_share*(nhermite+4)*(nlaguerre+2) = %d. \n",
	   pars_->i_share*(grids_->Nm+4)*(grids_->Nl+2));
    exit(1);
  }

  nn1 = grids_->NxNycNz;         nt1 = min(grids_->NxNycNz, 32) ;   nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nl;              nt2 = min(grids_->Nl, 4 )      ;   nb2 = 1 + (nn2-1)/nt2;
  nn3 = grids_->Nm;              nt3 = min(grids_->Nm, 4 )      ;   nb3 = 1 + (nn3-1)/nt3;

  dimBlockh = dim3(nt1, nt2, nt3);
  dimGridh  = dim3(nb1, nb2, nb3);
  
  
}

Linear::Linear(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids), closures(nullptr), grad_par(nullptr)
{
  ks = true;
  
  dB = dim3(min(128, grids_->Naky), 1, 1);
  dG = dim3(1+(grids_->Naky-1)/dB.x, 1, 1);
}

Linear::~Linear()
{
  if (closures) delete closures;
  if (grad_par) delete grad_par;

  if (upar_bar)   cudaFree(upar_bar);
  if (uperp_bar)  cudaFree(uperp_bar);
  if (t_bar)      cudaFree(t_bar);
}


void Linear::rhs(MomentsG* G, Fields* f, MomentsG* GRhs) {

  if (ks) {
    rhs_ks <<< dG, dB >>> (G->G(), GRhs->G(), grids_->ky, pars_->eps_ks);
    return;
  }

  // calculate conservation terms for collision operator
  int nn1 = grids_->NxNycNz;  int nt1 = min(nn1, 256);  int nb1 = 1 + (nn1-1)/nt1;
  if (pars_->collisions)  conservation_terms <<< nb1, nt1 >>>
			    (upar_bar, uperp_bar, t_bar, G->G(), f->phi, geo_->kperp2, G->zt(), G->r2());

  // to be safe, start with zeros on RHS
  GRhs->set_zero();

  // Free-streaming requires parallel FFTs, so do that first
  streaming_rhs <<< dGs, dBs >>> (G->G(), f->phi, geo_->kperp2, G->r2(), geo_->gradpar, G->vt(), G->zt(), GRhs->G());
  grad_par->dz(GRhs);
  
  // calculate most of the RHS
  cudaFuncSetAttribute(rhs_linear, cudaFuncAttributeMaxDynamicSharedMemorySize, 12*1024*sizeof(cuComplex));    
  rhs_linear<<<dimGrid, dimBlock, sharedSize>>>
      	(G->G(), f->phi, upar_bar, uperp_bar, t_bar,
        geo_->kperp2, geo_->cv_d, geo_->gb_d, geo_->bgrad, 
	 grids_->ky, G->vt(), G->zt(), G->tz(), G->nu(), G->tp(), G->up(), G->fp(), G->r2(), 
	 GRhs->G());

  // closures
  switch (pars_->closure_model_opt) {case Closure::none : break; closures->apply_closures(G, GRhs);}
  
  // hypercollisions
  if(pars_->hypercollisions) hypercollisions<<<dimGrid,dimBlock>>>(G->G(),
								   pars_->nu_hyper_l,
								   pars_->nu_hyper_m,
								   pars_->p_hyper_l,
								   pars_->p_hyper_m, GRhs->G());
  // hyper in k-space
  if(pars_->hyper) hyperdiff <<<dimGridh,dimBlockh>>>(G->G(), grids_->kx, grids_->ky,
						      pars_->nu_hyper, pars_->D_hyper, GRhs->G());
}






