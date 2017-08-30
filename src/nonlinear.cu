#include "nonlinear.h"

Nonlinear::Nonlinear(Grids* grids, Geometry* geo) :
  grids_(grids), geo_(geo)
{
  laguerre = new LaguerreTransform(grids_);
}

Nonlinear::~Nonlinear() 
{
  delete laguerre;
}

Nonlinear::nlps5d()
{
  grad_perp->dx(G, grids_->Nz*grids_->Nmoms, dG_dx);
  grad_perp->dy(G, grids_->Nz*grids_->Nmoms, dG_dy);

  laguerre->toGrid(dG_dx, dg_dx);
  laguerre->toGrid(dG_dy, dg_dy);

  laguerre->j0(phi, J0phi);
  grad_perp->dx(J0phi, grids_->Nz*(laguerre->J+1), dJ0phi_dx);
  grad_perp->dy(J0phi, grids_->Nz*(laguerre->J+1), dJ0phi_dy);

  bracket(dg_dx, dJ0phi_dy, dg_dy, dJ0Phi_dx, g_res);

  laguerre->toSpectral(g_res, G_res);
}
