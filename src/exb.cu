#include "exb.h"
//=======================================
// exb
// object for handling flow shear terms.
//=======================================
ExB_GK::ExB_GK(Parameters* pars_, Grids* grids, Geometry* geo) :
  pars_(pars_), grids_(grids), geo_(geo), phi_tmp(nullptr), g_tmp(nullptr)
{ 
  checkCuda(cudaMalloc(&phi_tmp, sizeof(cuComplex)*grids_->NxNycNz));
  checkCuda(cudaMalloc(&g_tmp  , sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl*grids_->Nm));

  //unsigned int nxkyz = grids_->NxNycNz;
  //unsigned int nlag = grids_->Nj;
  //int nbx = min(32, grids_->NxNycNz);   int ngx = 1 + (grids_->NxNycNz-1)/nbx;
  //int nby = min(16, grids_->Nj);   int ngy = 1 + (grids_->Nj-1)/nby;
  //dBk = dim3(nbx, nby, 1);
  //dGk = dim3(ngx, ngy, 1);
  // Dimensions for the field_shift kernels.
  //nt1 = pars_->i_share;   int nb1 = 1 + (nbx-1)/nt1;
  // JFP: this is likely wrong.
  //dimBlockfield = nt1;
  //dimGridfield = nb1;
  
  // Nx*Nyc kernels
  dimBlock_xy = dim3(32,16);
  dimGrid_xy  = dim3(1+(grids->Nyc-1)/dimBlock_xy.x, 1+(grids->Nx-1)/dimBlock_xy.y);
  
  // Nx*Nyc*Nz kernels
  dimBlock_xyz = dim3(32, 4, 4);
  dimGrid_xyz  = dim3(1+(grids->Nyc-1)/dimBlock_xyz.x, 1+(grids->Nx-1)/dimBlock_xyz.y, 1+(grids->Nz-1)/dimBlock_xyz.z);

  // Nx*Nyc*Nz*Nl*Nm kernels
  int nn1 = grids_->Nyc;             int nt1 = min(nn1, 16);    int nb1 = (nn1-1)/nt1 + 1;
  int nn2 = grids_->Nx*grids_->Nz;   int nt2 = min(nn2, 16);    int nb2 = (nn2-1)/nt2 + 1;
  int nn3 = grids_->Nm*grids_->Nl;   int nt3 = min(nn3,  4);    int nb3 = (nn3-1)/nt3 + 1;
  
  dimBlock_xyzlm = dim3(nt1, nt2, nt3);
  dimGrid_xyzlm  = dim3(nb1, nb2, nb3);

  CP_TO_GPU (grids_->x, grids_->x_h, sizeof(float)*grids_->Nx); //find a better place for this? only need to do it once for exb/ntft

}
ExB_GK::~ExB_GK()
{
  //if (closures) delete closures;
  //if (favg)       cudaFree(favg);
  if (phi_tmp)  cudaFree(phi_tmp);
  if (g_tmp)    cudaFree(g_tmp);
}
void ExB_GK::flow_shear_shift(MomentsG* G, Fields* f, double dt)
{
  // shift moments and fields in kx to account for ExB shear
  kxstar_phase_shift<<<dimBlock_xy, dimGrid_xy>>>(grids_->kxstar, grids_->kxbar_ikx_new, grids_->kxbar_ikx_old, grids_->ky, grids_->x, grids_->phasefac_exb, pars_->g_exb, dt, pars_->x0, pars_->ExBshear_phase);
  // update geometry
  if (pars_->nonTwist) {
    geo_shift_ntft<<<dimBlock_xyz, dimGrid_xyz>>>(grids_->kxstar, grids_->ky, geo_->cv_d, geo_->gb_d, geo_->kperp2,
                             geo_->cvdrift, geo_->cvdrift0, geo_->gbdrift, geo_->gbdrift0, geo_->omegad,
                             geo_->gds2, geo_->gds21, geo_->gds22, geo_->bmagInv, pars_->shat,
			     geo_->ftwist, geo_->deltaKx, geo_->m0, pars_->x0, grids_->iKx,
			     pars_->g_exb, dt, grids_->kx);
  } else {
    geo_shift<<<dimBlock_xyz, dimGrid_xyz>>>(grids_->kxstar, grids_->ky, geo_->cv_d, geo_->gb_d, geo_->kperp2,
                             geo_->cvdrift, geo_->cvdrift0, geo_->gbdrift, geo_->gbdrift0, geo_->omegad,
                             geo_->gds2, geo_->gds21, geo_->gds22, geo_->bmagInv, pars_->shat);
  }
  // shift fields
  CP_TO_GPU (phi_tmp, f->phi,  sizeof(cuComplex)*grids_->NxNycNz);
  CP_TO_GPU (g_tmp,   G->G(),  sizeof(cuComplex)*grids_->NxNycNz*grids_->Nl*grids_->Nm);
  field_shift<<<dimGrid_xyz,dimBlock_xyz>>>    (f->phi, phi_tmp, grids_->kxbar_ikx_new, grids_->kxbar_ikx_old, pars_->g_exb);
  g_shift <<< dimGrid_xyzlm, dimBlock_xyzlm>>> (G->G(), g_tmp, grids_->kxbar_ikx_new, grids_->kxbar_ikx_old, pars_->g_exb);
  //if (pars_->fapar > 0.) {
  //  field_shift<<<<dimGridfield,dimBlockfield>>>(f->apar,grids_->kxbar_ikx);
  //}
  //if (pars_->fbpar > 0.) field_shift<<<dimGridfield,dimBlockfield>>>(f->bpar,kxbar_ikx); // JFP: note: to update once we have bpar.
  // shift dist function, batching in m.
  //for(int m=grids_->m_lo; m<grids_->m_up; m++) {
  //  int m_local = m - grids_->m_lo;
  //  printf("m_local = %d \n", m_local);
  //  g_shift <<< dimGrid_xyzl, dimBlock_xyzl >>> (G->Gm(m_local),grids_->kxbar_ikx, m_local);
  //}
}
