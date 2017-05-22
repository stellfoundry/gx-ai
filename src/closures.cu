#include "closures.h"
#include "device_funcs.h"
#include "cuda_constants.h"
#include "get_error.h"

__global__ void toroidal_closures(cuComplex* g, cuComplex* gRhs, float* omegad, cuComplex* nu);

Beer42::Beer42(Grids* grids, float* omegad):
    grids_(grids), omegad_(omegad)
{
  // set up parallel derivatives, including |kpar|
  grad_par = new GradParallel(grids_);
  abs_grad_par = new GradParallel(grids_, true);

  cudaMalloc((void**) &tmp, sizeof(cuComplex)*grids_->NxNycNz);

  D_par = 2.*sqrt(M_PI)/(3.0*M_PI-8.0);
  D_perp = sqrt(M_PI)/2.;
  Beta_par = (32.-9.*M_PI)/(3.*M_PI-8.);

  cuComplex nu_h[11];
  nu_h[0].x = 0.; // dummy
  nu_h[0].y = 0.; // dummy
  nu_h[1].x=2.019;
  nu_h[1].y=-1.620;
  nu_h[2].x=.433;  
  nu_h[2].y= 1.018;
  nu_h[3].x=-.256; 
  nu_h[3].y=1.487; 
  nu_h[4].x=-.070; 
  nu_h[4].y=-1.382;
  nu_h[5].x=-8.927;
  nu_h[5].y=12.649;
  nu_h[6].x= 8.094;
  nu_h[6].y= 12.638-11.;
  nu_h[7].x= 13.720;
  nu_h[7].y= 5.139-3.;
  nu_h[8].x= 3.368; 
  nu_h[8].y= -8.110;
  nu_h[9].x= 1.974; 
  nu_h[9].y= -1.984-1.;
  nu_h[10].x= 8.269;
  nu_h[10].y= 2.060-7.;

  cudaMalloc((void**) &nu, sizeof(cuComplex)*11);
  cudaMemcpy(nu, nu_h, sizeof(cuComplex)*11, cudaMemcpyHostToDevice);

  // 1d thread blocks over xyz
  dimBlock = 512;
  dimGrid = grids_->NxNycNz/dimBlock.x+1;
  
}

Beer42::~Beer42() {
}

int Beer42::apply_closures(Moments* m, Moments* mRhs) 
{
  // mask unevolved moments
  cudaMemset(mRhs->gHL(2,1), 0., sizeof(cuComplex)*grids_->NxNycNz);
  cudaMemset(mRhs->gHL(3,1), 0., sizeof(cuComplex)*grids_->NxNycNz);

  // parallel terms (each must be done separately because of FFTs)
  grad_par->eval(m->gHL(2,0), tmp);
  add_scaled_singlemom_kernel<<<dimGrid,dimBlock>>>
      (mRhs->gHL(3,0), 1., mRhs->gHL(3,0), -Beta_par/sqrt(3.), tmp);

  abs_grad_par->eval(m->gHL(3,0), tmp);
  add_scaled_singlemom_kernel<<<dimGrid,dimBlock>>>
      (mRhs->gHL(3,0), 1., mRhs->gHL(3,0), -sqrt(2.)*D_par, tmp);

  abs_grad_par->eval(m->gHL(1,1), tmp);
  add_scaled_singlemom_kernel<<<dimGrid,dimBlock>>>
      (mRhs->gHL(1,1), 1., mRhs->gHL(1,1), -sqrt(2.)*D_perp, tmp);
  
  // toroidal terms
  toroidal_closures<<<dimGrid,dimBlock>>>(m->ghl, mRhs->ghl, omegad_, nu);

  return 0;
}

# define LM(L, M) idxyz + nx*nyc*nz*(M) + nx*nyc*nz*nlaguerre*(L)
__global__ void toroidal_closures(cuComplex* g, cuComplex* gRhs, float* omegad, cuComplex* nu)
{
  unsigned int idxyz = get_id1();

  if(idxyz<nx*nyc*nz) {

    const cuComplex iomegad = make_cuComplex(0., omegad[idxyz]);
    const float abs_omegad = abs(omegad[idxyz]);

    gRhs[LM(2,0)] = gRhs[LM(2,0)]
      - sqrtf(2)*abs_omegad*( nu[1].x*sqrtf(2)*g[LM(2,0)] + nu[2].x*g[LM(0,1)] )
      - sqrtf(2)* iomegad * ( nu[1].y*sqrtf(2)*g[LM(2,0)] + nu[2].y*g[LM(0,1)] );

    gRhs[LM(0,1)] = gRhs[LM(0,1)]
      - 2.*abs_omegad*( nu[3].x*sqrtf(2)*g[LM(2,0)] + nu[4].x*g[LM(0,1)] )
      - 2.* iomegad * ( nu[3].y*sqrtf(2)*g[LM(2,0)] + nu[4].y*g[LM(0,1)] );

    gRhs[LM(3,0)] = gRhs[LM(3,0)]
      - 1./sqrtf(6)*abs_omegad*( nu[5].x*g[LM(1,0)] + nu[6].x*sqrtf(6)*g[LM(3,0)] + nu[7].x*g[LM(1,1)] )
      - 1./sqrtf(6)* iomegad * ( nu[5].y*g[LM(1,0)] + nu[6].y*sqrtf(6)*g[LM(3,0)] + nu[7].y*g[LM(1,1)] );

    gRhs[LM(1,1)] = gRhs[LM(1,1)]
      - abs_omegad*( nu[8].x*g[LM(1,0)] + nu[9].x*sqrtf(6)*g[LM(3,0)] + nu[10].x*g[LM(1,1)] )
      -  iomegad * ( nu[8].y*g[LM(1,0)] + nu[9].y*sqrtf(6)*g[LM(3,0)] + nu[10].y*g[LM(1,1)] );
  }

}
