#include "device_funcs.h"
#include "cuda_constants.h"


__device__ float flr(float b) {
  float gam0 = g0(b);
  float gam1 = g1(b);
  float gam = sqrt(gam0);
  float rg = gam1/gam0;
  return -b/2. * (1.-rg) * gam;
}

__device__ float flr2(float b) {
  float gam0 = g0(b);
  float gam1 = g1(b);
  float gam = sqrt(gam0);
  float rg = gam1/gam0;
  return -(b/2.) * (1.+(1.-rg)*(1.-(b/2.)*(3.+rg))) * gam;
}

__device__ float nwgt(float b) {return 1./(1.+b/2.);}

__device__ float twgt(float b) {
  float d = 1./(1.+b/2.);
  return -b/2. * d * d;
}

__device__ float wgt(float b) {
  float gam0 = g0(b);
  return 1.-gam0;
}

__device__ float omegaD(float rho_vt, float kx, float ky, float shat, float gb,float gb0,float cv, float cv0)
{
  float shatInv;
  //if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  //} else {
  //  shatInv = 1.;
  //}

  
  return rho_vt* ( ky * (gb + cv) + kx * shatInv * (gb0 + cv0) );
  
}

__device__ float b(float rho, float kx, float ky, float shat, float gds2, float gds21, float gds22, float bmagInv)
{
  float shatInv;
  //if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  //} else {
  //  shatInv = 1.;
  //}

  float b = ( ky * (ky*gds2 + 2*kx*shatInv*gds21) + pow(kx*shatInv,2)*gds22 ) * pow(bmagInv,2) * pow(rho,2);
  return b;
}

__device__ float c(float kx, float gds22, float qsf, float eps, float bmagInv, float shat, float rho)
{ 
  float shatInv;
  if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  } else {
    shatInv = 1.;
  }
  
  // -k_r rho_pol / B
  return kx*shatInv*sqrt(gds22)*qsf/eps*bmagInv*rho;
}
