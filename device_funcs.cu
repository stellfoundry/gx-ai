#include "device_funcs.h"

__device__ unsigned int get_id1(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ unsigned int get_id2(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ unsigned int get_id3(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}

__device__ double kperp2(Geometry::kperp2_struct* kp2, int ix, int iy, int iz, int is) {
  float shatInv = 1./kp2->shat;

  return ( kp2->ky[iy] * (kp2->ky[iy]*kp2->gds2[iz] + 2.*kp2->kx[ix]*shatInv*kp2->gds21[iz]) + pow(kp2->kx[ix]*shatInv,2)*kp2->gds22[iz] )
                     * pow(kp2->bmagInv[iz],2) * pow(kp2->species[is].rho,2);
}

__device__ unsigned long long factorial(int m) {
  if(m<2) return (unsigned long long) 1;
  return (unsigned long long) m*factorial(m-1);
}

__device__ double Jflr(int m, double b) {
  return 1./factorial(m)*pow(-b/2.,m)*expf(-b/2.);
}

__device__ double g0(double b) {

  double tol = 1.e-7;
  double tk, b2, b2sq;
  double g, x, xi, err;

  if (b < tol) {return 1.0;}

  b2 = 0.5 * b;
  b2sq = b2 * b2;
  tk = __expf(-b);
  g = tk;

  x = 1.;
  err = 1.;
  
  while (err > tol) {
    xi = 1./x;
    tk  = tk * b2sq * xi * xi;
    g += tk;
    x  += 1.;
    err = abs(tk/g);
  }
  
  if(g<tol) g=tol; 
  return g;

}

__device__ double g1(double b) {

  double tol = 1.e-7;
  double tk, b2, b2sq;
  double g, x, xi, xp1i, err;

  if (b < tol) {return 0.0;}

  b2 = 0.5 * b;
  b2sq = b2 * b2;
  tk = __expf(-b) * b2;
  g = tk;

  x = 1.;
  err = 1.;
  
  while (err > tol) {
    xi = 1./x;
    xp1i=1./(1.+x);
    tk  = tk * b2sq * xi * xp1i;
    g += tk;
    x  += 1.;
    err = abs(tk/g);
  }
  
  if(g<tol) g=tol; 
  return g;

}

__device__ double sgam0 (double b) {return sqrt(g0(b));}

__device__ double flr(double b) {
  double gam0 = g0(b);
  double gam1 = g1(b);
  double gam = sqrt(gam0);
  double rg = gam1/gam0;
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

__device__ double b(float rho, float kx, float ky, float shat, float gds2, float gds21, float gds22, float bmagInv)
{
  float shatInv;
  //if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  //} else {
  //  shatInv = 1.;
  //}

  double b = ( ky * (ky*gds2 + 2*kx*shatInv*gds21) + pow(kx*shatInv,2)*gds22 ) * pow(bmagInv,2) * pow(rho,2);
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

__host__ __device__ bool operator>(cuComplex f, cuComplex g)
{
  return f.x*f.x+f.y*f.y > g.x*g.x+g.y*g.y;
}

__host__ __device__ bool operator<(cuComplex f, cuComplex g)
{
  return f.x*f.x+f.y*f.y < g.x*g.x+g.y*g.y;
}


__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g) 
{
  return cuCaddf(f,g);
} 

__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g)
{
  return cuCsubf(f,g);
}  

__host__ __device__ cuComplex operator-(cuComplex f)
{
  cuComplex zero;
  zero.x = 0.;
  zero.y = 0.;
  return cuCsubf(zero,f);
}  
/*
__host__ __device__ cuComplex operator*(float scaler, cuComplex f) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, float scaler) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}
*/
__host__ __device__ cuComplex operator*(double scaler, cuComplex f) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, double scaler) 
{
  cuComplex result;
  result.x = scaler*f.x;
  result.y = scaler*f.y;
  return result;
}

__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g)
{
  return cuCmulf(f,g);
}

__host__ __device__ cuComplex operator/(cuComplex f, float scaler)
{
  cuComplex result;
  result.x = f.x / scaler;
  result.y = f.y / scaler;
  return result;
}

__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) 
{
  return cuCdivf(f,g);
}

__device__ int get_ikx(int idx) {
  if( idx<nx/2+1 )
    return idx;
  else
    return idx-nx;
}

