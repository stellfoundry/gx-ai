#include "device_funcs.h"
#include "cuda_constants.h"

__device__ unsigned int get_id1(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ unsigned int get_id2(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ unsigned int get_id3(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}


// use stirling's approximation
__host__ __device__ float factorial(int m) {
  if(m<2) return 1.;
  else return sqrtf(2.*M_PI*m)*powf(m,m)*expf(-m)*(1.+1./(12.*m)+1./(288.*m*m));
}

__device__ float Jflr(int l, float b, bool enforce_JL_0) {
  if (l<0) return 0.;
  else if (l>=nl && enforce_JL_0) return 0;
  else return 1./factorial(l)*pow(-0.5*b, l)*expf(-b/2.);
}

__device__ float g0(float b) {

  float tol = 1.e-7;
  float tk, b2, b2sq;
  float g, x, xi, err;

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

__device__ float g1(float b) {

  float tol = 1.e-7;
  float tk, b2, b2sq;
  float g, x, xi, xp1i, err;

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

__device__ float sgam0 (float b) {return sqrt(g0(b));}

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

__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g)
{
  return cuCmulf(f,g);
}

__host__ __device__ cuDoubleComplex operator*(cuDoubleComplex f, cuDoubleComplex g)
{
  return cuCmul(f,g);
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

__host__ __device__ cuDoubleComplex operator/(cuDoubleComplex f, cuDoubleComplex g) 
{
  return cuCdiv(f,g);
}

__device__ int get_ikx(int idx) {
  if( idx<nx/2+1 )
    return idx;
  else
    return idx-nx;
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz];
  }
}

__global__ void add_scaled_singlemom_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2, double c3, cuComplex* m3)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz] + c3*m3[idxyz];
  }
}

__global__ void add_scaled_kernel(cuComplex* res, double c1, cuComplex* m1, double c2, cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int globalIdx = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[globalIdx] = c1*m1[globalIdx] + c2*m2[globalIdx];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
                 double c1, cuComplex* m1, double c2, cuComplex* m2, 
                 double c3, cuComplex* m3, double c4, cuComplex* m4,
                 double c5, cuComplex* m5)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int globalIdx = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[globalIdx] = c1*m1[globalIdx] + c2*m2[globalIdx] 
                         + c3*m3[globalIdx] + c4*m4[globalIdx] 
                         + c5*m5[globalIdx];
        }
      }
    }
  }
}

__global__ void scale_kernel(cuComplex* res, cuComplex* mom, double scalar)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int globalIdx = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[globalIdx] = scalar*mom[globalIdx];
        }
      }
    }
  }
}

__global__ void scale_kernel(cuComplex* res, cuComplex* mom, cuComplex scalar)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int globalIdx = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[globalIdx] = scalar*mom[globalIdx];
        }
      }
    }
  }
}

__global__ void scale_singlemom_kernel(cuComplex* res, cuComplex* mom, cuComplex scalar)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    res[idxyz] = scalar*mom[idxyz];
  }
}

__global__ void reality_kernel(cuComplex* g) 
{
  for(int s = 0; s < nspecies; s++) {
    for (int lm = threadIdx.z; lm < nm*nl; lm += blockDim.z) {
      for (int idz = threadIdx.y; idz<nz; idz+=blockDim.y) {
        for (int idx = threadIdx.x; idx<nx/2; idx+=blockDim.x) {
          int globalIdx = nyc*(idx + nx*idz) + nx*nyc*nz*lm + nx*nyc*nz*nl*nm*s; 
          int globalIdx2 = nyc*(nx-idx + nx*idz) + nx*nyc*nz*lm + nx*nyc*nz*nl*nm*s; 
          if(idx!=0) {
            g[globalIdx2].x = g[globalIdx].x;
            g[globalIdx2].y = -g[globalIdx].y;
          }
        }
      }
    }
  }
}

__global__ void reality_singlemom_kernel(cuComplex* mom) 
{
  for (int idz = threadIdx.y; idz<nz; idz+=blockDim.y) {
    for (int idx = threadIdx.x; idx<nx/2; idx+=blockDim.x) {
      unsigned int index = (ny/2+1)*idx + nx*(ny/2+1)*idz;
      unsigned int index2 = (ny/2+1)*(nx-idx) + nx*(ny/2+1)*idz;
      if(idx!=0) {
        mom[index2].x = mom[index].x;
        mom[index2].y = -mom[index].y;
      }
    }
  }
}

__device__ float atomicMaxFloat(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
