#include "device_funcs.h"
#include "cuda_constants.h"

__device__ unsigned int get_id1(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ unsigned int get_id2(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ unsigned int get_id3(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}


// use stirling's approximation
// switch to tgamma, or gsl? 
__host__ __device__ float factorial(int m) {
  if(m<2) return 1.;
  if(m==2) return 2.;
  if(m==3) return 6.;
  if(m==4) return 24.;
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
    x += 1.;
    err = abs(tk/g);
  }
  
  if(g<tol) g=tol; 
  return g;

}

__device__ float sgam0 (float b) {return sqrt(g0(b));}

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

__device__ bool not_fixed_eq(int idxyz) {
  int idxy_fixed = iky_fixed + ikx_fixed*nyc;
  if ( idxyz%(nx*nyc) == idxy_fixed )
    return false;
  else
    return true;
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

__global__ void add_scaled_singlemom_kernel(cuComplex* res, cuComplex c1, cuComplex* m1, cuComplex c2, cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    res[idxyz] = c1*m1[idxyz] + c2*m2[idxyz];
  }
}

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, bool eqfix)
{
  unsigned int idxyz = get_id1();
  if(not_fixed_eq(idxyz) && idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2,
				  double c3, cuComplex* m3)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig]
	          + c3 * m3[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2,
				  double c3, cuComplex* m3, bool eqfix)
{
  unsigned int idxyz = get_id1();
  if(not_fixed_eq(idxyz) && idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig]
	          + c3 * m3[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig] 
	          + c3 * m3[ig]
	          + c4 * m4[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4, bool eqfix)
{
  unsigned int idxyz = get_id1();
  if(not_fixed_eq(idxyz) && idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
	          + c2 * m2[ig] 
	          + c3 * m3[ig]
	          + c4 * m4[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4,
				  double c5, cuComplex* m5)
{
  unsigned int idxyz = get_id1();
  if(idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
   	          + c2 * m2[ig] 
	          + c3 * m3[ig]
	          + c4 * m4[ig] 
	          + c5 * m5[ig];
        }
      }
    }
  }
}

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, cuComplex* m1,
				  double c2, cuComplex* m2, 
				  double c3, cuComplex* m3,
				  double c4, cuComplex* m4,
				  double c5, cuComplex* m5, bool eqfix)
{
  unsigned int idxyz = get_id1();
  if(not_fixed_eq(idxyz) && idxyz<nx*nyc*nz) {
    for(int s = 0; s < nspecies; s++) {
      for (int m = threadIdx.z; m < nm; m += blockDim.z) {
        for (int l = threadIdx.y; l < nl; l += blockDim.y) {
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = c1 * m1[ig]
   	          + c2 * m2[ig] 
	          + c3 * m3[ig]
	          + c4 * m4[ig] 
	          + c5 * m5[ig];
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
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = scalar*mom[ig];
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
          int ig = idxyz + nx*nyc*nz*l + nx*nyc*nz*nl*m + nx*nyc*nz*nl*nm*s; 
          res[ig] = scalar*mom[ig];
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
    for (int lm = threadIdx.z; lm < nm*nl;     lm += blockDim.z) {
      for (int idz = threadIdx.y; idz<nz;     idz += blockDim.y) {
        for (int idx = threadIdx.x; idx<nx/2; idx += blockDim.x) {
          int ig  = nyc*(idx    + nx*idz) + nx*nyc*nz*lm + nx*nyc*nz*nl*nm*s; 
          int ig2 = nyc*(nx-idx + nx*idz) + nx*nyc*nz*lm + nx*nyc*nz*nl*nm*s; 
	  //	  if (ig<nx*nyc*nz*nl*nm*s && ig2<nx*nyc*nz*nl*nm*s) {
	    if(idx==0) {
	      g[ig].x = 0.;
	      g[ig].y = 0.;
	    } else if(idx!=0) {
	      g[ig2].x = g[ig].x;
	      g[ig2].y = -g[ig].y;
	    }
	    //	  }
	}
      }
    }
  }
}

__global__ void reality_singlemom_kernel(cuComplex* mom) 
{
  for (int idz = threadIdx.y; idz<nz; idz+=blockDim.y) {
    for (int idx = threadIdx.x; idx<nx/2; idx+=blockDim.x) {
      unsigned int index  = (ny/2+1)*idx      + nx*(ny/2+1)*idz;
      unsigned int index2 = (ny/2+1)*(nx-idx) + nx*(ny/2+1)*idz;
      if(idx==0) {
	mom[index].x = 0.;
	mom[index].y = 0.;
      } else if(idx!=0) {
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
__global__ void calc_bgrad(float* bgrad, float* bgrad_temp, float* bmag, float scale)
{
  unsigned int idx = get_id1();
  if (idx < nz) bgrad[idx] = ( bgrad_temp[idx] / bmag[idx] ) * scale;
}

__global__ void init_kperp2(float* kperp2, float* kx, float* ky,
			    float* gds2, float* gds21, float* gds22,
			    float* bmagInv, float shat) 
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  float shatInv = 1./shat; // Needs a test for zero

  if(idy<nyc && idx<nx && idz<nz) {
    unsigned int idxyz = idy + nyc*idx + nx*nyc*idz;
    kperp2[idxyz] = ( ky[idy] * ( ky[idy] * gds2[idz] 
                      + 2. * kx[idx] * shatInv * gds21[idz]) 
                      + pow( kx[idx] * shatInv, 2) * gds22[idz] ) 
                      * pow( bmagInv[idz], 2); 
  }
}

__global__ void set_mask(cuComplex *g)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  if (idx>(nx-1)/3 && idx<2*nx/3 && idy>(ny-1)/3 && idy<ny) {
    unsigned int ixy = idy + nyc*idx;
    for (int k=0; k<nz*nm*nl*nspecies; k++) {
      unsigned index = ixy*k;
      g[index].x = 0.;
      g[index].y = 0.;
    }
  }
}


__global__ void init_omegad(float* omegad, float* cv_d, float* gb_d, float* kx, float* ky,
			    float* cv, float* gb, float* cv0, float* gb0, float shat) 
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  float shatInv = 1./shat; // BD this needs an exception for shat = 0.

  // cv/gb is the y-directed part of the curvature / grad-B drift
  // cv0/gb0 is the part proportional to the theta_0, aka the x-directed component

  if(idy<nyc && idx<nx && idz<nz) {
    unsigned int idxyz = idy + nyc*idx + nx*nyc*idz;
    cv_d[idxyz] = ky[idy] * cv[idz] + kx[idx] * shatInv * cv0[idz] ;     
    gb_d[idxyz] = ky[idy] * gb[idz] + kx[idx] * shatInv * gb0[idz] ;
    omegad[idxyz] = cv_d[idxyz] + gb_d[idxyz];
  }
}
