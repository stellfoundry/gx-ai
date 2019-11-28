#pragma once
#include "device_funcs.h"
#include "geometry.h"
#include "species.h"
#include "cuda_constants.h"

# define G_(XYZ, L, M, S) g[(XYZ) + nx*nyc*nz*(L) + nx*nyc*nz*nl*(M) + nx*nyc*nz*nl*nm*(S)]
__global__ void real_space_density(cuComplex* nbar, cuComplex* g, float *kperp2, specie *species) 
{
  unsigned int idxyz = get_id1();

  if(idxyz<nx*nyc*nz) {
    nbar[idxyz] = make_cuComplex(0., 0.);
    for(int is=0; is<nspecies; is++) {
      float b_s = kperp2[idxyz]*species[is].rho2;

      for(int l=0; l<nl; l++) {
        // sum over l for m=0
        // G_(...) is defined by macro above
        nbar[idxyz] = nbar[idxyz] + Jflr(l,b_s)*G_(idxyz, l, 0, is);
      }
    }
  }
}

// We should probably specify tau_e == T_e/T_ref = 1/ti_ov_te since the paper does it that way
__global__ void qneutAdiab_part1(cuComplex* PhiAvgNum_tmp, cuComplex* nbar, float* kperp2, float* jacobian, specie* species, float ti_ov_te)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

    if( !(idy==0 && idx==0) && idy<nyc && idx<nx && idz<nz ) {

      unsigned int index = idy + nyc*idx + nx*nyc*idz;

      float pfilter2 = 0.;

      for(int is=0; is<nspecies; is++) {
        specie s = species[is];
        pfilter2 += s.dens*s.z*s.zt*( 1. - g0(kperp2[index]*s.rho2) );
      }

      PhiAvgNum_tmp[index] = ( nbar[index] / (ti_ov_te + pfilter2 ) ) * jacobian[idz];    
    }
}

__global__ void qneutAdiab_part2(cuComplex* Phi, cuComplex* PhiAvgNum_tmp, cuComplex* nbar,
				 float* PhiAvgDenom, float* kperp2, float* jacobian,
				 specie* species, float ti_ov_te)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  
  if( !(idy==0 && idx==0) && idy<nyc && idx<nx && idz<nz ) {
    
    unsigned int index = idy + nyc*idx + nx*nyc*idz;
    unsigned int idxy  = idy + nyc*idx;
        
    float pfilter2 = 0.;
    
    for(int is=0; is<nspecies; is++) {
      specie s = species[is];
      pfilter2 += s.dens*s.z*s.zt*( 1. - g0(kperp2[index]*s.rho2) );
    }

    cuDoubleComplex PhiAvgNum_zSum;
    PhiAvgNum_zSum.x = (double) 0.;
    PhiAvgNum_zSum.y = (double) 0.;
    
    
    // BD This is not guaranteed to be local
    for(int i=0; i<nz; i++) {
      PhiAvgNum_zSum.x = (double) PhiAvgNum_zSum.x + PhiAvgNum_tmp[idxy + i*nx*nyc].x;
      PhiAvgNum_zSum.y = (double) PhiAvgNum_zSum.y + PhiAvgNum_tmp[idxy + i*nx*nyc].y;
    }
    
    cuDoubleComplex PhiAvg;
    if(idy == 0 && idx!=0) { 
      PhiAvg.x = PhiAvgNum_zSum.x/( (double)PhiAvgDenom[idx] ); 
      PhiAvg.y = PhiAvgNum_zSum.y/( (double)PhiAvgDenom[idx] ); 
    }
    else {
      PhiAvg.x = 0.; PhiAvg.y = 0.;
    }
    
    Phi[index].x = ( nbar[index].x + ti_ov_te*PhiAvg.x ) / (ti_ov_te + pfilter2);
    Phi[index].y = ( nbar[index].y + ti_ov_te*PhiAvg.y ) / (ti_ov_te + pfilter2);

    //    Phi[index].x = idx;
    //    Phi[index].y = idy;
    
  }
}

__global__ void calc_phiavgdenom(float* PhiAvgDenom, float* PhiAvgDenom_tmpXZ, 
                                 float* kperp2, float* jacobian, specie* species, float ti_ov_te)
{   
  unsigned int idy = 0;
  unsigned int idx = get_id1();
  unsigned int idz = get_id2();
  
  if( idy==0 && idx!=0 && idx<nx && idz<nz ) {
    
    unsigned int idxz = idx + nx*idz;
    
    float pfilter2 = 0.;
    
    for(int is=0; is<nspecies; is++) {
      specie s = species[is];
      pfilter2 += s.dens*s.z*s.zt*( 1. - g0(kperp2[nyc*idxz]*s.rho2) );
    }
    
    PhiAvgDenom_tmpXZ[idxz] = pfilter2*jacobian[idz]/( ti_ov_te + pfilter2 );
    
    __syncthreads();
    
    PhiAvgDenom[idx] = 0.;
    
    for(int i=0; i<nz; i++) {
      PhiAvgDenom[idx] = PhiAvgDenom[idx] + PhiAvgDenom_tmpXZ[idx + nx*i];
    }
  }
}

__global__ void add_source(cuComplex* f, float source)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  
  if( !(idy==0 && idx==0) && idy<nyc && idx<nx && idz<nz ) {
    unsigned int index = idy + nyc*idx + nx*nyc*idz;
    f[index].x = f[index].x + source;
  }
}
