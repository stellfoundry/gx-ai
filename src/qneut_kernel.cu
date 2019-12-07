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
    int idy = idxyz%nyc;
    int idx = (idxyz/nyc)%nx;
    if ( unmasked(idx, idy) ) {
      for(int is=0; is<nspecies; is++) {
	float b_s = kperp2[idxyz]*species[is].rho2;
	for(int l=0; l<nl; l++) {
	  // sum over l for m=0
	  // G_(...) is defined by macro above
	  // Each thread does the full Laguerre sum for a particular (kx, ky, z) element
	  nbar[idxyz] = nbar[idxyz] + Jflr(l,b_s)*G_(idxyz, l, 0, is);
	}	
      }
    }
  }
}

__global__ void qneutAdiab_part1(cuComplex* PhiAvgNum_tmp, cuComplex* nbar,
				 float* kperp2, float* jacobian, specie* species, float ti_ov_te)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  if ( unmasked(idx, idy) && idz < nz) {
  
    unsigned int index = idy + nyc*idx + nx*nyc*idz;
    float pfilter2 = 0.;
    
    // BD This species stuff will need to be changed when species are shared over GPUs. 
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
  
  if ( unmasked(idx, idy) && idz < nz) {

    unsigned int index = idy + nyc*idx + nx*nyc*idz;
    unsigned int idxy  = idy + nyc*idx;

    float pfilter2 = 0.;
    
    // BD multiple GPUs will require change here
    for(int is=0; is<nspecies; is++) {
      specie s = species[is];
      pfilter2 += s.dens*s.z*s.zt*( 1. - g0(kperp2[index]*s.rho2) );
    }

    // This is okay because PhiAvgNum_zSum is local to each thread
    cuDoubleComplex PhiAvgNum_zSum;
    PhiAvgNum_zSum.x = (double) 0.;
    PhiAvgNum_zSum.y = (double) 0.;

    for(int i=0; i<nz; i++) {
      PhiAvgNum_zSum.x = (double) PhiAvgNum_zSum.x + PhiAvgNum_tmp[idxy + i*nx*nyc].x;
      PhiAvgNum_zSum.y = (double) PhiAvgNum_zSum.y + PhiAvgNum_tmp[idxy + i*nx*nyc].y;
    }
    
    cuDoubleComplex PhiAvg;	
    if (idy == 0 && idx != 0) {
      PhiAvg.x = PhiAvgNum_zSum.x/( (double) PhiAvgDenom[idx] ); 
      PhiAvg.y = PhiAvgNum_zSum.y/( (double) PhiAvgDenom[idx] ); 
    } else {
      PhiAvg.x = 0.; PhiAvg.y = 0.;
    }
    
    Phi[index].x = ( nbar[index].x + ti_ov_te*PhiAvg.x ) / (ti_ov_te + pfilter2);
    Phi[index].y = ( nbar[index].y + ti_ov_te*PhiAvg.y ) / (ti_ov_te + pfilter2);
  }
}

__global__ void calc_phiavgdenom(float* PhiAvgDenom, float* kperp2,
				 float* jacobian, specie* species, float ti_ov_te)
{   
  unsigned int idx = get_id1();
  
  if ( idx > 0 && idx < nx) {
    int ikx = get_ikx(idx);
    if (ikx <  (nx-1)/3+1 && ikx > -(nx-1)/3-1) {

      float pfilter2;
      PhiAvgDenom[idx] = 0.;  

      for(int i=0; i<nz; i++) {
	pfilter2 = 0.;
	for (int is=0; is<nspecies; is++) {
	  specie s = species[is];
	  pfilter2 += s.dens*s.z*s.zt*(1. - g0(kperp2[(idx + i*nx)*nyc]*s.rho2));
	}	
	PhiAvgDenom[idx] = PhiAvgDenom[idx] + jacobian[i] * pfilter2 / (ti_ov_te + pfilter2);
      }
    }
  }
}

// Is this really set up correctly? 
__global__ void add_source(cuComplex* f, float source)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();
  
  if ( unmasked(idx, idy) && idz < nz) {
    unsigned int index = idy + nyc*idx + nx*nyc*idz;
    f[index].x = f[index].x + source;
  }
}

__global__ void qneutAdiab(cuComplex* Phi, cuComplex* nbar,
			   float* kperp2, float* jacobian,
			   specie* species, float ti_ov_te)
{
  unsigned int idy = get_id1();
  unsigned int idx = get_id2();
  unsigned int idz = get_id3();

  // BD all these qneut routines now assume we are working with the dealiased grids only
  if ( unmasked(idx, idy) && idz < nz) {

    unsigned int index = idy + nyc*idx + nx*nyc*idz;
    float pfilter2 = 0.;
    
    // BD Check this for parallel correctness when nspecies > 1
    for(int is=0; is<nspecies; is++) {
      specie s = species[is];
      pfilter2 += s.dens*s.z*s.zt*( 1. - g0(kperp2[index]*s.rho2) );
    }
    
    Phi[index] = ( nbar[index] / (ti_ov_te + pfilter2 ) ) * jacobian[idz];
  }
}

