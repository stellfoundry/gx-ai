__global__ void qneut(cuComplex* Phi, cuComplex* nbartot_field, cuComplex* n_e, specie* s,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, float tau)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {


      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      unsigned int idxy = idy + (ny/2+1)*idx;
        
      float pfilter2 = 0.;    
      float bidx;
      
      for(int i=0; i<nspecies; i++) {
        bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
        pfilter2 = pfilter2 + s[i].dens*s[i].z*s[i].zt*( 1. - g0(bidx) );
      }
    
      Phi[index] = ( nbartot_field[index] - n_e[index] ) / pfilter2;
    }
  

}

__global__ void qneutETG(cuComplex* Phi, cuComplex* nbartot_field, specie* s, 
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, float tau)
{  
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {


      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
        
      float pfilter2 = 0.;    
      float bidx;
    
      for(int i=0; i<nspecies; i++) {
        bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
        pfilter2 = pfilter2 + s[i].dens*s[i].z*s[i].zt*( 1. - g0(bidx) );
      }

      
      Phi[index] = ( nbartot_field[index] ) / (tau + pfilter2);

     }
  }
      
}

__global__ void qneutAdiab(cuComplex* Phi, cuComplex* PhiAvgNum_tmp, cuComplex* nbartot_field, float* PhiAvgDenom, float* jacobian, specie* s, 
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, float tau)
{  
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {


      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      unsigned int idxy = idy + (ny/2+1)*idx;
        
      float pfilter2 = 0.;    
      float bidx;
      
      for(int i=0; i<nspecies; i++) {
        bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
        pfilter2 = pfilter2 + s[i].dens*s[i].z*s[i].zt*( 1. - g0(bidx) );
      }

      PhiAvgNum_tmp[index] = ( nbartot_field[index] / (tau + pfilter2 ) ) * jacobian[idz];

      //since nz<= dimBlock.z, we can sync and sum over z without having to call a new kernel
      __syncthreads();

      cuDoubleComplex PhiAvgNum_zSum;
      PhiAvgNum_zSum.x = (double) 0.;
      PhiAvgNum_zSum.y = (double) 0.;

      for(int i=0; i<nz; i++) {
        PhiAvgNum_zSum.x = (double) PhiAvgNum_zSum.x + PhiAvgNum_tmp[idxy + i*nx*(ny/2+1)].x;
        PhiAvgNum_zSum.y = (double) PhiAvgNum_zSum.y + PhiAvgNum_tmp[idxy + i*nx*(ny/2+1)].y;
      }

      cuDoubleComplex PhiAvg;
      if(idy == 0 && idx!=0) { 
        PhiAvg.x = PhiAvgNum_zSum.x/( (double)PhiAvgDenom[idx] ); 
        PhiAvg.y = PhiAvgNum_zSum.y/( (double)PhiAvgDenom[idx] ); 
      }
      else {
        PhiAvg.x = 0.; PhiAvg.y = 0.;
      }
      
      //Phi[index] = nbartot_field[index];
      Phi[index].x = ( nbartot_field[index].x + tau*PhiAvg.x ) / (tau + pfilter2);
      Phi[index].y = ( nbartot_field[index].y + tau*PhiAvg.y ) / (tau + pfilter2);

     }
      
}

__global__ void qneutAdiab_part1(cuComplex* PhiAvgNum_tmp, cuComplex* nbartot_field, float* jacobian, specie* s,
                      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, float tau)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {


      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      //unsigned int idxy = idy + (ny/2+1)*idx;

      float pfilter2 = 0.;
      float bidx;

      for(int i=0; i<nspecies; i++) {
        bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
        pfilter2 = pfilter2 + s[i].dens*s[i].z*s[i].zt*( 1. - g0(bidx) );
      }

      PhiAvgNum_tmp[index] = ( nbartot_field[index] / (tau + pfilter2 ) ) * jacobian[idz];
    
    }
}

__global__ void qneutAdiab_part2(cuComplex* Phi, cuComplex* PhiAvgNum_tmp, cuComplex* nbartot_field, float* PhiAvgDenom, float* jacobian, specie* s,
                      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, float tau)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {


      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      unsigned int idxy = idy + (ny/2+1)*idx;
        
      float pfilter2 = 0.;    
      float bidx;
      
      for(int i=0; i<nspecies; i++) {
        bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
        pfilter2 = pfilter2 + s[i].dens*s[i].z*s[i].zt*( 1. - g0(bidx) );
      }
  

      cuDoubleComplex PhiAvgNum_zSum;
      PhiAvgNum_zSum.x = (double) 0.;
      PhiAvgNum_zSum.y = (double) 0.;

      for(int i=0; i<nz; i++) {
        PhiAvgNum_zSum.x = (double) PhiAvgNum_zSum.x + PhiAvgNum_tmp[idxy + i*nx*(ny/2+1)].x;
        PhiAvgNum_zSum.y = (double) PhiAvgNum_zSum.y + PhiAvgNum_tmp[idxy + i*nx*(ny/2+1)].y;
      }

      cuDoubleComplex PhiAvg;
      if(idy == 0 && idx!=0) { 
        PhiAvg.x = PhiAvgNum_zSum.x/( (double)PhiAvgDenom[idx] ); 
        PhiAvg.y = PhiAvgNum_zSum.y/( (double)PhiAvgDenom[idx] ); 
      }
      else {
        PhiAvg.x = 0.; PhiAvg.y = 0.;
      }
      
      //Phi[index] = nbartot_field[idxy+27*nx*(ny/2+1)];
      Phi[index].x = ( nbartot_field[index].x + tau*PhiAvg.x ) / (tau + pfilter2);
      Phi[index].y = ( nbartot_field[index].y + tau*PhiAvg.y ) / (tau + pfilter2);

    }
}

/*__global__ void qneutAdiab_part1(cuComplex* Num_tmp, cuComplex* Denom_field, float tau, cuComplex* Dens_i, cuComplex* Tprp_i, float* jacobian, specie s,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {

      float bidx = b(s.rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      
      Num_tmp[index] = ( ( Dens_i[index]/(1.+bidx/2.) - (bidx*Tprp_i[index])/(2.*pow(1.+bidx/2.,2)) ) 
      			/ (tau - g0(bidx) + 1.) ) * jacobian[idz];
      
       
      
      
      Denom_field[index].x = ( ( 1. - g0(bidx) ) / ( tau - g0(bidx) + 1. ) ) * jacobian[idz];
      
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<zthreads ) {
        
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(s.rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);

	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
      
	Num_tmp[index] = ( ( Dens_i[index]/(1.+bidx/2.) - (bidx*Tprp_i[index])/(2.*pow(1.+bidx/2.,2)) ) / (tau - g0(bidx) + 1.) ) * jacobian[IDZ];


	Denom_field[index].x = ( ( 1. - g0(bidx) ) / ( tau - g0(bidx) + 1. ) ) * jacobian[IDZ];

      }
    }
  }      
      
} 

__global__ void qneutAdiab_part2(cuComplex* Phi, cuComplex* Num_tmp, cuComplex* Denom_field, float tau, cuComplex* Dens_i, cuComplex* Tprp_i, specie s,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {

      float bidx = b(s.rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      unsigned int idxy = idy + (ny/2+1)*idx;
      
      
      //sum numerator and denominator over z. zSum = zSum[idxy] = zSum(kx,ky).
      cuComplex zSum_Num;
      float zSum_Denom = 0.;
      zSum_Num.x = 0.;
      zSum_Num.y = 0.;
      for(int i=0; i<nz; i++) {
	zSum_Num = zSum_Num + Num_tmp[idxy + nx*(ny/2+1)*i];
	zSum_Denom = zSum_Denom + Denom_field[idxy + nx*(ny/2+1)*i].x;
      }       
            
      cuComplex PhiAvg;
      if(idy == 0 && idx!=0) { PhiAvg = zSum_Num/zSum_Denom; }
      else {PhiAvg.x = 0.; PhiAvg.y = 0.;}
      
      Phi[index] = ( Dens_i[index]/(1.+bidx/2.) - (bidx*Tprp_i[index])/(2.*pow(1.+bidx/2.,2)) + tau*PhiAvg ) 
      			/ (tau - g0(bidx) + 1.);
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<zthreads ) {
        
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(s.rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);

	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	unsigned int idxy = idy + (ny/2+1)*idx;
      
	//sum over z
	cuComplex zSum_Num;
	float zSum_Denom = 0.;
	zSum_Num.x = 0;
	zSum_Num.y = 0;
	for(int i=0; i<nz; i++) {
	  zSum_Num = zSum_Num + Num_tmp[idxy + nx*(ny/2+1)*i];
	  zSum_Denom = zSum_Denom + Denom_field[idxy + nx*(ny/2+1)*i].x;	  
	}       

	cuComplex PhiAvg;
	if(idy == 0 && idx!=0) { PhiAvg = zSum_Num/zSum_Denom; }
	else {PhiAvg.x = 0; PhiAvg.y = 0;}

	Phi[index] = ( Dens_i[index]/(1.+bidx/2.) - (bidx*Tprp_i[index])/(2.*pow(1.+bidx/2.,2))  + tau*PhiAvg ) / (tau - g0(bidx) + 1.);
      }
    }
  }      
      
} 
*/            

  
__global__ void nbar(cuComplex* nbar, cuComplex* Dens, cuComplex* Tprp, 
		      specie s, float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy(); 
  unsigned int idx = get_idx();
  unsigned int idz = get_idz(); 
  
  
    if( idy<(ny/2+1) && idx<nx && idz<nz ) {

      float bidx = b(s.rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      nbar[index] = (Dens[index]/(1.+bidx/2.) - bidx*Tprp[index]/(2.*pow(1.+bidx/2.,2.)))*s.dens*s.z;
      
    }
}    

__global__ void phiavgdenom(float* PhiAvgDenom, float* PhiAvgDenom_tmpXZ, float* jacobian, specie* s, float* kx, float* ky, float shat, float* gds2, float* gds21, float* gds22, float* bmagInv,float tau)
{   
  unsigned int idy = 0;
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if( idy==0 && idx!=0 && idx<nx && idz<nz ) {
 
  unsigned int idxz = idx + nx*idz;
  
  float pfilter2 = 0.;
  float bidx;
    
    for(int i=0; i<nspecies; i++) {
      bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
      pfilter2 = pfilter2 + s[i].dens*s[i].z*s[i].zt*( 1. - g0(bidx) );
    }
   
    PhiAvgDenom_tmpXZ[idxz] = pfilter2*jacobian[idz]/( tau + pfilter2 );
 
    __syncthreads();

    PhiAvgDenom[idx] = 0.;

    for(int i=0; i<nz; i++) {
      PhiAvgDenom[idx] = PhiAvgDenom[idx] + PhiAvgDenom_tmpXZ[idx + nx*i];
    }
    
  }

}


__global__ void ampere(cuComplex* Apar, cuComplex* ubartot_field, cuComplex* upar_e, float beta_e,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, float tau)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {


      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      unsigned int idxy = idy + (ny/2+1)*idx;
        
      float bidx;
      
      bidx = b(1., kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]); // just kperp^2, not (kperp rho)^2

      Apar[index] = tau*beta_e*( ubartot_field[index] - upar_e[index] ) / ( 2. * bidx );
      // DOES THIS NEED ADDITIONAL NORMALIZATIONS ??? also check ubartot
    
   }   

}






