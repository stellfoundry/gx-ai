__global__ void qneut(cuComplex* Phi, cuComplex* Dens_e, cuComplex* Dens_i, cuComplex* Tprp_i, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy(); 
  unsigned int idx = get_idx();
  unsigned int idz = get_idz(); 
  
  
  if(nz<=zthreads) {
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      //CHECK SIGNS
      Phi[index] = ( Dens_e[index] - Dens_i[index]/(1.+bidx/2.) + (bidx*Tprp_i[index])/(2.*pow(1.+bidx/2.,2)) ) / (g0(bidx) - 1.); 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<zthreads ) {

	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	Phi[index] = ( Dens_e[index] - Dens_i[index]/(1.+bidx/2.) + (bidx*Tprp_i[index])/(2.*pow(1.+bidx/2.,2)) ) / (g0(bidx) - 1.);
      }
    }
  }      
      
}     

__global__ void qneutETG(cuComplex* Phi, cuComplex* nbartot_field, specie* s, 
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv, float tau)
{  
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zblockthreads) {
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {


      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
        
      float pfilter2 = 0.;    
    
      for(int i=0; i<nspecies; i++) {
        float bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
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
  
  if(nz<=zblockthreads) {
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {


      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      unsigned int idxy = idy + (ny/2+1)*idx;
        
      float pfilter2 = 0.;    
    
      for(int i=0; i<nspecies; i++) {
        float bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
        pfilter2 = pfilter2 + s[i].dens*s[i].z*s[i].zt*( 1. - g0(bidx) );
      }

      PhiAvgNum_tmp[index] = ( nbartot_field[index] / (tau + pfilter2 ) ) * jacobian[idz];

      //since nz<= dimBlock.z, we can sync and sum over z without having to use a new kernel
      __syncthreads();

      cuComplex PhiAvgNum_zSum;
      PhiAvgNum_zSum.x = 0.;
      PhiAvgNum_zSum.y = 0.;

      for(int i=0; i<nz; i++) {
        PhiAvgNum_zSum = PhiAvgNum_zSum + PhiAvgNum_tmp[idxy + i*nx*(ny/2+1)];
      }

      cuComplex PhiAvg;
      if(idy == 0 && idx!=0) { PhiAvg = PhiAvgNum_zSum/PhiAvgDenom[idx]; }
      else {PhiAvg.x = 0.; PhiAvg.y = 0.;}
      
      Phi[index] = ( nbartot_field[index] + tau*PhiAvg ) / (tau + pfilter2);

     }
  }
      
}

__global__ void qneutAdiab_part1(cuComplex* Num_tmp, cuComplex* Denom_field, float tau, cuComplex* Dens_i, cuComplex* Tprp_i, float* jacobian, specie s,
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
      
       
      /*//like fixFFT
      if((idy==0 || idy==ny/2) && idx>nx/2) {
        Num_tmp[index].x = 0;
	Num_tmp[index].y = 0;
      }      
      if((idy==0 || idy==ny/2) && idx==nx/2) {
        Num_tmp[index] = .5*Num_tmp[index];
      }*/
      
      
      
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

	/*
	//like fixFFT
	if((idy==0 || idy==ny/2) && idx>nx/2) {
          Num_tmp[index].x = 0;
	  Num_tmp[index].y = 0;
	}      
	if((idy==0 || idy==ny/2) && idx==nx/2) {
          Num_tmp[index] = .5*Num_tmp[index];
	}*/

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
            

  
__global__ void nbar(cuComplex* nbar, cuComplex* Dens, cuComplex* Tprp, 
		      specie s, float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy(); 
  unsigned int idx = get_idx();
  unsigned int idz = get_idz(); 
  
  
  if(nz<=zthreads) {
    if( idy<(ny/2+1) && idx<nx && idz<nz ) {

      float bidx = b(s.rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      nbar[index] = (Dens[index]/(1.+bidx/2.) - bidx*Tprp[index]/(2*pow(1.+bidx/2.,2)))*s.dens*s.z;
      
    }
  }
}    

__global__ void phiavgdenom(float* PhiAvgDenom, float* jacobian, specie* s, float* kx, float* ky, float shat, float* gds2, float* gds21, float* gds22, float* bmagInv,float tau)
{   
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();

  if( idy==0 && idx!=0 && idx<nx ) {
 
  float pfilter2 = 0.;
  float tmp = 0.;

  for(int idz=0; idz<nz; idz++) {
    
    for(int i=0; i<nspecies; i++) {
      float bidx = b(s[i].rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);
      pfilter2 = pfilter2 + s[i].dens*s[i].z*s[i].zt*( 1. - g0(bidx) );
    }
   
    tmp = pfilter2*jacobian[idz]/( tau + pfilter2 );
    PhiAvgDenom[idx] = PhiAvgDenom[idx] + tmp;
  }
 
  }

}








