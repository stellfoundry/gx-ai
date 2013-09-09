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

//nSpecies=1, ions only
__global__ void qneut(cuComplex* Phi, float tau, cuComplex* Dens_i, cuComplex* Tprp_i, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      Phi[index] = ( Dens_i[index]/(1.+bidx/2.) - (bidx*Tprp_i[index])/(2.*pow(1.+bidx/2.,2)) ) 
      			/ (tau - g0(bidx) + 1.);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<zthreads ) {

	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	Phi[index] = ( Dens_i[index]/(1.+bidx/2.) - (bidx*Tprp_i[index])/(2.*pow(1.+bidx/2.,2)) ) 
			/ (tau - g0(bidx) + 1.);
      }
    }
  }      
      
}    

__global__ void qneutAdiab_part1(cuComplex* Num_tmp, cuComplex* Denom_field, float tau, cuComplex* Dens_i, cuComplex* Tprp_i, float* jacobian, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

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

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);

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

__global__ void qneutAdiab_part2(cuComplex* Phi, cuComplex* Num_tmp, cuComplex* Denom_field, float tau, cuComplex* Dens_i, cuComplex* Tprp_i, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if( !(idy==0 && idx==0) && idy<(ny/2+1) && idx<nx && idz<nz ) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

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

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);

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
            

__global__ void phi_n(cuComplex* res, cuComplex* phi, float tprim, float rho, float fprim,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * ( fprim*sgam0(bidx) + tprim * flr(bidx) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
	unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * ( fprim*sgam0(bidx) + tprim * flr(bidx) );
      }
    }
  }    	    
}  


__global__ void phi_u(cuComplex* res, cuComplex* Phi, float rho,
		      float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = Phi[index] * sgam0( bidx );      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {

	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = Phi[index] * sgam0(bidx);
      }
    }
  }    	    
}    

__global__ void phi_tpar(cuComplex* res, cuComplex* phi, float tprim, float rho,
			 float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * sgam0(bidx) * tprim;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * sgam0(bidx) * tprim;
      }
    }
  }    	    
}  

__global__ void phi_tperp(cuComplex* res, cuComplex* phi, float tprim, float rho, float fprim,
			  float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy(); //blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * ( tprim*( sgam0(bidx) + flr2(bidx) ) + fprim*flr(bidx) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * ( tprim*( sgam0(bidx) + flr2(bidx) ) + fprim*flr(bidx) );
      }
    }
  }    	    
}

//also known as phi_qperp
__global__ void phi_flr(cuComplex* res, cuComplex* phi, float rho,
			float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * flr(bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * flr(bidx);
      }
    }
  }    	    
}    

__global__ void phi_flr2(cuComplex* res, cuComplex* phi, float rho,
			 float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * flr2(bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * flr2(bidx);
      }
    }
  }    	    
}    

__global__ void phi_nd(cuComplex* res, cuComplex* phi, float zt, float rho,
		       float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * ( zt * (2.*sgam0(bidx) + flr(bidx)) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * ( zt * (2.*sgam0(bidx) + flr(bidx)) );
      }
    }
  }    	    
}    

__global__ void phi_tpard(cuComplex* res, cuComplex* phi, float zt, float rho,
			  float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * ( zt * 2. * sgam0(bidx) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * ( zt * 2. * sgam0(bidx) );
      }
    }
  }    	    
}    

__global__ void phi_tperpd(cuComplex* res, cuComplex* phi, float zt, float rho,
			   float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * ( zt * ( sgam0(bidx) + 2.*flr(bidx) + flr2(bidx) ) );
      }
    }
  }    	    
} 

__global__ void phi_qperpb(cuComplex* res, cuComplex* phi, float rho,
			   float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = phi[index] * (flr2(bidx) - flr(bidx));
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
	unsigned int IDZ = idz + zthreads*i;

	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*IDZ;
	
	res[index] = phi[index] * (flr2(bidx) - flr(bidx));
      }
    }
  }    	    
}      


  
   
