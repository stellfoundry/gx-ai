//i*kz*f(ky,kx,kz)
__global__ void zderiv(cuComplex* res, cuComplex* f, float* kz)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
            
      //result(ky,kx,kz)= i*kz*f(ky,kx,kz)
      cuComplex tmp;
      tmp.x = -kz[idz]*f[index].y;
      tmp.y = kz[idz]*f[index].x;    
      res[index] = tmp;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	unsigned int IDZ = idz + i*zthreads;
	
	cuComplex tmp;
	tmp.x = -kz[IDZ]*f[index].y;
	tmp.y = kz[IDZ]*f[index].x;    
	res[index] = tmp; 
      }
    }
  }    	
}  

__global__ void zderiv(cuComplex* res, cuComplex* f, float* kz, int nx, int ny, int nz)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
            
      //result(ky,kx,kz)= i*kz*f(ky,kx,kz)
      cuComplex tmp;
      tmp.x = -kz[idz]*f[index].y;
      tmp.y = kz[idz]*f[index].x;    
      res[index] = tmp;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	unsigned int IDZ = idz + i*zthreads;
	
	cuComplex tmp;
	tmp.x = -kz[IDZ]*f[index].y;
	tmp.y = kz[IDZ]*f[index].x;    
	res[index] = tmp; 
      }
    }
  }    	
}  
//i*kz*B(kz)
__global__ void zderivB(cuComplex* res, cuComplex* B, float* kz)
{
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idz<(nz/2+1)) {
     
      //result(kz)= i*kz*f(kz)
      cuComplex tmp;
      tmp.x = -kz[idz]*B[idz].y;
      tmp.y = kz[idz]*B[idz].x;  
      res[idz] = tmp;  
    }
  }
  else {
    for(int i=0; i<(nz/2+1)/zthreads; i++) {
      if(idz<zthreads) {   
        unsigned int IDZ = idz + i*zthreads; 	
	cuComplex tmp;
	tmp.x = -kz[IDZ]*B[IDZ].y;
        tmp.y = kz[IDZ]*B[IDZ].x;
	res[IDZ] = tmp;  
      }
    }
  }    	
}  
  			        	  
__global__ void mask_Z(cufftComplex* mult) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
    unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    
    
    if( idy>(ny-1)/3 || ( idx>(nx-1)/3 && idx<2*(nx)/3+1 ) || ( idz>(nz-1)/3 && idz<2*(nz)/3+1 ) ) {
      mult[index].x = 0;
      mult[index].y = 0;
    }  
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
     unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
     unsigned int IDZ = idz + zthreads*i;
    
     if( idy>(ny-1)/3 || ( idx>(nx-1)/3 && idx<2*(nx)/3+1 ) || ( IDZ>(nz-1)/3 && IDZ<2*(nz)/3+1 ) ) {
       mult[index].x = 0;
       mult[index].y = 0;
     }  
    }
   }
  }
    
}   

