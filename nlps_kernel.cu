__global__ void NLPSderivX_abs(cuComplex* fdx, cuComplex* f, float* kx)                        
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
     unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

     //df/dx
    fdx[index] = -abs(kx[idx])*f[index];			
   }
  } 
  else {
   for(int i=0; i<nz/zthreads; i++) { 
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
    unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;

    //df/dx
     fdx[index] = -abs(kx[idx])*f[index];
    }
   }
  } 
}  

__global__ void NLPSderivY_abs(cuComplex* fdx, cuComplex* f, float* ky)                        
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
     unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

     //df/dx
     fdx[index] = abs(ky[idy])*f[index];
   }
  } 
  else {
   for(int i=0; i<nz/zthreads; i++) { 
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
    unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;

    //df/dx
     fdx[index] = abs(ky[idy])*f[index];
    }
   }
  } 
}  

__global__ void NLPSderivX(cuComplex* fdx, cuComplex* f, float* kx)                        
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
     unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

     //df/dx
     fdx[index].x = -kx[idx]*f[index].y;			
     fdx[index].y =  kx[idx]*f[index].x;   
   }
  } 
  else {
   for(int i=0; i<nz/zthreads; i++) { 
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
    unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;

    //df/dx
    fdx[index].x = -kx[idx]*f[index].y;			
    fdx[index].y =  kx[idx]*f[index].x;			
    }
   }
  } 
}  

__global__ void NLPSderivY(cuComplex* fdy, cuComplex* f, float* ky)                        
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
     unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    
    
     //df/dy
     fdy[index].x = -ky[idy]*f[index].y;			
     fdy[index].y =  ky[idy]*f[index].x;			
    
    }
  } 
  else {
   for(int i=0; i<nz/zthreads; i++) { 
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
    unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
    
    //df/dy
    fdy[index].x = -ky[idy]*f[index].y;			
    fdy[index].y =  ky[idy]*f[index].x;			
    		
    }
   }
  } 
}  

__global__ void mask(cuComplex* f) 
{
  unsigned int idy = get_idy();
  int idx = get_idx();
  unsigned int idz = get_idz();
  
  //nx_unmasked = 2*((nx-1)/3)+1
  //ny_unmasked = (ny-1)/3+1
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
    unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    
    int ikx = get_ikx(idx);
    
    if( idy>(ny-1)/3 || ikx>(nx-1)/3 || ikx<-(nx-1)/3) {
      f[index].x = 0.;
      f[index].y = 0.;
    }  
    
    //also zero the kx=ky=0 mode
    if( idx==0 && idy==0 ) {
      f[index].x = 0;
      f[index].y = 0;
    }  
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
     unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
    
    int ikx = get_ikx(idx);
    
    if( idy>(ny-1)/3 || ikx>(nx-1)/3 || ikx<-(nx-1)/3) {
       f[index].x = 0;
       f[index].y = 0;
     }  
     //also zero the kx=ky=0 mode
     if( idx==0 && idy==0 ) {
       f[index].x = 0;
       f[index].y = 0;
     }  
    }
   }
  }
    
}      
  
__global__ void mask(float* f) 
{
  unsigned int idy = get_idy();
  int idx = get_idx();
  unsigned int idz = get_idz();
  
  //nx_unmasked = 2*((nx-1)/3)+1
  //ny_unmasked = (ny-1)/3+1
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
    unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    
    
    int ikx = get_ikx(idx);
    
    if( idy>(ny-1)/3 || ikx>(nx-1)/3 || ikx< -(nx-1)/3) {
      f[index] = 0.;
    }  
    
    //also zero the kx=ky=0 mode
    if( idx==0 && idy==0 ) {
      f[index] = 0.;
    }  
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
     unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
    
    int ikx = get_ikx(idx);
    
    if( idy>(ny-1)/3 || ikx>(nx-1)/3 || ikx<-(nx-1)/3) {
       f[index] = 0;
     }  
     //also zero the kx=ky=0 mode
     if( idx==0 && idy==0 ) {
       f[index] = 0;
     }  
    }
   }
  }
    
}      

__global__ void mask(float* f, int nx, int ny, int nz) 
{
  unsigned int idy = get_idy();
  int idx = get_idx();
  unsigned int idz = get_idz();
  
  //nx_unmasked = 2*((nx-1)/3)+1
  //ny_unmasked = (ny-1)/3+1
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
    unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    
    
    int ikx = get_ikx(idx);
    
    if( idy>(ny-1)/3 || ikx>(nx-1)/3 || ikx< -(nx-1)/3) {
      f[index] = 0.;
    }  
    
    //also zero the kx=ky=0 mode
    if( idx==0 && idy==0 ) {
      f[index] = 0.;
    }  
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
     unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
    
    int ikx = get_ikx(idx);
    
    if( idy>(ny-1)/3 || ikx>(nx-1)/3 || ikx<-(nx-1)/3) {
       f[index] = 0;
     }  
     //also zero the kx=ky=0 mode
     if( idx==0 && idy==0 ) {
       f[index] = 0;
     }  
    }
   }
  }
    
}      

__global__ void bracket(float* result, float* fdxgdy,  
                      float* fdy, float* gdx, float scaler)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny) && idx<nx && idz<nz ) {
    unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
    
    
    result[index] = scaler*( fdxgdy[index] - (fdy[index])*(gdx[index]) );  
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny) && idx<nx && idz<zthreads ) {
    unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*ny*zthreads*i;
    
    
    result[index] = scaler*( fdxgdy[index] - (fdy[index])*(gdx[index]) );  
    }
   }
  } 
 
}  

__global__ void reality(cuComplex* f) 
{
  unsigned int idy = 0;
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if( idx>0 &&  idx<(nx/2+1) && idz<nz) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
	unsigned int index2 = idy + (ny/2+1)*(nx-idx) + nx*(ny/2+1)*idz;
	
	/*
	float reavg = .5*(f[index].x + f[index2].x);
	float imavg = .5*(f[index].y - f[index2].y);
	f[index].x = reavg;
	f[index2].x = reavg;
	f[index].y = imavg;
	f[index2].y = -imavg;*/
	
	f[index2].x = f[index].x;
	f[index2].y = -f[index].y;
    }
  }
}
  

__global__ void realityAll(cuComplex* f) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if( idy<(ny/2+1) && idx>0 &&  idx<(nx/2+1) && idz<nz) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
	unsigned int index2 = idy + (ny/2+1)*(nx-idx) + nx*(ny/2+1)*idz;
	
	/*
	float reavg = .5*(f[index].x + f[index2].x);
	float imavg = .5*(f[index].y - f[index2].y);
	f[index].x = reavg;
	f[index2].x = reavg;
	f[index].y = imavg;
	f[index2].y = -imavg;*/
	
	f[index2].x = f[index].x;
	f[index2].y = -f[index].y;
    }
  }
}


