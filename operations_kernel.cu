//zeroes complex arrays
__global__ void zeroC(cuComplex* f) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
    int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    
    f[index].x = 0;
    f[index].y = 0;
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
    int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
    
    f[index].x = 0;
    f[index].y = 0;
    }
   }
  }    
}


//zeroes complex arrays
__global__ void zeroC(cuComplex* f, int nx, int ny, int nz) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
    int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    
    f[index].x = 0;
    f[index].y = 0;
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
    int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
    
    f[index].x = 0;
    f[index].y = 0;
    }
   }
  }    
}


    
//zeroes real arrays
__global__ void zero(float* f) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny) && idx<nx && idz<nz) {
    int index = idy + (ny)*idx + nx*(ny)*idz;
    
    f[index] = 0;
    
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny) && idx<nx && idz<zthreads) {
    int index = idy + (ny)*idx + nx*(ny)*idz + nx*ny*zthreads*i;
    
    f[index] = 0;
    
    }
   }
  }     
}    

//zeroes real arrays
__global__ void zero(int* f, int nx, int ny, int nz) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny) && idx<nx && idz<nz) {
    int index = idy + (ny)*idx + nx*(ny)*idz;
    
    f[index] = 0;
    
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny) && idx<nx && idz<zthreads) {
    int index = idy + (ny)*idx + nx*(ny)*idz + nx*ny*zthreads*i;
    
    f[index] = 0;
    
    }
   }
  }     
}  

//zeroes real arrays
__global__ void zero(float* f, int nx, int ny, int nz) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny) && idx<nx && idz<nz) {
    int index = idy + (ny)*idx + nx*(ny)*idz;
    
    f[index] = 0;
    
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny) && idx<nx && idz<zthreads) {
    int index = idy + (ny)*idx + nx*(ny)*idz + nx*ny*zthreads*i;
    
    f[index] = 0;
    
    }
   }
  }     
}  

//zero last gridpoint in z
__global__ void zeroNz(cuComplex* f)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  
  if( idy<(ny/2+1) && idx<nx ) {
    unsigned int idxy = idy + (ny/2+1)*idx;
    
    f[idxy + nx*(ny/2+1)*(nz-1)].x = 0.;
    f[idxy + nx*(ny/2+1)*(nz-1)].y = 0;
  }
}


//add (a=1) or subtract (a=-1) complex arrays
__global__ void addsubt(cuComplex* result, cuComplex* f, cuComplex* g, int a)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      
      if(a == 1) {
        result[index] = f[index] + g[index];
      }	
      if(a == -1) {
        result[index] = f[index] - g[index];
      }	
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        if(a == 1) {
          result[index] = f[index] + g[index];
	  }
        if(a == -1) {
          result[index] = f[index] - g[index];
	  }
      }
    }
  }
}   

__global__ void accum(cuComplex* accum, cuComplex* f, int a)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      
      if(a == 1) {
        accum[index] = accum[index] + f[index];
      }	
      if(a == -1) {
        accum[index] = accum[index] - f[index];
      }	
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        if(a == 1) {
          accum[index] = accum[index] + f[index];
	  }
        if(a == -1) {
          accum[index] = accum[index] - f[index];
	  }
      }
    }
  }
} 

__global__ void accum(cuComplex* accum, cuComplex* f, int a, int nx, int ny, int nz)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      
      if(a == 1) {
        accum[index] = accum[index] + f[index];
      }	
      if(a == -1) {
        accum[index] = accum[index] - f[index];
      }	
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        if(a == 1) {
          accum[index] = accum[index] + f[index];
	  }
        if(a == -1) {
          accum[index] = accum[index] - f[index];
	  }
      }
    }
  }
} 

__global__ void accum(float* accum, float* f, int a, int nx, int ny, int nz)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      
      if(a == 1) {
        accum[index] = accum[index] + f[index];
      }	
      if(a == -1) {
        accum[index] = accum[index] - f[index];
      }	
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        if(a == 1) {
          accum[index] = accum[index] + f[index];
	  }
        if(a == -1) {
          accum[index] = accum[index] - f[index];
	  }
      }
    }
  }
} 

//two fields
__global__ void add_scaled(cuComplex* result, float fscaler, cuComplex* f, float gscaler, cuComplex* g)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index];
      }
    }
  }
} 

//two fields, complex scalars
__global__ void add_scaled(cuComplex* result, cuComplex fscaler, cuComplex* f, cuComplex gscaler, cuComplex* g)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index];
      }
    }
  }
} 

__global__ void add_scaled(float* result, float fscaler, float* f, float gscaler, float* g,int nx, int ny, int nz)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index];
      }
    }
  }
} 

//two fields, complex scalars, ONLY KY=0
__global__ void add_scaled_Ky0(cuComplex* result, float fscaler, cuComplex* f, float gadd)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy==0 && idx!=0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index].x = fscaler*f[index].x + gadd;      
      result[index].y = fscaler*f[index].y + gadd;      
    }
  }
} 
//two fields, complex scalars, ONLY KY=0
__global__ void add_scaled_Ky0(cuComplex* result, cuComplex fscaler, cuComplex* f, cuComplex gscaler, cuComplex* g)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy==0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy==0 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index];
      }
    }
  }
} 

//two fields, complex scalars, ONLY KY=0
__global__ void add_scaled_Ky0(cuComplex* result, float fscaler, cuComplex* f, float gscaler, cuComplex* g)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy==0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy==0 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index];
      }
    }
  }
} 

//three fields
__global__ void add_scaled(cuComplex* result, float fscaler, cuComplex* f, 
                                              float gscaler, cuComplex* g,
					      float hscaler, cuComplex* h)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index];
      }
    }
  }
} 

//three fields, complex scalars
__global__ void add_scaled(cuComplex* result, cuComplex fscaler, cuComplex* f, 
                                              cuComplex gscaler, cuComplex* g,
					      cuComplex hscaler, cuComplex* h)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index];
      }
    }
  }
} 

//three fields, complex scalars, ONLY KY=0
__global__ void add_scaled_Ky0(cuComplex* result, cuComplex fscaler, cuComplex* f, 
                                              cuComplex gscaler, cuComplex* g,
					      cuComplex hscaler, cuComplex* h)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy==0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy==0 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index];
      }
    }
  }
} 

__global__ void add_scaled_Ky0(cuComplex* result, float fscaler, cuComplex* f, 
                                              float gscaler, cuComplex* g,
					      float hscaler, cuComplex* h)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy==0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy==0 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index];
      }
    }
  }
} 

//four fields
__global__ void add_scaled(cuComplex* result, float fscaler, cuComplex* f, 
                                              float gscaler, cuComplex* g,
					      float hscaler, cuComplex* h,
					      float jscaler, cuComplex* j)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index] + jscaler*j[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index] + jscaler*j[index];
      }
    }
  }
}   

//four fields, KY=0 ONLY
__global__ void add_scaled_Ky0(cuComplex* result, float fscaler, cuComplex* f, 
                                              float gscaler, cuComplex* g,
					      float hscaler, cuComplex* h,
					      float jscaler, cuComplex* j)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy==0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index] + jscaler*j[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy==0 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index] + jscaler*j[index];
      }
    }
  }
}   
  
//multiply a complex array by a scaler


//this DOES preserve b
//scaler * b(ky,kx,z)
__global__ void scale(cuComplex* result,cuComplex* b, float scaler)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + (ny/2+1)*(nx)*idz;
    
      result[index] = scaler*b[index];
    }
  }
    
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	result[index] = scaler*b[index];
      }
    }
  }    	
} 

__global__ void scale(cuComplex* result,cuComplex* b, double scaler)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + (ny/2+1)*(nx)*idz;
    
      result[index] = scaler*b[index];
    }
  }
    
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	result[index] = scaler*b[index];
      }
    }
  }    	
} 

//can specify nx, ny, nz
__global__ void scale(cuComplex* result,cuComplex* b, float scaler, int nx, int ny, int nz)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + (ny/2+1)*(nx)*idz;
    
      result[index] = scaler*b[index];
    }
  }
    
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	result[index] = scaler*b[index];
      }
    }
  }    	
} 

//scaler * B(z)
__global__ void scaleRealZ(float* result, float* B, float scaler) 
{
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idz<nz) {   
      result[idz] = scaler*B[idz];
    }
  }
    
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idz<zthreads) {
        unsigned int IDZ = idz + zthreads*i;
	
	result[IDZ] = scaler*B[IDZ];
      }
    }
  }    	
}  

//scaler * b(y,x,z)
__global__ void scaleReal(float* result, cufftReal* b, float scaler)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + ny*idx + ny*nx*idz;
      
      result[index] = scaler*b[index];
    }
  } 
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
	result[index] = scaler*b[index]; 
      }
    }
  }     
}    

//can specify nx, ny, nz
__global__ void scaleReal(float* result, float* b, float scaler, int nx, int ny, int nz)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + ny*idx + ny*nx*idz;
      
      result[index] = scaler*b[index];
    }
  } 
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
	result[index] = scaler*b[index]; 
      }
    }
  }     
}    


//multiply by kPerp^2
__global__ void multKPerp(cuComplex* fK, cuComplex* f, float* kPerp2, int a)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      unsigned int kidx = idy + (ny/2+1)*idx;
      
      if(a == 1) {
        fK[index] = f[index] * kPerp2[kidx];
      }	
      if(a == -1) {
        fK[index] = f[index] * -kPerp2[kidx];
      }	 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	unsigned int kidx = idy + (ny/2+1)*idx;
	
        if(a == 1) {
          fK[index] = f[index] * kPerp2[kidx];
	  }
        if(a == -1) {
          fK[index] = f[index] * -kPerp2[kidx];
	  }
        
      }
    }
  }
}  

__global__ void multKPerp(float* fK, float* f, float* kPerp2, int a)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      unsigned int kidx = idy + (ny/2+1)*idx;
      
      if(a == 1) {
        fK[index] = f[index] * kPerp2[kidx];
      }	
      if(a == -1) {
        fK[index] = f[index] * -kPerp2[kidx];
      }	 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	unsigned int kidx = idy + (ny/2+1)*idx;
	
        if(a == 1) {
          fK[index] = f[index] * kPerp2[kidx];
	  }
        if(a == -1) {
          fK[index] = f[index] * -kPerp2[kidx];
	  }
        
      }
    }
  }
}           

//f(kx) * f(ky,kx,z)
__global__ void multKx(cuComplex* res, cuComplex* f, float* kx) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = f[index] * kx[idx];
      		 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        res[index] = f[index] * kx[idx];
        
      }
    }
  }
}   

//f(ky) * f(ky,kx,z)
__global__ void multKy(cuComplex* res, cuComplex* f, float* ky) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = f[index] * ky[idy];
      		 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        res[index] = f[index] * ky[idy];
        
      }
    }
  }
}   

//f(ky) * f(ky,kx,z)
__global__ void multKy(cuComplex* res, cuComplex* f, float* ky, int nx, int ny, int nz) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = f[index] * ky[idy];
      		 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        res[index] = f[index] * ky[idy];
        
      }
    }
  }
}   

//f(ky,kx,z)/ky
__global__ void divKy(float* res, float* f, float* ky, int nx, int ny, int nz) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = f[index] / ky[idy];
      		 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        res[index] = f[index] / ky[idy];
        
      }
    }
  }
}   

//f(z)*f(ky,kx,z)
__global__ void multZ(cuComplex* res, cuComplex* f, float* z) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = f[index] * z[idz];
      		 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	unsigned int IDZ = idz + i*zthreads;
	
        res[index] = f[index] * z[IDZ];
        
      }
    }
  }
} 


__global__ void multZ(float* res, float* f, float* z, int nx, int ny, int nz) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = f[index] * z[idz];
      		 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	unsigned int IDZ = idz + i*zthreads;
	
        res[index] = f[index] * z[IDZ];
        
      }
    }
  }
} 


__global__ void multdiv(cuComplex* result, cuComplex* f, cuComplex* g, int a)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      
      if(a == 1) {
        result[index] = f[index] * g[index];
      }	
      if(a == -1) {
        result[index] = f[index] / g[index];
      }	
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        if(a == 1) {
          result[index] = f[index] * g[index];
	  }
        if(a == -1) {
          result[index] = f[index] / g[index];
	  }
      }
    }
  }
}

__global__ void multdiv(cuComplex* result, cuComplex* f, float* g, int a)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      
      if(a == 1) {
        result[index] = f[index] * g[index];
      }	
      if(a == -1) {
        result[index] = f[index] / g[index];
      }	
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        if(a == 1) {
          result[index] = f[index] * g[index];
	  }
        if(a == -1) {
          result[index] = f[index] / g[index];
	  }
      }
    }
  }
}

__global__ void multdiv(cuComplex* result, cuComplex* f, cuComplex* g, int nx, int ny, int nz, int a)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      
      if(a == 1) {
        result[index] = f[index] * g[index];
      }	
      if(a == -1) {
        result[index] = f[index] / g[index];
      }	
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        if(a == 1) {
          result[index] = f[index] * g[index];
	  }
        if(a == -1) {
          result[index] = f[index] / g[index];
	  }
      }
    }
  }
}

__global__ void multdiv(float* result, float* f, float* g, int nx, int ny, int nz, int a)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      
      if(a == 1) {
        result[index] = f[index] * g[index];
      }	
      if(a == -1) {
        result[index] = (float) f[index] / g[index];
      }	
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        if(a == 1) {
          result[index] = f[index] * g[index];
	  }
        if(a == -1) {
          result[index] = (float) f[index] / g[index];
	  }
      }
    }
  }
}

__global__ void multdiv(float* result, float* f, float* g, int a)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
      
      
      if(a == 1) {
        result[index] = f[index] * g[index];
      }	
      if(a == -1) {
        result[index] = f[index] / g[index];
      }	
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
        if(a == 1) {
          result[index] = f[index] * g[index];
	  }
        if(a == -1) {
          result[index] = f[index] / g[index];
	  }
      }
    }
  }
}

  
 
//squares a complex
__global__ void squareComplex(cuComplex* res, cuComplex* f)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index].x = f[index].x*f[index].x + f[index].y * f[index].y;
      res[index].y = 0;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        res[index].x = f[index].x*f[index].x + f[index].y * f[index].y;
	res[index].y = 0;
      }
    }
  }
}    

__global__ void squareComplex(float* res, cuComplex* f)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = f[index].x*f[index].x + f[index].y * f[index].y;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        res[index] = f[index].x*f[index].x + f[index].y * f[index].y;
      }
    }
  }
}  

__global__ void abs(float* res, float* f)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index] = abs( f[index] );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        res[index] = abs( f[index] );
      }
    }
  }
}      


//fixes roundoff errors after fft
__global__ void roundoff(cuComplex* f, float max)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if( abs(f[index].x) < max)
        f[index].x = 0.0f;
      if( abs(f[index].y) < max)
        f[index].y = 0.0f;
	
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        if( abs(f[index].x) < max)
          f[index].x = 0.0f;
        if( abs(f[index].y) < max)
          f[index].y = 0.0f;	
      }
    }
  }

}     

__global__ void PfirschSchluter(cuComplex* Qps, cuComplex* Q, float psfac, float* kx, float* gds22, float
				qsf, float eps, float* bmagInv, cuComplex* T_fluxsurfavg, float shat)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  float shatInv;
  if (abs(shat)>1.e-8) {
    shatInv = 1./shat;
  } else {
    shatInv = 1.;
  }
  
  if(nz<=zthreads) {
    if(idy<ny && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      //psfac is 3 or 1 depending on whether using Qpar or Qprp, respectively
       
      if(idy==0) {
	//double check signs... k_r = -kx for ky=0?
		
	cuComplex tmp;
	tmp.x = Q[index].x + psfac*(-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*T_fluxsurfavg[idx].y;
	tmp.y = Q[index].y - psfac*(-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*T_fluxsurfavg[idx].x;
	Qps[index] = tmp;
      }
      else {
        Qps[index] = Q[index];
      }
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
	if(idy==0) {
	  //double check signs... k_r = -kx for ky=0?
	  cuComplex tmp;
	  tmp.x = Q[index].x + psfac*(-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*T_fluxsurfavg[idx].y;
	  tmp.y = Q[index].y - psfac*(-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*T_fluxsurfavg[idx].x;
	  Qps[index] = tmp;
        }
	else {
	  Qps[index] = Q[index];
	}
      }
    }
  }
}
            
        
__global__ void SmagorinskyDiffusion(cuComplex* result, cuComplex* field, float D, 
            float rho, float *kx, float *ky, float shat, float *gds2, float *gds21, float *gds22, float *bmagInv)     
{
  unsigned int idy = get_idy(); 
  unsigned int idx = get_idx();
  unsigned int idz = get_idz(); 
  
  
  if(nz<=zthreads) {
    if( idy<(ny/2+1) && idx<nx && idz<nz ) {

      float bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      result[index] = field[index] / (1. + D*bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
	float bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
	result[index] = field[index] / (1. + D*bidx);
      }
    }
  }
}

__global__ void sqrtX(float* result, float* f)
{
  unsigned int idx = get_idx();
  
  if(idx < nx) {
    result[idx] = sqrtf( f[idx] );
  }
}

__global__ void sqrtY(float* result, float* f)
{
  unsigned int idy = get_idy();
  
  if(idy < ny/2+1) {
    result[idy] = sqrtf( f[idy] );
  }
}

__global__ void sqrtZ(float* result, float* f)
{
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idz < nz) {
      result[idz] = sqrtf( f[idz] );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idz<zthreads) {
        unsigned int IDZ = idz + zthreads*i;
	
	result[IDZ] = sqrtf( f[IDZ] );
      }
    }
  }
      
}

__global__ void sqrtXY(float* result, float* f)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  
  if(idy < ny/2+1 && idx<nx) {
    unsigned int idxy = idy + (ny/2+1)*idx;
    result[idxy] = sqrtf( f[idxy] );
  }
}

/*
__global__ void bounds(cuComplex* f, int nLinks, int nChains, int* kxCover, int* kyCover)
{
  unsigned int idy = get_idy(); 
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(  
      
*/      

