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

__global__ void add_scaled(float* result, float fscaler, float* f, float gscaler, float* g)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
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

__global__ void add_scaled(cuComplex* result, float fscaler, cuComplex* f, float gscaler, cuComplex* g,int nx, int ny, int nz)
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
__global__ void add_scaled(cuComplex* result, float fscaler, float* f, float gscaler, cuComplex* g,int nx, int ny, int nz)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index].x = fscaler*f[index] + gscaler*g[index].x;      
      result[index].y = gscaler*g[index].y;      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index].x = fscaler*f[index] + gscaler*g[index].x;      
        result[index].y = gscaler*g[index].y;      
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

//five fields
__global__ void add_scaled(cuComplex* result, float fscaler, cuComplex* f, 
                                              float gscaler, cuComplex* g,
					      float hscaler, cuComplex* h,
					      float jscaler, cuComplex* j,
					      float kscaler, cuComplex* k)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index] + jscaler*j[index] + kscaler*k[index];      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        result[index] = fscaler*f[index] + gscaler*g[index] + hscaler*h[index] + jscaler*j[index] + kscaler*k[index];
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

__global__ void scaleI(cuComplex* result,cuComplex* b, float scaler)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + (ny/2+1)*(nx)*idz;
 
      cuComplex I;
      I.x = 0.;
      I.y = 1.;   

      result[index] = I*scaler*b[index];
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
__global__ void scaleRealZ(float* result, float* B, double scaler) 
{
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idz<nz) {   
      result[idz] = (double) scaler*B[idz];
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

__global__ void multKx4(float* res, float* f, float* kx) 
{
  unsigned int idx = get_idx();
  
    if(idx<nx) {
      
      double kx4 = kx[idx]*kx[idx]*kx[idx]*kx[idx];

      res[idx] = f[idx] * kx4;
      		 
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
        if(g[index]!=0.) {
          result[index] = f[index] / g[index];
        }
        else result[index] = 0.;
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

__global__ void magnitude_xy(float* mag, float* f_x, float* f_y)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
      
      mag[index] = sqrt( f_x[index]*f_x[index] + f_y[index] * f_y[index] );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        mag[index] = sqrt( f_x[index]*f_x[index] + f_y[index] * f_y[index] );
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
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
      
      res[index] = abs( f[index] );
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
        res[index] = abs( f[index] );
      }
    }
  }
}      

__global__ void abs(cuComplex* res, cuComplex* f)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      res[index].x = cuCabsf( f[index] );
      res[index].y = 0.;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        res[index].x = cuCabsf( f[index] );
        res[index].y = 0.;
      }
    }
  }
}      

__global__ void abs_sgn(float* res, float* f, float* g_sgn)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
      
      int sgn = 0;
  
      if(abs(g_sgn[index])>1.e-5) {
        if(g_sgn[index]>0) sgn=1;
        if(g_sgn[index]<0) sgn=-1;
      }

      res[index] = abs(f[index]) * sgn;
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
        int sgn = 0;

        if(g_sgn[index]>0) sgn=1;
        if(g_sgn[index]<0) sgn=-1;

        res[index] = abs(f[index]) * sgn;
      }
    }
  }
}      

__global__ void abs_sgn(cuComplex* res, cuComplex* f, cuComplex* g)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(cuCabsf(g[index]) != 0) {
        res[index] = cuCabsf(f[index]) * g[index] / cuCabsf(g[index]);
      } else {
        res[index].x = 0.;
        res[index].y = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;

        if(cuCabsf(g[index]) != 0) {
          res[index] = cuCabsf(f[index]) * g[index] / cuCabsf(g[index]);
        } else {
          res[index].x = 0.;
          res[index].y = 0.;
        }
      }
    }
  }
}      

__global__ void sgn(cuComplex* res, cuComplex* g)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      //float fac=sqrt(2.);
      float fac=1.;

      if(cuCabsf(g[index]) > 0.) {
        res[index] = fac*g[index] / cuCabsf(g[index]);
      } else {
        res[index].x = 0.;
        res[index].y = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;

        if(cuCabsf(g[index]) != 0) {
          res[index] = g[index] / cuCabsf(g[index]);
        } else {
          res[index].x = 0.;
          res[index].y = 0.;
        }
      }
    }
  }
}      

__global__ void sgn(float* res, float* g)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<ny && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
     
      int sgn = 0;
      if(g[index]>0.) sgn = 1;
      if(g[index]<0.) sgn = -1;
 
      res[index] = sgn;      

    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;

      int sgn = 0;
      if(g[index]>0.) sgn = 1;
      if(g[index]<0.) sgn = -1;
 
      res[index] = sgn;      
      }
    }
  }
}      
__global__ void mult_sgn(cuComplex* res, cuComplex* f, cuComplex* g)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(cuCabsf(g[index]) != 0) {
        res[index] = f[index] * g[index] / cuCabsf(g[index]);
      } else {
        res[index].x = 0.;
        res[index].y = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;

        if(cuCabsf(g[index]) != 0) {
          res[index] = cuCabsf(f[index]) * g[index] / cuCabsf(g[index]);
        } else {
          res[index].x = 0.;
          res[index].y = 0.;
        }
      }
    }
  }
}      

__global__ void mult_sgn(float* res, float* f, float* g)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
      
      if(abs(g[index]) != 0) {
        res[index] = f[index] * g[index] / abs(g[index]);
      } else {
        res[index] = 0.;
      }
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;

        if(abs(g[index]) != 0) {
          res[index] = f[index] * g[index] / abs(g[index]);
        } else {
          res[index] = 0.;
        }
      }
    }
  }
}      

__global__ void norm(float* res, float* fx, float* fy)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
      
      res[index] = sqrt(fx[index]*fx[index] + fy[index]*fy[index]);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
        res[index] = sqrt(fx[index]*fx[index] + fy[index]*fy[index]);
      }
    }
  }
}      



__global__ void mult_I(cuComplex* f)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
      
      cuComplex I;
      I.x = 0.;
      I.y = 1.;

      f[index] = I*f[index];
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
        cuComplex I;
        I.x = 0.;
        I.y = 1.;

        f[index] = I*f[index];
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
				qsf, float eps, float* bmagInv, cuComplex* T, float shat, float rho)
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
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      //psfac is 3 or 1 depending on whether using Qpar or Qprp, respectively
       
      if(idy==0) {
	//double check signs... k_r = -kx for ky=0?
		
	cuComplex tmp;
	tmp.x = Q[index].x + psfac*(-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho*T[index].y;
	tmp.y = Q[index].y - psfac*(-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho*T[index].x;
	Qps[index] = tmp;
      }
      else {
        Qps[index] = Q[index];
      }
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny/2+1 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
	if(idy==0) {
	  //double check signs... k_r = -kx for ky=0?
	  cuComplex tmp;
	  tmp.x = Q[index].x + psfac*(-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*T[idx].y;
	  tmp.y = Q[index].y - psfac*(-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*T[idx].x;
	  Qps[index] = tmp;
        }
	else {
	  Qps[index] = Q[index];
	}
      }
    }
  }
}


__global__ void PfirschSchluter_fsa(cuComplex* Qps, cuComplex* Q, float psfac, float* kx, float* gds22, float
				qsf, float eps, float* bmagInv, cuComplex* T_fluxsurfavg, float shat, float rho)
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
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      //psfac is 3 or 1 depending on whether using Qpar or Qprp, respectively
       
      if(idy==0) {
	//double check signs... k_r = -kx for ky=0?
		
	cuComplex tmp;
	tmp.x = Q[index].x + psfac*(-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho*T_fluxsurfavg[idx].y;
	tmp.y = Q[index].y - psfac*(-kx[idx])*shatInv*sqrt(gds22[idz])*qsf/eps*bmagInv[idz]*rho*T_fluxsurfavg[idx].x;
	Qps[index] = tmp;
      }
      else {
        Qps[index] = Q[index];
      }
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny/2+1 && idx<nx && idz<zthreads) {
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
            
__global__ void new_varenna_zf(cuComplex* Qps, cuComplex* Q, float* kx, float* gds22, float
				qsf, float eps, float* bmagInv, cuComplex* T, float shat, float rho)
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
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
       
      if(idy==0) {
	//double check signs... k_r = -kx for ky=0?
    
        float kr = (-kx[idx])*shatInv*sqrt(gds22[idz]);		

	cuComplex tmp;
	tmp.x = Q[index].x + T[index].y/(kr*qsf/eps*rho);
	tmp.y = Q[index].y - T[index].x/(kr*qsf/eps*rho);
	Qps[index] = tmp;
      }
      else {
        Qps[index] = Q[index];
      }
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny/2+1 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
	if(idy==0) {
	  //double check signs... k_r = -kx for ky=0?
	  cuComplex tmp;
	  tmp.x = Q[index].x + (-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*T[idx].y;
	  tmp.y = Q[index].y - (-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*T[idx].x;
	  Qps[index] = tmp;
        }
	else {
	  Qps[index] = Q[index];
	}
      }
    }
  }
}


__global__ void new_varenna_zf_fsa(cuComplex* Qps, cuComplex* Q, float* kx, float* gds22, float
				qsf, float eps, float* bmagInv, cuComplex* T_fluxsurfavg, float shat, float rho)
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
    if(idy<ny/2+1 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;

      if(idy==0) {
	//double check signs... k_r = -kx for ky=0?

        float kr = (-kx[idx])*shatInv*sqrt(gds22[idz]);		
		
	cuComplex tmp;
	tmp.x = Q[index].x + T_fluxsurfavg[idx].y/(kr*qsf/eps*rho);
	tmp.y = Q[index].y - T_fluxsurfavg[idx].x/(kr*qsf/eps*rho);
	Qps[index] = tmp;
      }
      else {
        Qps[index] = Q[index];
      }
      
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny/2+1 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
	if(idy==0) {
	  //double check signs... k_r = -kx for ky=0?
	  cuComplex tmp;
	  tmp.x = Q[index].x + (-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*T_fluxsurfavg[idx].y;
	  tmp.y = Q[index].y - (-kx[idx])*shatInv*sqrt(gds22[IDZ])*qsf/eps*bmagInv[IDZ]*T_fluxsurfavg[idx].x;
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

      double bidx = b(rho, kx[idx], ky[idy], shat, gds2[idz], gds21[idz], gds22[idz], bmagInv[idz]);

      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      result[index] = field[index] / (1. + D*bidx);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<ny && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
	double bidx = b(rho, kx[idx], ky[idy], shat, gds2[IDZ], gds21[IDZ], gds22[IDZ], bmagInv[IDZ]);
	
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
      //result[idz] = sqrt(f[idz]);
      result[idz] = sqrt(f[idz]);
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idz<zthreads) {
        unsigned int IDZ = idz + zthreads*i;
	
	result[IDZ] = sqrt( f[IDZ] );
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

__global__ void replace_ky0(cuComplex* f, cuComplex* f_ky0)
{
  
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();

  if(idx<nx && idz<nz) {
    unsigned int index_ky0 = 0 + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    unsigned int idxz = idx + nx*idz;
    f[index_ky0] = f_ky0[idxz];
  
  }
}

//use this if input f(ky=0) is unpadded
__global__ void replace_ky0_nopad(cuComplex* f, cuComplex* f_ky0)
{
  
  int idx = get_idx();
  //unsigned int idy = 0;
  unsigned int idz = get_idz();

  unsigned int ntheta0 = 1 + 2*((nx-1)/3);

  if(idx<nx && idz<nz) {
    int index_ky0 = 0 + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    int idx_nopad;
    int idxz_nopad;
  
    int ikx = get_ikx(idx);

    if(ikx <= (nx-1)/3 && ikx>=0) {
      idx_nopad = idx;
      idxz_nopad = idx_nopad + ntheta0*idz;
      f[index_ky0] = f_ky0[idxz_nopad];
    } else if(ikx >= -(nx-1)/3 && ikx < 0) {
      idx_nopad = idx - nx + ntheta0;
      idxz_nopad = idx_nopad + ntheta0*idz;
      f[index_ky0] = f_ky0[idxz_nopad];
    }
    else { //pad with zeros for mask in middle
      f[index_ky0].x = 0.;
      f[index_ky0].y = 0.;
    }

  //  if(idx<(nx-1)/3+1) {  // (ntheta0-1)/2+1
  //    idx_nopad = idx;
  //    idxz_nopad = idx_nopad + ntheta0*idz;
  //    f[index_ky0] = f_ky0[idxz_nopad];
  //  } else if(idx>2*((nx-1)/3)) {
  //    idx_nopad = idx - nx + ntheta0;
  //    idxz_nopad = idx_nopad + ntheta0*idz;
  //    f[index_ky0] = f_ky0[idxz_nopad];
  //  }
  //  else { //pad with zeros for mask in middle
  //    f[index_ky0].x = 0.;
  //    f[index_ky0].y = 0.;
  //  }
  }
}

__global__ void getky0(cuComplex* res_ky0kxz, cuComplex* f_kykxz)
{
  unsigned int idx = get_idx();
  //unsigned int idy = 0;
  unsigned int idz = get_idz();

  if(idx<nx && idz<nz) {
    unsigned int index_ky0 = 0 + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    unsigned int idxz = idx + nx*idz;
    res_ky0kxz[idxz] = f_kykxz[index_ky0];
  }
}

//use this if want result f(ky=0) to be unpadded
__global__ void getky0_nopad(cuComplex* res_ky0kxz, cuComplex* f_kykxz)
{
  int idx = get_idx();
  //unsigned int idy = 0;
  unsigned int idz = get_idz();

  int ntheta0 = 1 + 2*((nx-1)/3);

  if(idx<nx && idz<nz) {
    int index_ky0 = 0 + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    int idx_nopad;
    int idxz_nopad;

    int ikx = get_ikx(idx);

    if(ikx <= (nx-1)/3 && ikx>=0) {
      idx_nopad = idx;
      idxz_nopad = idx_nopad + ntheta0*idz;
      res_ky0kxz[idxz_nopad] = f_kykxz[index_ky0];
    } else if(ikx >= -(nx-1)/3 && ikx<0) {
      idx_nopad = idx - nx + ntheta0;
      idxz_nopad = idx_nopad + ntheta0*idz;
      res_ky0kxz[idxz_nopad] = f_kykxz[index_ky0];
    }

   // if(idx<(nx-1)/3+1) {  // (ntheta0-1)/2+1
   //   idx_nopad = idx;
   //   idxz_nopad = idx_nopad + ntheta0*idz;
   //   res_ky0kxz[idxz_nopad] = f_kykxz[index_ky0];
   // } else if(idx>2*((nx-1)/3)) {
   //   idx_nopad = idx - nx + ntheta0;
   //   idxz_nopad = idx_nopad + ntheta0*idz;
   //   res_ky0kxz[idxz_nopad] = f_kykxz[index_ky0];
   // }
  }
}

__global__ void replace_fixed_mode(cuComplex* f, cuComplex* fixed, int iky, int ikx, float S)
{

  unsigned int idz = get_idz();

  if(idz<nz) {
    f[iky+(ny/2+1)*ikx+nx*(ny/2+1)*idz] = S*fixed[idz];
  }

}

__global__ void iso_fixed_mode(cuComplex* f, cuComplex* fixed, int iky, int ikx)
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(idy<(ny/2+1) && idx<nx && idz<nz) {
    int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    if(!(idy==iky && idx==ikx)) {
      f[index].x = 0.;
      f[index].y = 0.;
    }
  }

}
__global__ void get_fixed_mode(cuComplex* fixed, cuComplex* f, int iky, int ikx)
{

  unsigned int idz = get_idz();

  if(idz<nz) {
    fixed[idz] = f[iky+(ny/2+1)*ikx+nx*(ny/2+1)*idz];
  }

}

__global__ void get_fixed_mode(cuComplex* fixed, cuComplex* f, int iky, int ikx, int iz)
{

    int idz = iz+nz/2+1;
    fixed[idz] = f[iky+(ny/2+1)*ikx+nx*(ny/2+1)*idz];

}

__global__ void set_fixed_amplitude_withz(cuComplex* phi_fixed, cuComplex* dens_fixed, cuComplex* upar_fixed, cuComplex* tpar_fixed, cuComplex* tprp_fixed, cuComplex* qpar_fixed, cuComplex* qprp_fixed, cuComplex phi_test_in) 
{
  
  
//  double scaler_rVeal = phi_test_in.x / phi_fixed[nz/2+1].x;
//  double scaler_imag = phi_test_in.y / phi_fixed[nz/2+1].y;
//  
//  cuComplex phi_fixed_z0; 
//  cuComplex dens_fixed_z0;
//  cuComplex upar_fixed_z0;
//  cuComplex tpar_fixed_z0;
//  cuComplex tprp_fixed_z0;
//  cuComplex qpar_fixed_z0;
//  cuComplex qprp_fixed_z0;
//
//   phi_fixed_z0.x  = phi_fixed[nz/2+1].x * scaler_real;
//   dens_fixed_z0.x = dens_fixed[nz/2+1].x * scaler_real;
//   upar_fixed_z0.x = upar_fixed[nz/2+1].x * scaler_real;
//   tpar_fixed_z0.x = tpar_fixed[nz/2+1].x * scaler_real;
//   tprp_fixed_z0.x = tprp_fixed[nz/2+1].x * scaler_real;
//   qpar_fixed_z0.x = qpar_fixed[nz/2+1].x * scaler_real;
//   qprp_fixed_z0.x = qprp_fixed[nz/2+1].x * scaler_real;
//   phi_fixed_z0.y  = phi_fixed[nz/2+1].y * scaler_imag;
//   dens_fixed_z0.y = dens_fixed[nz/2+1].y * scaler_imag;
//   upar_fixed_z0.y = upar_fixed[nz/2+1].y * scaler_imag;
//   tpar_fixed_z0.y = tpar_fixed[nz/2+1].y * scaler_imag;
//   tprp_fixed_z0.y = tprp_fixed[nz/2+1].y * scaler_imag;
//   qpar_fixed_z0.y = qpar_fixed[nz/2+1].y * scaler_imag;
//   qprp_fixed_z0.y = qprp_fixed[nz/2+1].y * scaler_imag;
  

  unsigned int idz = get_idz();
 
  if(idz<nz) {
    //determine the scaler multiplier needed to set phi to the desired amplitude for each z

    double scaler_real = (double) phi_test_in.x / (double) phi_fixed[nz/2+1].x; 
    double scaler_imag = (double) phi_test_in.y / (double) phi_fixed[nz/2+1].y;

    cuComplex phi_fixed_out =  make_cuComplex( phi_fixed[idz].x * scaler_real, phi_fixed[idz].y * scaler_imag);
    cuComplex dens_fixed_out =  make_cuComplex( dens_fixed[idz].x * scaler_real, dens_fixed[idz].y * scaler_imag);
    cuComplex upar_fixed_out =  make_cuComplex( upar_fixed[idz].x * scaler_real, upar_fixed[idz].y * scaler_imag);
    cuComplex tpar_fixed_out =  make_cuComplex( tpar_fixed[idz].x * scaler_real, tpar_fixed[idz].y * scaler_imag);
    cuComplex tprp_fixed_out =  make_cuComplex( tprp_fixed[idz].x * scaler_real, tprp_fixed[idz].y * scaler_imag);
    cuComplex qpar_fixed_out =  make_cuComplex( qpar_fixed[idz].x * scaler_real, qpar_fixed[idz].y * scaler_imag);
    cuComplex qprp_fixed_out =  make_cuComplex( qprp_fixed[idz].x * scaler_real, qprp_fixed[idz].y * scaler_imag);
   
    phi_fixed[idz] = phi_fixed_out;
    dens_fixed[idz] = dens_fixed_out;
    upar_fixed[idz] = upar_fixed_out;
    tpar_fixed[idz] = tpar_fixed_out;
    tprp_fixed[idz] = tprp_fixed_out;
    qpar_fixed[idz] = qpar_fixed_out;
    qprp_fixed[idz] = qprp_fixed_out;
  
    //phi_fixed[idz].x = (double)  phi_fixed[idz].x * scaler_real;
    //dens_fixed[idz].x = (double) dens_fixed[idz].x * scaler_real;
    //upar_fixed[idz].x = (double) upar_fixed[idz].x * scaler_real;
    //tpar_fixed[idz].x = (double) tpar_fixed[idz].x * scaler_real;
    //tprp_fixed[idz].x = (double) tprp_fixed[idz].x * scaler_real;
    //qpar_fixed[idz].x = (double) qpar_fixed[idz].x * scaler_real;
    //qprp_fixed[idz].x = (double) qprp_fixed[idz].x * scaler_real;
    //phi_fixed[idz].y = (double)  phi_fixed[idz].y * scaler_imag;
    //dens_fixed[idz].y = (double) dens_fixed[idz].y * scaler_imag;
    //upar_fixed[idz].y = (double) upar_fixed[idz].y * scaler_imag;
    //tpar_fixed[idz].y = (double) tpar_fixed[idz].y * scaler_imag;
    //tprp_fixed[idz].y = (double) tprp_fixed[idz].y * scaler_imag;
    //qpar_fixed[idz].y = (double) qpar_fixed[idz].y * scaler_imag;
    //qprp_fixed[idz].y = (double) qprp_fixed[idz].y * scaler_imag;

//    phi_fixed[idz] = phi_fixed_z0;
//    dens_fixed[idz] = dens_fixed_z0;
//    upar_fixed[idz] = upar_fixed_z0;
//    tpar_fixed[idz] = tpar_fixed_z0;
//    tprp_fixed[idz] = tprp_fixed_z0;
//    qpar_fixed[idz] = qpar_fixed_z0;
//    qprp_fixed[idz] = qprp_fixed_z0;

  }
}
__global__ void set_fixed_amplitude(cuComplex* phi_fixed, cuComplex* dens_fixed, cuComplex* upar_fixed, cuComplex* tpar_fixed, cuComplex* tprp_fixed, cuComplex* qpar_fixed, cuComplex* qprp_fixed, cuComplex phi_test_in) 
{
  
  
  double scaler_real = phi_test_in.x / phi_fixed[nz/2+1].x;
  double scaler_imag = phi_test_in.y / phi_fixed[nz/2+1].y;
  
  cuComplex phi_fixed_z0; 
  cuComplex dens_fixed_z0;
  cuComplex upar_fixed_z0;
  cuComplex tpar_fixed_z0;
  cuComplex tprp_fixed_z0;
  cuComplex qpar_fixed_z0;
  cuComplex qprp_fixed_z0;

   phi_fixed_z0.x  = phi_fixed[nz/2+1].x * scaler_real;
   dens_fixed_z0.x = dens_fixed[nz/2+1].x * scaler_real;
   upar_fixed_z0.x = upar_fixed[nz/2+1].x * scaler_real;
   tpar_fixed_z0.x = tpar_fixed[nz/2+1].x * scaler_real;
   tprp_fixed_z0.x = tprp_fixed[nz/2+1].x * scaler_real;
   qpar_fixed_z0.x = qpar_fixed[nz/2+1].x * scaler_real;
   qprp_fixed_z0.x = qprp_fixed[nz/2+1].x * scaler_real;
   phi_fixed_z0.y  = phi_fixed[nz/2+1].y * scaler_imag;
   dens_fixed_z0.y = dens_fixed[nz/2+1].y * scaler_imag;
   upar_fixed_z0.y = upar_fixed[nz/2+1].y * scaler_imag;
   tpar_fixed_z0.y = tpar_fixed[nz/2+1].y * scaler_imag;
   tprp_fixed_z0.y = tprp_fixed[nz/2+1].y * scaler_imag;
   qpar_fixed_z0.y = qpar_fixed[nz/2+1].y * scaler_imag;
   qprp_fixed_z0.y = qprp_fixed[nz/2+1].y * scaler_imag;
  

  unsigned int idz = get_idz();
 
  if(idz<nz) {
    //determine the scaler multiplier needed to set phi to the desired amplitude for each z
/*
    double scaler_real = (double) phi_test_in.x / (double) phi_fixed[nz/2+1].x; 
    double scaler_imag = (double) phi_test_in.y / (double) phi_fixed[nz/2+1].y;
  
    phi_fixed[idz].x = (double)  phi_fixed[idz].x * scaler_real;
    dens_fixed[idz].x = (double) dens_fixed[idz].x * scaler_real;
    upar_fixed[idz].x = (double) upar_fixed[idz].x * scaler_real;
    tpar_fixed[idz].x = (double) tpar_fixed[idz].x * scaler_real;
    tprp_fixed[idz].x = (double) tprp_fixed[idz].x * scaler_real;
    qpar_fixed[idz].x = (double) qpar_fixed[idz].x * scaler_real;
    qprp_fixed[idz].x = (double) qprp_fixed[idz].x * scaler_real;
    phi_fixed[idz].y = (double)  phi_fixed[idz].y * scaler_imag;
    dens_fixed[idz].y = (double) dens_fixed[idz].y * scaler_imag;
    upar_fixed[idz].y = (double) upar_fixed[idz].y * scaler_imag;
    tpar_fixed[idz].y = (double) tpar_fixed[idz].y * scaler_imag;
    tprp_fixed[idz].y = (double) tprp_fixed[idz].y * scaler_imag;
    qpar_fixed[idz].y = (double) qpar_fixed[idz].y * scaler_imag;
    qprp_fixed[idz].y = (double) qprp_fixed[idz].y * scaler_imag;
*/
    phi_fixed[idz] = phi_fixed_z0;
    dens_fixed[idz] = dens_fixed_z0;
    upar_fixed[idz] = upar_fixed_z0;
    tpar_fixed[idz] = tpar_fixed_z0;
    tprp_fixed[idz] = tprp_fixed_z0;
    qpar_fixed[idz] = qpar_fixed_z0;
    qprp_fixed[idz] = qprp_fixed_z0;

  }
}

__global__ void scale_ky0(cuComplex* f, float scaler) 
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(idx<nx && idy==0 && idz<nz) {
    unsigned int index = idy + (ny/2+1)*idx + (ny/2+1)*nx*idz;
    f[index] = f[index]*scaler;
  }

}

__global__ void scale_ky_neq_0(cuComplex* result, cuComplex* f, float scaler) 
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(idx<nx && idy>0 && idy<(ny/2+1) && idz<nz) {
    unsigned int index = idy + (ny/2+1)*idx + (ny/2+1)*nx*idz;
    result[index] = f[index]*scaler;
  }

}
__global__ void scale_ky_neq_0(cuComplex* f, float scaler) 
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(idx<nx && idy>0 && idy<(ny/2+1) && idz<nz) {
    unsigned int index = idy + (ny/2+1)*idx + (ny/2+1)*nx*idz;
    f[index] = f[index]*scaler;
  }

}

__global__ void scale_ky_neq_0(cuComplex* f, float scaler, int nx, int ny, int nz) 
{

  unsigned int idx = get_idx();
  unsigned int idy = get_idy();
  unsigned int idz = get_idz();

  if(idx<nx && idy>0 && idy<(ny/2+1) && idz<nz) {
    unsigned int index = idy + (ny/2+1)*idx + (ny/2+1)*nx*idz;
    f[index] = f[index]*scaler;
  }

}
__global__ void sqr_div_real(float* result, float* a, float* b) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny) && idx<nx && idz<nz) {
    int index = idy + (ny)*idx + nx*(ny)*idz;
    
    //if(abs(a[index]) < 1.e-10 || abs(b[index]) < 1.e-10 ) result[index] = 0.;
    //else 
      result[index] = 2.*a[index]*a[index];///b[index];
    
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny) && idx<nx && idz<zthreads) {
    int index = idy + (ny)*idx + nx*(ny)*idz + nx*ny*zthreads*i;
    
    result[index] = a[index]*a[index]/b[index];
    
    }
   }
  }     
}    

__global__ void sqr_div_complex(cuComplex* result, cuComplex* a, cuComplex* b, float fac) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny) && idx<nx && idz<nz) {
    int index = idy + (ny)*idx + nx*(ny)*idz;
    
    //if(abs(a[index]) < 1.e-7 || abs(b[index]) < 1.e-7 ) result[index] = 0.;
    //else 
      //cuComplex num = fac*a[index]*a[index]*cuConjf(b[index]);
        cuComplex den = b[index]*cuConjf(b[index]);
        if(den.x == 0.) {result[index].x = 0.; result[index].y = 0.;}
      //if(abs(den.x) < 1.e-5) {result[index].x = 0.; result[index].y = 0.;}
        else 
          result[index] = fac*a[index]*a[index]/b[index];
    
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny) && idx<nx && idz<zthreads) {
    int index = idy + (ny)*idx + nx*(ny)*idz + nx*ny*zthreads*i;
    
    result[index] = a[index]*a[index]/b[index];
    
    }
   }
  }     
}    

__global__ void sqr_complex(cuComplex* result, cuComplex* a, float fac) 
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
    int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    
    //if(abs(a[index]) < 1.e-7 || abs(b[index]) < 1.e-7 ) result[index] = 0.;
    //else 
      //cuComplex num = fac*a[index]*a[index]*cuConjf(b[index]);
      //if(abs(den.x) < 1.e-5) {result[index].x = 0.; result[index].y = 0.;}
        //else 
          result[index] = fac*a[index]*a[index];
    
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny) && idx<nx && idz<zthreads) {
    int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
    
    result[index] = fac*a[index]*a[index];
    
    }
   }
  }     
}    


__global__ void zonal_only(cuComplex* f)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
   if(idy<(ny/2+1) && idx<nx && idz<nz) {
    int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
    
    if(idy != 0) {
      f[index].x = 0.;
      f[index].y = 0.;
    }
    
   }
  }
  else {
   for(int i=0; i<nz/zthreads; i++) {
    if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
    int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
    
      if(idy != 0) {
        f[index].x = 0.;
        f[index].y = 0.;
      }
    
    }
   }
  }     
}    


__global__ void copy_ky(cuComplex* f, float* ky)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
 
      f[index].x = ky[idy];
      f[index].y = 0.;
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
      f[index].x = ky[idy];
      f[index].y = 0.;
	
      }
    }
  }
} 

__global__ void mult_abs1_2(float* result, float* f1, float* f2)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;
      
      result[index] = abs(f1[index])*f2[index];
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
        result[index] = abs(f1[index])*f2[index];
      }
    }
  }
}

__global__ void mult_1_sgn2_3(float* result, float* f1, float* f2, float* f3)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;

      int sgn_f2 = 0;

      if(f2[index]>0) sgn_f2=1;
      if(f2[index]<0) sgn_f2=-1;

      result[index] = f1[index]*sgn_f2*f3[index];
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
      int sgn_f2 = 0;

      if(f2[index]>0) sgn_f2=1;
      if(f2[index]<0) sgn_f2=-1;

        result[index] = f1[index]*sgn_f2*f3[index];
      }
    }
  }
}


__global__ void hilbert_v(float* result, float* vx, float* vy)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny)*idx + nx*(ny)*idz;

      float x = 2.*M_PI*X0_d*(idx)/nx;
      float y = 2.*M_PI*Y0_d*(idy)/ny;

      float v_abs = sqrt( vx[index]*vx[index] + vy[index]*vy[index] );
      float v_dot_x = vx[index]*x + vy[index]*y;

      if( v_abs < 1.e-7 || index%2==0 ) result[index]=0.; 
      else result[index] = -1./tan(v_dot_x/v_abs);

    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny)*idx + nx*(ny)*idz + nx*(ny)*zthreads*i;
	
      }
    }
  }
}

