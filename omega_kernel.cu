__global__ void iOmegaStar(cuComplex* result, cuComplex* f, float* ky)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      cuComplex tmp;
      tmp.x = -ky[idy] * f[index].y;
      tmp.y =  ky[idy] * f[index].x;      
      result[index] = tmp;		 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
        cuComplex tmp;
	tmp.x = -ky[idy] * f[index].y;
	tmp.y =  ky[idy] * f[index].x;      
	result[index] = tmp;
        
      }
    }
  }   
}

__global__ void iOmegaD(cuComplex* result, cuComplex* f, float rho, float vt, float* kx,float* ky,
						float shat, float* gb,float* gb0,float* cv,float* cv0)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      cuComplex tmp;
      tmp.x = -omegaD(rho,vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]) * f[index].y;
      tmp.y =  omegaD(rho,vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]) * f[index].x;
      result[index] = tmp;		 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
        cuComplex tmp;
	tmp.x = -omegaD(rho,vt,kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]) * f[index].y;
	tmp.y =  omegaD(rho,vt,kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]) * f[index].x;
	result[index] = tmp;
        
      }
    }
  }     
}

__global__ void iOmegaD_ky0(cuComplex* result, cuComplex* f, float rho, float vt, float* kx,float* ky,
						float shat, float* gb,float* gb0,float* cv,float* cv0)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();
  
  if(nz<=zthreads) {
    if(idy==0 && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      cuComplex tmp;
      tmp.x = -omegaD(rho,vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]) * f[index].y;
      tmp.y =  omegaD(rho,vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz]) * f[index].x;
      result[index] = tmp;
      		 
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy==0 && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
	
        cuComplex tmp;
	tmp.x = -omegaD(rho,vt,kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]) * f[index].y;
	tmp.y =  omegaD(rho,vt,kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ]) * f[index].x;
	result[index] = tmp;
        
      }
    }
  }     
}

__global__ void absOmegaD(cuComplex* result, cuComplex* f, float rho, float vt, float* kx,float* ky,
						float shat, float* gb,float* gb0,float* cv,float* cv0, bool varenna)
{
  unsigned int idy = get_idy();
  unsigned int idx = get_idx();
  unsigned int idz = get_idz();  
  
  if(nz<=zthreads) {
    if(idy<(ny/2+1) && idx<nx && idz<nz) {
      unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz;
      
      if(idy == 0 && varenna) {
        result[index].x = 0;
	result[index].y = 0;
      }
      else {
        result[index] = abs(omegaD(rho,vt,kx[idx],ky[idy],shat,gb[idz],gb0[idz],cv[idz],cv0[idz])) * f[index];
      }
      
    }
  }
  else {
    for(int i=0; i<nz/zthreads; i++) {
      if(idy<(ny/2+1) && idx<nx && idz<zthreads) {
        unsigned int index = idy + (ny/2+1)*idx + nx*(ny/2+1)*idz + nx*(ny/2+1)*zthreads*i;
	
	unsigned int IDZ = idz + zthreads*i;
        
	if(idy == 0 && varenna) {
          result[index].x = 0;
	  result[index].y = 0;
	}
	else {
	  result[index] = abs(omegaD(rho,vt,kx[idx],ky[idy],shat,gb[IDZ],gb0[IDZ],cv[IDZ],cv0[IDZ])) * f[index];
        }
        
      }
    }
  }   
}
