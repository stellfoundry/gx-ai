__global__ void deriv(cufftComplex* f, cufftComplex* fdx, cufftComplex* fdy, float* kx, float* ky)                        
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
     //df/dx
     fdy[index].x = -ky[idy]*f[index].y;			
     fdy[index].y =  ky[idy]*f[index].x;			
    
     //df/dy
     fdx[index].x = -kx[idx]*f[index].y;			
     fdx[index].y =  kx[idx]*f[index].x;   
   }
  } 
  else {
   for(int i=0; i<Nz/zThreads; i++) { 
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
    
    //df/dx
    fdy[index].x = -ky[idy]*f[index].y;			
    fdy[index].y =  ky[idy]*f[index].x;			
    
    //df/dy
    fdx[index].x = -kx[idx]*f[index].y;			
    fdx[index].y =  kx[idx]*f[index].x;			
    }
   }
  } 
}  

__global__ void mask(cufftComplex* mult) 
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
    unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    
    if( (idy>(Ny/3+1) || (idx>(Nx/3+1) && idx<(2*Nx/3+1))) ) {
      mult[index].x = 0;
      mult[index].y = 0;
    }  
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
     unsigned int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
    
    
     if( (idy>(Ny/3+1) || (idx>(Nx/3+1) && idx<(2*Nx/3+1))) ) {
       mult[index].x = 0;
       mult[index].y = 0;
     }  
    }
   }
  }
    
}      
  
  

__global__ void bracket(cufftReal* mult, cufftReal* fdx, cufftReal* fdy, 
                      cufftReal* gdx, cufftReal* gdy, float scaler)
{
  unsigned int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  unsigned int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  unsigned int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
   if(idy<(Ny) && idx<Nx && idz<Nz ) {
    unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz;
    
    
    mult[index] = scaler*( (fdx[index])*(gdy[index]) - (fdy[index])*(gdx[index]) );  
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny) && idx<Nx && idz<zThreads ) {
    unsigned int index = idy + (Ny)*idx + Nx*(Ny)*idz + Nx*Ny*zThreads*i;
    
    
    mult[index] = scaler*( (fdx[index])*(gdy[index]) - (fdy[index])*(gdx[index]) );  
    }
   }
  } 
 
}  

     					      
__global__ void zeroC(cufftComplex* f) 
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
   if(idy<(Ny/2+1) && idx<Nx && idz<Nz) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz;
    
    f[index].x = 0;
    f[index].y = 0;
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny/2+1) && idx<Nx && idz<zThreads) {
    int index = idy + (Ny/2+1)*idx + Nx*(Ny/2+1)*idz + Nx*(Ny/2+1)*zThreads*i;
    
    f[index].x = 0;
    f[index].y = 0;
    }
   }
  }    
}    

__global__ void zero(cufftReal* f) 
{
  int idy = __umul24(blockIdx.y,blockDim.y)+threadIdx.y;
  int idx = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
  int idz = __umul24(blockIdx.z,blockDim.z)+threadIdx.z;
  
  if(Nz<=zThreads) {
   if(idy<(Ny) && idx<Nx && idz<Nz) {
    int index = idy + (Ny)*idx + Nx*(Ny)*idz;
    
    f[index] = 0;
    
   }
  }
  else {
   for(int i=0; i<Nz/zThreads; i++) {
    if(idy<(Ny) && idx<Nx && idz<zThreads) {
    int index = idy + (Ny)*idx + Nx*(Ny)*idz + Nx*Ny*zThreads*i;
    
    f[index] = 0;
    
    }
   }
  }   
  
}    

void getfcn(cufftComplex* fcn_d)
{
  cufftComplex *fcnC;
  fcnC = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
  cudaMemcpy(fcnC, fcn_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz, cudaMemcpyDeviceToHost);
  //for(int k=0; k<(Nz); k++) { 
    for(int j=0; j<Nx/2+1; j++) { 
      for(int i=0; i<Ny/2+1; i++) {  
	int index = i + (Ny/2+1)*(j); // Nx*(Ny/2+1)*k;
	
	if(!(fcnC[index].x<.00001 && fcnC[index].y<.00001 && fcnC[index].x>-.00001 && fcnC[index].y>-.00001)) {
	
	//printf("F(%d,%d,%.2f)...
	  printf("F(%g,%g)=%.5f + i*%.5f  %d\n", (float) i/X0, (float) j/X0, //2*M_PI*(float)(k-Nz/2)/Nz,
	                     fcnC[index].x, fcnC[index].y, index);
        }
      }
      //printf("\n");
    }  
    for(int j=-Nx/2+1; j<0; j++) {
      for(int i=0; i<Ny/2+1; i++) {
        int index = (i) + (Ny/2+1)*(j+Nx);// + Nx*(Ny/2+1)*k;
	
	if(!(fcnC[index].x<.00001 && fcnC[index].y<.00001 && fcnC[index].x>-.00001 && fcnC[index].y>-.00001)) {

	  printf("F(%g,%g)=%.5f + i*%.5f  %d\n", (float) i/X0, (float) j/X0, //2*M_PI*(float)(k-Nz/2)/Nz, 
	                   fcnC[index].x, fcnC[index].y, index);
        }
      }
        
      //printf("\n");
    }
  //}  
  free(fcnC);
} 

void getfcn(cufftReal* fcn_d) {
  cufftReal *fcn;
  fcn = (cufftReal*) malloc(sizeof(cufftReal)*Ny*Nx*Nz);
  cudaMemcpy(fcn, fcn_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
  
  for(int k=0; k<Nz; k++) {  
   for(int j=0; j<Nx; j++) {
    for(int i=0; i<Ny; i++) {
      int index = i + Ny*j + Nx*Ny*k;
      printf("f(%.2fPI,%.2fPI)=%.3f ", 2*(float)(i-Ny/2)/Ny, 2*(float)(j-Nx/2)/Nx, fcn[index]);     
      }
      printf("\n");
    } printf("\n");
   } 
  free(fcn); 
}      

void getfcnComplexPadded(cufftComplex* fcn_d)
{
  cufftComplex *fcnC;
  
  fcnC = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny)*Nx*Nz);
  for(int i=0; i<Nx*Ny*Nz; i++) {
    fcnC[i].x = 0;
    fcnC[i].y = 0;
  }
  cudaMemcpy(fcnC, fcn_d, sizeof(cufftComplex)*(Ny)*Nx*Nz, cudaMemcpyDeviceToHost);
  //for(int k=0; k<(Nz); k++) { 
    /*for(int j=0; j<Nx/2+1; j++) { 
      for(int i=0; i<Ny; i++) {  
	int index = i + (Ny)*(j); // Nx*(Ny/2+1)*k;
	
	if(!(fcnC[index].x<.001 && fcnC[index].y<.001 && fcnC[index].x>-.001 && fcnC[index].y>-.001)) {
	
	//printf("F(%d,%d,%.2f)...
	  printf("F(%d,%d)=%.5f+i*%.5f  %d\n", i, j, //2*M_PI*(float)(k-Nz/2)/Nz,
	                     fcnC[index].x, fcnC[index].y, index);
        }
      }
      //printf("\n");
    }  
    for(int j=-Nx/2+1; j<0; j++) {
      for(int i=0; i<Ny; i++) {
        int index = (i) + (Ny)*(j+Nx);// + Nx*(Ny/2+1)*k;
	
	if(!(fcnC[index].x<.001 && fcnC[index].y<.001 && fcnC[index].x>-.001 && fcnC[index].y>-.001)) {

	  printf("F(%d,%d)=%.5f+i*%.5f  %d\n", i, j, //2*M_PI*(float)(k-Nz/2)/Nz, 
	                   fcnC[index].x, fcnC[index].y, index);
        }
      }
        
      //printf("\n");
    } */
  //}  
  
  for(int i=0; i<Nx*Ny*Nz; i++) {
    if(!(fcnC[i].x<.001 && fcnC[i].y<.001 && fcnC[i].x>-.001 && fcnC[i].y>-.001)) 
      printf("F(%d)=%.5f+i*%.5f\n", i, fcnC[i].x, fcnC[i].y);
  }  
  
  free(fcnC);
}     
     
                           
    
