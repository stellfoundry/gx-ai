void getfcn(cuComplex* fcn_d)
{
  cuComplex *fcnC;
  fcnC = (cuComplex*) malloc(sizeof(cuComplex)*(Ny/2+1)*Nx*Nz);
  cudaMemcpy(fcnC, fcn_d, sizeof(cuComplex)*(Ny/2+1)*Nx*Nz, cudaMemcpyDeviceToHost);
  for(int k=0; k<(Nz); k++) { 
    for(int j=0; j<Nx/2+1; j++) { 
      for(int i=0; i<Ny/2+1; i++) {  
	int index = i + (Ny/2+1)*(j) + Nx*(Ny/2+1)*k;
	
	
	if(!(fcnC[index].x<.00001 && fcnC[index].y<.00001 && fcnC[index].x>-.00001 && fcnC[index].y>-.00001)) {
	
	//printf("F(%d,%d,%.2f)...
	  printf("F(%g,%g,%g)=%e + i*%e  %d\n", (float) i, (float) j, 2*M_PI*(float)(k-Nz/2)/Nz,
	                                   fcnC[index].x, fcnC[index].y, index);
        }
      }
      //printf("\n");
    }  
    for(int j=-Nx/2+1; j<0; j++) {
      for(int i=0; i<Ny/2+1; i++) {
        int index = (i) + (Ny/2+1)*(j+Nx) + Nx*(Ny/2+1)*k;
	
	
	if(!(fcnC[index].x<.00001 && fcnC[index].y<.00001 && fcnC[index].x>-.00001 && fcnC[index].y>-.00001)) {

	  printf("F(%g,%g,%g)=%e + i*%e  %d\n", (float) i, (float) j, 2*M_PI*(float)(k-Nz/2)/Nz, 
	                   fcnC[index].x, fcnC[index].y, index);
        }
      }
        
      //printf("\n");
    }
  }  
  free(fcnC);
} 

void getfcnALL(cufftComplex* fcn_d)
{
  cufftComplex *fcnC;
  fcnC = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
  cudaMemcpy(fcnC, fcn_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz, cudaMemcpyDeviceToHost);
  //for(int k=0; k<(Nz); k++) { 
    for(int j=0; j<Nx/2+1; j++) { 
      for(int i=0; i<Ny/2+1; i++) {  
	int index = i + (Ny/2+1)*(j);// + Nx*(Ny/2+1)*k;
	
	//if(!(fcnC[index].x<.00001 && fcnC[index].y<.00001 && fcnC[index].x>-.00001 && fcnC[index].y>-.00001)) {
	
	//printf("F(%d,%d,%.2f)...
	  printf("F(%g,%g)=%.5f + i*%.5f  %d  ", (float) i, (float) j, //Zp*2*M_PI*(float)(k-Nz/2)/Nz,
	                     fcnC[index].x, fcnC[index].y, index);
        //}
      }
      printf("\n");
    }  
    for(int j=-Nx/2+1; j<0; j++) {
      for(int i=0; i<Ny/2+1; i++) {
        int index = (i) + (Ny/2+1)*(j+Nx);// + Nx*(Ny/2+1)*k;
	
	//if(!(fcnC[index].x<.00001 && fcnC[index].y<.00001 && fcnC[index].x>-.00001 && fcnC[index].y>-.00001)) {

	  printf("F(%g,%g)=%.5f + i*%.5f  %d  ", (float) i, (float) j, //Zp*2*M_PI*(float)(k-Nz/2)/Nz, 
	                   fcnC[index].x, fcnC[index].y, index);
        //}
      }
        
      printf("\n");
    }
  //}  
  free(fcnC);
} 

void getfcnZCOMPLEX(cufftComplex* fcn_d)
{
  cufftComplex *fcnC;
  fcnC = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
  cudaMemcpy(fcnC, fcn_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz, cudaMemcpyDeviceToHost);
  for(int k=0; k<Nz/2+1; k++) { 
    for(int j=0; j<Nx/2+1; j++) { 
      for(int i=0; i<Ny/2+1; i++) {  
	int index = i + (Ny/2+1)*(j) + Nx*(Ny/2+1)*k;
	
	if(!(fcnC[index].x<.00001 && fcnC[index].y<.00001 && fcnC[index].x>-.00001 && fcnC[index].y>-.00001)) {
	
	//printf("F(%d,%d,%.2f)...
	  printf("F(%g,%g,%g)=%.5f + i*%.5f  %d\n", (float) i/Y0, (float) j/X0, (float)k/Zp,
	                     fcnC[index].x, fcnC[index].y, index);
        }
      }
      //printf("\n");
    }  
    for(int j=-Nx/2+1; j<0; j++) {
      for(int i=0; i<Ny/2+1; i++) {
        int index = (i) + (Ny/2+1)*(j+Nx) + Nx*(Ny/2+1)*k;
	
	if(!(fcnC[index].x<.00001 && fcnC[index].y<.00001 && fcnC[index].x>-.00001 && fcnC[index].y>-.00001)) {

	  printf("F(%g,%g,%g)=%.5f + i*%.5f  %d\n", (float) i/Y0, (float) j/X0, (float)k/Zp, 
	                   fcnC[index].x, fcnC[index].y, index);
        }
      }
        
      //printf("\n");
    }
  } 
  for(int k=-Nz/2+1; k<0; k++) { 
    for(int j=0; j<Nx/2+1; j++) { 
      for(int i=0; i<Ny/2+1; i++) {  
	int index = i + (Ny/2+1)*(j) + Nx*(Ny/2+1)*(k+Nz);
	
	if(!(fcnC[index].x<.00001 && fcnC[index].y<.00001 && fcnC[index].x>-.00001 && fcnC[index].y>-.00001)) {
	
	//printf("F(%d,%d,%.2f)...
	  printf("F(%g,%g,%g)=%.5f + i*%.5f  %d\n", (float) i/Y0, (float) j/X0, (float)k/Zp,
	                     fcnC[index].x, fcnC[index].y, index);
        }
      }
      //printf("\n");
    }  
    for(int j=-Nx/2+1; j<0; j++) {
      for(int i=0; i<Ny/2+1; i++) {
        int index = (i) + (Ny/2+1)*(j+Nx) + Nx*(Ny/2+1)*(k+Nz);
	
	if(!(fcnC[index].x<.00001 && fcnC[index].y<.00001 && fcnC[index].x>-.00001 && fcnC[index].y>-.00001)) {

	  printf("F(%g,%g,%g)=%.5f + i*%.5f  %d\n", (float) i/Y0, (float) j/X0, (float)k/Zp, 
	                   fcnC[index].x, fcnC[index].y, index);
        }
      }
        
      //printf("\n");
    }
  }  
  free(fcnC);
} 

void getfcnZCOMPLEX_Covering(cufftComplex* fcn_d, int nLinks, int nChains, int* ky, int* kx, float* kz_d) 
{
  cufftComplex *fcnC; 
  float *kz;
  fcnC = (cufftComplex*) malloc(sizeof(cufftComplex)*Nz*nLinks*nChains);
  kz = (float*) malloc(sizeof(float)*Nz*nLinks);
  cudaMemcpy(fcnC, fcn_d, sizeof(cufftComplex)*Nz*nLinks*nChains, cudaMemcpyDeviceToHost);
  cudaMemcpy(kz, kz_d, sizeof(float)*nLinks*Nz, cudaMemcpyDeviceToHost);  //SEG FAULT
  for(int n=0; n<nChains; n++) { 
    for(int p=0; p<nLinks; p++) { 
      for(int i=0; i<Nz; i++) {  
	int index = i + p*Nz + n*Nz*nLinks;
	
	if( ky[p+nLinks*n] == 11 && (kx[p+nLinks*n] == 11 || kx[p+nLinks*n] == 106) ) {
	//if(!(fcnC[index].x<.0001 && fcnC[index].y<.0001 && fcnC[index].x>-.0001 && fcnC[index].y>-.0001)) {
	
	  //printf("F(%d,%d,%.2f)...
	  printf("F(%g,%g,%g)=%.5f + i*%.5f  %d,%d,%d\n", (float) ky[p+nLinks*n]/Y0, (float) kx[p+nLinks*n]/X0, kz[i+p*Nz],
	                     fcnC[index].x, fcnC[index].y, i,p,n);
        }
      }      
    }
  }
  free(fcnC); free(kz);
}     

void getfcn_Covering(cufftComplex* fcn_d, int nLinks, int nChains, int* ky, int* kx)
{
  cufftComplex *fcnC;
  fcnC = (cufftComplex*) malloc(sizeof(cufftComplex)*nLinks*nChains*Nz);
  cudaMemcpy(fcnC, fcn_d, sizeof(cufftComplex)*nLinks*nChains*Nz, cudaMemcpyDeviceToHost);
  for(int n=0; n<nChains; n++) { 
    for(int p=0; p<nLinks; p++) { 
      for(int i=0; i<Nz; i++) {  
	int index = i + p*Nz + n*Nz*nLinks;
	
	if(!(fcnC[index].x<.0001 && fcnC[index].y<.0001 && fcnC[index].x>-.0001 && fcnC[index].y>-.0001)) {
		
	  printf("F(%g,%g,%g)=%.5f + i*%.5f  %d\n", (float) ky[p+nLinks*n]/Y0, (float) kx[p+nLinks*n]/X0, 2*M_PI*(float)(i-Nz/2)/Nz,
	                     fcnC[index].x, fcnC[index].y, index);
        }
      }
      //printf("\n");
    }
  }    
    
  free(fcnC);
} 


void getfcn(float* fcn_d) {
  cufftReal *fcn;
  fcn = (float*) malloc(sizeof(float)*Ny*Nx*Nz);
  cudaMemcpy(fcn, fcn_d, sizeof(float)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
  
  for(int k=0; k<Nz; k++) {  
   for(int j=0; j<Nx; j++) {
    for(int i=0; i<Ny; i++) {
      int index = i + Ny*j + Nx*Ny*k;
      printf("f(%.2fPI,%.2fPI)=%.3e ", 2*(float)(i-Ny/2)/Ny, 2*(float)(j-Nx/2)/Nx, fcn[index]);     
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
     
                                             
   
