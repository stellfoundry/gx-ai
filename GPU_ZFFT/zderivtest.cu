#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

#include <zfft_kernel.cu>
#include <zderiv.cu>

//void getfcnC(cufftComplex* fcn, cufftComplex* fcn_d, int Ny, int Nx, int Nz);

cufftReal* ZDERIVtest(int akx, int aky, int akz, int asin, int acos, int Ny, int Nx, int Nz) 
{
    //host variables
    cufftReal *a, *b;
    //cufftComplex *test_complex;
    
    float *y, *x, *z;
    a = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    b = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    y = (float*) malloc(sizeof(float)*Ny);                                 //
    x = (float*) malloc(sizeof(float)*Nx);				   //
    z = (float*) malloc(sizeof(float)*Nz);
    //test_complex = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
    
    
    
    
    //device variables
    cufftReal *a_d, *b_d;
    cufftComplex *a_complex_d, *b_complex_d;
    float scaler;
    cudaMalloc((void**) &a_d, sizeof(cufftReal)*Nx*Ny*Nz); 
    cudaMalloc((void**) &b_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &a_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);
    cudaMalloc((void**) &b_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);
    cudaMalloc((void**) &scaler, sizeof(float));
    
    int block_size_x=16; int block_size_y=16; 
    int dimGridx, dimGridy;

    dim3 dimBlock(block_size_x, block_size_y);
    if(Nx/dimBlock.x == 0) {dimGridx = 1;}
    else dimGridx = Nx/dimBlock.x;
    if(Ny/dimBlock.y == 0) {dimGridy = 1;}
    else dimGridy = Ny/dimBlock.y;
    dim3 dimGrid(dimGridx, dimGridy);   
    
    for(int k=0; k<Nz; k++) {
     for(int j=0; j<Nx; j++) {
      for(int i=0; i<Ny; i++) {
      
      
      y[i] = 2*M_PI*(float)(i-Ny/2)/Ny;                             //  
      x[j] = 2*M_PI*(float)(j-Nx/2)/Nx;				    //
      z[k] = 2*M_PI*(float)(k-Nz/2)/Nz;				    //
      int index = i + Ny*j + Ny*Nx*k;
      
      //we start with a field a of the form a(y,x,z)    
      a[index]= acos*cos(aky*y[i] + akx*x[j] + akz*z[k]) + 		//
                asin*sin(aky*y[i] + akx*x[j] + akz*z[k]);	        //
      
      
      
      //printf("f(%.2fPI,%.2fPI,%.2fPI)=%.3f: %d ", y[i]/M_PI, x[j]/M_PI, z[k]/M_PI, a[index], index);
      
      }
      //printf("\n");
     }
    }
    
    cudaMemcpy(a_d, a, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
    
    
    cufftHandle plan;
    cufftHandle plan2;
    int n[2] = {Nx, Ny};
    
    cufftPlanMany(&plan, 2,n,NULL,1,0,NULL,1,0,CUFFT_R2C,Nz);
    cufftPlanMany(&plan2,2,n,NULL,1,0,NULL,1,0,CUFFT_C2R,Nz);
    
    cufftExecR2C(plan, a_d, a_complex_d);
    /*  printf("a(ky,kx,z)\n");
    getfcnC(test_complex, a_complex_d, Ny, Nx, Nz);
    printf("\n"); */ 
    //now we have a field of the form a(ky,kx,z)
    
    /**************************************************************/
    // the ZDERIV sub-routine takes an input field a(ky,kx,z) and //
    // outputs a field b(ky,kx,z) = da/dz                         // 
    /**************************************************************/ 
    b_complex_d = ZDERIV(a_complex_d, b_complex_d, Ny, Nx, Nz);
    
    /*  printf("b(ky,kx,z)\n");
    getfcnC(test_complex, b_complex_d, Ny, Nx, Nz);
    printf("\n"); */ 
    
    cufftExecC2R(plan2, b_complex_d, b_d);
    
    
    scaler = (float)1/(Ny*Nx);
    
    scaleReal<<<dimGrid,dimBlock>>>(b_d, scaler, Ny, Nx, Nz);
    
    cudaMemcpy(b, b_d, sizeof(cufftReal)*Ny*Nx*Nz, cudaMemcpyDeviceToHost);
    
    //we return a real field b(y,x,z)
    return b;
    
}    

/*   //prints a complex function    
void getfcnC(cufftComplex* fcn, cufftComplex* fcn_d, int Ny, int Nx, int Nz) {
  cudaMemcpy(fcn, fcn_d, sizeof(cufftComplex)*(Ny/2+1)*(Nx)*Nz, cudaMemcpyDeviceToHost);
    
   for(int k=0; k<(Nz); k++) { 
    for(int j=0; j<Nx/2+1; j++) { 
      for(int i=0; i<Ny/2+1; i++) {  
	int index = i + (Ny/2+1)*(j)+Nx*(Ny/2+1)*k;
	
	printf("F(%d,%d,%.2f)=%.3f+i*%.3f: %d ", i, j, 2*M_PI*(float)(k-Nz/2)/Nz, fcn[index].x, fcn[index].y, index);
      }
      printf("\n");
    }  
    for(int j=-Nx/2+1; j<0; j++) {
      for(int i=0; i<Ny/2+1; i++) {
        int index = (i) + (Ny/2+1)*(j+Nx)+Nx*(Ny/2+1)*k;
	
	printf("F(%d,%d,%.2f)=%.3f+i*%.3f: %d ", i, j, 2*M_PI*(float)(k-Nz/2)/Nz, fcn[index].x, fcn[index].y, index);
      }
        
      printf("\n");
    }
   } 
}       */
