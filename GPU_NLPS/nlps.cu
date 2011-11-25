

//  This subroutine takes as inputs two fields in (ky,kx,z) space and calculates
//  the nlps bracket, and returns it in (ky,kx,z) space

/***** LINES CHANGED FOR X <-> Y MARKED BY // *******/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cufft.h"

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

#include <nlps_kernel.cu>

//void getfcnC(cufftComplex* fcn, cufftComplex* fcn_d, int Nx, int Ny, int Nz);
//void getfcn(cufftReal* fcn, cufftReal* fcn_d, int Nx, int Ny, int Nz);

//void kInit(float *k, float *k, float *k, float *k, int N, int N, int N);

//kernels declared and defined in nlps_kernel.cu
/* __global__ void multiply(cufftComplex *c);

__global__ void deriv(cufftComplex *c, cufftComplex *d, cufftComplex *d, 
                      cufftComplex *c, cufftComplex *d, cufftComplex *d, 
                      float *k, float *k, int N, int N); 

__global__ void scalemult(cufftReal *f, cufftReal *f, cufftReal *f, cufftReal *f, cufftReal *f,
                      float a, int N, int N); */

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
cufftComplex* NLPS(cufftComplex *f_complex_d, cufftReal *fdxR_d, cufftReal *fdyR_d, 
                   cufftComplex *g_complex_d, cufftReal *gdxR_d, cufftReal *gdyR_d, 
		   float *kx_d, float *ky_d, int Ny, int Nx, int Nz, int isave) 
{
    //host variables
    //everything done on device
    
    
    //device variables
    
    
    cufftComplex *fdy_d, *fdx_d;
    cufftComplex *gdy_d, *gdx_d;
    float scaler;
    cufftReal *multR_d;
    cufftComplex *mult_d;
    cudaMalloc((void**) &fdx_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
    cudaMalloc((void**) &fdy_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz); 
    cudaMalloc((void**) &gdx_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
    cudaMalloc((void**) &gdy_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);      				
    cudaMalloc((void**) &scaler, sizeof(float));
    cudaMalloc((void**) &multR_d, sizeof(cufftReal)*Ny*Nx*Nz);
    cudaMalloc((void**) &mult_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
    
    
    int block_size_x=2; int block_size_y=2; 
    int dimGridx, dimGridy;

    dim3 dimBlock(block_size_x, block_size_y);
    if(Ny/dimBlock.x == 0) {dimGridx = 1;}
    else dimGridx = Nx/dimBlock.x;
    if(Nx/dimBlock.y == 0) {dimGridy = 1;}
    else dimGridy = Ny/dimBlock.y;
    dim3 dimGrid(dimGridx, dimGridy); 
    
    cufftHandle plan;
    cufftHandle plan2;
    int n[2] = {Nx, Ny};
    
    
    cufftPlanMany(&plan, 2,n,NULL,1,0,NULL,1,0,CUFFT_R2C,Nz);
    cufftPlanMany(&plan2,2,n,NULL,1,0,NULL,1,0,CUFFT_C2R,Nz);

    
    
    
    
    if(isave != 1) {      
      deriv<<<dimGrid, dimBlock>>> (f_complex_d, fdx_d, fdy_d, kx_d, ky_d, Ny, Nx, Nz);
      cufftExecC2R(plan2, fdy_d, fdyR_d);
      cufftExecC2R(plan2, fdx_d, fdxR_d);
    }  
    
    if(isave != 2) {
      deriv<<<dimGrid, dimBlock>>> (g_complex_d, gdx_d, gdy_d, kx_d, ky_d, Ny, Nx, Nz);    
      cufftExecC2R(plan2, gdy_d, gdyR_d);
      cufftExecC2R(plan2, gdx_d, gdxR_d);
    }  
     
    
    
    scaler = (float)1/(Nx*Nx*Ny*Ny);
    
    bracket<<<dimGrid,dimBlock>>> (multR_d, fdxR_d, fdyR_d, gdxR_d, gdyR_d, scaler, Ny, Nx, Nz);
    
    cufftExecR2C(plan, multR_d, mult_d);  
    
    ///////////////////////////////////////////////
    //  mask kernel
    
    mask<<<dimGrid,dimBlock>>>(mult_d, Ny, Nx, Nz);
    
    ///////////////////////////////////////////////
    
    
    cufftDestroy(plan);
    cufftDestroy(plan2);
    
    
    cudaFree(multR_d);
    cudaFree(fdy_d), cudaFree(fdx_d);
    cudaFree(gdy_d), cudaFree(gdx_d);
    
    
    //cudaFree(scaler);
       
    return mult_d;
}



