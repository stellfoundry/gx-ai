/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>
#include <cufft.h>


// includes, kernels
#include <fft_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);
void getfcn(float* a, float* a, int N);
void getfcnC(cufftComplex *c, cufftComplex *c, int N);

__global__ void real2complex( float *a, cufftComplex *c, int N);

__global__ void complex2real( cufftComplex *c, float *a, int N);

__global__ void multiply(cufftComplex *c);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    cufftReal *fcn, *gcn;
    cufftComplex *gcn_complex;
    float *k;
    int N = 8;
    fcn = (cufftReal*) malloc(sizeof(cufftReal)*N*N);
    gcn = (cufftReal*) malloc(sizeof(cufftReal)*N*N);
    k = (float*) malloc(sizeof(float)*N*N);
    gcn_complex = (cufftComplex*) malloc(sizeof(cufftComplex)*(N)*(N));
    
    cufftReal *fcn_d;
    cudaMalloc((void**) &fcn_d, sizeof(cufftReal)*N*N);
    cufftComplex *fcn_complex;
    cudaMalloc((void**) &fcn_complex, sizeof(cufftComplex)*(N)*(N));
    
    //set fcn on host
    
    for(int j=0; j<N; j++) {
      for(int i=0; i<N; i++) {
      
      //1 + 3*cos((float)(2*3.14159265*i/N)) - cos((float)(6*3.14159265*i/N));
      k[i] = 2*M_PI*(float)(i-N/2)/N;
      k[j] = 2*M_PI*(float)(j-N/2)/N;
      int index = i + N*j;
      
      /* fcn[index] = 0+ 3*cos(1*2*M_PI*(i-N/2)/N) + 0*cos(2*2*M_PI*(i-N/2)/N) + 0*cos(3*2*M_PI*(i-N/2)/N)
               + 0*cos(4*2*M_PI*(i-N/2)/N) + 0*cos(5*2*M_PI*(i-N/2)/N) +
	       0*cos(7*2*M_PI*(i-N/2)/N) + sin(5*2*M_PI*(i-N/2)/N); */
      fcn[index]= 1*cos( 1*k[i] + 0*k[j]) + 3*cos(2*k[i] + 1*k[j]);
      printf("f(%d,%d)= %f   ", i, j, fcn[index]);
      
      //fcn_complex[i].x = fcn[i];
      //fcn_complex[i].y = 0.f;
      }
      printf("\n");
    }
    printf("\n");
    cudaMemcpy(fcn_d, fcn, sizeof(cufftReal)*N*N, cudaMemcpyHostToDevice);
    
    
    //return 0;
    
    
    //return 0;
      
    /* for(int i=0; i<N; ++i) {
      fcn_complex[i].x = fcn_d[i];
      fcn_complex[i].y = 0.f;
      } */
      
    int block_size_x=4; int block_size_y=2; 

    dim3 dimBlock(block_size_x, block_size_y);
    dim3 dimGrid(N/dimBlock.x, N/dimBlock.y);
    
    //real2complex<<<dimGrid, dimBlock>>>(fcn_d, fcn_complex, N);  
    
    cufftHandle plan;
    cufftHandle plan2;
    
    cufftPlan2d(&plan, N, N, CUFFT_R2C);
    cufftPlan2d(&plan2, N, N, CUFFT_C2R);
    
    cufftExecR2C(plan, fcn_d, fcn_complex);
    
    getfcnC(gcn_complex, fcn_complex, N);
    printf("\n");
    
    
    
    
    //multiply<<<dimGrid, dimBlock>>> (fcn_complex);
    
    cufftExecC2R(plan2, fcn_complex, fcn_d);
    
    //complex2real<<<dimGrid, dimBlock>>>(fcn_complex, fcn_d, N);
    
    getfcn(gcn, fcn_d, N);
    
    
    /* for(int i=0; i<N; ++i) {
      fcn[i] = fcn_complex[i].x;
      printf("x=%d: %f\n", i, fcn[i]);
      } */
    
    cufftDestroy(plan);
    cufftDestroy(plan2);
    
    cudaFree(fcn_d); cudaFree(fcn_complex);
    
      
    
      


}

void getfcn(cufftReal* fcn, cufftReal* fcn_d, int N) {
  cudaMemcpy(fcn, fcn_d, sizeof(cufftReal)*N*N, cudaMemcpyDeviceToHost);
    
  for(int j=0; j<N; j++) {
    for(int i=0; i<N; i++) {
      int index = i + N*j;
      printf("f(%d,%d)= %f   ", i, j, fcn[index]);     
      }
      printf("\n");
    }
  }  
    
void getfcnC(cufftComplex* fcn, cufftComplex* fcn_d, int N) {
  cudaMemcpy(fcn, fcn_d, sizeof(cufftComplex)*(N)*(N), cudaMemcpyDeviceToHost);
    
    for(int j=0; j<N/2+1; j++) {
      for(int i=0; i<N; i++) { 
        int index = i + N*j;
	printf("F(%d,%d)= %f   ", i, j, fcn[index].x);
      }
      printf("\n");
    }
    }    

__global__ void real2complex(float *a, cufftComplex *c, int N)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if(idx<N)
    {
    
    c[idx].x = a[idx];
    c[idx].y = 0.f;
    }
}

__global__ void multiply(cufftComplex *c) {
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  c[idx].x = 2*c[idx].x;
  c[idx].y = 2*c[idx].y;
  }

__global__ void complex2real(cufftComplex *c, float* a, int N)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if(idx<N)
    {
    
    a[idx] = c[idx].x;
    //c[idx].y = 0.f;
    }
}
////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    bool bTestResult = true;

    shrQAStart(argc, argv);

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cutilSafeCall( cudaSetDevice( cutGetMaxGflopsDeviceId() ) );

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    unsigned int num_threads = 32;
    unsigned int mem_size = sizeof( float) * num_threads;

    // allocate host memory
    float* h_idata = (float*) malloc( mem_size);
    // initalize the memory
    for( unsigned int i = 0; i < num_threads; ++i) 
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float* d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for result
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));

    // setup execution parameters
    dim3  grid( 1, 1, 1);
    dim3  threads( num_threads, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads, mem_size >>>( d_idata, d_odata);
    
    
    
    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    // copy result from device to host
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, sizeof( float) * num_threads,
                                cudaMemcpyDeviceToHost) );

    cutilCheckError( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));

    // compute reference solution
    float* reference = (float*) malloc( mem_size);
    computeGold( reference, h_idata, num_threads);

    // check result
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test
        cutilCheckError( cutWriteFilef( "./data/regression.dat",
                                      h_odata, num_threads, 0.0));
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        bTestResult = (bool)cutComparef( reference, h_odata, num_threads);
    }

    // cleanup memory
    free( h_idata);
    free( h_odata);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));

    cutilDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, (bTestResult ? QA_PASSED : QA_FAILED) );
}
