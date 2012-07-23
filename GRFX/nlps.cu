//void getfcn(cufftComplex* fcn_d);
//void getfcn(cufftReal* f);

 
//void getfcn(cufftReal* fcn, cufftReal* fcn_d, int Nx, int Ny, int Nz);

//void kInit(float *k, float *k, float *k, float *k, int N, int N, int N);



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
void NLPS(cufftComplex *result, cufftComplex *f, cufftComplex *g, float *kx, float *ky)
{
    //host variables
    //everything done on device
    
    
    //device variables
    float scaler;   				
    cudaMalloc((void**) &scaler, sizeof(float));
    //other variables declared, allocated, and de-allocated throughout program to preserve memory

       
    cufftHandle plan;
    cufftHandle plan2;
    int n[2] = {Nx, Ny};
    
    
    cufftPlanMany(&plan, 2,n,NULL,1,0,NULL,1,0,CUFFT_R2C,Nz);
    cufftPlanMany(&plan2,2,n,NULL,1,0,NULL,1,0,CUFFT_C2R,Nz);
     
      
    cufftComplex *dy, *dx;
    cufftReal *fdxR, *fdyR;
    cufftReal *gdxR, *gdyR;
    cudaMalloc((void**) &gdxR, sizeof(cufftReal)*Ny*Nx*Nz);
    cudaMalloc((void**) &gdyR, sizeof(cufftReal)*Ny*Nx*Nz);
    cudaMalloc((void**) &dx, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
    cudaMalloc((void**) &dy, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);     
    cudaMalloc((void**) &fdxR, sizeof(cufftReal)*Ny*Nx*Nz);    
    cudaMalloc((void**) &fdyR, sizeof(cufftReal)*Ny*Nx*Nz);
    if(debug) getError("After allocations");

        
    deriv<<<dimGrid, dimBlock>>> (f, dx, dy, kx, ky);    
    if(debug) getError("After first kernel");
    cufftExecC2R(plan2, dy, fdyR);
    if(debug) getError("After first FFT");
    cufftExecC2R(plan2, dx, fdxR);
    
    
    deriv<<<dimGrid, dimBlock>>> (g, dx, dy, kx, ky); 
    cufftExecC2R(plan2, dy, gdyR);
    cufftExecC2R(plan2, dx, gdxR);
    //scaling for these FFTs done in bracket kernel

    cudaFree(dy); cudaFree(dx);

    
    cufftReal *resultR;
    
    cudaMalloc((void**) &resultR, sizeof(cufftReal)*Ny*Nx*Nz);
    

    
    //scaler = (float)1 / (Nx*Nx*Ny*Ny);
    
    
    
    
    
    bracket<<<dimGrid, dimBlock>>> (resultR, fdxR, fdyR, gdxR, gdyR, .25);
    
    
    cufftExecR2C(plan, resultR, result);  
    scaler = (float) 1/(Nx*Ny/2);
    scale<<<dimGrid,dimBlock>>>(result,scaler);
    
    ///////////////////////////////////////////////
    //  mask kernel
    
    if(MASK) {
      mask<<<dimGrid,dimBlock>>>(result);
    }
    else {
      if(!quiet) printf("\nNO MASK\n");
    }
    
    ///////////////////////////////////////////////
    
    
    
    cufftDestroy(plan);
    cufftDestroy(plan2);
    
    
    cudaFree(resultR), cudaFree(fdxR), cudaFree(fdyR);
    cudaFree(gdxR), cudaFree(gdyR);
    
    
       
    
}

