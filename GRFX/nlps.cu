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
    
    
    
    /*int block_size_x=2; int block_size_y=2; 
    int dimGridx, dimGridy;

    dim3 dimBlock(block_size_x, block_size_y);
    if(Ny/dimBlock.x == 0) {dimGridx = 1;}
    else dimGridx = Nx/dimBlock.x;
    if(Nx/dimBlock.y == 0) {dimGridy = 1;}
    else dimGridy = Ny/dimBlock.y;
    dim3 dimGrid(dimGridx, dimGridy); */ 
    
    int dev;
    struct cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop,dev);
    int zThreads = prop.maxThreadsDim[2];
    int totalThreads = prop.maxThreadsPerBlock;   
    
    int xy = totalThreads/Nz;
    int blockxy = sqrt(xy);
    //dimBlock = threadsPerBlock, dimGrid = numBlocks
    dim3 dimBlock(blockxy,blockxy,Nz);
    if(Nz>zThreads) {
      dimBlock.x = sqrt(totalThreads/zThreads);
      dimBlock.y = sqrt(totalThreads/zThreads);
      dimBlock.z = zThreads;
    }  
    
    dim3 dimGrid(Nx/dimBlock.x+1,Ny/dimBlock.y+1,1);
    //if(dimGrid.x == 0) {dimGrid.x = 1;}
    //if(dimGrid.y == 0) {dimGrid.y = 1;}
    //if(dimGrid.z == 0) {dimGrid.z = 1;}  
    
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
    
    
    
    
    deriv<<<dimGrid, dimBlock>>> (f, dx, dy, kx, ky);
    
    scaler = (float) 1/(Nx*Ny);
    
    cufftExecC2R(plan2, dy, fdyR);
    cufftExecC2R(plan2, dx, fdxR);
    
    //scaleReal<<<dimGrid,dimBlock>>>(fdxR,scaler);
    //scaleReal<<<dimGrid,dimBlock>>>(fdyR,scaler);

    
    deriv<<<dimGrid, dimBlock>>> (g, dx, dy, kx, ky); 
    cufftExecC2R(plan2, dy, gdyR);
    cufftExecC2R(plan2, dx, gdxR);
    
    //scaling for these FFTs done in bracket kernel

    
    //scaleReal<<<dimGrid,dimBlock>>>(gdxR,scaler);
    //scaleReal<<<dimGrid,dimBlock>>>(gdyR,scaler);

    
    cudaFree(dy); cudaFree(dx);
      
    
    
    
    cufftReal *resultR;
    
    cudaMalloc((void**) &resultR, sizeof(cufftReal)*Ny*Nx*Nz);
    

    
    scaler = (float)1 / (Nx*Nx*Ny*Ny);
    
    //zero<<<dimGrid, dimBlock>>> (resultR);
    
    
    
    bracket<<<dimGrid, dimBlock>>> (resultR, fdxR, fdyR, gdxR, gdyR, .25);
    
    
    cufftExecR2C(plan, resultR, result);  
    scaler = (float) 1/(Nx*Ny/2);
    scale<<<dimGrid,dimBlock>>>(result,scaler);
    
    ///////////////////////////////////////////////
    //  mask kernel
    
    mask<<<dimGrid,dimBlock>>>(result);
    
    ///////////////////////////////////////////////
    
    //roundoff<<<dimGrid,dimBlock>>>(result,.00001);
    //zeromode<<<dimGrid,dimBlock>>>(result);
    
    cufftDestroy(plan);
    cufftDestroy(plan2);
    
    
    cudaFree(resultR), cudaFree(fdxR), cudaFree(fdyR);
    cudaFree(gdxR), cudaFree(gdyR);
    
    //cudaFree(scaler);
       
    
}

