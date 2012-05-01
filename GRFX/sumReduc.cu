void sumReduc(cufftComplex* result, cufftComplex* f, cufftComplex* padded) 
{
    int dev;
    struct cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop,dev);
    int zThreads = prop.maxThreadsDim[2];
    int totalThreads = prop.maxThreadsPerBlock;   
    
    int xy = totalThreads/Nz;
    int blockxy = sqrt(xy);
    //dimBlock = threadsPerBlock, dimGrid = numBlocks
    dim3 dimBlock2(blockxy,blockxy,Nz);
    if(Nz>zThreads) {
      dimBlock2.x = sqrt(totalThreads/zThreads);
      dimBlock2.y = sqrt(totalThreads/zThreads);
      dimBlock2.z = zThreads;
    }  
    
    dim3 dimGrid2(Nx/dimBlock2.x+1,Ny/dimBlock2.y+1,1);
    
    cudaMemcpy(padded,f,sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToDevice);
    
    cleanPadded<<<dimGrid2,dimBlock2>>>(padded);
    
    //block size is 8*8*8=512, so that all of each block fits in shared memory
    dim3 dimBlock(8,8,8);
    //gridx is the number of blocks configured
    int gridx = (Nx*Ny*Nz)/512;
    
    if (Nx*Ny*Nz <= 512) {
      dimBlock.x = Nx;
      dimBlock.y = Ny;
      dimBlock.z = Nz;
      gridx = 1;
    }  
    
    dim3 dimGrid(gridx,1,1);
    
    sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*512>>>(padded, padded);
    
    while(dimGrid.x > 512) {
      dimGrid.x = dimGrid.x / 512;
      sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*512>>>(padded, padded);
    }  
    
    dimBlock.x = dimGrid.x;
    dimGrid.x = 1;
    dimBlock.y = dimBlock.z = 1;
    sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*512>>>(padded,padded);  
    
    cudaMemcpy(result, padded, sizeof(cufftComplex), cudaMemcpyDeviceToHost);

}    
