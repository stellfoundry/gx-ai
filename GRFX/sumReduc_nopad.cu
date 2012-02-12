void sumReduc_nopad(cufftComplex* result, cufftComplex* f) 
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
    
    sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(f, f);
    
    while(dimGrid.x > 512) {
      dimGrid.x = dimGrid.x / 512;
      sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(f, f);
    }  
    
    dimBlock.x = dimGrid.x;
    dimGrid.x = 1;
    dimBlock.y = dimBlock.z = 1;
    sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(f,f);  
    
    cudaMemcpy(result, f, sizeof(cufftComplex), cudaMemcpyDeviceToHost);

}    
