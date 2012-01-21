void maxReduc(cufftComplex* max, cufftComplex* f, cufftComplex* padded) 
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
    
    dim3 dimBlock(8,8,8);
    //dimBlock.x=dimBlock.y=dimBlock.z=8;
      //dim3 dimGrid(Nx/dimBlock.x+1,Ny/dimBlock.y+1,1);
    int gridx = (Nx*Ny*Nz)/512;
    
    if (Nx*Ny*Nz <= 512) {
      dimBlock.x = Nx;
      dimBlock.y = Ny;
      dimBlock.z = Nz;
      gridx = 1;
    }  
    
    dim3 dimGrid(gridx,1,1);
    //dimGrid.x=gridx;
    //dimGrid.y=dimGrid.z=1;
    
    maximum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(padded, padded);
    
    while(dimGrid.x > 512) {
      dimGrid.x = dimGrid.x / 512;
      // dimGrid.x = 8
      maximum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(padded, padded);
      // result = 8 elements
    }  
    
    dimBlock.x = dimGrid.x;
    dimGrid.x = 1;
    dimBlock.y = dimBlock.z = 1;
    maximum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(padded,padded);  
    
    cudaMemcpy(max, padded, sizeof(cufftComplex), cudaMemcpyDeviceToHost);

}    
