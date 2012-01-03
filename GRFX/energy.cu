void energy(cufftComplex* totEnergy_h, cufftComplex* kinEnergy_h, cufftComplex* magEnergy_h,
              cufftComplex* kPhi, cufftComplex* kA, 
	      cufftComplex* zp, cufftComplex* zm, float* kPerp2)
{
    cufftComplex *padded;
    cudaMalloc((void**) &padded, sizeof(cufftComplex)*Nx*Ny*Nz);
    
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
    
    
    addsubt<<<dimGrid,dimBlock>>> (kPhi, zp, zm, 1);
    //kPhi = zp+zm
        
    scale<<<dimGrid,dimBlock>>> (kPhi, .5);
    //kPhi = .5*(zp+zm) = phi
    
    addsubt<<<dimGrid,dimBlock>>> (kA, zp, zm, -1);
    //kA = zp-zm
    
    scale<<<dimGrid,dimBlock>>> (kA, .5);
    //kA = .5*(zp-zm) = A
    
    //since the R2C FFT duplicates some elements when ky=0 or ky=Ny/2, we have to fix this by zeroing the duplicate elements
    //before integrating
    fixFFT<<<dimGrid,dimBlock>>>(kPhi);
    fixFFT<<<dimGrid,dimBlock>>>(kA);
    
    squareComplex<<<dimGrid,dimBlock>>> (kPhi);
    //kPhi = phi**2
    
     
    squareComplex<<<dimGrid,dimBlock>>> (kA);
    //kA = A**2
    
    multKPerp<<<dimGrid,dimBlock>>> (kPhi, kPhi, kPerp2,-1);
    //kPhi = (kperp**2) * (phi**2)
    
    
    
    multKPerp<<<dimGrid,dimBlock>>> (kA, kA, kPerp2,-1);
    //kA = (kperp**2) * (A**2)
    
    
    // integrate kPhi to find kinetic energy
    cudaMemcpy(padded,kPhi,sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
    zeroPadded<<<dimGrid,dimBlock>>>(padded);
    
    
        
    dimBlock.x=dimBlock.y=dimBlock.z=8;
      //dim3 dimGrid(Nx/dimBlock.x+1,Ny/dimBlock.y+1,1);
    int gridx = (Nx*Ny*Nz)/512;
    
    if (Nx*Ny*Nz < 512) {
      dimBlock.x = Nx;
      dimBlock.y = Ny;
      dimBlock.z = Nz;
      gridx = 1;
    }  
    
    dimGrid.x=gridx;
    dimGrid.y=dimGrid.z=1;
    
    sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(padded, padded);
    // result = 4096 elements = gridx

    
    while(dimGrid.x > 512) {
      dimGrid.x = dimGrid.x / 512;
      // dimGrid.x = 8
      sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(padded, padded);
      // result = 8 elements
    }  
    
    if(dimGrid.x != 1) {
      dimBlock.x = dimGrid.x;
      dimGrid.x = 1;
      dimBlock.y = dimBlock.z = 1;
      sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(padded,padded);
    }  
    
    
    
    cudaMemcpy(kinEnergy_h, padded, sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    
    // integrate kA to find magnetic energy
    cudaMemcpy(padded,kA,sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
    zeroPadded<<<dimGrid,dimBlock>>>(padded);
    
    dimBlock.x=dimBlock.y=dimBlock.z=8;
      //dim3 dimGrid(Nx/dimBlock.x+1,Ny/dimBlock.y+1,1);
    gridx = (Nx*Ny*Nz)/512;
    
    if (Nx*Ny*Nz < 512) {
      dimBlock.x = Nx;
      dimBlock.y = Ny;
      dimBlock.z = Nz;
      gridx = 1;
    }  
    dimGrid.x=gridx;
    dimGrid.y=dimGrid.z=1;
    
    sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(padded, padded);
    // result = 4096 elements = gridx
    
    while(dimGrid.x > 512) {
      dimGrid.x = dimGrid.x / 512;
      // dimGrid.x = 8
      sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(padded, padded);
      // result = 8 elements
    }  
    
    if(dimGrid.x != 1) {
      dimBlock.x = dimGrid.x;
      dimGrid.x = 1;
      dimBlock.y = dimBlock.z = 1;
      sum<<<dimGrid,dimBlock,sizeof(cufftComplex)*8*8*8>>>(padded,padded);
    } 
    
    cudaMemcpy(magEnergy_h, padded, sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    
    totEnergy_h[0].x = kinEnergy_h[0].x + magEnergy_h[0].x;
    
    cudaFree(padded);
    
}    
     
    
