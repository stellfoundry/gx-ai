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
    sumReduc(kinEnergy_h, kPhi, padded);
    
    
    // integrate kA to find magnetic energy
    sumReduc(magEnergy_h, kA, padded);
    
    //calculate total energy
    totEnergy_h[0].x = kinEnergy_h[0].x + magEnergy_h[0].x;
    
    cudaFree(padded);
    
    
}    
     
    
