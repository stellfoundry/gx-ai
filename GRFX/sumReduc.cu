void sumReduc(cufftComplex* result, cufftComplex* f, cufftComplex* padded) 
{    
    cudaMemcpy(padded,f,sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToDevice);
    
    cleanPadded<<<dimGrid,dimBlock>>>(padded);
    
    //block size is 8*8*8=512, so that all of each block fits in shared memory
    dim3 dimBlockReduc(8,8,8);
    //gridx is the number of blocks configured
    int gridx = (Nx*Ny*Nz)/512;
    
    if (Nx*Ny*Nz <= 512) {
      dimBlockReduc.x = Nx;
      dimBlockReduc.y = Ny;
      dimBlockReduc.z = Nz;
      gridx = 1;
    }  
    
    dim3 dimGridReduc(gridx,1,1);
    
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cufftComplex)*512>>>(padded, padded);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      sum<<<dimGridReduc,dimBlockReduc,sizeof(cufftComplex)*512>>>(padded, padded);
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cufftComplex)*512>>>(padded,padded);  
    
    cudaMemcpy(result, padded, sizeof(cufftComplex), cudaMemcpyDeviceToHost);

}    
