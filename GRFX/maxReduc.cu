void maxReduc(cufftComplex* max, cufftComplex* f, cufftComplex* padded) 
{

    cudaMemcpy(padded,f,sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToDevice);
    
    cleanPadded<<<dimGrid,dimBlock>>>(padded);
    
    dim3 dimBlockReduc(8,8,8);
    //dimBlockReduc.x=dimBlockReduc.y=dimBlockReduc.z=8;
      //dim3 dimGridReduc(Nx/dimBlockReduc.x+1,Ny/dimBlockReduc.y+1,1);
    int gridx = (Nx*Ny*Nz)/512;
    
    if (Nx*Ny*Nz <= 512) {
      dimBlockReduc.x = Nx;
      dimBlockReduc.y = Ny;
      dimBlockReduc.z = Nz;
      gridx = 1;
    }  
    
    dim3 dimGridReduc(gridx,1,1);
    //dimGridReduc.x=gridx;
    //dimGridReduc.y=dimGridReduc.z=1;
    
    maximum<<<dimGridReduc,dimBlockReduc,sizeof(cufftComplex)*8*8*8>>>(padded, padded);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      // dimGridReduc.x = 8
      maximum<<<dimGridReduc,dimBlockReduc,sizeof(cufftComplex)*8*8*8>>>(padded, padded);
      // result = 8 elements
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    maximum<<<dimGridReduc,dimBlockReduc,sizeof(cufftComplex)*8*8*8>>>(padded,padded);  
    
    cudaMemcpy(max, padded, sizeof(cufftComplex), cudaMemcpyDeviceToHost);

}    
