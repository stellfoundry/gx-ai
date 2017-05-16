void sumReduc_nopad(cufftComplex* result, cufftComplex* f, int Nx, int Ny, int Nz) 
{
  
  dim3 dimBlock2(8,8,8);          // block size is 8*8*8=512, so that all of each block fits in shared memory  
  int gridx = (Nx*Ny*Nz)/512;     // gridx is the number of blocks configured
    
  if (Nx*Ny*Nz <= 512) {
    dimBlock2.x = Nx;
    dimBlock2.y = Ny;
    dimBlock2.z = Nz;
    gridx = 1;
  }  
    
  dim3 dimGrid2(gridx,1,1);
    
  sum <<<dimGrid2, dimBlock2, sizeof(cufftComplex)*8*8*8>>> (f, f);
    
  while(dimGrid2.x > 512) {
    dimGrid2.x = dimGrid2.x / 512;
    sum <<<dimGrid2, dimBlock2, sizeof(cufftComplex)*8*8*8>>> (f, f);
  }  
    
  dimBlock2.x = dimGrid2.x;
  dimGrid2.x = 1;
  dimBlock2.y = dimBlock2.z = 1;
  sum <<<dimGrid2, dimBlock2, sizeof(cufftComplex)*8*8*8>>> (f,f);  
    
  cudaMemcpy(result, f, sizeof(cufftComplex), cudaMemcpyDeviceToHost);

}    
