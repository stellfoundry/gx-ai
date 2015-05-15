template <class T>
T maxReduc (T *data, unsigned int nn, T *idata, T* odata)
{
  unsigned int size;
  int threads = min (nn, maxThreads);
  int blocks = (nn+threads-1)/threads;
  //T *idata, *odata;
  T max;
  //int sum, *tmp;

  //if (!overwrite) cudaMalloc( (void**)&idata, nn*sizeof(T) );
  //cudaMalloc( (void**)&odata, nn*sizeof(T) );
  getNumBlocksAndThreads (nn, blocks, threads);
  MAXreduce_wrapper<T> (nn, threads, blocks, data, odata);

  size = blocks;
  while (size > 1) {
    getNumBlocksAndThreads (size, blocks, threads);
    //printf ("size= %d, blocks= %d, threads= %d\n", size, blocks, threads);
    //if (overwrite) {
    //  // swap device pointers
    //  //tmp=data; data=odata; odata=tmp;
    //  swapargs (data, odata);
    //  MAXreduce_wrapper<T> (size, threads, blocks, data, odata);
    //} else {
      //tmp=idata; idata=odata; odata=tmp;
      swapargs (idata, odata);
      MAXreduce_wrapper<T> (size, threads, blocks, idata, odata);
    //}
    if (scheme < 3) size = (size + threads - 1) / threads;
    else            size = (size + (threads*2-1)) / (threads*2);
  }

  
  cudaMemcpy( &max, odata, sizeof(T), cudaMemcpyDeviceToHost );
  //cudaFree (odata);
  //if (!overwrite) cudaFree (idata);
  return max;
  
  
}

template <class T>
T maxReduc (T *data, unsigned int nn, bool overwrite)
{
  unsigned int size;
  int threads = min (nn, maxThreads);
  int blocks = (nn+threads-1)/threads;
  T *idata, *odata;
  T max;
  //int sum, *tmp;

  if (!overwrite) cudaMalloc( (void**)&idata, nn*sizeof(T) );
  cudaMalloc( (void**)&odata, nn*sizeof(T) );
  getNumBlocksAndThreads (nn, blocks, threads);
  MAXreduce_wrapper<T> (nn, threads, blocks, data, odata);

  size = blocks;
  while (size > 1) {
    getNumBlocksAndThreads (size, blocks, threads);
    //printf ("size= %d, blocks= %d, threads= %d\n", size, blocks, threads);
    if (overwrite) {
      // swap device pointers
      //tmp=data; data=odata; odata=tmp;
      swapargs (data, odata);
      MAXreduce_wrapper<T> (size, threads, blocks, data, odata);
    } else {
      //tmp=idata; idata=odata; odata=tmp;
      swapargs (idata, odata);
      MAXreduce_wrapper<T> (size, threads, blocks, idata, odata);
    }
    if (scheme < 3) size = (size + threads - 1) / threads;
    else            size = (size + (threads*2-1)) / (threads*2);
  }

  
  cudaMemcpy( &max, odata, sizeof(T), cudaMemcpyDeviceToHost );
  cudaFree (odata);
  if (!overwrite) cudaFree (idata);
  return max;
  
  
}


/*void maxReduc(cuComplex* max, cuComplex* f, cuComplex* padded) 
{    
  cudaMemcpy(padded,f,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToDevice);    
  cleanPadded<<<dimGrid,dimBlock>>>(padded,Nx,Ny,Nz);

  dim3 dimBlock2(8,8,8);
  int gridx = (Nx*Ny*Nz)/512;
    
  if (Nx*Ny*Nz <= 512) {
    dimBlock2.x = Nx;
    dimBlock2.y = Ny;
    dimBlock2.z = Nz;
    gridx = 1;
  }  
    
  dim3 dimGrid2(gridx,1,1);
    
  maximum <<<dimGrid2, dimBlock2, sizeof(cuComplex)*8*8*8>>>(padded, padded);
    
  while(dimGrid2.x > 512) {
    dimGrid2.x = dimGrid2.x / 512;                                                // dimGrid2.x = 8
    maximum <<<dimGrid2, dimBlock2, sizeof(cuComplex)*8*8*8>>>(padded, padded);	  // result = 8 elements
  }  
    
  dimBlock2.x = dimGrid2.x;
  dimGrid2.x = 1;
  dimBlock2.y = dimBlock2.z = 1;
  maximum <<<dimGrid2, dimBlock2, sizeof(cuComplex)*8*8*8>>>(padded,padded);  
    
  cudaMemcpy(max, padded, sizeof(cuComplex), cudaMemcpyDeviceToHost);
}    */
