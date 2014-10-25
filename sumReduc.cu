template <class T>
inline T sumReduc (T *data, unsigned int nn, bool overwrite)
{
  unsigned int size;
  int threads = min (nn, maxThreads);
  int blocks = (nn+threads-1)/threads;
  T *idata, *odata;
  T sum;
  //int sum, *tmp;

  if (!overwrite) cudaMalloc( (void**)&idata, nn*sizeof(T) );
  cudaMalloc( (void**)&odata, nn*sizeof(T) );
  getNumBlocksAndThreads (nn, blocks, threads);
  //printf ("Total Sum, before first reduc: size= %d, blocks= %d, threads= %d\n", nn, blocks, threads);
  reduce_wrapper<T> (nn, threads, blocks, data, odata);

  size = blocks;
  while (size > 1) {
    getNumBlocksAndThreads (size, blocks, threads);
    //printf ("Total Sum: size= %d, blocks= %d, threads= %d\n", size, blocks, threads);
    if (overwrite) {
      // swap device pointers
      //tmp=data; data=odata; odata=tmp;
      swapargs (data, odata);
      reduce_wrapper<T> (size, threads, blocks, data, odata);
    } else {
      //tmp=idata; idata=odata; odata=tmp;
      swapargs (idata, odata);
      reduce_wrapper<T> (size, threads, blocks, idata, odata);
    }
    if (scheme < 3) size = (size + threads - 1) / threads;
    else            size = (size + (threads*2-1)) / (threads*2);
  }

  
  cudaMemcpy( &sum, odata, sizeof(T), cudaMemcpyDeviceToHost );
  cudaFree (odata);
  if (!overwrite) cudaFree (idata);
  return sum;
  
  
}

template <class T>
inline void sumReduc_Partial (T *sum, T *data, unsigned int nn, unsigned int outsize, bool overwrite)
{
  unsigned int size;
  int threads = min (nn, maxThreads);
  int blocks = (nn+threads-1)/threads;
  T *idata, *odata;
  //int sum, *tmp;
  
  
  if (!overwrite) cudaMalloc( (void**)&idata, nn*sizeof(T) );
  cudaMalloc( (void**)&odata, nn*sizeof(T) );
  
  
  getNumBlocksAndThreads_partial (nn, blocks, threads);
  //printf ("Partial Sum, before first reduc: size= %d, blocks= %d, threads= %d\n", nn, blocks, threads);
  if(blocks<outsize) {
    //printf("blocks<outsize. changing to blocks=outsize, recalculating threads\n");
    blocks=outsize;
    getThreads_partial(nn,blocks,threads);
    //printf ("Partial Sum, before first reduc: size= %d, blocks= %d, threads= %d\n", nn, blocks, threads);
  }
  reduce_wrapper_partial<T> (nn, threads, blocks, data, odata);

  size = blocks;
  while (size > outsize) {
    getNumBlocksAndThreads_partial (size, blocks, threads);
    if(blocks<outsize) {
      blocks=outsize;
      getThreads_partial(size,blocks,threads);
    }
    //printf ("Partial Sum: size= %d, blocks= %d, threads= %d\n", size, blocks, threads);
    if (overwrite) {
      // swap device pointers
      //tmp=data; data=odata; odata=tmp;
      swapargs (data, odata);
      reduce_wrapper_partial<T> (size, threads, blocks, data, odata);
    } else {
      //tmp=idata; idata=odata; odata=tmp;
      swapargs (idata, odata);
      reduce_wrapper_partial<T> (size, threads, blocks, idata, odata);
    }
    size = (size + threads - 1) / threads;
  }

  
  cudaMemcpy(sum, odata, sizeof(T)*outsize, cudaMemcpyDeviceToDevice );
  cudaFree (odata);
  if (!overwrite) cudaFree (idata);
  
  
}

inline void sumReduc_Partial_complex (cuComplex *sum, cuComplex *data, unsigned int nn, unsigned int outsize, bool overwrite)
{
  unsigned int size;
  int threads = min (nn, maxThreads);
  int blocks = (nn+threads-1)/threads;
  cuComplex *idata, *odata;
  //int sum, *tmp;
  
  
  if (!overwrite) cudaMalloc( (void**)&idata, nn*sizeof(cuComplex) );
  cudaMalloc( (void**)&odata, nn*sizeof(cuComplex) );
  
  
  getNumBlocksAndThreads_partial (nn, blocks, threads);
  //printf ("Partial Sum, before first reduc: size= %d, blocks= %d, threads= %d\n", nn, blocks, threads);
  if(blocks<outsize) {
    //printf("blocks<outsize. changing to blocks=outsize, recalculating threads\n");
    blocks=outsize;
    getThreads_partial(nn,blocks,threads);
    //printf ("Partial Sum, before first reduc: size= %d, blocks= %d, threads= %d\n", nn, blocks, threads);
  }
  reduce_wrapper_partial_complex (nn, threads, blocks, data, odata);

  size = blocks;
  while (size > outsize) {
    getNumBlocksAndThreads_partial (size, blocks, threads);
    if(blocks<outsize) {
      blocks=outsize;
      getThreads_partial(size,blocks,threads);
    }
    //printf ("Partial Sum: size= %d, blocks= %d, threads= %d\n", size, blocks, threads);
    if (overwrite) {
      // swap device pointers
      //tmp=data; data=odata; odata=tmp;
      swapargs (data, odata);
      reduce_wrapper_partial_complex (size, threads, blocks, data, odata);
    } else {
      //tmp=idata; idata=odata; odata=tmp;
      swapargs (idata, odata);
      reduce_wrapper_partial_complex (size, threads, blocks, idata, odata);
    }
    size = (size + threads - 1) / threads;
  }

  
  cudaMemcpy(sum, odata, sizeof(cuComplex)*outsize, cudaMemcpyDeviceToDevice );
  cudaFree (odata);
  if (!overwrite) cudaFree (idata);
  
  
}

/* void sumReduc(cufftComplex* result_h, cufftComplex* f, cufftComplex* padded, int Nx, int Ny, int Nz) 
{    
    cudaMemcpy(padded,f,sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToDevice);
    
    cleanPadded<<<dimGrid,dimBlock>>>(padded,Nx,Ny,Nz);
    
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
    
    cudaMemcpy(result_h, padded, sizeof(cufftComplex), cudaMemcpyDeviceToHost);

}    

void sumReduc(float* result_h, float* f, float* padded, int Nx, int Ny, int Nz) 
{    
    cudaMemcpy(padded,f,sizeof(float)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToDevice);
    
    cleanPadded<<<dimGrid,dimBlock>>>(padded,Nx,Ny,Nz);
    
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
    
    sum<<<dimGridReduc,dimBlockReduc,sizeof(float)*512>>>(padded, padded);
    
    while(dimGridReduc.x > 512) {
      dimGridReduc.x = dimGridReduc.x / 512;
      sum<<<dimGridReduc,dimBlockReduc,sizeof(cufftComplex)*512>>>(padded, padded);
    }  
    
    dimBlockReduc.x = dimGridReduc.x;
    dimGridReduc.x = 1;
    dimBlockReduc.y = dimBlockReduc.z = 1;
    sum<<<dimGridReduc,dimBlockReduc,sizeof(cufftComplex)*512>>>(padded,padded);  
    
    cudaMemcpy(result_h, padded, sizeof(cufftComplex), cudaMemcpyDeviceToHost);

}    
 */
