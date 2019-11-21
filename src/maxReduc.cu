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

  
  cudaMemcpyAsync( &max, odata, sizeof(T), cudaMemcpyDeviceToHost,0 );
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

