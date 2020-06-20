template <class T>
T maxReduc (T *data, unsigned int nn, T *idata, T* odata)
{
  unsigned int size;
  int threads = min (nn, maxThreads);
  int blocks = (nn+threads-1)/threads;

  T max;

  getNumBlocksAndThreads (nn, blocks, threads);
  MAXreduce_wrapper<T> (nn, threads, blocks, data, odata);

  size = blocks;
  while (size > 1) {
    getNumBlocksAndThreads (size, blocks, threads);
    swapargs (idata, odata);
    MAXreduce_wrapper<T> (size, threads, blocks, idata, odata);
    
    size = (size + (threads*2-1)) / (threads*2);
  }
  
  cudaMemcpyAsync( &max, odata, sizeof(T), cudaMemcpyDeviceToHost,0 );
  return max;
}

