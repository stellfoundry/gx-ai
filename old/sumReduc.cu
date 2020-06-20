// NEW: pass in idata and odata instead of alloc/dealloc them. 
// idata and odata must be temp arrays of type T and size nn!
// *** using same array for idata and odata does NOT appear to affect precision ***
template <class T>
inline T sumReduc (T *data, unsigned int nn, T *idata, T* odata)
{
  unsigned int size;
  int threads = min (nn, maxThreads);
  int blocks = (nn+threads-1)/threads;

  T sum;

  getNumBlocksAndThreads (nn, blocks, threads);
  reduce_wrapper<T> (nn, threads, blocks, data, odata);

  size = blocks;
  while (size > 1) {
    getNumBlocksAndThreads (size, blocks, threads);
    swapargs (idata, odata);
    reduce_wrapper<T> (size, threads, blocks, idata, odata);

    if (scheme < 3) size = (size + threads - 1) / threads;
    else            size = (size + (threads*2-1)) / (threads*2);
  }

  cudaMemcpy( &sum, odata, sizeof(T), cudaMemcpyDeviceToHost);
  return sum;  
}



// sums over innermost index; e.g. can do f(z) = sum_x F(x,z), or f(x,z) = sum_y F(y,x,z)
// cannot be used for other way around... cannot do f(x) = sum_z F(x,z) ! in this case, use a kernel with an explicit looped sum, e.g. sumZ
// NEW: pass in idata and odata instead of alloc/dealloc them. 
// idata and odata must be temp arrays of type T and size nn!
// *** using same array for idata and odata does not appear to affect precision ***
template <class T>
inline void sumReduc_Partial (T *sum, T *data, unsigned int nn, unsigned int outsize, T* idata, T* odata)
{
  unsigned int size;
  int threads = min (nn, maxThreads);
  int blocks = (nn+threads-1)/threads;
  
  getNumBlocksAndThreads_partial (nn, blocks, threads);
  if(blocks<outsize) {
    blocks=outsize;
    getThreads_partial(nn,blocks,threads);
  }
  reduce_wrapper_partial<T> (nn, threads, blocks, data, odata);

  size = blocks;
  while (size > outsize) {
    getNumBlocksAndThreads_partial (size, blocks, threads);
    if(blocks<outsize) {
      blocks=outsize;
      getThreads_partial(size,blocks,threads);
    }
    swapargs (idata, odata);
    reduce_wrapper_partial<T> (size, threads, blocks, idata, odata);
    size = (size + threads - 1) / threads;
  }
  
  cudaMemcpy(sum, odata, sizeof(T)*outsize, cudaMemcpyDeviceToDevice );
}

