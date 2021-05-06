#include "nca.h"

nca::nca(int N, int Nwrite) :
  N_(N), Nwrite_(Nwrite)
{
  data = nullptr;  cpu = nullptr;  tmp = nullptr; z_tmp = nullptr;
  write = false;
  write_v_time = false;
  xydata = false;
  time_start[0] = 0;
  time_start[1] = 0;
  time_start[2] = 0;
  time_start[3] = 0;
  time_start[4] = 0;
  time_start[5] = 0;

  time_count[0] = 1;

  if (N == 0) return;
  
  if (N > 0) {
    cudaMalloc (&data, sizeof(float) * N);
    if (Nwrite > 0) {
      cudaMallocHost  (&tmp,  sizeof(float) * N);
      cudaMallocHost  (&cpu,  sizeof(float) * Nwrite);
    } else {
      cudaMallocHost  (&cpu,  sizeof(float) * N);      
    }
  } else {
    N = -N;
    if (Nwrite > 0) {
      cudaMallocHost (&z_tmp,  sizeof(cuComplex) * N);
      cudaMallocHost (&cpu,    sizeof(float)     * Nwrite);
    }
  }  
}
nca::~nca() {
  if (data)  cudaFree     ( data   );
  if (tmp)   cudaFreeHost ( tmp    );
  if (cpu)   cudaFreeHost ( cpu    );
  if (z_tmp) cudaFreeHost ( z_tmp  );
}
void nca::increment_ts(void) {time_start[0] += 1;}
