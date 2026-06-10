#include "grad_parallel.h"
#include <cmath>
#define GGPF <<< dGf, dBf >>> 
#define GGP2 <<< dGd, dBd >>> 
#define GGP <<< dG, dB >>>

GradParallelPeriodic::GradParallelPeriodic(Grids* grids) :
  grids_(grids)
{
  // (ky, kx, theta) <-> (ky, kx, kpar)
  cufftCreate(&zft_plan_forward);
  cufftCreate(&zft_plan_inverse);
  cufftCreate(&dz_plan_forward);
  cufftCreate(&dz_plan_inverse);
  cufftCreate(&dz2_plan_forward);
  cufftCreate(&abs_dz_plan_forward);

  int n = grids_->Nz; 			// size of FFT
  int isize = grids_->NxNycNz;		// size of input data
  int osize = grids_->NxNycNz;		// size of output data
  int dim = 1;				// 1 dimensional
  int istride = grids_->NxNyc;		// distance between two elements in a batch 
					// = distance between (ky,kx,z=1) and (ky,kx,z=2) = Nx*(Ny/2+1)
  int idist = 1;			// idist = distance between first element of consecutive batches 
					// = distance between (ky=1,kx=1,z=1) and (ky=2,kx=1,z=1) = 1
  int ostride = grids_->NxNyc;
  int odist = 1;
  int batchsize = grids_->NxNyc;	// number of consecutive transforms
  size_t workSize;

  cufftMakePlanMany(zft_plan_forward, dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany(zft_plan_inverse, dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany( dz_plan_forward, dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany( dz_plan_inverse, dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany(dz2_plan_forward, dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);
  cufftMakePlanMany(abs_dz_plan_forward,
 		                      dim, &n, &isize, istride, idist, &osize, ostride, odist, CUFFT_C2C, batchsize, &workSize);

  // set up callback functions
  cudaDeviceSynchronize();

  cufftCallbackStoreC zfts_callbackPtr_h;
  cufftCallbackStoreC i_kz_callbackPtr_h;
  cufftCallbackStoreC mkz2_callbackPtr_h;
  cufftCallbackStoreC abs_kz_callbackPtr_h;
  checkCuda(cudaMemcpyFromSymbol(&zfts_callbackPtr_h,   GPU_SYMBOL(zfts_callbackPtr),   sizeof(zfts_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&i_kz_callbackPtr_h,   GPU_SYMBOL(i_kz_callbackPtr),   sizeof(i_kz_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&mkz2_callbackPtr_h,   GPU_SYMBOL(mkz2_callbackPtr),   sizeof(mkz2_callbackPtr_h)));
  checkCuda(cudaMemcpyFromSymbol(&abs_kz_callbackPtr_h, GPU_SYMBOL(abs_kz_callbackPtr), sizeof(abs_kz_callbackPtr_h)));

  checkCuda(cufftXtSetCallback(   zft_plan_forward, (void**)   &zfts_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz));
  checkCuda(cufftXtSetCallback(    dz_plan_forward, (void**)   &i_kz_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kzp));
  checkCuda(cufftXtSetCallback(abs_dz_plan_forward, (void**) &abs_kz_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kzp));
  checkCuda(cufftXtSetCallback(   dz2_plan_forward, (void**)   &mkz2_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kzp));
  cudaDeviceSynchronize();

  int nn1, nt1, nb1, nn2, nt2, nb2, nn3, nt3, nb3;

  nn1 = grids_->NxNyc;                          nt1 = min(nn1, 32);         nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nz;                             nt2 = min(nn2, 32);         nb2 = 1 + (nn2-1)/nt2;
  nn3 = grids_->Nmoms;                          nt3 = min(nn3, 1);          nb3 = 1 + (nn3-1)/nt3;

  dBd = dim3(nt1, nt2, nt3);
  dGd = dim3(nb1, nb2, nb3);
  
  nn1 = grids_->NxNyc;                          nt1 = min(nn1, 32);         nb1 = 1 + (nn1-1)/nt1;
  nn2 = grids_->Nz;                             nt2 = min(nn2, 32);         nb2 = 1 + (nn2-1)/nt2;

  dBf = dim3(nt1, nt2, 1);
  dGf = dim3(nb1, nb2, 1);
  
}

GradParallelPeriodic::~GradParallelPeriodic() {
  cufftDestroy(zft_plan_forward);
  cufftDestroy(zft_plan_inverse);
  cufftDestroy( dz_plan_forward);
  cufftDestroy( dz_plan_inverse);
  cufftDestroy( dz2_plan_forward);
  cufftDestroy(abs_dz_plan_forward);
}

// Dealias in kz
void GradParallelPeriodic::dealias(MomentsG* G)
{
  for (int i = 0; i < grids_->Nmoms; i++) cufftExecC2C(zft_plan_forward, G->G(i), G->G(i), CUFFT_FORWARD);
  kz_dealias GGP2 (G->G(), grids_->kzm, grids_->Nmoms);
  for (int i = 0; i < grids_->Nmoms; i++) cufftExecC2C(zft_plan_inverse, G->G(i), G->G(i), CUFFT_INVERSE);  
}

// Dealias in kz
void GradParallelPeriodic::dealias(cuComplex* f)
{
  cufftExecC2C(zft_plan_forward, f, f, CUFFT_FORWARD);
  int one  = 1;
  kz_dealias GGPF (f, grids_->kzm, one);
  cufftExecC2C(zft_plan_inverse, f, f, CUFFT_INVERSE);  
}

// Fourier transform all moments 
void GradParallelPeriodic::zft(MomentsG* G)
{
  // for now, loop over all l and m because cannot batch 
  for(int i = 0; i < grids_->Nmoms; i++) cufftExecC2C(zft_plan_forward, G->G(i), G->G(i), CUFFT_FORWARD);
}

void GradParallelPeriodic::zft_inverse(MomentsG* G)
{
  // for now, loop over all l and m because cannot batch 
  for(int i = 0; i < grids_->Nmoms; i++) cufftExecC2C(zft_plan_inverse, G->G(i), G->G(i), CUFFT_INVERSE);
}

// Fourier transform for a single moment
void GradParallelPeriodic::zft(cuComplex* mom, cuComplex* res)
{
  cufftExecC2C(zft_plan_forward, mom, res, CUFFT_FORWARD);
}
/*
// inverse Fourier transform for a single moment
void GradParallelPeriodic::zft_inverse(cuComplex* mom, cuComplex* res)
{
  cufftExecC2C(zft_plan_inverse, mom, res, CUFFT_INVERSE);
}
*/

// FFT and derivative for all moments
void GradParallelPeriodic::dz(MomentsG* G, MomentsG* res, bool accumulate)
{
  // FFT and derivative on parallel term
  // i*kz*G calculated via callback, defined as part of dz_plan_forward
  // for now, loop over all l and m because cannot batch 
  // eventually will optimize by first transposing so that z is fastest index

  for(int i = 0; i < grids_->Nmoms; i++) {
    // forward FFT (z -> kz) & multiply by i kz (via callback)
    cufftExecC2C(dz_plan_forward, G->G(i), G->G(i), CUFFT_FORWARD);

    // backward FFT (kz -> z)
    cufftExecC2C(dz_plan_inverse, G->G(i), G->G(i), CUFFT_INVERSE);
  }
}

// FFT and two derivatives for all moments
void GradParallelPeriodic::dz2(MomentsG* G)
{
  // FFT and second derivative on parallel term
  // -kz*kz*G calculated via callback, defined as part of dz2_plan_forward
  // for now, loop over all l and m because cannot batch 
  // eventually will optimize by first transposing so that z is fastest index

  for(int i = 0; i < grids_->Nmoms; i++) {
    // forward FFT (z -> kz) & multiply by -kz**2 (via callback)
    cufftExecC2C(dz2_plan_forward, G->G(i), G->G(i), CUFFT_FORWARD);

    // backward FFT (kz -> z)
    cufftExecC2C(dz_plan_inverse, G->G(i), G->G(i), CUFFT_INVERSE);
  }
}

// FFT and two derivatives for a single moment
void GradParallelPeriodic::dz2(cuComplex* mom, cuComplex* res)
{
  cufftExecC2C(dz2_plan_forward, mom, res, CUFFT_FORWARD);
  cufftExecC2C(dz_plan_inverse, res, res, CUFFT_INVERSE);
}

// FFT and derivative for a single moment
void GradParallelPeriodic::dz(cuComplex* mom, cuComplex* res, bool accumulate)
{
  cufftExecC2C(dz_plan_forward, mom, res, CUFFT_FORWARD);
  cufftExecC2C(dz_plan_inverse, res, res, CUFFT_INVERSE);
}

// FFT and |kz| operator for a single moment
void GradParallelPeriodic::abs_dz(cuComplex* mom, cuComplex* res, bool accumulate)
{
  cufftExecC2C(abs_dz_plan_forward, mom, res, CUFFT_FORWARD);
  cufftExecC2C(dz_plan_inverse, res, res, CUFFT_INVERSE);
}

// FFT only for a single moment -- deprecated. Should change to zft, dropping dir parameter
void GradParallelPeriodic::fft_only(cuComplex* mom, cuComplex* res, int dir)
{
  // use dz_plan_inverse since it does not multiply by i kz via callback 
  cufftExecC2C(dz_plan_inverse, mom, res, dir);
}

GradParallelLocal::GradParallelLocal(Grids* grids) :
  grids_(grids)
{
  dB = 512;
  dG = 1 + (grids_->NxNycNz-1)/dB.x;
  kpar = 1./((float) grids_->Zp);
  mkpar2 = -kpar/((float) grids_->Zp);
}

void GradParallelLocal::dz(MomentsG *G, MomentsG *res, bool accumulate)
{
  G->scale(make_cuComplex(0.,kpar));
}

void GradParallelLocal::dz2(MomentsG *G)
{
  G->scale(mkpar2);
}

void GradParallelLocal::zft(MomentsG *G) {return;}
void GradParallelLocal::zft(cuComplex* mom, cuComplex* res) {
  scale_singlemom_kernel GGP (res, mom, make_cuComplex(1.,0.));
}
void GradParallelLocal::zft_inverse(MomentsG *G) {return;}
//void GradParallelLocal::zft_inverse(MomentsG *G, cuComplex* res) {return;}

// single moment
void GradParallelLocal::dz(cuComplex* mom, cuComplex* res, bool accumulate) {
  scale_singlemom_kernel GGP (res, mom, make_cuComplex(0.,kpar));
}
// single moment
void GradParallelLocal::abs_dz(cuComplex* mom, cuComplex* res, bool accumulate) {
  scale_singlemom_kernel GGP (res, mom, make_cuComplex(kpar,0.));
}
void GradParallelLocal::dz2(cuComplex* mom, cuComplex* res) {
  scale_singlemom_kernel GGP (res, mom, mkpar2);
}

__global__ void pack_theta_plane(cuComplex* buf, const cuComplex* f, int iz, int nmoms, int nxy, int nz)
{
  int i = get_id1();
  int n = nxy*nmoms;
  if (i < n) {
    int imom = i/nxy;
    int ixy = i - imom*nxy;
    buf[i] = f[ixy + nxy*(iz + nz*imom)];
  }
}

__global__ void theta_fd_kernel(cuComplex* res, const cuComplex* f, const cuComplex* left,
                                const cuComplex* right, float inv_2dz, int nmoms,
                                int nxy, int nz, bool accumulate)
{
  int i = get_id1();
  int n = nxy*nz*nmoms;
  if (i < n) {
    int imom = i/(nxy*nz);
    int rem = i - imom*nxy*nz;
    int iz = rem/nxy;
    int ixy = rem - iz*nxy;

    cuComplex fl = (iz == 0)    ? left[ixy + nxy*imom]  : f[ixy + nxy*((iz-1) + nz*imom)];
    cuComplex fr = (iz == nz-1) ? right[ixy + nxy*imom] : f[ixy + nxy*((iz+1) + nz*imom)];
    cuComplex val = (fr - fl) * inv_2dz;
    res[i] = accumulate ? res[i] + val : val;
  }
}

__global__ void theta_fd2_kernel(cuComplex* res, const cuComplex* f, const cuComplex* left,
                                 const cuComplex* right, float inv_dz2, int nmoms,
                                 int nxy, int nz)
{
  int i = get_id1();
  int n = nxy*nz*nmoms;
  if (i < n) {
    int imom = i/(nxy*nz);
    int rem = i - imom*nxy*nz;
    int iz = rem/nxy;
    int ixy = rem - iz*nxy;

    cuComplex fl = (iz == 0)    ? left[ixy + nxy*imom]  : f[ixy + nxy*((iz-1) + nz*imom)];
    cuComplex fr = (iz == nz-1) ? right[ixy + nxy*imom] : f[ixy + nxy*((iz+1) + nz*imom)];
    res[i] = (fl - 2.0f*f[i] + fr) * inv_dz2;
  }
}

GradParallelThetaFD::GradParallelThetaFD(Parameters* pars, Grids* grids) :
  pars_(pars), grids_(grids), send_left_(nullptr), send_right_(nullptr),
  recv_left_(nullptr), recv_right_(nullptr), tmp_(nullptr)
{
  if(!pars_->boundary_option_periodic) {
    printf("theta decomposition currently supports only periodic parallel boundary conditions.\n");
    printf("Twist-and-shift endpoint communication still needs a dedicated implementation.\n");
    exit(1);
  }

  nxy_ = grids_->NxNyc;
  max_moms_ = max(1, grids_->Nmoms);
  float dz = 2.0f * M_PI * pars_->Zp / grids_->Nz_glob;
  inv_2dz_ = 1.0f/(2.0f*dz);
  inv_dz2_ = 1.0f/(dz*dz);

  size_t halo_size = sizeof(cuComplex)*nxy_*max_moms_;
  cudaMalloc((void**) &send_left_, halo_size);
  cudaMalloc((void**) &send_right_, halo_size);
  cudaMalloc((void**) &recv_left_, halo_size);
  cudaMalloc((void**) &recv_right_, halo_size);
  cudaMalloc((void**) &tmp_, sizeof(cuComplex)*grids_->NxNycNz*max_moms_);
}

GradParallelThetaFD::~GradParallelThetaFD()
{
  if(send_left_) cudaFree(send_left_);
  if(send_right_) cudaFree(send_right_);
  if(recv_left_) cudaFree(recv_left_);
  if(recv_right_) cudaFree(recv_right_);
  if(tmp_) cudaFree(tmp_);
}

void GradParallelThetaFD::exchange(cuComplex* f, int nmoms)
{
  int n = nxy_*nmoms;
  int nt = min(256, n);
  int nb = 1 + (n-1)/nt;

  pack_theta_plane <<<nb, nt>>> (send_left_,  f, 0,            nmoms, nxy_, grids_->Nz);
  pack_theta_plane <<<nb, nt>>> (send_right_, f, grids_->Nz-1, nmoms, nxy_, grids_->Nz);

  if(grids_->nprocs_z == 1) {
    CP_ON_GPU(recv_left_,  send_right_, sizeof(cuComplex)*n);
    CP_ON_GPU(recv_right_, send_left_,  sizeof(cuComplex)*n);
    return;
  }

  int left = (grids_->iproc_z - 1 + grids_->nprocs_z) % grids_->nprocs_z;
  int right = (grids_->iproc_z + 1) % grids_->nprocs_z;
  size_t count = 2*n;

  ncclGroupStart();
  ncclSend(send_left_,  count, ncclFloat, left,  grids_->ncclComm_z, 0);
  ncclRecv(recv_right_, count, ncclFloat, right, grids_->ncclComm_z, 0);
  ncclSend(send_right_, count, ncclFloat, right, grids_->ncclComm_z, 0);
  ncclRecv(recv_left_,  count, ncclFloat, left,  grids_->ncclComm_z, 0);
  ncclGroupEnd();
  cudaStreamSynchronize(0);
}

void GradParallelThetaFD::dz(MomentsG* G, MomentsG* res, bool accumulate)
{
  exchange(G->G(), grids_->Nmoms);
  int n = nxy_*grids_->Nz*grids_->Nmoms;
  int nt = min(256, n);
  int nb = 1 + (n-1)/nt;
  cuComplex* out = (G->G() == res->G()) ? tmp_ : res->G();
  if(out == tmp_ && accumulate) CP_ON_GPU(tmp_, res->G(), sizeof(cuComplex)*n);
  theta_fd_kernel <<<nb, nt>>> (out, G->G(), recv_left_, recv_right_, inv_2dz_, grids_->Nmoms, nxy_, grids_->Nz, accumulate);
  if(out == tmp_) CP_ON_GPU(res->G(), tmp_, sizeof(cuComplex)*n);
}

void GradParallelThetaFD::dz(cuComplex* m, cuComplex* res, bool accumulate)
{
  exchange(m, 1);
  int n = nxy_*grids_->Nz;
  int nt = min(256, n);
  int nb = 1 + (n-1)/nt;
  cuComplex* out = (m == res) ? tmp_ : res;
  if(out == tmp_ && accumulate) CP_ON_GPU(tmp_, res, sizeof(cuComplex)*n);
  theta_fd_kernel <<<nb, nt>>> (out, m, recv_left_, recv_right_, inv_2dz_, 1, nxy_, grids_->Nz, accumulate);
  if(out == tmp_) CP_ON_GPU(res, tmp_, sizeof(cuComplex)*n);
}

void GradParallelThetaFD::dz2(MomentsG* G)
{
  exchange(G->G(), grids_->Nmoms);
  int n = nxy_*grids_->Nz*grids_->Nmoms;
  int nt = min(256, n);
  int nb = 1 + (n-1)/nt;
  theta_fd2_kernel <<<nb, nt>>> (tmp_, G->G(), recv_left_, recv_right_, inv_dz2_, grids_->Nmoms, nxy_, grids_->Nz);
  CP_ON_GPU(G->G(), tmp_, sizeof(cuComplex)*n);
}

void GradParallelThetaFD::dz2(cuComplex* m, cuComplex* res)
{
  exchange(m, 1);
  int n = nxy_*grids_->Nz;
  int nt = min(256, n);
  int nb = 1 + (n-1)/nt;
  cuComplex* out = (m == res) ? tmp_ : res;
  theta_fd2_kernel <<<nb, nt>>> (out, m, recv_left_, recv_right_, inv_dz2_, 1, nxy_, grids_->Nz);
  if(out == tmp_) CP_ON_GPU(res, tmp_, sizeof(cuComplex)*n);
}

void GradParallelThetaFD::zft(cuComplex* m, cuComplex* res)
{
  int n = grids_->NxNycNz;
  int nt = min(256, n);
  int nb = 1 + (n-1)/nt;
  scale_singlemom_kernel <<<nb, nt>>> (res, m, make_cuComplex(1.,0.));
}

GradParallel1D::GradParallel1D(Grids* grids) :
  grids_(grids)
{
  // (theta) <-> (kpar)
  cufftCreate(&dz_plan_forward);
  cufftCreate(&dz_plan_inverse);

  cufftCreate(&dz2_plan_forward);
  
  // MFM: Plan for 1d FFT
  cufftPlan1d(&dz_plan_forward, grids_->Nz, CUFFT_R2C, 1);
  cufftPlan1d(&dz_plan_inverse, grids_->Nz, CUFFT_C2R, 1);

  cufftPlan1d(&dz2_plan_forward, grids_->Nz, CUFFT_R2C, 1);

  cudaDeviceSynchronize();
  cufftCallbackStoreC i_kz_1d_callbackPtr_h;
  checkCuda(cudaMemcpyFromSymbol(&i_kz_1d_callbackPtr_h, GPU_SYMBOL(i_kz_1d_callbackPtr), sizeof(i_kz_1d_callbackPtr_h)));
  checkCuda(cufftXtSetCallback(dz_plan_forward, (void**) &i_kz_1d_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz));
  cudaDeviceSynchronize();

  cufftCallbackStoreC mkz2_1d_callbackPtr_h;
  checkCuda(cudaMemcpyFromSymbol(&mkz2_1d_callbackPtr_h, GPU_SYMBOL(mkz2_1d_callbackPtr), sizeof(mkz2_1d_callbackPtr_h)));
  checkCuda(cufftXtSetCallback(dz2_plan_forward, (void**) &mkz2_1d_callbackPtr_h, CUFFT_CB_ST_COMPLEX, (void**)&grids_->kz));
  cudaDeviceSynchronize();

  cudaMalloc((void**) &b_complex, sizeof(cuComplex)*(grids_->Nz/2+1));
}

GradParallel1D::~GradParallel1D() {
  cufftDestroy(dz_plan_forward);
  cufftDestroy(dz_plan_inverse);
  cufftDestroy(dz2_plan_forward);
  cudaFree(b_complex);
}

void GradParallel1D::dz1D(float* b)  // even tho cuda 11+ overwrites inputs, this is ok
{
  checkCuda(cufftExecR2C(dz_plan_forward, b, b_complex)); 
  checkCuda(cufftExecC2R(dz_plan_inverse, b_complex, b));
}
