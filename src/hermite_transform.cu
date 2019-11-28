#include "hermite_transform.h"

HermiteTransform::HermiteTransform(Grids* grids, int batch_size) :
  grids_(grids), M(grids->Nm), batch_size_(batch_size)
{
  float *toGrid_h, *toSpectral_h, *roots_h;
  cudaMallocHost((void**) &toGrid_h,     sizeof(float)*M*M);
  cudaMallocHost((void**) &toSpectral_h, sizeof(float)*M*M);
  cudaMallocHost((void**) &roots_h,      sizeof(float)*M);

  cudaMalloc((void**) &toGrid,     sizeof(float)*M*M);
  cudaMalloc((void**) &toSpectral, sizeof(float)*M*M);
  cudaMalloc((void**) &roots,      sizeof(float)*M);

  initTransforms(toGrid_h, toSpectral_h, roots_h);
  CP_TO_GPU(toGrid,     toGrid_h,     sizeof(float)*M*M);
  CP_TO_GPU(toSpectral, toSpectral_h, sizeof(float)*M*M);
  CP_TO_GPU(roots,      roots_h,      sizeof(float)*M);


  cublasCreate(&handle);
  cudaFreeHost(toGrid_h);
  cudaFreeHost(toSpectral_h);
  cudaFreeHost(roots_h);
}

HermiteTransform::~HermiteTransform()
{
  cudaFree(toGrid);
  cudaFree(toSpectral);
  cudaFree(roots);
}

int HermiteTransform::initTransforms(float* toGrid_h, float* toSpectral_h, float* roots_h)
{
  int i, j;
  int Msq = M*M;
  double Jacobi[Msq];
  double stmp;
  double *P_tmp = (double *) calloc(Msq, sizeof(double));
  
  for (j=0; j<Msq; j++) Jacobi[j] = 0.0;

  for (i = 0; i < M; i ++) {
    stmp = sqrt(double (i+1));
    Jacobi[1 + i * (M+1)] = Jacobi[M + i * (M+1)] = stmp;
  } 

  gsl_matrix_view m = gsl_matrix_view_array (Jacobi, M, M); // defined type gsl_matrix_view
  gsl_vector *eval = gsl_vector_alloc (M);
  gsl_matrix *evec = gsl_matrix_alloc (M, M);
  gsl_eigen_symmv_workspace *wrk = gsl_eigen_symmv_alloc (M);
  gsl_eigen_symmv (&m.matrix, eval, evec, wrk);                 // & returns address for pointer
  gsl_eigen_symmv_free (wrk);
  gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

  // eval: eigenvalues of Jacobi matrix; roots of the M^th Hermite
  // evec: Hermite cardinal polynomials; use these to get the weights 

  gsl_matrix *poly = gsl_matrix_alloc (M, M);               // pointer to defined type gsl_matrix
  gsl_matrix_set_zero (poly);

  int mm;
  double x_j, wgt, Mmat;

  for (j=0; j<M; j++) {
    x_j = gsl_vector_get (eval, j); 
    //    printf("roots_h[%d]=%g \n",j,x_j);
    roots_h[j] = (float) x_j; 

    wgt = pow (gsl_matrix_get (evec, 0, j), 2); // square first element of j_th eigenvector

    // evaluate the m-th polynomial at x(j) = x_j and multiply by weight(j) as needed
    for (mm=0; mm<M; mm++) {
      Mmat = gsl_sf_hermite_prob(mm, x_j);

      toGrid_h[mm + M*j] = (float) Mmat;
      Mmat = Mmat * wgt;
      toSpectral_h[j + M*mm] = (float) Mmat;
    }
  }
  return 0; 
}

int HermiteTransform::transformToGrid(float* G_in, float* g_res)
{
  int m = grids_->NxNycNz*grids_->Nl;
  int n = M;
  int k = M;
  float alpha = 1.;
  int lda = m;
  int strideA = m*M;
  int ldb = M;
  int strideB = 0;
  float beta = 0.;
  int ldc = m;
  int strideC = m*M;
  return cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
     m, n, k, &alpha,
     G_in, lda, strideA,
     toGrid, ldb, strideB,
     &beta, g_res, ldc, strideC,
     batch_size_); 
}

int HermiteTransform::transformToSpectral(float* g_in, float* G_res)
{
  int m = grids_->NxNycNz*grids_->Nl;
  int n = M;
  int k = M;
  float alpha = 1.;
  int lda = m;
  int strideA = m*M;
  int ldb = M;
  int strideB = 0;
  float beta = 0.;
  int ldc = m;
  int strideC = m*M;

  return cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
     m, n, k, &alpha,
     g_in, lda, strideA,
     toSpectral, ldb, strideB,
     &beta, G_res, ldc, strideC,
     batch_size_); 
}



