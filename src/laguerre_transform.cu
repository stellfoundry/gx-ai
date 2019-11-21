#include "laguerre_transform.h"

LaguerreTransform::LaguerreTransform(Grids* grids, int batch_size) :
  grids_(grids), L(grids->Nl-1), J((3*L-1)/2), batch_size_(batch_size)
{
  float *toGrid_h, *toSpectral_h, *roots_h;
  cudaMallocHost((void**) &toGrid_h,     sizeof(float)*(L+1)*(J+1));
  cudaMallocHost((void**) &toSpectral_h, sizeof(float)*(L+1)*(J+1));
  cudaMallocHost((void**) &roots_h,      sizeof(float)*(J+1));

  cudaMalloc((void**) &toGrid,     sizeof(float)*(L+1)*(J+1));
  cudaMalloc((void**) &toSpectral, sizeof(float)*(L+1)*(J+1));
  cudaMalloc((void**) &roots,      sizeof(float)*(J+1));

  initTransforms(toGrid_h, toSpectral_h, roots_h);
  CP_TO_GPU(toGrid,     toGrid_h,     sizeof(float)*(L+1)*(J+1));
  CP_TO_GPU(toSpectral, toSpectral_h, sizeof(float)*(L+1)*(J+1));
  CP_TO_GPU(roots,      roots_h,      sizeof(float)*(J+1));

  cublasCreate(&handle);
  cudaFreeHost(toGrid_h);
  cudaFreeHost(toSpectral_h);
  cudaFreeHost(roots_h);
}

LaguerreTransform::~LaguerreTransform()
{
  cudaFree(toGrid);
  cudaFree(toSpectral);
  cudaFree(roots);
}

int LaguerreTransform::initTransforms(float* toGrid_h, float* toSpectral_h, float* roots_h)
{
  int i, j;
  int Jsq = (J+1) * (J+1);
  double Jacobi[Jsq];

  for (j=0; j<Jsq; j++) Jacobi[j] = 0.0;

  for (i = 0; i < J+1; i ++) {
    Jacobi[i * (J+2)] = 2 * i + 1;
    Jacobi[1 + i * (J+2)] = i + 1;
    Jacobi[J+1 + i * (J+2)] = i + 1;
  } 

  /*
  for (i = 0; i < J+1; i ++) {
    for (j = 0; j < J+1; j ++) {
      printf("%f \t",Jacobi[i*(J+1)+j]);
    } 
    printf("\n");
  }
  */

  gsl_matrix_view m = gsl_matrix_view_array (Jacobi, J+1, J+1); // defined type gsl_matrix_view
  //  gsl_vector *weights = gsl_vector_alloc (J+1);                 // pointer to a structure
  gsl_vector *eval = gsl_vector_alloc (J+1);
  gsl_matrix *evec = gsl_matrix_alloc (J+1, J+1);
  gsl_eigen_symmv_workspace * wrk = gsl_eigen_symmv_alloc (J+1);
  gsl_eigen_symmv (&m.matrix, eval, evec, wrk);                 // & returns address for pointer
  gsl_eigen_symmv_free (wrk);
  gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_ABS_ASC);

  // eval: eigenvalues of Jacobi matrix; roots of the (J+1)th Laguerre
  // evec: Laguerre cardinal polynomials; use these to get the weights 

  gsl_matrix *poly = gsl_matrix_alloc (J+1, J+1);               // pointer to defined type gsl_matrix
  gsl_matrix_set_zero (poly);
  for (i=0; i<J+1; i++) {      
    for (j=0; j<i+1; j++) {
      // i-th polynomial, j-th coefficient
      double tmp = gsl_sf_choose (i, j);      
      //      printf("tmp = %d\n",tmp);
      tmp *= gsl_pow_int(-1.0, j) / gsl_sf_fact(j) * gsl_pow_int(-1.0, i);
      //      printf("tmp=%g \n",tmp);
      gsl_matrix_set(poly, i, j, tmp);
    }
  }
  int ell;
  double x_i, wgt, Lmat;
  gsl_vector_view polyvec;

  for (j=0; j<J+1; j++) {
    x_i = gsl_vector_get (eval, j); 
    //    printf("roots_h[%d]=%g \n",j,x_i);
    roots_h[j] = (float) x_i; // Used in argument of J0
    wgt = pow (gsl_matrix_get (evec, 0, j), 2); // square first element of j_th eigenvector
    //    gsl_vector_set (weights, j, wgt);

    // evaluate the ell-th polynomial at x(j) = x_j and multiply by weight(j) as needed
    for (ell=0; ell<L+1; ell++) {
      polyvec = gsl_matrix_row (poly, ell);
      Lmat = gsl_poly_eval(polyvec.vector.data, ell+1, x_i);

      toGrid_h[ell + (L+1)*j] = (float) Lmat;
      Lmat = Lmat * wgt;
      toSpectral_h[j + (J+1)*ell] = (float) Lmat;
    }	
  }
  return 0; 
}

int LaguerreTransform::transformToGrid(float* G_in, float* g_res)
{
  int m = grids_->NxNyNz;
  int n = J+1;
  int k = L+1;
  float alpha = 1.;
  int lda = grids_->NxNyNz;
  int strideA = grids_->NxNyNz*(L+1);
  int ldb = L+1;
  int strideB = 0;
  float beta = 0.;
  int ldc = grids_->NxNyNz;
  int strideC = grids_->NxNyNz*(J+1);
  return cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
     m, n, k, &alpha,
     G_in, lda, strideA,
     toGrid, ldb, strideB,
     &beta, g_res, ldc, strideC,
     batch_size_); 
}

int LaguerreTransform::transformToSpectral(float* g_in, float* G_res)
{
  int m = grids_->NxNyNz;
  int n = L+1;
  int k = J+1;
  float alpha = 1.;
  int lda = grids_->NxNyNz;
  int strideA = grids_->NxNyNz*(J+1);
  int ldb = J+1;
  int strideB = 0;
  float beta = 0.;
  int ldc = grids_->NxNyNz;
  int strideC = grids_->NxNyNz*(L+1);

  // column-major order, according to the manuals. WTF?
  // So I should transpose everything? 

  // COLUMN MAJOR
  // G(l) = T(l, j) g(j)   (for every k)
  // G(k, l) = g(k, j) T(j, l)  <<<<<< Note that T is transposed...
  // G has leading dimension K and a new block starts K*L away 
  // g has leading dimension K and a new block starts K*J away 
  // T has leading dimension J and a stride of zero 

  // T has to be T[j + (J+1)*l] for this to hold together. 
  // ... and it does; the Python numby flatten trick does that. 

  // C = A B ==> lda=K, strideA = K*J;  ldc=K, strideC=K*L; ldb=J and strideB=0
  // A has K rows so ::::::::: m = K   ////////////// yes
  // B has L columns so :::::: n = L   ////////////// yes b/c L = L+1 for some python reason
  // A has J columns so :::::: k = J   ////////////// yes b/c J = J+1 for some unknown reason

  // lda=K  ///////////////////////// yes
  // strideA=K*(J+1) /////////////// yes
  // ldb=J+1        //////////////// yes
  // strideB=0 ////////////////////// yes
  // ldc=K  ///////////////////////// yes
  // strideC=K*(L+1) //////////////  yes

  return cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
     m, n, k, &alpha,
     g_in, lda, strideA,
     toSpectral, ldb, strideB,
     &beta, G_res, ldc, strideC,
     batch_size_); 
}



