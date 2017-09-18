#include <Python.h>
#include <numpy/arrayobject.h>
#include "laguerre_transform.h"

LaguerreTransform::LaguerreTransform(Grids* grids, int batch_size) :
  grids_(grids), L(grids->Nl-1), J((3*L-1)/2), batch_size_(batch_size)
{
  float *toGrid_h, *toSpectral_h, *roots_h;
  cudaMallocHost((void**) &toGrid_h, sizeof(float)*(L+1)*(J+1));
  cudaMallocHost((void**) &toSpectral_h, sizeof(float)*(L+1)*(J+1));
  cudaMallocHost((void**) &roots_h, sizeof(float)*(J+1));

  cudaMalloc((void**) &toGrid, sizeof(float)*(L+1)*(J+1));
  cudaMalloc((void**) &toSpectral, sizeof(float)*(L+1)*(J+1));
  cudaMalloc((void**) &roots, sizeof(float)*(J+1));

  initTransforms(toGrid_h, toSpectral_h, roots_h);
  cudaMemcpy(toGrid, toGrid_h, sizeof(float)*(L+1)*(J+1), cudaMemcpyHostToDevice);
  cudaMemcpy(toSpectral, toSpectral_h, sizeof(float)*(L+1)*(J+1), cudaMemcpyHostToDevice);
  cudaMemcpy(roots, roots_h, sizeof(float)*(J+1), cudaMemcpyHostToDevice);

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

// toGrid = toGrid[l + (L+1)*j] = Psi^l(x_j)
// toSpectral = toSpectral[j + (J+1)*l] = w_j Psi_l(x_j)
int LaguerreTransform::initTransforms(float* toGrid, float* toSpectral, float* roots)
{
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pReturn;
    PyObject *toGrid_py, *toSpectral_py, *roots_py;

    const char* filename = "laguerre_quadrature";
    const char* funcname = "laguerre_quadrature";

    Py_Initialize();
    pName = PyString_FromString(filename);

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, funcname);

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(1);
            PyTuple_SetItem(pArgs, 0, PyInt_FromLong(L));

            pReturn = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pReturn != NULL && PyTuple_Check(pReturn)) {

		toGrid_py = PyTuple_GetItem(pReturn, 0);
		toSpectral_py = PyTuple_GetItem(pReturn, 1);
                roots_py = PyTuple_GetItem(pReturn, 2);

                memcpy(toGrid, (float*) PyArray_DATA(toGrid_py), sizeof(float)*(J+1)*(L+1));
                memcpy(toSpectral, (float*) PyArray_DATA(toSpectral_py), sizeof(float)*(J+1)*(L+1));
                memcpy(roots, (float*) PyArray_DATA(roots_py), sizeof(float)*(J+1));

                Py_DECREF(toSpectral_py);
                Py_DECREF(toGrid_py);
                Py_DECREF(roots_py);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
            // this causes segfaults...
            //Py_DECREF(pReturn);
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"%s\"\n", funcname);
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", filename);
        return 1;
    }
    // also causes segfaults...
    //Py_Finalize();
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
  return cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
     m, n, k, &alpha,
     g_in, lda, strideA,
     toSpectral, ldb, strideB,
     &beta, G_res, ldc, strideC,
     batch_size_);
}
