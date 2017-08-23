#include <Python.h>
#include <numpy/arrayobject.h>
#include "laguerre_transform.h"

LaguerreTransform::LaguerreTransform(Grids* grids) :
  grids_(grids)
{
  L = grids->Nlaguerre - 1;
  J = (3*L-1)/2;
  float *toGrid_h = NULL, *toSpectral_h = NULL;
  // don't need to allocate these since they are allocated in python
  //cudaMallocHost((void**) &toGrid_h, sizeof(float)*(L+1)*(J+1));
  //cudaMallocHost((void**) &toSpectral_h, sizeof(float)*(L+1)*(J+1));

  cudaMalloc((void**) &toGrid, sizeof(float)*(L+1)*(J+1));
  cudaMalloc((void**) &toSpectral, sizeof(float)*(L+1)*(J+1));

  initTransforms(toGrid_h, toSpectral_h);
  cudaMemcpy(toGrid, toGrid_h, sizeof(float)*(L+1)*(J+1), cudaMemcpyHostToDevice);
  cudaMemcpy(toSpectral, toSpectral_h, sizeof(float)*(L+1)*(J+1), cudaMemcpyHostToDevice);

  cublasCreate(&handle);
  cudaFreeHost(toGrid_h);
  cudaFreeHost(toSpectral_h);
}

LaguerreTransform::~LaguerreTransform()
{
  cudaFree(toGrid);
  cudaFree(toSpectral);
}

// toGrid = toGrid[l + (L+1)*j] = Psi^l(x_j)
// toSpectral = toSpectral[j + (J+1)*l] = w_j Psi_l(x_j)
int LaguerreTransform::initTransforms(float* toGrid, float* toSpectral)
{
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pReturn;
    PyObject *toGrid_py, *toSpectral_py;

    const char* filename = "laguerre_quadrature";
    const char* funcname = "laguerre_quadrature";

    Py_Initialize();
    //PyRun_SimpleString("import sys");
    //PyRun_SimpleString("sys.path.append(\".\")");
    pName = PyString_FromString(filename);

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, funcname);
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(1);
            PyTuple_SetItem(pArgs, 0, PyInt_FromLong(L));

            pReturn = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pReturn != NULL && PyTuple_Check(pReturn)) {

		toGrid_py = PyTuple_GetItem(pReturn, 0);
		toSpectral_py = PyTuple_GetItem(pReturn, 1);

                toGrid = (float*) PyArray_DATA(toGrid_py);
                toSpectral = (float*) PyArray_DATA(toSpectral_py);

                Py_DECREF(toSpectral_py);
                Py_DECREF(toGrid_py);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
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
    Py_Finalize();
    return 0;
}

int LaguerreTransform::transformToGrid(Moments* m)
{
//  return cublasCgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//     grids_->NxNyNz, J+1, L+1, 1.,
//     m->ghl, grids_->NxNyNz, grids_->NxNyNz*(L+1),
//     toGrid, J+1, 0,
//     0., m->ghl, grids_->NxNyNz, grids_->NxNyNz*(L+1), 
//     grids_->Nhermite);
  return 0;
}

int LaguerreTransform::transformToSpectral(Moments* m)
{
  return 0;
}
