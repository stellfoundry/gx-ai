#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <unordered_map>
#include <vector>
#include <iterator>
#include "grids.h"

class Red {
public:
  virtual ~Red() {};
  virtual void Sum(float *f, float* res, int i=0) {};
  virtual void Max(float *f, float* res) {}; 
  virtual void Sum(double *f, double* res, int i=0) {};
  virtual void Max(double *f, double* res) {}; 
  virtual void MatVec(double* res, double* A, double* x) {};
  virtual void MatMat(double* res, double* M1, double* M2) {};
  
  void * Addwork;     uint64_t sizeAdd;    uint64_t sizeWork;    
  cudaDataType_t cfloat = CUDA_R_32F;
  cudaDataType_t dfloat = CUDA_R_64F;
  cutensorComputeType_t typeCompute64 = CUTENSOR_R_MIN_64F;
  cutensorComputeType_t typeCompute = CUTENSOR_R_MIN_32F;
  cutensorOperator_t opAdd = CUTENSOR_OP_ADD;
  cutensorOperator_t opMax = CUTENSOR_OP_MAX;
  cutensorHandle_t handle; 
  cutensorContractionFind_t find;
    
  float alpha = 1.0;  double alpha64 = 1.0;
  float beta  = 0.0;  double beta64  = 0.0;

};

class Grid_Species_Reduce : public Red {
 public:
  Grid_Species_Reduce(Grids *grids, std::vector<int> spectra);
  ~Grid_Species_Reduce();
  void Sum(float *f, float* res, int i=0);

 private:
  Grids *grids_;
  std::vector<int> spectra_;

  std::vector<int32_t> initialized;

  cutensorTensorDescriptor_t dP;  std::vector<cutensorTensorDescriptor_t> desc; 
  std::vector<int64_t> extent_P;  std::vector<std::vector<int64_t>> extents; 

  std::unordered_map<int32_t, int64_t> extent;
  std::vector<int32_t> Pmode{'y', 'x', 'z', 's'};

  int32_t nPmode = Pmode.size(); // for integrals of (1-Gamma_0) Phi**2
  std::vector<std::vector<int32_t>> pModes{{'s'},
					   {'x', 's'},
					   {'y', 's'},
					   {'y', 'x', 's'},
					   {'y', 'x', 's'},
					   {'z', 's'},
					   {'z', 's'}};							       
};

class Grid_Reduce : public Red {
 public:
  Grid_Reduce(Grids *grids, std::vector<int> s);
  ~Grid_Reduce();
  void Sum(float *f, float* res, int i=0);

 private:
  Grids *grids_;
  std::vector<int> spectra_;
  std::vector<int32_t> initialized;

  cutensorTensorDescriptor_t dI;  std::vector<cutensorTensorDescriptor_t> desc;
  std::vector<int64_t> extent_I;  std::vector<std::vector<int64_t>> extents; 
  std::unordered_map<int32_t, int64_t> extent;

  std::vector<int32_t> Imode{'y', 'x', 'z'};
  int32_t nImode = Imode.size(); // for integrals of Phi**2
  
  std::vector<std::vector<int32_t>> iModes{{},
					   {'x'},
					   {'y'},
					   {'y', 'x'},
					   {'y', 'x'},
					   {'z'},
					   {'z'}};

};

class All_Reduce : public Red {
 public:
  All_Reduce(Grids *grids, std::vector<int> s);
  ~All_Reduce();
  void Sum(float *f, float* res, int i=0);

 private:
  Grids *grids_;
  std::vector<int> spectra_;
  std::vector<int32_t> initialized;

  cutensorTensorDescriptor_t dW;  std::vector<cutensorTensorDescriptor_t> desc; 
  std::vector<int64_t> extent_W;  std::vector<std::vector<int64_t>> extents; 
  std::unordered_map<int32_t, int64_t> extent;

  std::vector<int32_t> Wmode{'y', 'x', 'z', 'l', 'm', 's'};
  int32_t nWmode = Wmode.size(); // for integrals of g**2, all species, or all (m,s), or all (l,m,s), etc.
  
  std::vector<std::vector<int32_t>> Modes{{'s'},
					  {'x', 's'},
					  {'y', 's'},
					  {'z', 's'},
					  {'l', 's'},
					  {'m', 's'},
					  {'l', 'm', 's'},
					  {'y', 'x', 's'},
					  {'y', 'x', 's'},
					  {'z', 's'},
					  {'z', 's'}};

};

class Block_Reduce : public Red {
 public:
  Block_Reduce(int N);
  ~Block_Reduce();
  void Sum(float *f, float* res, int i=0);
  void Max(float *f, float* res);

 private:
  int N_;    void * Maxwork;    uint64_t sizeMax;   

  bool first_Max = true;
  bool first_Sum = true;
  
  cutensorTensorDescriptor_t dA, dB;
  std::unordered_map<int32_t, int64_t> extent;
  std::vector<int64_t> extent_A, extent_B;

  std::vector<int32_t> Amode{'a'};
  std::vector<int32_t> Bmode{};

  int32_t nAmode = Amode.size(); // for single-block data input (such as for CFL condition)
  int32_t nBmode = Bmode.size(); // scalar output for contiguous data
  
};

class dBlock_Reduce : public Red {
 public:
  dBlock_Reduce(int N);
  ~dBlock_Reduce();
  void Sum(double *f, double* res, int i=0);
  void Max(double *f, double* res);

 private:
  int N_;    void * Maxwork;    uint64_t sizeMax;   

  bool first_Max = true;
  bool first_Sum = true;
  
  cutensorTensorDescriptor_t dA, dB;
  std::unordered_map<int32_t, int64_t> extent;
  std::vector<int64_t> extent_A, extent_B;

  std::vector<int32_t> Amode{'a'};
  std::vector<int32_t> Bmode{};

  int32_t nAmode = Amode.size(); // for single-block data input (such as for CFL condition)
  int32_t nBmode = Bmode.size(); // scalar output for contiguous data
  
};

class Species_Reduce : public Red {
 public:
  Species_Reduce(int N, int nspecies);
  ~Species_Reduce();
  void Sum(float *f, float* res, int i=0);

 private:
  int N_;
  bool first_Sum = true;
  
  cutensorTensorDescriptor_t dQ, dR;
  std::unordered_map<int32_t, int64_t> extent;
  std::vector<int64_t> extent_Q, extent_R;  

  std::vector<int32_t> Qmode{'a', 's'};
  std::vector<int32_t> Rmode{'s'};

  int32_t nQmode = Qmode.size(); // for single-block data input (such as flux)
  int32_t nRmode = Rmode.size(); // for single-block data input (such as flux)
  
};

class DenseM : public Red {
 public:
  DenseM(int N, int M);
  ~DenseM();
  void MatVec(double* res, double* Mat, double* vec);
  void MatMat(double* res, double* M1, double* M2);

 private:
  int M_;  int N_;
  void * Multwork;    uint64_t sizeWork; 
  void * MMwork;      uint64_t sizeMM;

  bool first_MV = true;
  bool first_MM = true;
  
  cutensorTensorDescriptor_t dM, dV, dW, dX, dY, dZ;
  cutensorContractionDescriptor_t dMV, dMM;
  cutensorContractionPlan_t MVplan, MMplan;

  std::unordered_map<int32_t, int64_t> extent;
  std::vector<int64_t> extent_Y, extent_M, extent_V, extent_W, extent_X, extent_Z;

  // G = P R2 or Y[M] = M[M x N] * V[N]
  std::vector<int32_t> Ymode{'g'};
  std::vector<int32_t> Mmode{'g', 'r'};
  std::vector<int32_t> Vmode{'r'};

  // X[M x N] = W [M x N] Z [ N x N ]  
  std::vector<int32_t> Xmode{'g', 's'};
  std::vector<int32_t> Wmode{'g', 'r'};
  std::vector<int32_t> Zmode{'r', 's'};
  
  int32_t nYmode = Ymode.size(); 
  int32_t nMmode = Mmode.size(); 
  int32_t nVmode = Vmode.size(); 
  int32_t nWmode = Wmode.size(); 
  int32_t nXmode = Xmode.size(); 
  int32_t nZmode = Zmode.size(); 
};
