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
  virtual void Sum(float *f, float* res, int i=0) = 0; // each tensor reduction has to provide a sum method
  virtual void Max(float *f, float* res) {};  // the max operation is not required

  void * Addwork;     uint64_t sizeAdd;    uint64_t sizeWork;    
  cudaDataType_t cfloat = CUDA_R_32F;;
  cutensorComputeType_t typeCompute = CUTENSOR_R_MIN_32F;
  cutensorOperator_t opAdd = CUTENSOR_OP_ADD;
  cutensorOperator_t opMax = CUTENSOR_OP_MAX;
  cutensorHandle_t handle; 
  float alpha = 1.;
  float beta  = 0.;
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
