#include "reductions.h"
#include <iostream>

// ======= Grid_Species_Reduce ==========
Grid_Species_Reduce::Grid_Species_Reduce(Grids *grids, std::vector<int> spectra) :
  grids_(grids), spectra_(spectra)
{
  Addwork = nullptr;     sizeWork = 0;         sizeAdd = 0;

  int J;  J = spectra_.size();
  initialized.assign(J, 0);   desc.resize(J);   extents.resize(J);
  
  extent['y'] = grids_->Nyc;
  extent['x'] = grids_->Nx;
  extent['z'] = grids_->Nz;
  extent['s'] = grids_->Nspecies;;

  for (auto mode : Pmode) extent_P.push_back(extent[mode]);;
  for (int j = 0; j < J; j++) {
    if (spectra_[j] == 1) {
      for (auto mode : pModes[j]) extents[j].push_back(extent[mode]);

      cutensorInit(&handle);
      cutensorInitTensorDescriptor(&handle, &dP, nPmode, extent_P.data(), NULL, cfloat, CUTENSOR_OP_ABS);
      cutensorInitTensorDescriptor(&handle, &desc[j], pModes[j].size(), extents[j].data(), NULL, cfloat, CUTENSOR_OP_ABS);
    }
  }
}

Grid_Species_Reduce::~Grid_Species_Reduce()
{
  if (Addwork) cudaFree(Addwork);
}

// ======== Grid_Species_Reduce  ==============
void Grid_Species_Reduce::Sum(float* P2, float* res, int ispec)
{
  if (initialized[ispec] == 0) {
  
    cutensorReductionGetWorkspace(&handle, P2, &dP, Pmode.data(),
				  res, &desc[ispec], pModes[ispec].data(),
				  res, &desc[ispec], pModes[ispec].data(),
				  opAdd, typeCompute, &sizeAdd);
    if (sizeAdd > sizeWork) {
      sizeWork = sizeAdd;
      if (Addwork) cudaFree (Addwork);
      if (cudaSuccess != cudaMalloc(&Addwork, sizeWork)) {
	Addwork = nullptr;	sizeWork = 0;
      }
    }
    initialized[ispec]  = 1;
  }
  
  cutensorReduction(&handle,
		    (const void*) &alpha, P2, &dP, Pmode.data(),
		    (const void*) &beta,  res,  &desc[ispec], pModes[ispec].data(),
		    res,  &desc[ispec], pModes[ispec].data(),
		    opAdd, typeCompute, Addwork, sizeWork, 0);
}		     

// ======= Grid_Reduce ==========
Grid_Reduce::Grid_Reduce(Grids *grids, std::vector<int> spectra) :
  grids_(grids), spectra_(spectra)
{
  Addwork = nullptr;     sizeWork = 0;         sizeAdd = 0;

  int J;  J = spectra_.size();
  initialized.assign(J, 0);   desc.resize(J);   extents.resize(J);
  sizeWork = 0;               sizeAdd = 0;          

  extent['y'] = grids_->Nyc;
  extent['x'] = grids_->Nx;
  extent['z'] = grids_->Nz;
  
  for (auto mode : Imode) extent_I.push_back(extent[mode]);
  for (int j = 0; j < J; j++) {
    if (spectra_[j] == 1) {
      for (auto mode : iModes[j]) extents[j].push_back(extent[mode]);

      //      printf("0 =? %d \n",iModes[0].size());
      
      cutensorInit(&handle);
      cutensorInitTensorDescriptor(&handle, &dI, nImode, extent_I.data(), NULL, cfloat, CUTENSOR_OP_ABS);
      cutensorInitTensorDescriptor(&handle, &desc[j], iModes[j].size(), extents[j].data(), NULL, cfloat, CUTENSOR_OP_ABS);

    }
  }
}

Grid_Reduce::~Grid_Reduce()
{
  if (Addwork) cudaFree(Addwork);
}

void Grid_Reduce::Sum(float* I2, float* res, int ispec)
{

  if (initialized[ispec] == 0) {
  
    cutensorReductionGetWorkspace(&handle, I2, &dI, Imode.data(),
				  res, &desc[ispec], iModes[ispec].data(),
				  res, &desc[ispec], iModes[ispec].data(),
				  opAdd, typeCompute, &sizeAdd);
    if (sizeAdd > sizeWork) {
      sizeWork = sizeAdd;
      if (Addwork) cudaFree (Addwork);
      if (cudaSuccess != cudaMalloc(&Addwork, sizeWork)) {
	Addwork = nullptr;	sizeWork = 0;
      }
    }
    initialized[ispec]  = 1;
  } 

  cutensorReduction(&handle,
		    (const void*) &alpha, I2, &dI, Imode.data(),
		    (const void*) &beta,  res,  &desc[ispec], iModes[ispec].data(),
		    res,  &desc[ispec], iModes[ispec].data(),
		    opAdd, typeCompute, Addwork, sizeWork, 0);
}

// ======= All_Reduce ==========
All_Reduce::All_Reduce(Grids *grids, std::vector<int> spectra) :
  grids_(grids), spectra_(spectra)
{
  Addwork = nullptr;      sizeWork = 0;         sizeAdd = 0;

  int J;  J = spectra_.size();
  initialized.assign(J, 0);   desc.resize(J);   extents.resize(J);
  
  extent['y'] = grids_->Nyc;
  extent['x'] = grids_->Nx;
  extent['z'] = grids_->Nz;
  extent['l'] = grids_->Nl;
  extent['m'] = grids_->Nm;
  extent['s'] = grids_->Nspecies;;

  for (auto mode : Wmode) extent_W.push_back(extent[mode]);;

  // Build tensor descriptions for partial summations here
  for (int j = 0; j < J; j++) {
    if (spectra_[j] == 1) {
      for (auto mode : Modes[j]) extents[j].push_back(extent[mode]);

      cutensorInit(&handle);
      cutensorInitTensorDescriptor(&handle, &dW, nWmode, extent_W.data(), NULL, cfloat, CUTENSOR_OP_ABS);
      cutensorInitTensorDescriptor(&handle, &desc[j], Modes[j].size(), extents[j].data(), NULL, cfloat, CUTENSOR_OP_ABS);
    }
  }
}

All_Reduce::~All_Reduce()
{
  if (Addwork) cudaFree(Addwork);
}

void All_Reduce::Sum(float* W, float* res, int ispec) 
{

  if (initialized[ispec] == 0) {

    // Get size of workspace that will be used, stored in sizeAdd (sizeWork)
    cutensorReductionGetWorkspace(&handle, W, &dW, Wmode.data(),
				  res, &desc[ispec], Modes[ispec].data(),
				  res, &desc[ispec], Modes[ispec].data(),
				  opAdd, typeCompute, &sizeAdd);
  
    // if the size is larger than currently allocated (starting with unallocated) space, free
    // the old one (if it is allocated) and allocate the larger space
    // Assume it is fine to use the larger work space freely. 
    //    printf("Workspace allocation: %d \t with size %d \n",ispec,sizeAdd);
    if (sizeAdd > sizeWork) {
      sizeWork = sizeAdd;
      if (Addwork) cudaFree (Addwork);
      if (cudaSuccess != cudaMalloc(&Addwork, sizeWork)) {
	Addwork = nullptr;	sizeWork = 0;
      }
      //      printf("work size = %d \n", sizeWork);
    }
    initialized[ispec] = 1;
  } 

  //  printf("Reduction: %d \n",ispec);
  cutensorReduction(&handle,
		    (const void*) &alpha, W,   &dW,          Wmode.data(),
		    (const void*) &beta,  res, &desc[ispec], Modes[ispec].data(),
		    res,  &desc[ispec], Modes[ispec].data(),
		    opAdd, typeCompute, Addwork, sizeWork, 0);
  // The final argument in this call is the stream used for the calculation
}

//============ Block_Reduce ==============
Block_Reduce::Block_Reduce(int N) : N_(N)
{
  Addwork = nullptr;      sizeAdd = 0;
  Maxwork = nullptr;      sizeMax = 0;
  
  extent['a'] = N_;
  extent['s'] = 1;
  for (auto mode : Amode) extent_A.push_back(extent[mode]); // incoming tensor assuming data is contiguous
  for (auto mode : Bmode) extent_B.push_back(extent[mode]); // target scalar output

  cutensorInit(&handle);
  cutensorInitTensorDescriptor(&handle, &dA, nAmode, extent_A.data(), NULL, cfloat, CUTENSOR_OP_ABS);
  cutensorInitTensorDescriptor(&handle, &dB, nBmode, extent_B.data(), NULL, cfloat, CUTENSOR_OP_ABS);
}

Block_Reduce::~Block_Reduce()
{
  if (Addwork) cudaFree(Addwork);
  if (Maxwork) cudaFree(Maxwork);  
}

void Block_Reduce::Max(float* A2, float* B)
{
  // calculate reduction, B = max(|A2|), over first few indices

  if (first_Max) {
    
    // get workspace (sizeMax) for a max (opMax) over |P2|
    cutensorReductionGetWorkspace(&handle,
				  A2, &dA, Amode.data(),
				  B,  &dB, Bmode.data(),
				  B,  &dB, Bmode.data(),
				  opMax, typeCompute, &sizeMax);
    
    if (cudaSuccess != cudaMalloc(&Maxwork, sizeMax)) {
      Maxwork = nullptr;      sizeMax = 0;
    }
    first_Max = false;
  }

  cutensorReduction(&handle,
		    (const void*) &alpha, A2, &dA, Amode.data(),
		    (const void*) &beta,  B,  &dB, Bmode.data(),
		    B,  &dB, Bmode.data(),
		    opMax, typeCompute, Maxwork, sizeMax, 0);
}

void Block_Reduce::Sum(float* A, float* B, int i)
{
  // calculate full reduction, B = sum A

  if (first_Sum) {
    
    cutensorReductionGetWorkspace(&handle,
				  A,  &dA, Amode.data(),
				  B,  &dB, Bmode.data(),
				  B,  &dB, Bmode.data(),
				  opAdd, typeCompute, &sizeAdd);
    
    if (cudaSuccess != cudaMalloc(&Addwork, sizeAdd)) {
      Addwork = nullptr;      sizeAdd = 0;
    }
    first_Sum = false;
  }

  cutensorReduction(&handle,
		    (const void*) &alpha, A, &dA, Amode.data(),
		    (const void*) &beta,  B, &dB, Bmode.data(),
		    B, &dB, Bmode.data(),
		    opAdd, typeCompute, Addwork, sizeAdd, 0);
}

//============ Species_Reduce ==============
Species_Reduce::Species_Reduce(int N, int nspecies) : N_(N)
{
  Addwork = nullptr;     sizeAdd = 0; 
  
  extent['a'] = N_;
  extent['s'] = nspecies;
  for (auto mode : Qmode) extent_Q.push_back(extent[mode]); // incoming tensor without abs value
  for (auto mode : Rmode) extent_R.push_back(extent[mode]); // target species scalar output

  cutensorInit(&handle);  
  cutensorInitTensorDescriptor(&handle, &dQ, nQmode, extent_Q.data(), NULL, cfloat, CUTENSOR_OP_IDENTITY);
  cutensorInitTensorDescriptor(&handle, &dR, nRmode, extent_R.data(), NULL, cfloat, CUTENSOR_OP_IDENTITY);
}

Species_Reduce::~Species_Reduce()
{
  if (Addwork) cudaFree(Addwork);
}

void Species_Reduce::Sum(float* Q, float* R, int i)
{
  // calculate reduction, R = sum Q, leaving results sorted by species only

  if (first_Sum) {
    
    // get workspace (sizeAdd) for a sum (opSum) over Q
    cutensorReductionGetWorkspace(&handle,
				  Q,  &dQ, Qmode.data(),
				  R,  &dR, Rmode.data(),
				  R,  &dR, Rmode.data(),
				  opAdd, typeCompute, &sizeAdd);
    
    if (cudaSuccess != cudaMalloc(&Addwork, sizeAdd)) {
      Addwork = nullptr;      sizeAdd = 0;
    }
    first_Sum = false;
  }

  cutensorReduction(&handle,
		    (const void*) &alpha, Q, &dQ, Qmode.data(),
		    (const void*) &beta,  R, &dR, Rmode.data(),
		    R, &dR, Rmode.data(),
		    opAdd, typeCompute, Addwork, sizeAdd, 0);
}




// __host__ ​cudaError_t cudaPointerGetAttributes ( cudaPointerAttributes* attributes, const void* ptr )
