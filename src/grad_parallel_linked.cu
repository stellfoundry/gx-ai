#include "grad_parallel.h"
#include "cuda_constants.h"
#include "device_funcs.h"
#include "cufftXt.h"
#include "cufft.h"
#include <stdlib.h>

__global__ void linkedCopy(cuComplex* G, cuComplex* G_linked, int nLinks, int nChains, int* ikx, int* iky, int nMoms);
__global__ void linkedCopyBack(cuComplex* G_linked, cuComplex* G, int nLinks, int nChains, int* ikx, int* iky, int nMoms);

__device__ void i_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  unsigned int idz = offset;
  cuComplex Ikz = make_cuComplex(0., kz[idz]);
  // zp*kz[1] = 1/nLinks[c] (for normalization)
  ((cuComplex*)dataOut)[offset] = Ikz*element/nz*zp*kz[1];
}

__device__ void abs_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr)
{
  float *kz = (float*) kzData;
  unsigned int idz = offset;
  // zp*kz[1] = 1/nLinks[c] (for normalization)
  ((cuComplex*)dataOut)[offset] = abs(kz[idz])*element/nz*zp*kz[1];
}

__global__ void init_kzLinked(float* kz, int nLinks)
{
  for(int i=0; i<nz*nLinks; i++) {
    if(i<nz*nLinks/2+1) {
      kz[i] = (float) i/(zp*nLinks);
    } else {
      kz[i] = (float) (i-nz*nLinks)/(zp*nLinks);
    }
  }
}

__managed__ cufftCallbackStoreC i_kzLinked_callbackPtr = i_kzLinked;
__managed__ cufftCallbackStoreC abs_kzLinked_callbackPtr = abs_kzLinked;

GradParallelLinked::GradParallelLinked(Grids* grids, int jtwist)
 : grids_(grids)
{
  int naky = grids_->Naky;
  int ntheta0 = grids_->Nakx;

  int idxRight[naky*ntheta0];
  int idxLeft[naky*ntheta0];

  int linksR[naky*ntheta0];
  int linksL[naky*ntheta0];

  int n_k[naky*ntheta0];

  nClasses = get_nClasses(idxRight, idxLeft, linksR, linksL, n_k, naky, ntheta0, jtwist);

  nLinks = (int*) malloc(sizeof(int)*nClasses);
  nChains = (int*) malloc(sizeof(int)*nClasses);

  get_nLinks_nChains(nLinks, nChains, n_k, nClasses, naky, ntheta0);

  ikxLinked_h = (int**) malloc(sizeof(int)*nClasses);
  ikyLinked_h = (int**) malloc(sizeof(int)*nClasses);
  for(int c=0; c<nClasses; c++) {
    ikxLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    ikyLinked_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
  }

  kFill(nClasses, nChains, nLinks, ikyLinked_h, ikxLinked_h, linksL, linksR, idxRight, naky, ntheta0);

  gradpar_plan_forward = (cufftHandle*) malloc(sizeof(cufftHandle)*nClasses);
  gradpar_plan_inverse = (cufftHandle*) malloc(sizeof(cufftHandle)*nClasses);
  gradpar_plan_forward_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle)*nClasses);
  abs_gradpar_plan_forward_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle)*nClasses);
  gradpar_plan_inverse_singlemom = (cufftHandle*) malloc(sizeof(cufftHandle)*nClasses);
  dimGrid = (dim3*) malloc(sizeof(dim3)*nClasses);
  dimBlock = (dim3*) malloc(sizeof(dim3)*nClasses);

  // these are arrays of pointers to device memory
  cudaMallocHost((void**) &ikxLinked, sizeof(int*)*nClasses);
  cudaMallocHost((void**) &ikyLinked, sizeof(int*)*nClasses);
  cudaMallocHost((void**) &G_linked, sizeof(cuComplex*)*nClasses);
  cudaMallocHost((void**) &kzLinked, sizeof(float*)*nClasses);
  for(int c=0; c<nClasses; c++) {
    // allocate and copy into device memory
    cudaMalloc((void**) &ikxLinked[c], sizeof(int)*nLinks[c]*nChains[c]);
    cudaMalloc((void**) &ikyLinked[c], sizeof(int)*nLinks[c]*nChains[c]);
    cudaMemcpy(ikxLinked[c], ikxLinked_h[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyHostToDevice);
    cudaMemcpy(ikyLinked[c], ikyLinked_h[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyHostToDevice);

    cudaMalloc((void**) &G_linked[c], sizeof(cuComplex)*grids_->Nz*nLinks[c]*nChains[c]*grids_->Nl*grids_->Nm);
    cudaMemset(G_linked[c], 0., sizeof(cuComplex)*grids_->Nz*nLinks[c]*nChains[c]*grids_->Nl*grids_->Nm);

    cudaMalloc((void**) &kzLinked[c], sizeof(float)*grids_->Nz*nLinks[c]);
    cudaMemset(kzLinked[c], 0., sizeof(float)*grids_->Nz*nLinks[c]);

    // set up transforms
    cufftCreate(&gradpar_plan_forward[c]);
    cufftCreate(&gradpar_plan_inverse[c]);
    cufftCreate(&gradpar_plan_forward_singlemom[c]);
    cufftCreate(&abs_gradpar_plan_forward_singlemom[c]);
    cufftCreate(&gradpar_plan_inverse_singlemom[c]);
    int size = nLinks[c]*grids_->Nz;
    size_t workSize;
    cufftMakePlanMany(gradpar_plan_forward[c], 1, &size, 
		      NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c]*grids_->Nl*grids_->Nm, &workSize);
    cufftMakePlanMany(gradpar_plan_inverse[c], 1, &size, 
		      NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c]*grids_->Nl*grids_->Nm, &workSize);
    cufftMakePlanMany(gradpar_plan_forward_singlemom[c], 1, &size, 
		      NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c], &workSize);
    cufftMakePlanMany(abs_gradpar_plan_forward_singlemom[c], 1, &size, 
		      NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c], &workSize);
    cufftMakePlanMany(gradpar_plan_inverse_singlemom[c], 1, &size, 
		      NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 
                      nChains[c], &workSize);

    // initialize kzLinked
    init_kzLinked<<<1,1>>>(kzLinked[c], nLinks[c]);

    // set up callback functions
    cufftXtSetCallback(gradpar_plan_forward[c], (void**) &i_kzLinked_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);
    cufftXtSetCallback(gradpar_plan_forward_singlemom[c], (void**) &i_kzLinked_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);
    cufftXtSetCallback(abs_gradpar_plan_forward_singlemom[c], (void**) &abs_kzLinked_callbackPtr, CUFFT_CB_ST_COMPLEX, (void**)&kzLinked[c]);

    dimBlock[c] = dim3(32,4,4);
    dimGrid[c] = dim3(grids_->Nz/dimBlock[c].x+1, nLinks[c]*nChains[c]/dimBlock[c].y+1, grids_->Nmoms/dimBlock[c].z+1);
  }
}

GradParallelLinked::~GradParallelLinked()
{
  free(nLinks);
  free(nChains);
  free(dimBlock);
  free(dimGrid);
  for(int c=0; c<nClasses; c++) {
    cufftDestroy(gradpar_plan_forward[c]);
    cufftDestroy(gradpar_plan_inverse[c]);
    cufftDestroy(gradpar_plan_forward_singlemom[c]);
    cufftDestroy(abs_gradpar_plan_forward_singlemom[c]);
    cufftDestroy(gradpar_plan_inverse_singlemom[c]);
    free(ikxLinked_h[c]);
    free(ikyLinked_h[c]);
    cudaFree(ikxLinked[c]);
    cudaFree(ikyLinked[c]);
    cudaFree(kzLinked[c]);
    cudaFree(G_linked[c]);
  }
  free(gradpar_plan_forward);
  free(gradpar_plan_inverse);
  free(gradpar_plan_forward_singlemom);
  free(abs_gradpar_plan_forward_singlemom);
  free(gradpar_plan_inverse_singlemom);
  free(ikxLinked_h);
  free(ikyLinked_h);
  cudaFreeHost(ikxLinked);
  cudaFreeHost(ikyLinked);
  cudaFreeHost(G_linked);
  cudaFreeHost(kzLinked);
}

void GradParallelLinked::dz(MomentsG* G) 
{
  for(int c=0; c<nClasses; c++) {
    // each "class" has a different number of links in the chains, and a different number of chains.
    linkedCopy<<<dimGrid[c],dimBlock[c]>>>(G->G(), G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
    cufftExecC2C(gradpar_plan_forward[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(gradpar_plan_inverse[c], G_linked[c], G_linked[c], CUFFT_INVERSE);
    linkedCopyBack<<<dimGrid[c],dimBlock[c]>>>(G_linked[c], G->G(), nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], grids_->Nmoms);
  }
}

// for a single moment m
void GradParallelLinked::dz(cuComplex* m, cuComplex* res)
{
  int nMoms=1;
  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopy<<<dimGrid[c],dimBlock[c]>>>(m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
    cufftExecC2C(gradpar_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(gradpar_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);
    linkedCopyBack<<<dimGrid[c],dimBlock[c]>>>(G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}

// for a single moment m
void GradParallelLinked::abs_dz(cuComplex* m, cuComplex* res)
{
  int nMoms=1;
  for(int c=0; c<nClasses; c++) {
    // these only use the G(0,0) part of G_linked
    linkedCopy<<<dimGrid[c],dimBlock[c]>>>(m, G_linked[c], nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
    cufftExecC2C(abs_gradpar_plan_forward_singlemom[c], G_linked[c], G_linked[c], CUFFT_FORWARD);
    cufftExecC2C(gradpar_plan_inverse_singlemom[c], G_linked[c], G_linked[c], CUFFT_INVERSE);
    linkedCopyBack<<<dimGrid[c],dimBlock[c]>>>(G_linked[c], res, nLinks[c], nChains[c], ikxLinked[c], ikyLinked[c], nMoms);
  }
}

__global__ void linkedCopy(cuComplex* G, cuComplex* G_linked, int nLinks, int nChains, int* ikx, int* iky, int nMoms)
{
  unsigned int idz = get_id1();
  unsigned int idk = get_id2();
  unsigned int idlm = get_id3();

  if(idz<nz && idk<nLinks*nChains && idlm<nMoms) {
    unsigned int idlink = idz + nz*idk + nz*nLinks*nChains*idlm;
    unsigned int globalIdx = iky[idk] + nyc*ikx[idk] + idz*nx*nyc + idlm*nx*nyc*nz;

    // seems hopeless to make these accesses coalesced. how bad is it?
    G_linked[idlink] = G[globalIdx];
  }
}

__global__ void linkedCopyBack(cuComplex* G_linked, cuComplex* G, int nLinks, int nChains, int* ikx, int* iky, int nMoms)
{
  unsigned int idz = get_id1();
  unsigned int idk = get_id2();
  unsigned int idlm = get_id3();

  if(idz<nz && idk<nLinks*nChains && idlm<nMoms) {
    unsigned int idlink = idz + nz*idk + nz*nLinks*nChains*idlm;
    unsigned int globalIdx = iky[idk] + nyc*ikx[idk] + idz*nx*nyc + idlm*nx*nyc*nz;

    G[globalIdx] = G_linked[idlink];
  }
}

int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

int GradParallelLinked::get_nClasses(int *idxRight, int *idxLeft, int *linksR, int *linksL, int *n_k, int naky, int ntheta0, int jshift0)
{  
  int idx0, idxL, idxR;
  for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      
      //map indices to kx indices
      if(idx < (ntheta0+1)/2 ) {
        idx0 = idx;     
      } else {
        idx0 = idx - ntheta0;
      }       
              
      if(idy == 0) {                 
        idxL = idx0;
	idxR = idx0;
      } else {
        // signs here are correct according to Mike Beer's thesis
        idxL = idx0 + idy*jshift0;
	idxR = idx0 - idy*jshift0;
      }
      
      //remap to usual indices
      if(idxL >= 0 && (idxL) < (ntheta0+1)/2) {
        idxLeft[idy + naky*idx] = idxL;
      } else if( (idxL+ntheta0)>=(ntheta0+1)/2 && (idxL+ntheta0)<ntheta0 ) {
        idxLeft[idy + naky*idx] = idxL + ntheta0;                   //nshift
      } else {
        idxLeft[idy + naky*idx] = -1;
      }
      
      if(idxR >= 0 && idxR < (ntheta0+1)/2) {
        idxRight[idy + naky*idx] = idxR;
      } else if( (idxR+ntheta0)>=(ntheta0+1)/2 && (idxR+ntheta0)<ntheta0 ) {
        idxRight[idy + naky*idx] = idxR + ntheta0;
      } else {
        idxRight[idy + naky*idx] = -1;
      }
    }
  }
  
  /*for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("idxLeft[%d,%d]= %d  ", idy, idx, idxLeft[idy + naky*idx]);
    }
    printf("\n");
  }
  for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("idxRight[%d,%d]= %d  ", idy, idx, idxRight[idy + naky*idx]);
    }
    printf("\n");
  } */  
  
  for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      
      
      //count the links for each region
      //linksL = number of links to the left of current position
      
      linksL[idy + naky*idx] = 0;     
      
      
      int idx_star = idx;
      
      
      
      while(idx_star != idxLeft[idy + naky*idx_star] && idxLeft[idy + naky*(idx_star)] >= 0) {
        //increment left links counter, and move to next link to left
	//until idx of link to left is negative or same as current idx
	linksL[idy + naky*idx]++;
	idx_star = idxLeft[idy + naky*(idx_star)];
      }	  
      

      //linksR = number of links to the right
      linksR[idy + naky*idx] = 0;     
      idx_star = idx;
      while(idx_star != idxRight[idy + naky*idx_star] && idxRight[idy + naky*idx_star] >= 0) {
        linksR[idy + naky*idx]++;
        idx_star = idxRight[idy + naky*idx_star];
      }	       
    }
  }
  
  /* for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("linksL[%d,%d]= %d  ", idy, idx, linksL[idy + naky*idx]);
    }
    printf("\n");
  }
  for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("linksR[%d,%d]= %d  ", idy, idx, linksR[idy + naky*idx]);
    }
    printf("\n");
  } */
  
  //now we set up class array
  //nClasses = # of classes  
  
  //first count number of links for each (kx,ky)
  int k = 0;
  
  for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      n_k[k] = 1 + linksL[idy + naky*idx] + linksR[idy + naky*idx];
      k++;
    }
  }
  
  /*for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("nLinks[%d,%d]= %d  ", idy, idx, n_k[idy+naky*idx]);
    }
    printf("\n");
  }*/
    
  //count how many unique values of n_k there are, which is the number of classes
  
  //sort...
  qsort(n_k, naky*ntheta0, sizeof(int), compare);   
  
  //then count
  int nClasses = 1;
  for(int k=0; k<naky*ntheta0-1; k++) {
    if(n_k[k] != n_k[k+1])
      nClasses= nClasses + 1;
  }
  return nClasses;
  
}

void GradParallelLinked::get_nLinks_nChains(int *nLinks, int *nChains, int *n_k, int nClasses, int naky, int ntheta0)
{
  for(int c=0; c<nClasses; c++) {
    nChains[c] = 1;
    nLinks[c] = 0;
  }
  
  //fill the nChains and nLinks arrays
  int c = 0;
  for(int k=1; k<naky*ntheta0; k++) {
    if(n_k[k] == n_k[k-1])
      nChains[c]++;
    else {
      nLinks[c] = n_k[k-1];
      nChains[c] = nChains[c]/nLinks[c];    //why???
      c++;
    }
  }
  c = nClasses-1;
  nLinks[c] = n_k[naky*ntheta0-1];
  nChains[c] = nChains[c]/nLinks[c];
  
}  


void kt2ki(int idy, int idx, int *c, int *p, int* linksL, int* linksR, int nClasses, int* nLinks, int naky)
{
  //get nLinks in the current chain
  int np_k = 1 + linksL[idy + naky*idx] + linksR[idy + naky*idx];
  
  //find which class corresponds to this nLinks
  for(int i=0; i<nClasses; i++) {
    if(nLinks[i] == np_k) {
      *c= i;
      break;
    }
  }
  
  *p = linksL[idy + naky*idx];
} 


void fill(int *ky, int *kx, int idy, int idx, int *idxRight, int c, int p, int n, int naky, int ntheta0, int nshift, int nLinks) {
  int idx0;
  if(idx < (ntheta0+1)/2)
    idx0=idx;
  else
    idx0=idx+nshift;
  
  ky[p+nLinks*n] = idy;              
  kx[p+nLinks*n] = idx0;
  int idxR=idx;
  
  for(p=1; p<nLinks; p++) {
    idxR = idxRight[idy + naky*idxR];     
    
    ky[p + nLinks*n] = idy;
    if(idxR < (ntheta0+1)/2) {      
      kx[p + nLinks*n] = idxR;
    } else {
      kx[p + nLinks*n] = idxR+nshift;
    }  
  }
}   
  
void GradParallelLinked::kFill(int nClasses, int *nChains, int *nLinks, int **ky, int **kx, int *linksL, int *linksR, int *idxRight, int naky, int ntheta0) 
{
  int nshift = grids_->Nx-ntheta0;
  //fill the kx and ky arrays
  for(int ic=0; ic<nClasses; ic++) {
    int n = 0;
    int  p, c;
    for(int idy=0; idy<naky; idy++) {
      for(int idx=0; idx<ntheta0; idx++) {
        kt2ki(idy, idx, &c, &p, linksL, linksR, nClasses, nLinks, naky);
     	if(c==ic) {	  
	  if(p==0) {	 
	      	    
	    fill(ky[c], kx[c], idy, idx, idxRight, c, p, n, naky,ntheta0, nshift,nLinks[c]);
	    
	    n++;	     
	  }
	}
      }
    }
  }
}      	

void GradParallelLinked::linkPrint() {
  for(int c=0; c<nClasses; c++) {
    for(int n=0; n<nChains[c]; n++) {
      for(int p=0; p<nLinks[c]; p++) {
        if(ikxLinked_h[c][p+nLinks[c]*n]<grids_->Nx/2+1) {
          printf("(%d,%d) ", ikyLinked_h[c][p+nLinks[c]*n], ikxLinked_h[c][p+nLinks[c]*n]);
        }
        else {
          printf("(%d,%d) ",ikyLinked_h[c][p+nLinks[c]*n], ikxLinked_h[c][p+nLinks[c]*n]-grids_->Nx);	
        }
        if(ikxLinked_h[c][p+nLinks[c]*n]>(grids_->Nx-1)/3 && ikxLinked_h[c][p+nLinks[c]*n]<2*(grids_->Nx/3)+1) {
          printf("->DEALIASING ERROR");
        }	
        /* *counter= *counter+1; */
      }
      printf("\n");
    }
    printf("\n\n");
  }
}


          

