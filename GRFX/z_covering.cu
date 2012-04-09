void ZDERIVcovering(cufftComplex *result, cufftComplex* f)
{
  //initialize the kx and ky arrays for the covering space  
  int nClasses;
  int *nLinks;
  int *nChains;
  int *linksR;
  int *linksL;
  int *idxRight;
  int *idxLeft;
  
  //Nx=Ny=16;
  
  //printf("\nm\n");
  //each class has nChains[c] chains, all with the same length, nLinks[c]
  
  //covering(nClasses, nChains, nLinks, linksL, linksR, idxLeft, idxRight);
  
  int naky = 1 + Ny/3;
  int ntheta0 = 1 + 2*Nx/3;   // mask in the middle of array... problems?
  int nshift = Nx - ntheta0;
  int jshift0;
  
  ////////////////////////////
  //get idxLeft and idxRight//
  ////////////////////////////
  
  jshift0 = 1;
  
  cudaMalloc((void**) &idxLeft, sizeof(int)*naky*ntheta0);
  cudaMalloc((void**) &idxRight, sizeof(int)*naky*ntheta0);
  
  for(int i=0; i<ntheta0; i++) {
    idxLeft[0+naky*i] = -1;
    idxRight[0+naky*i] = -1;
  }
  
  int idx0, idxL, idxR;
  //using idx0 fixes mask problem?
  
  for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      //map indices to kx indices
      if(idx > (ntheta0)/2 ) {
        idx0 = idx - ntheta0;     
      } else {
        idx0 = idx;
      }       
              
      if(idy == 0) {                 
        idxL = idx0;
	idxR = idx0;
      } else {
        idxL = idx0 + idy*jshift0;
	idxR = idx0 - idy*jshift0;
      }
      
      //remap to usual indices
      if(idxL >= 0 && (idxL) < ntheta0/2+1) {
        idxLeft[idy + naky*idx] = idxL;
      } else if( (idxL+ntheta0)>(ntheta0)/2 && (idxL+ntheta0)<ntheta0 ) {
        idxLeft[idy + naky*idx] = idxL + ntheta0;                   //nshift
      } else {
        idxLeft[idy + naky*idx] = -1;
      }
      
      if(idxR >= 0 && idxR < (ntheta0)/2+1) {
        idxRight[idy + naky*idx] = idxR;
      } else if( (idxR+ntheta0)>(ntheta0)/2 && (idxR+ntheta0)<ntheta0 ) {
        idxRight[idy + naky*idx] = idxR + ntheta0;
      } else {
        idxRight[idy + naky*idx] = -1;
      }
    }
  }    
  
  for(int idx=0; idx<ntheta0; idx++) {
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
  }
  
      
  /////////////////////////
  //get linksR and linksL//
  /////////////////////////
  
  cudaMalloc((void**) &linksR, sizeof(int)*naky*ntheta0);
  cudaMalloc((void**) &linksL, sizeof(int)*naky*ntheta0);
  
  for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      
      //count the links for each region
      //linksL = number of links to the left of current position
      linksL[idy + naky*idx] = 0;
      /*if(idx < ntheta0/2+1)
        idx0=idx;
      else
        idx0=idx+nshift;
      int idx_star = idx0;*/
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
        /*    
        //increment left links counter, and move to next link to left
	//until idx of link to left is negative or same as current idx
      bool breaker = true;
      do {
	if(idx_star < ntheta0/2+1) {
	  if(idx_star != idxLeft[idy + naky*idx] && idxLeft[idy + naky*(idx_star)] >= 0) {
	    linksL[idy + naky*idx]++;
	    idx_star = idxLeft[idy + naky*(idx_star)];
	  } else breaker = false;  
	} else {
	  if(idx_star != idxLeft[idy + naky*idx] && idxLeft[idy + naky*(idx_star-nshift)] >= 0) {
	    linksL[idy + naky*idx]++;
	    idx_star = idxLeft[idy + naky*(idx_star-nshift)];
	  } else breaker = false;  
	}  
      }	while(breaker);  
      

      //linksR = number of links to the right
      linksR[idy + naky*idx] = 0;
      idx_star = idx0;
      breaker = true;
      do {
	if(idx_star < ntheta0/2+1) {
	  if(idx_star != idxRight[idy + naky*idx] && idxRight[idy + naky*(idx_star)] >= 0) {
	    linksR[idy + naky*idx]++;
	    idx_star = idxRight[idy + naky*(idx_star)];
	  } else breaker = false;  
	} else {
	  if(idx_star != idxRight[idy + naky*idx] && idxRight[idy + naky*(idx_star-nshift)] >= 0) {
	    linksR[idy + naky*idx]++;
	    idx_star = idxRight[idy + naky*(idx_star-nshift)];
	  } else breaker = false;  
	}  
      }	while(breaker);  
      */  
      
    }
  }
  
  for(int idx=0; idx<ntheta0; idx++) {
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
  }
  
  //now we set up class arrays
  //nClasses = # of classes
  //nLinks[c] = # of links in each chain in the class
  //nChains[c] = # of chains in the class
  
  
  //first count number of links for each (kx,ky)
  int k = 0;
  int* n_k;
  cudaMalloc((void**) &n_k, sizeof(int)*naky*ntheta0);
  for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      n_k[k] = 1 + linksL[idy + naky*idx] + linksR[idy + naky*idx];
      k++;
    }
  }
  
  for(int idx=0; idx<ntheta0; idx++) {
    for(int idy=0; idy<naky; idy++) {
      printf("nLinks[%d,%d]= %d  ", idy, idx, n_k[idy+naky*idx]);
    }
    printf("\n");
  }
    
  //count how many unique values of n_k there are, which is the number of classes
  
  //sort...
  qsort(n_k, naky*ntheta0, sizeof(int), compare);   
  
  //then count
  nClasses = 1;
  for(int k=0; k<naky*ntheta0-1; k++) {
    if(n_k[k] != n_k[k+1])
      nClasses++;
  }
  
  cudaMalloc((void**) &nChains, sizeof(int)*nClasses);
  cudaMalloc((void**) &nLinks, sizeof(int)*nClasses);
  
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
  
  printf("nClasses=%d  ", nClasses);
  printf("nLinks[0]=%d  ", nLinks[0]);
  printf("nChains[0]=%d  \n", nChains[0]);


  ///////////////////////////////////////////////////////////
  
  int *ky[nClasses];
  int *kx[nClasses];
  for(int c=0; c<nClasses; c++) {
    cudaMalloc((void**) &ky[c], sizeof(int)*nLinks[c]*nChains[c]);
    cudaMalloc((void**) &kx[c], sizeof(int)*nLinks[c]*nChains[c]);
  }  
    
  kFill(nClasses, nChains, nLinks, ky, kx, linksL, linksR, idxRight); 
  
  for(int c=0; c<nClasses; c++) {
    printf("\n");
    kPrint(nClasses, nLinks[c], nChains[c], ky[c], kx[c],c);
    zTransformCovering(nLinks[c], nChains[c], ky[c], kx[c], f);
  }    	
  
  
 
} 


