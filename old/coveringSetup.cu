	#include <stdlib.h>
int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}
  
void getNClasses(int *nClasses, int *idxRight, int *idxLeft, int *linksR, int *linksL, int *n_k, int naky, int ntheta0, int jshift0)
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
  *nClasses = 1;
  for(int k=0; k<naky*ntheta0-1; k++) {
    if(n_k[k] != n_k[k+1])
      *nClasses= *nClasses + 1;
  }
  
}

void getNLinksChains(int *nLinks, int *nChains, int *n_k, int nClasses, int naky, int ntheta0)
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
  
  /*if(debug) {
    printf("\n\nnClasses=%d  \n", nClasses);
    for(int i=0; i<nClasses; i++) {
      printf("nLinks[%d]=%d  ", i, nLinks[i]);
      printf("nChains[%d]=%d  \n", i, nChains[i]);
    }  
  } */
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
  
void kFill(int nClasses, int *nChains, int *nLinks, int **ky, int **kx, int *linksL, int *linksR, int *idxRight, int naky, int ntheta0) 
{
  int nshift = Nx-ntheta0;
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

void kPrint(int nLinks, int nChains, int *ky, int *kx) {
  for(int n=0; n<nChains; n++) {
    for(int p=0; p<nLinks; p++) {
      if(kx[p+nLinks*n]<Nx/2+1) {
        printf("(%d,%d) ", ky[p+nLinks*n], kx[p+nLinks*n]);
      }
      else {
        printf("(%d,%d) ",ky[p+nLinks*n], kx[p+nLinks*n]-Nx);	
      }
      if(kx[p+nLinks*n]>(Nx-1)/3 && kx[p+nLinks*n]<2*(Nx/3)+1) {
        printf("->DEALIASING ERROR");
      }	
      /* *counter= *counter+1; */
    }
    printf("\n");
  }
  printf("\n\n");
}


          

