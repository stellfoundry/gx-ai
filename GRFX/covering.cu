#include <stdlib.h>
int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
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
  if(idx <ntheta0/2+1)
    idx0=idx;
  else
    idx0=idx+nshift;
  ky[p+nLinks*n] = idy;  
  kx[p+nLinks*n] = idx0;
  int idxR=idx0;
  for(p=1; p<nLinks; p++) {
    idxR = idxRight[idy + naky*idxR];
    ky[p + nLinks*n] = idy;
    if(idxR < ntheta0/2+1) {
      kx[p + nLinks*n] = idxR;
    } else {
      kx[p + nLinks*n] = idxR+nshift;
    }  
  }
}   
  
void kFill(int nClasses, int *nChains, int *nLinks, int *ky[], int *kx[], int *linksL, int *linksR, int *idxRight) 
{
  int naky = 1 + Ny/3;
  int ntheta0 = 1 + 2*Nx/3;
  int nshift = Nx-ntheta0;
  //fill the kx and ky arrays
  for(int ic=0; ic<nClasses; ic++) {
    int n = 0;
    int  p, c;
    for(int idx=0; idx<ntheta0; idx++) {
      for(int idy=0; idy<naky; idy++) {
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

void kPrint(int nClasses, int nLinks, int nChains, int *ky, int *kx, int c) {
  for(int n=0; n<nChains; n++) {
    for(int p=0; p<nLinks; p++) {
      printf("(%d,%d) ",ky[p+nLinks*n], kx[p+nLinks*n]);
    }
    printf("\n");
  }
  printf("\n\n");
}


          

