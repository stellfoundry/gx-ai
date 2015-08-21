#include "cufft.h"
#include <stdio.h>

int getError() {
	//printf("Getting error...\n");
  printf("\n%s\n",cudaGetErrorString(cudaGetLastError()));
  //if (cudaGetErrorString(cudaGetLastError()) == "no error")
    return cudaGetLastError();
  //else exit(1);
}    

int getError(char* message) {
//	printf("Getting error...\n");
  printf("\n%s: %s\n",message, cudaGetErrorString(cudaGetLastError()));
  //if (cudaGetErrorString(cudaGetLastError()) == "no error")
    return cudaGetLastError();
  //else exit(1);
}     

int getError(char* message,int i) {
  printf("\n%d: %s: %s\n",i,message, cudaGetErrorString(cudaGetLastError()));
  //if (cudaGetErrorString(cudaGetLastError()) == "no error")
    return cudaGetLastError();
  //else exit(1);
}    
