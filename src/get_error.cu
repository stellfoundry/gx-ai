#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include "cufft.h"
#include <stdio.h>
#include "standard_headers.h"
#include "get_error.h"

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

int getError(info_struct * info, char* message) {
//	printf("Getting error...\n");
  printf("\n%d (jid %d): %s: %s\n",info->gpuID, info->job_id, message, cudaGetErrorString(cudaGetLastError()));
  //if (cudaGetErrorString(cudaGetLastError()) == "no error")
    return cudaGetLastError();
  //else exit(1);
}     

int getError(info_struct * info, char* message, int i) {
//	printf("Getting error...\n");
  printf("\n%d (jid %d): %s: %d: %s\n",info->gpuID, info->job_id, message, i, cudaGetErrorString(cudaGetLastError()));
  //if (cudaGetErrorString(cudaGetLastError()) == "no error")
    return cudaGetLastError();
  //else exit(1);
}     
