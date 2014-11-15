inline void ZDerivCovering(cufftComplex *result, cufftComplex* f, int** kx, int** ky, 
                    cuComplex** g_covering, float** kz_covering,
		    char* abs, cufftHandle* plan_covering)
{

  if(NO_ZDERIV_COVERING) {
    zeroC<<<dimGrid,dimBlock>>>(result);
  }
  /*if(LINEAR && Zp!=1) {
    reality<<<dimGrid,dimBlock>>>(f);
    ZDeriv(result, f, kz); 
    reality<<<dimGrid,dimBlock>>>(result);
  }*/
  else {
	  reality<<<dimGrid,dimBlock>>>(f);
          mask<<<dimGrid,dimBlock>>>(f);
	  for(int c=0; c<nClasses; c++) {  
	      
	      // can use a separate stream for each class, do some classes at the same time. 
              // ^ This seems to make things slower! We will just use the default stream, 0.
	      ZTransformCovering(nLinks[c], nChains[c], ky[c], kx[c],f,result,g_covering[c],kz_covering[c],abs, plan_covering[c], 0, dimGridCovering[c], dimBlockCovering);

              //Only need this stuff if using multiple streams...
	      //cudaEventRecord(end_of_zderiv[c], zstreams[c]);
              //cudaStreamWaitEvent(0, end_of_zderiv[c], 0); //make sure linear stream waits until each zstream is finished before going on
          }    
              //Only need this stuff if using multiple streams...
	      //cudaEventRecord(end_of_zderiv[0], 0);
              //cudaEventSynchronize(end_of_zderiv[0]);

	  scale<<<dimGrid,dimBlock>>>(result, result, gradpar);
	  
          mask<<<dimGrid,dimBlock>>>(result);
	  reality<<<dimGrid,dimBlock>>>(result); 	   
  }
} 


