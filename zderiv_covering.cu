void ZDerivCovering(cufftComplex *result, cufftComplex* f, int** kx, int** ky, 
                    cuComplex** g_covering, float** kz_covering,
		    char* abs, cufftHandle* plan_covering)
{

  if(LINEAR && Zp!=1) {
    reality<<<dimGrid,dimBlock>>>(f);
    ZDeriv(result, f, kz); 
    reality<<<dimGrid,dimBlock>>>(result);
  }
  else {
	  reality<<<dimGrid,dimBlock>>>(f);
	  for(int c=0; c<nClasses; c++) {  
	      
	      // can use a separate stream for each class, do some classes at the same time. 
	      ZTransformCovering(nLinks[c], nChains[c], ky[c], kx[c],f,result,g_covering[c],kz_covering[c],abs, plan_covering[c], streams[c]);
	      
	  } 
	  cudaEventRecord(end_of_zderiv, 0);
	  cudaEventSynchronize(end_of_zderiv);   //wait until all streams are finished before going on
	 
	  scale<<<dimGrid,dimBlock>>>(result, result, gradpar);
	  
	  reality<<<dimGrid,dimBlock>>>(result); 	   
  }
} 


