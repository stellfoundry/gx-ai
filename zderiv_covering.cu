void ZDerivCovering(cufftComplex *result, cufftComplex* f, int** kx, int** ky, 
                    cuComplex** g_covering, float** kz_covering,
		    char* abs, cufftHandle* plan_covering)
{

  reality<<<dimGrid,dimBlock>>>(f);
  
  for(int c=0; c<nClasses; c++) {  
       
      ZTransformCovering(nLinks[c], nChains[c], ky[c], kx[c],f,result,g_covering[c],kz_covering[c],abs, plan_covering[c]);
  
  } 
  
  scale<<<dimGrid,dimBlock>>>(result, result, gradpar);
  
  reality<<<dimGrid,dimBlock>>>(result); 	   
 
} 


