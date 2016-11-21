inline void ZDerivCovering(cuComplex *result, cuComplex* f, grids_struct * grids_h, grids_struct * grids_hd, grids_struct * grids_d, 
		    char* abs, cufftHandle* plan_covering)
{

  int** kx = grids_hd->kxCover;
  int** ky = grids_hd->kyCover;
  int** kx_d = grids_hd->kxCover_d;
  int** ky_d = grids_hd->kyCover_d;
  cuComplex** g_covering = grids_hd->g_covering;
  cuComplex** g_covering_d = grids_hd->g_covering_d;
  float** kz_covering = grids_hd->kz_covering;
  float** kz_covering_d = grids_hd->kz_covering_d;
  int* nLinks = grids_h->nLinks;
  int* nChains = grids_h->nChains;
  int* nLinks_d = grids_hd->nLinks;
  int* nChains_d = grids_hd->nChains;

  float* covering_scaler_d = grids_hd->covering_scaler;

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

if(!zderiv_loop) {
#ifdef PROFILE
PUSH_RANGE("zderiv_covering_all",4);
#endif
          zeroCovering_all<<<dimGridCovering_all,dimBlockCovering>>>(g_covering_d, nLinks_d, nChains_d, icovering, nClasses);
          coveringCopy_all<<<dimGridCovering_all,dimBlockCovering>>>(g_covering_d, nLinks_d, nChains_d, ky_d, kx_d, f, icovering, nClasses);

          for(int c=0; c<nClasses; c++) {
              cufftExecC2C(plan_covering[c], g_covering[c], g_covering[c], CUFFT_FORWARD);
          }

          if(abs=="abs") {
            zderiv_abs_covering_all<<<dimGridCovering_all, dimBlockCovering>>> (g_covering_d, nLinks_d, nChains_d, kz_covering_d, icovering, nClasses);
          } else if(abs=="invert") {
            zderiv_covering_invert_all<<<dimGridCovering_all, dimBlockCovering>>> (g_covering_d, nLinks_d, nChains_d, kz_covering_d, icovering, nClasses);
          } else {
            zderiv_covering_all<<<dimGridCovering_all, dimBlockCovering>>> (g_covering_d, nLinks_d, nChains_d, kz_covering_d, icovering, nClasses);
          }  
        
          for(int c=0; c<nClasses; c++) {
              if(nLinks[c]==1) reality_covering<<<dimGridCovering[c],dimBlockCovering>>>(g_covering[c]);
              cufftExecC2C(plan_covering[c], g_covering[c], g_covering[c], CUFFT_INVERSE);
          }
              
          scale_covering_all<<<dimGridCovering_all,dimBlockCovering>>>(g_covering_d, nLinks_d, nChains_d, covering_scaler_d, nClasses);
     
          coveringCopyBack_all<<<dimGridCovering_all,dimBlockCovering>>>(result, nLinks_d, nChains_d, ky_d, kx_d, g_covering_d, nClasses);
#ifdef PROFILE
POP_RANGE;
#endif
}
else {
#ifdef PROFILE
PUSH_RANGE("zderiv_covering loop",5);
#endif
	  for(int c=0; c<nClasses; c++) {  
	      
	      // can use a separate stream for each class, do some classes at the same time. 
              // ^ This seems to make things slower! We will just use the default stream, 0.
              // It is slower to use multiple streams here because launch overhead is not hidden. 
	      ZTransformCovering(nLinks[c], nChains[c], ky[c], kx[c],f,result,g_covering[c],kz_covering[c],abs, 
                     plan_covering[c], 0, dimGridCovering[c], dimBlockCovering);

              //I think this stuff immediately below isn't right. If using multiple streams, leave this commented, uncomment stuff about streams outside loop

              ////Only need this stuff if using multiple streams...
	      ////cudaEventRecord(end_of_zderiv[c], zstreams[c]);
              ////cudaStreamWaitEvent(0, end_of_zderiv[c], 0); //make sure linear stream waits until each zstream is finished before going on
          }    
#ifdef PROFILE
POP_RANGE;
#endif
}
              //Only need this stuff if using multiple streams...
	      //cudaEventRecord(end_of_zderiv[0], 0);
              //cudaEventSynchronize(end_of_zderiv[0]);

          if(abs=="invert") {
	    scale<<<dimGrid,dimBlock>>>(result, result, 1./gradpar);
	  } else {
	    scale<<<dimGrid,dimBlock>>>(result, result, gradpar);
          }
          mask<<<dimGrid,dimBlock>>>(result);
	  reality<<<dimGrid,dimBlock>>>(result); 	   
  }
} 


