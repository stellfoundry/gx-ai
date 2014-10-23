inline void fieldWrite(cuComplex* f_d, cuComplex* f_h, char* ext, char* filename)
{
  strcpy(filename,out_stem);
  strcat(filename,ext);
  FILE* out = fopen(filename,"w+");
  cudaMemcpy(f_h,f_d,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToHost);
  fprintf(out, "#\tz (1)\t\t\tky (2)\t\t\tkx (3)\t\t\tRe (4)\t\t\tIm (5)\t\t\t");  
  fprintf(out, "\n");
  int blockid = 0;
  for(int i=0; i<(Nx-1)/3+1; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      fprintf(out, "\n#%d\n\n", blockid);
      blockid++;      
      for(int k=0; k<=Nz; k++) {
        int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
	//if(index!=0){
	  if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -z_h[0], ky_h[j], kx_h[i], f_h[j+(Ny/2+1)*i].x, f_h[j+(Ny/2+1)*i].y); 
	  else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", z_h[k], ky_h[j], kx_h[i], f_h[index].x, f_h[index].y);    	  
        //}
      }     
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      fprintf(out, "\n#%d\n\n", blockid);
      blockid++;
      for(int k=0; k<=Nz; k++) {
        int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
	if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -z_h[0], ky_h[j], kx_h[i], f_h[j+(Ny/2+1)*i].x, f_h[j+(Ny/2+1)*i].y); 
	else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", z_h[k], ky_h[j], kx_h[i], f_h[index].x, f_h[index].y);    	  
      }    
    }
  }
  fclose(out);
}

inline void fieldWrite(cuComplex* f_d, cuComplex* f_h, char* ext, char* filename, int Nx, int Ny, int Nz)
{
  strcpy(filename,out_stem);
  strcat(filename,ext);
  FILE* out = fopen(filename,"w+");
  cudaMemcpy(f_h,f_d,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,cudaMemcpyDeviceToHost);
  fprintf(out, "#\tz (1)\t\t\tky (2)\t\t\tkx (3)\t\t\tRe (4)\t\t\tIm (5)\t\t\t");  
  fprintf(out, "\n");
  int blockid = 0;
  for(int i=0; i<(Nx-1)/3+1; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      fprintf(out, "\n#%d\n\n", blockid);
      blockid++;      
      for(int k=0; k<=Nz; k++) {
        int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
	//if(index!=0){
	  if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -z_h[0], ky_h[j], kx_h[i], f_h[j+(Ny/2+1)*i].x, f_h[j+(Ny/2+1)*i].y); 
	  else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", z_h[k], ky_h[j], kx_h[i], f_h[index].x, f_h[index].y);    	  
        //}
      }     
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      fprintf(out, "\n#%d\n\n", blockid);
      blockid++;
      for(int k=0; k<=Nz; k++) {
        int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
	if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -z_h[0], ky_h[j], kx_h[i], f_h[j+(Ny/2+1)*i].x, f_h[j+(Ny/2+1)*i].y); 
	else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", z_h[k], ky_h[j], kx_h[i], f_h[index].x, f_h[index].y);    	  
      }    
    }
  }
  fclose(out);
}

inline void fieldWrite_nopad(cuComplex* f_nopad_d, cuComplex* f_nopad_h, char* ext, char* filename, int Nx, int Ny, int Nz, int ntheta0, int naky)
{
  strcpy(filename,out_stem);
  strcat(filename,ext);
  FILE* out = fopen(filename,"w+");
  cudaMemcpy(f_nopad_h,f_nopad_d,sizeof(cuComplex)*ntheta0*(naky)*Nz,cudaMemcpyDeviceToHost);
  fprintf(out, "#\tz (1)\t\t\tky (2)\t\t\tkx (3)\t\t\tRe (4)\t\t\tIm (5)\t\t\t");  
  fprintf(out, "\n");
  int blockid = 0;
  for(int i=0; i<(Nx-1)/3+1; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      fprintf(out, "\n#%d\n\n", blockid);
      blockid++;      
      for(int k=0; k<=Nz; k++) {
        int index = j+(naky)*i+(naky)*ntheta0*k;
	//if(index!=0){
	  if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -z_h[0], ky_h[j], kx_h[i], f_nopad_h[j+(naky)*i].x, f_nopad_h[j+(naky)*i].y); 
	  else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", z_h[k], ky_h[j], kx_h[i], f_nopad_h[index].x, f_nopad_h[index].y);    	  
        //}
      }     
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      fprintf(out, "\n#%d\n\n", blockid);
      blockid++;
      for(int k=0; k<=Nz; k++) {
        int index = j+(naky)*(i-Nx+ntheta0)+(naky)*ntheta0*k;
	if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -z_h[0], ky_h[j], kx_h[i], f_nopad_h[j+(naky)*(i-Nx+ntheta0)].x, f_nopad_h[j+(naky)*(i-Nx+ntheta0)].y); 
	else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", z_h[k], ky_h[j], kx_h[i], f_nopad_h[index].x, f_nopad_h[index].y);    	  
      }    
    }
  }
  fclose(out);
}
inline void fieldWrite_nopad_h(cuComplex* f_nopad_h, char* ext, char* filename, int Nx, int Ny, int Nz, int ntheta0, int naky)
{
  strcpy(filename,out_stem);
  strcat(filename,ext);
  FILE* out = fopen(filename,"w+");
  fprintf(out, "#\tz (1)\t\t\tky (2)\t\t\tkx (3)\t\t\tRe (4)\t\t\tIm (5)\t\t\t");  
  fprintf(out, "\n");
  int blockid = 0;
  for(int i=0; i<(Nx-1)/3+1; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      fprintf(out, "\n#%d\n\n", blockid);
      blockid++;      
      for(int k=0; k<=Nz; k++) {
        int index = j+(naky)*i+(naky)*ntheta0*k;
	//if(index!=0){
	  if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -z_h[0], ky_h[j], kx_h[i], f_nopad_h[j+(naky)*i].x, f_nopad_h[j+(naky)*i].y); 
	  else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", z_h[k], ky_h[j], kx_h[i], f_nopad_h[index].x, f_nopad_h[index].y);    	  
        //}
      }     
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      fprintf(out, "\n#%d\n\n", blockid);
      blockid++;
      for(int k=0; k<=Nz; k++) {
        int index = j+(naky)*(i-Nx+ntheta0)+(naky)*ntheta0*k;
	if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -z_h[0], ky_h[j], kx_h[i], f_nopad_h[j+(naky)*(i-Nx+ntheta0)].x, f_nopad_h[j+(naky)*(i-Nx+ntheta0)].y); 
	else fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", z_h[k], ky_h[j], kx_h[i], f_nopad_h[index].x, f_nopad_h[index].y);    	  
      }    
    }
  }
  fclose(out);
}


inline void fieldWriteXY(float* f_d, char* fieldname, float dt)
{
  char filename[200];  
  sprintf(filename, "fields/%s/%s%g",fieldname,fieldname,dt);
  FILE* out = fopen(filename,"w+");
  float *f_h;
  f_h = (float*) malloc(sizeof(float)*Nx*(Ny/2+1));
  cudaMemcpy(f_h,f_d,sizeof(float)*Nx*(Ny/2+1),cudaMemcpyDeviceToHost);
  printf("fieldWrite %s%g\n\n", fieldname,dt);
  fprintf(out, "#\tz\t\t\tky\t\t\tkx\t\t\tRe(%s)\t\t\tIm(%s)\t\t\t",fieldname,fieldname);  
  fprintf(out, "\n");
  for(int i=0; i<(Nx-1)/3+1; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      
        int index = j+(Ny/2+1)*i;
	//if(index!=0){
	  fprintf(out, "\t%f\t\t%f\t\t%e\t\n", ky_h[j], kx_h[i], f_h[index]);    	  
        //}
      
            
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<(Ny-1)/3+1; j++) {
      for(int k=0; k<=Nz; k++) {
        int index = j+(Ny/2+1)*i;
	fprintf(out, "\t%f\t\t%f\t\t%e\t\n", ky_h[j], kx_h[i], f_h[index]);    	  
      }
           
    }
  }
  free(f_h);
  fclose(out);
}

inline void fieldWriteCovering(cuComplex* f_d, char* filename,int** kxCover,int** kyCover, int** kxCover_h, int** kyCover_h)
{ 
  for(int c=0; c<nClasses; c++) { 
    sprintf(filename, "%sphi_covering_nperiod%d.field",out_stem,nLinks[c]);
    FILE* out = fopen(filename,"w+"); 
    cuComplex *g_h;
    g_h = (cuComplex*) malloc(sizeof(cuComplex)*Nz*icovering*nLinks[c]*nChains[c]);
    cuComplex* g_d;
    cudaMalloc((inline void**) &g_d, sizeof(cuComplex)*(Nz*icovering*nLinks[c]*nChains[c]));
    int xy = totalThreads/nLinks[c];
    int blockxy = (int) sqrt(xy);  
    dim3 dimBlockCovering(blockxy,blockxy,nLinks[c]);
    if(nLinks[c]>zThreads) {
      dimBlockCovering.x = (int) sqrt(totalThreads/zThreads);
      dimBlockCovering.y = (int) sqrt(totalThreads/zThreads);
      dimBlockCovering.z = zThreads;
    }    
    dim3 dimGridCovering(Nz/dimBlockCovering.x+1,nChains[c]/dimBlockCovering.y+1,1);
    zeroCovering<<<dimGridCovering,dimBlockCovering>>>(g_d, nLinks[c], nChains[c], icovering);
    
    coveringCopy<<<dimGridCovering,dimBlockCovering>>> (g_d, nLinks[c], nChains[c], kyCover[c], kxCover[c], f_d, icovering);

    //normalize_covering<<<dimGridCovering, dimBlockCovering>>> (g_d, g_d, 1., nLinks[c], nChains[c]);
     
    cudaMemcpy(g_h,g_d,sizeof(cuComplex)*icovering*Nz*nLinks[c]*nChains[c], cudaMemcpyDeviceToHost);
    
    fprintf(out, "#\tz (1)\t\t\tky (2)\t\t\tkx (3)\t\t\tRe (4)\t\t\tIm (5)\t\t\t");  
    fprintf(out, "\n");
    int blockid = 0;
    for(int n=0; n<nChains[c]; n++) {
      fprintf(out, "\n#%d\n\n", blockid);
      blockid++;
      for(int j=0; j<icovering; j++) {
      for(int p=0; p<nLinks[c]; p++) {
        for(int i=0; i<Nz; i++) {
	  int zidx = i + p*Nz + j*nLinks[c]*Nz;
	  int kidx;
	  if(j==0) kidx = p + nLinks[c]*n;
	  if(j==1) kidx = (nLinks[c] - p - 1) + nLinks[c]*n;
	  int index = i + p*Nz + j*Nz*nLinks[c] + n*Nz*nLinks[c]*icovering;	  
	  float z_cover = 2*M_PI*(zidx-(Nz*icovering*nLinks[c])/2)/(Nz);	  
	  fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", z_cover, ky_h[ kyCover_h[c][kidx] ], kx_h[ kxCover_h[c][kidx] ], g_h[index].x, g_h[index].y); 	  
	}
      }
      }
      //periodic point   
      int zidx = 0;   
      int kidx = nLinks[c]*n;
      int index = n*Nz*nLinks[c];
      float z_cover = 2*M_PI*(zidx-(Nz*icovering*nLinks[c])/2)/(Nz);
      fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -z_cover, ky_h[ kyCover_h[c][kidx] ], kx_h[ kxCover_h[c][kidx] ], g_h[index].x, g_h[index].y);
      
    }
    cudaFree(g_d);
    free(g_h);
    fclose(out);
  }  
}

//only a certain class
inline void fieldWriteCovering(cuComplex* f_d, char* fieldname, float dt,int** kxCover,int** kyCover, int** kxCover_h, int** kyCover_h, int C)
{ 
  for(int c=C; c<C+1; c++) {
    char filename[200];  
    sprintf(filename, "fields/%s_covering/%s_covering%g_c%d",fieldname,fieldname,dt,c);
    FILE* out = fopen(filename,"w+"); 
    cuComplex *g_h;
    g_h = (cuComplex*) malloc(sizeof(cuComplex)*Nz*nLinks[c]*nChains[c]);
    cuComplex* g_d;
    cudaMalloc((inline void**) &g_d, sizeof(cuComplex)*(Nz*nLinks[c]*nChains[c]));
    int xy = totalThreads/nLinks[c];
    int blockxy = (int) sqrt(xy);  
    dim3 dimBlockCovering(blockxy,blockxy,nLinks[c]);
    if(nLinks[c]>zThreads) {
      dimBlockCovering.x = (int) sqrt(totalThreads/zThreads);
      dimBlockCovering.y = (int) sqrt(totalThreads/zThreads);
      dimBlockCovering.z = zThreads;
    }    
    dim3 dimGridCovering(Nz/dimBlockCovering.x+1,nChains[c]/dimBlockCovering.y+1,1);
    zeroCovering<<<dimGridCovering,dimBlockCovering>>>(g_d, nLinks[c], nChains[c],icovering);
    
    coveringCopy<<<dimGridCovering,dimBlockCovering>>> (g_d, nLinks[c], nChains[c], kyCover[c], kxCover[c], f_d,icovering);
    
    cudaMemcpy(g_h,g_d,sizeof(cuComplex)*Nz*nLinks[c]*nChains[c], cudaMemcpyDeviceToHost);
    
    printf("fieldWrite %s_covering%g_c%d\n\n", fieldname,dt,c);
    fprintf(out, "#\tz\t\t\tky\t\t\tkx\t\t\tRe(%s_covering)\t\t\tIm(%s_covering)\t\t\t",fieldname,fieldname);  
    fprintf(out, "\n");
    
    for(int n=0; n<nChains[c]; n++) {
      for(int p=0; p<nLinks[c]; p++) {
        for(int i=0; i<Nz; i++) {
	  int zidx = i + p*Nz;
	  int kidx = p + nLinks[c]*n;
	  int index = i + p*Nz + n*Nz*nLinks[c];	  
	  float z_cover = nLinks[c]*2*M_PI*(zidx-(Nz*nLinks[c])/2)/(nLinks[c]*Nz);	  
	  fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", z_cover, ky_h[ kyCover_h[c][kidx] ], kx_h[ kxCover_h[c][kidx] ], g_h[index].x, g_h[index].y); 	  
	}
      }
      //periodic point   
      int zidx = 0;   
      int kidx = nLinks[c]*n;
      int index = n*Nz*nLinks[c];
      float z_cover = nLinks[c]*2*M_PI*(zidx-(Nz*nLinks[c])/2)/(nLinks[c]*Nz);
      fprintf(out, "\t%f\t\t%f\t\t%f\t\t%e\t\t%e\t\n", -z_cover, ky_h[ kyCover_h[c][kidx] ], kx_h[ kxCover_h[c][kidx] ], g_h[index].x, g_h[index].y);
      fprintf(out,"\n\n");
    }
    cudaFree(g_d);
    free(g_h);
    fclose(out);
  }  
}

inline void phiWriteSetup(FILE* out)
{
  fprintf(out, "y\t\tx\t\tPhi(z=0)\n");
}

//save Phi(x,y,z=0)
inline void phiR_historyWrite(cuComplex* Phi, cuComplex* Phi_XYz0, float* PhiR_XYz0, float* PhiR_XYz0_h, 
			float runtime, FILE* phifile)
{
  get_z0<<<dimGrid,dimBlock>>>(Phi_XYz0, Phi);
  cufftExecC2R(XYplanC2R, Phi_XYz0, PhiR_XYz0);
  cudaMemcpy(PhiR_XYz0_h, PhiR_XYz0, sizeof(float)*Nx*Ny, cudaMemcpyDeviceToHost);
  fprintf(phifile,"t=%f\n", runtime);
  for(int j=0; j<=Nx; j++) {
    for(int i=0; i<=Ny; i++) {
      int index = i + Ny*j;
      if(j==Nx) index = i;
      if(i==Ny) index = Ny*j;
      if(i==Ny && j==Nx) index= 0;
      fprintf(phifile,"%.6f\t%.6f\t%f\n", 2*Y0*(i-Ny/2)/Ny, 2*X0*(j-Nx/2)/Nx, PhiR_XYz0_h[index]);
    }
    fprintf(phifile,"\n");
  }
  fprintf(phifile,"\n\n");
}


inline void boxAvg(cuComplex *fAvg, cuComplex *f, cuComplex **fBox, 
                  float dt, float *dtBox, int navg, int counter)
{
  float dtBoxSum;
  float dtBoxSumInv;
  
  zeroC<<<dimGrid,dimBlock>>>(fAvg,Nx,Ny,1);
  dtBoxSum = 0;
  
  if(counter<navg) {
    scale<<<dimGrid,dimBlock>>>(fBox[counter],f,dt,Nx,Ny,1);
    dtBox[counter] = dt;
    for(int t=0; t<counter+1; t++) {
      accum<<<dimGrid,dimBlock>>>(fAvg,fBox[t],1,Nx,Ny,1);
      dtBoxSum += dtBox[t];
    }
    if(dtBoxSum !=0) dtBoxSumInv = 1./dtBoxSum;
    else dtBoxSumInv = 1;
    scale<<<dimGrid,dimBlock>>>(fAvg,fAvg,dtBoxSumInv,Nx,Ny,1);
  }
  else {
    scale<<<dimGrid,dimBlock>>>(fBox[counter%navg],f,dt,Nx,Ny,1);
    dtBox[counter%navg] = dt;

    for(int t=0; t<navg; t++) {
      accum<<<dimGrid,dimBlock>>>(fAvg,fBox[t],1,Nx,Ny,1);
      dtBoxSum += dtBox[t];
    }
    if(dtBoxSum !=0) dtBoxSumInv = 1./dtBoxSum;
    else dtBoxSumInv = 1;
    scale<<<dimGrid,dimBlock>>>(fAvg,fAvg,dtBoxSumInv,Nx,Ny,1);
  }  
}

inline void boxAvg(float *fAvg, float *f, float **fBox, 
                  float dt, float *dtBox, int navg, int counter, int Nx, int Ny, int Nz)
{
  float dtBoxSum;
  float dtBoxSumInv;
  
  
  dtBoxSum = 0;
  
  if(counter<navg) {
    scaleReal<<<dimGrid,dimBlock>>>(fBox[counter],f,dt,Nx,Ny,Nz);
    zero<<<dimGrid,dimBlock>>>(fAvg,Nx,Ny,Nz); //fAvg and f can be same array if we wait to zero fAvg until here
    dtBox[counter] = dt;
    for(int t=0; t<counter+1; t++) {
      accum<<<dimGrid,dimBlock>>>(fAvg,fBox[t],1,Nx,Ny,Nz);
      dtBoxSum += dtBox[t];
    }
    if(dtBoxSum !=0) dtBoxSumInv = 1./dtBoxSum;
    else dtBoxSumInv = 1;
    scaleReal<<<dimGrid,dimBlock>>>(fAvg,fAvg,dtBoxSumInv,Nx,Ny,Nz);
  }
  else {
    scaleReal<<<dimGrid,dimBlock>>>(fBox[counter%navg],f,dt,Nx,Ny,Nz);
    zero<<<dimGrid,dimBlock>>>(fAvg,Nx,Ny,Nz); //fAvg and f can be same array if we wait to zero fAvg until here
    dtBox[counter%navg] = dt;

    for(int t=0; t<navg; t++) {
      accum<<<dimGrid,dimBlock>>>(fAvg,fBox[t],1,Nx,Ny,Nz);
      dtBoxSum += dtBox[t];
    }
    if(dtBoxSum !=0) dtBoxSumInv = 1./dtBoxSum;
    else dtBoxSumInv = 1;
    scaleReal<<<dimGrid,dimBlock>>>(fAvg,fAvg,dtBoxSumInv,Nx,Ny,Nz);
  }  
}

// scaler version of above
float boxAvg(float f, float* fBox, float dt, float* dtBox, int navg, int counter)
{
  float fAvg = 0;
  float dtBoxSum = 0;
  float dtBoxSumInv;
  
  if(counter<navg) {
    //scale<<<dimGrid,dimBlock>>>(fBox[counter],f,dt,Nx,Ny,1);
    fBox[counter] = f*dt;
    dtBox[counter] = dt;
    for(int t=0; t<counter+1; t++) {
      //accum<<<dimGrid,dimBlock>>>(fAvg,fBox[t],1,Nx,Ny,1);
      fAvg += fBox[t];
      dtBoxSum += dtBox[t];
    }
    if(dtBoxSum !=0) dtBoxSumInv = 1./dtBoxSum;
    else dtBoxSumInv = 1;
    //scale<<<dimGrid,dimBlock>>>(fAvg,fAvg,dtBoxSumInv,Nx,Ny,1);
    fAvg = fAvg*dtBoxSumInv; 
    return f;
  }
  else {
    //scale<<<dimGrid,dimBlock>>>(fBox[counter%navg],f,dt,Nx,Ny,1);
    fBox[counter%navg] = f*dt;
    dtBox[counter%navg] = dt;

    for(int t=0; t<navg; t++) {
      //accum<<<dimGrid,dimBlock>>>(fAvg,fBox[t],1,Nx,Ny,1);
      fAvg += fBox[t];
      dtBoxSum += dtBox[t];
    }
    if(dtBoxSum !=0) dtBoxSumInv = 1./dtBoxSum;
    else dtBoxSumInv = 1;
    //scale<<<dimGrid,dimBlock>>>(fAvg,fAvg,dtBoxSumInv,Nx,Ny,1);
    fAvg = fAvg*dtBoxSumInv;
    
    return fAvg;
  }    
}
  
//fill boxes for box averaging, but don't average.  
inline void boxFill(float *f, float **fBox, 
        float dt, float *dtBox, int navg, int counter, int Nx, int Ny, int Nz)
{
  scaleReal<<<dimGrid,dimBlock>>>(fBox[counter%navg],f,dt,Nx,Ny/2+1,1);
  dtBox[counter%navg] = dt;  
}

//average over box that's already been filled by boxFill()
inline void boxAvg_filled(float *fAvg, float **fBox, 
                  float *dtBox, int navg, int counter, int Nx, int Ny, int Nz)
{
  //sum boxes, then divide
  float dtBoxSum;
  float dtBoxSumInv;
  
  
  dtBoxSum = 0;
  
  if(counter<navg) {
    for(int t=0; t<counter+1; t++) {
      accum<<<dimGrid,dimBlock>>>(fAvg,fBox[t],1,Nx,Ny,Nz);
      dtBoxSum += dtBox[t];
    }
    if(dtBoxSum !=0) dtBoxSumInv = 1./dtBoxSum;
    else dtBoxSumInv = 1;
    scaleReal<<<dimGrid,dimBlock>>>(fAvg,fAvg,dtBoxSumInv,Nx,Ny,Nz);
  }
  else {
    for(int t=0; t<navg; t++) {
      accum<<<dimGrid,dimBlock>>>(fAvg,fBox[t],1,Nx,Ny,Nz);
      dtBoxSum += dtBox[t];
    }
    if(dtBoxSum !=0) dtBoxSumInv = 1./dtBoxSum;
    else dtBoxSumInv = 1;
    scaleReal<<<dimGrid,dimBlock>>>(fAvg,fAvg,dtBoxSumInv,Nx,Ny,Nz);
  }  
}
  
//sums over kx,ky by breaking into kx,ky array for each z and reducing. outputs as function of z.
inline void kxkySum(float* sum_tmpZ, cuComplex* f_tmp, float* f_tmpXY)
{
  
  //fixFFT<<<dimGrid,dimBlock>>>(f_tmp);
  
  float sum_i;
  
  for(int i=0; i<Nz; i++) {
    //for each z(i), copy f(kx,ky,z(i)) into f_tmpXY(kx,ky)
    zcopy<<<dimGrid,dimBlock>>>(f_tmpXY, f_tmp, i);
    sum_i = sumReduc(f_tmpXY, Nx*(Ny/2+1), false);  
    //assign sum_i to ith element of sum_tmpZ. (without copying)  
    assign<<<1,1>>>(sum_tmpZ, sum_i, i);
  }

}

//incredibly slow!!
inline void zSum(float* sum_tmpXY, cuComplex* f_tmp, float* f_tmpZ)
{
  float sum_i;
  
  for(int i=0; i<Nx*(Ny/2+1); i++) {
    kxkycopy<<<1,Nz>>>(f_tmpZ,f_tmp,i);
    sum_i = sumReduc(f_tmpZ, Nz, false);
    assign<<<1,1>>>(sum_tmpXY, sum_i, i);
  }  
  
}


//calculate field line average of f(kx,ky,z)
inline void volflux(cuComplex* f, cuComplex* g, cuComplex* tmp, float* flux_tmpXY)
{
  
  /*
  calcReIm<<<dimGrid,dimBlock>>>(tmp,f,g);
  // ReIm = Re(f)*Re(g) + Im(f)*Im(g) = (f)(g*)
  
  //fixFFT<<<dimGrid,dimBlock>>>(ReIm_tmp);
  //reality<<<dimGrid,dimBlock>>>(ReIm_tmp);

  
  multZ<<<dimGrid,dimBlock>>>(tmp, tmp, jacobian);
  
  //sum over z: sum_z[ReIm(kx,ky,z)] = sum(kx,ky)
  sumZ<<<dimGrid,dimBlock>>>(flux_tmpXY, tmp);
  
  scaleReal<<<dimGrid,dimBlock>>>(flux_tmpXY, flux_tmpXY, 1./fluxDen, Nx, Ny/2+1, 1);
  */    
  
  
  volflux<<<dimGrid,dimBlock>>>(flux_tmpXY, f, g, jacobian, 1./fluxDen);
  //volflux_part2<<<dimGrid,dimBlock>>>(flux_tmpXY, tmp, 1./fluxDen);
  

  
}

inline void volflux_zonal(cuComplex* f, cuComplex* g, float* flux_tmpX) 
{
  volflux_zonal<<<dimGrid,dimBlock>>>(flux_tmpX, f, g, jacobian, 1./(fluxDen*fluxDen));
}

inline void rms(float *A_rms, cuComplex* A, float* A_fsa_tmpXY)
{
  volflux<<<dimGrid,dimBlock>>>(A_fsa_tmpXY, A, A, jacobian, 1./fluxDen);
  float A2_rms = sumReduc(A_fsa_tmpXY, Nx*(Ny/2+1), false);
  *A_rms = sqrt(A2_rms);
}

inline void phase_angle(float *phase, cuComplex* A, cuComplex* B, float* tmpXY)
{
  float A_rms;
  float B_rms;

  rms(&A_rms, A, tmpXY);
  rms(&B_rms, B, tmpXY);
  
  volflux<<<dimGrid, dimBlock>>>(tmpXY,A,B, jacobian, 1./fluxDen);
  float AB_fsa = sumReduc(tmpXY, Nx*(Ny/2+1), false);
  *phase = AB_fsa / (A_rms*B_rms);
}   

inline void fluxes(float *pflux, float *qflux, float qflux1, float qflux2, cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp, cuComplex* Phi, 
            cuComplex* phi_tmp, cuComplex* vPhi_tmp, cuComplex* tmp, cuComplex* totPr_field, cuComplex* Pprp_field, cuComplex* nbar_field,
            float* tmpZ, float* tmpXY, specie s, float runtime, 
            float *qflux1_phase, float *qflux2_phase, float *Dens_phase, float *Tpar_phase, float *Tprp_phase)
{   
  
  add_scaled<<<dimGrid,dimBlock>>>(totPr_field, 1., Tprp, .5, Tpar, 1.5, Dens);
  phi_u<<<dimGrid,dimBlock>>>(phi_tmp,Phi,s.rho,kx,ky,shat,gds2,gds21,gds22,bmagInv);
  iOmegaStar<<<dimGrid,dimBlock>>>(vPhi_tmp,phi_tmp,ky); 
  mask<<<dimGrid,dimBlock>>>(vPhi_tmp);
  mask<<<dimGrid,dimBlock>>>(totPr_field);
  volflux(vPhi_tmp, totPr_field, tmp, tmpXY);
  qflux1 = sumReduc(tmpXY, Nx*(Ny/2+1), false);
 /* //if(write_phase) {  
    float vPhi_rms;
    float totPr_rms;
    float Dens_rms;
    float Tpar_rms;
    float Tprp_rms;
    float Dens_flux;
    float Tpar_flux;
    float Tprp_flux;

    rms(&vPhi_rms, vPhi_tmp, tmpXY);
    rms(&totPr_rms, totPr_field, tmpXY);

    *qflux1_phase = qflux1/(vPhi_rms*totPr_rms);

    volflux<<<dimGrid,dimBlock>>>(tmpXY,vPhi_tmp,Dens,jacobian,1./fluxDen);
    Dens_flux = sumReduc(tmpXY, Nx*(Ny/2+1), false);
    volflux<<<dimGrid,dimBlock>>>(tmpXY,vPhi_tmp,Tpar,jacobian,1./fluxDen);
    Tpar_flux = sumReduc(tmpXY, Nx*(Ny/2+1), false); 
    volflux<<<dimGrid,dimBlock>>>(tmpXY,vPhi_tmp,Tprp,jacobian, 1./fluxDen);
    Tprp_flux = sumReduc(tmpXY, Nx*(Ny/2+1), false);
    rms(&Dens_rms, Dens, tmpXY);
    rms(&Tpar_rms, Tpar, tmpXY);
    rms(&Tprp_rms, Tprp, tmpXY);

    *Dens_phase = Dens_flux /(vPhi_rms*Dens_rms);
    *Tpar_phase = Tpar_flux / (vPhi_rms*Tpar_rms);
    *Tprp_phase = Tprp_flux / (vPhi_rms*Tprp_rms);    
  //}
  */
  add_scaled<<<dimGrid,dimBlock>>>(Pprp_field, 1., Dens, 1., Tprp);
  phi_flr<<<dimGrid,dimBlock>>>(phi_tmp,Phi,s.rho,kx,ky,shat,gds2,gds21,gds22,bmagInv);
  iOmegaStar<<<dimGrid,dimBlock>>>(vPhi_tmp,phi_tmp,ky);
  mask<<<dimGrid,dimBlock>>>(vPhi_tmp);
  mask<<<dimGrid,dimBlock>>>(Pprp_field);
  volflux(vPhi_tmp, Pprp_field, tmp, tmpXY);
  qflux2 = sumReduc(tmpXY, Nx*(Ny/2+1), false);
  /*//if(write_phase) {  
    float Pprp_rms;

    rms(&vPhi_rms, vPhi_tmp, tmpXY);
    rms(&Pprp_rms, Pprp_field, tmpXY);
    *qflux2_phase = qflux2/(vPhi_rms*Pprp_rms);
  //}
  */
  *qflux = -(qflux1+qflux2)*s.dens*s.temp;
    
  //wpfx[s+nSpecies*time] = (qflux1+qflux2) * n[s] * temp[s];

  //calculate particle flux
  //nbar<<<dimGrid,dimBlock>>>(nbar_field, Dens, Tprp, Phi,s.rho,kx,ky,shat,gds2,gds21,gds22,bmagInv);
  iOmegaStar<<<dimGrid,dimBlock>>>(vPhi_tmp, Phi, ky);
  mask<<<dimGrid,dimBlock>>>(nbar_field);
  mask<<<dimGrid,dimBlock>>>(vPhi_tmp);
  volflux(vPhi_tmp, nbar_field, tmp, tmpXY);
  float pflux_tmp = sumReduc(tmpXY, Nx*(Ny/2+1), false);
  *pflux = pflux_tmp*s.dens;


}

//outputs as function of ky
inline void fluxes_k(float* wpfx_tmpY, float *flux1_tmpY,float *flux2_tmpY2,
	    float* wpfx_tmpXY, float* flux1_tmpXY, float* flux2_tmpXY2,
	    cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp, cuComplex* Phi, 
            cuComplex* phi_tmp, cuComplex* vPhi_tmp, cuComplex* tmp, cuComplex* totPr_field, 
	    cuComplex* Pprp_field, float* tmpZ, specie s)
{  
  
  
  add_scaled<<<dimGrid,dimBlock>>>(totPr_field, 1., Tprp, .5, Tpar, 1.5, Dens);
  phi_u<<<dimGrid,dimBlock>>>(phi_tmp,Phi,s.rho,kx,ky,shat,gds2,gds21,gds22,bmagInv);
  iOmegaStar<<<dimGrid,dimBlock>>>(vPhi_tmp,phi_tmp,ky);  
  mask<<<dimGrid,dimBlock>>>(vPhi_tmp);
  mask<<<dimGrid,dimBlock>>>(totPr_field);
  volflux(vPhi_tmp, totPr_field, tmp, flux1_tmpXY);
  sumX<<<dimGrid,dimBlock>>>(flux1_tmpY, flux1_tmpXY);
  
  
  add_scaled<<<dimGrid,dimBlock>>>(Pprp_field, 1., Dens, 1., Tprp);
  phi_flr<<<dimGrid,dimBlock>>>(phi_tmp,Phi,s.rho,kx,ky,shat,gds2,gds21,gds22,bmagInv);
  iOmegaStar<<<dimGrid,dimBlock>>>(vPhi_tmp,phi_tmp,ky);
  mask<<<dimGrid,dimBlock>>>(vPhi_tmp);
  mask<<<dimGrid,dimBlock>>>(Pprp_field);
  volflux(vPhi_tmp,Pprp_field, tmp, flux2_tmpXY2);
  sumX<<<dimGrid,dimBlock>>>(flux2_tmpY2, flux2_tmpXY2);
  
  add_scaled<<<dimGrid,dimBlock>>>(wpfx_tmpXY, s.dens*s.temp, flux1_tmpXY, 
  					       s.dens*s.temp, flux2_tmpXY2, Nx, Ny, 1);
  
  add_scaled<<<dimGrid,dimBlock>>>(wpfx_tmpY, s.dens*s.temp, flux1_tmpY, 
                                              s.dens*s.temp, flux2_tmpY2, 1, Ny, 1);
    
}

// outputs as function of kx and ky
inline void fluxes_kxky(float* flux_tmpXY, float *flux1_tmpXY,float *flux2_tmpXY2, cuComplex* Dens, cuComplex* Tpar, cuComplex* Tprp, cuComplex* Phi, 
            cuComplex* phi_tmp, cuComplex* vPhi_tmp, cuComplex* tmp, cuComplex* totPr_field, 
	    cuComplex* Pprp_field, float* tmpZ, specie s)
{  
  
  add_scaled<<<dimGrid,dimBlock>>>(totPr_field, 1., Tprp, .5, Tpar, 1.5, Dens);
  phi_u<<<dimGrid,dimBlock>>>(phi_tmp,Phi,s.rho,kx,ky,shat,gds2,gds21,gds22,bmagInv);
  iOmegaStar<<<dimGrid,dimBlock>>>(vPhi_tmp,phi_tmp,ky);  
  volflux(vPhi_tmp, totPr_field, tmp, flux1_tmpXY);
  
  
  
  add_scaled<<<dimGrid,dimBlock>>>(Pprp_field, 1., Dens, 1., Tprp);
  phi_flr<<<dimGrid,dimBlock>>>(phi_tmp,Phi,s.rho,kx,ky,shat,gds2,gds21,gds22,bmagInv);
  iOmegaStar<<<dimGrid,dimBlock>>>(vPhi_tmp,phi_tmp,ky);
  volflux(vPhi_tmp,Pprp_field, tmp, flux2_tmpXY2);

  add_scaled<<<dimGrid,dimBlock>>>(flux_tmpXY, s.dens*s.temp, flux1_tmpXY, 
  						s.dens*s.temp, flux2_tmpXY2, Nx, Ny, 1);
    
  //wpfx[s+nSpecies*time] = (flux1+flux2) * n[s] * tau[s];
}


inline void omegaWriteSetup(FILE* ofile, char* w)
{
  fprintf(ofile, "#\ttime(s)\t");
  int col = 2;
  for(int i=0; i<((Nx-1)/3+1); i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {      
      if(!(i==0 && j==0)) {
        fprintf(ofile, "\t\t\t%d:(%.3g,%.3g)", col, ky_h[j],kx_h[i]);
        col++;
      }
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      fprintf(ofile, "\t\t\t%d:(%.3g,%.3g)", col, ky_h[j],kx_h[i]);
      col++;
    }
  }
  
  fprintf(ofile, "\n");
}

//time history of growth rates
inline void omegaWrite(FILE* omegafile, FILE* gammafile, cuComplex* omega,float time)
{
  fprintf(omegafile, "\t%f", time);
  fprintf(gammafile, "\t%f", time);
  for(int i=0; i<((Nx-1)/3+1); i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j+(Ny/2+1)*i;
      if(index!=0) {
	if( isnan(omega[index].x) || isinf(omega[index].x) ) 
	  fprintf(omegafile, "\t\t\t%f\t", omega[index].x); 
	else
	  fprintf(omegafile, "\t\t\t%f", omega[index].x); 
	if( isnan(omega[index].y) || isinf(omega[index].y) )
	  fprintf(gammafile, "\t\t\t%f\t", omega[index].y); 
	else
	  fprintf(gammafile, "\t\t\t%f", omega[index].y); 
      }
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j+(Ny/2+1)*i;
      if( isnan(omega[index].x) || isinf(omega[index].x) ) 
	fprintf(omegafile, "\t\t\t%f\t", omega[index].x); 
      else
	fprintf(omegafile, "\t\t\t%f", omega[index].x); 
      if( isnan(omega[index].y) || isinf(omega[index].y) )
	fprintf(gammafile, "\t\t\t%f\t", omega[index].y); 
      else
	fprintf(gammafile, "\t\t\t%f", omega[index].y); 
    }
  }
  fprintf(omegafile, "\n");
  fprintf(gammafile, "\n");
}

//time history of growth rates
inline void omegaWrite(FILE* omegafile, FILE* gammafile, cuComplex* omegaSum, float dtSum, float time)
{
  fprintf(omegafile, "\t%f", time);
  fprintf(gammafile, "\t%f", time);
  for(int i=0; i<((Nx-1)/3+1); i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j+(Ny/2+1)*i;
      if(index!=0) {
	if( isnan(omegaSum[index].x) || isinf(omegaSum[index].x) ) 
	  fprintf(omegafile, "\t\t\t%f\t", omegaSum[index].x/dtSum); 
	else
	  fprintf(omegafile, "\t\t\t%f", omegaSum[index].x/dtSum); 
	if( isnan(omegaSum[index].y) || isinf(omegaSum[index].y) )
	  fprintf(gammafile, "\t\t\t%f\t", omegaSum[index].y/dtSum); 
	else
	  fprintf(gammafile, "\t\t\t%f", omegaSum[index].y/dtSum); 
      }
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j+(Ny/2+1)*i;
      if( isnan(omegaSum[index].x) || isinf(omegaSum[index].x) ) 
	fprintf(omegafile, "\t\t\t%f\t", omegaSum[index].x/dtSum); 
      else
	fprintf(omegafile, "\t\t\t%f", omegaSum[index].x/dtSum); 
      if( isnan(omegaSum[index].y) || isinf(omegaSum[index].y) )
	fprintf(gammafile, "\t\t\t%f\t", omegaSum[index].y/dtSum); 
      else
	fprintf(gammafile, "\t\t\t%f", omegaSum[index].y/dtSum); 
    }
  }
  fprintf(omegafile, "\n");
  fprintf(gammafile, "\n");
}

inline void kxkyTimeWrite(FILE* file, float* f, float time)
{
  if(time==0) {  
    fprintf(file, "#\ttime(s)\t");
    int col = 2;
    for(int i=0; i<((Nx-1)/3+1); i++) {
      for(int j=0; j<((Ny-1)/3+1); j++) {      
        if(!(i==0 && j==0)) {
          fprintf(file, "\t\t\t%d:(ky=%.3g,kx=%.3g)", col, ky_h[j],kx_h[i]);
          col++;
        }
      }
    }
    for(int i=2*Nx/3+1; i<Nx; i++) {
      for(int j=0; j<((Ny-1)/3+1); j++) {
        fprintf(file, "\t\t\t%d:(ky=%.3g,kx=%.3g)", col, ky_h[j],kx_h[i]);
        col++;
      }
    } 
    fprintf(file, "\n");
  }
  fprintf(file, "\t%f", time);
  for(int i=0; i<((Nx-1)/3+1); i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j+(Ny/2+1)*i;
      fprintf(file, "\t\t\t%e\t", f[index]);
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j+(Ny/2+1)*i;
      fprintf(file, "\t\t\t%e\t", f[index]);
    }
  }
  fprintf(file, "\n");
}

inline void kyTimeWrite(FILE* file, float* f, float time)
{
  if(time==0) {  
    fprintf(file, "#\ttime(s)\t");
    int col = 2;
      for(int j=0; j<((Ny-1)/3+1); j++) {      
          fprintf(file, "\t\t\t%d:(ky=%.3g)", col, ky_h[j]);
          col++;
      }
    fprintf(file, "\n");
  }
  fprintf(file, "\t%f", time);
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j;
      fprintf(file, "\t\t\t%e\t", f[index]);
    }
  fprintf(file, "\n");
}

//time history of flux
inline void fluxWrite(FILE* fluxfile, float* pflx, float* pflxAvg, float* wpfx, float* wpfxAvg, float Dnlpm, float Dnlpm_avg, float Phi_zf_kx1, float Phi_zf_kx1_avg, float Phi_zf_rms, float Phi_zf_rms_avg, float wpfxmax, float wpfxmin, 
		int converge_count, float time, specie* species)
{
  if(time == 0) {
    fprintf(fluxfile, "#\ttime(s)\t");
    for(int s=0; s<nSpecies; s++) {
      fprintf(fluxfile, "\tflux[%s]\t\tavgflux[%s]\t\tDnlpm\t\tDnlpm_avg\t\tphi_zf_kx1\t\tphi_zf_kx1_avg\t\tphi_zf_rms\t\tphi_zf_rms_avg",species[s].type,species[s].type);
    }
    fprintf(fluxfile,"\n");
  }
  
  fprintf(fluxfile, "\t%f",time);
  for(int s=0; s<nSpecies; s++) {
    fprintf(fluxfile,"\t%e\t\t%e\t\t%f\t\t%f\t\t%e\t\t%e\t\t%e\t\t%e", wpfx[s], wpfxAvg[s], Dnlpm, Dnlpm_avg, Phi_zf_kx1, Phi_zf_kx1_avg, Phi_zf_rms, Phi_zf_rms_avg);
  }
  fprintf(fluxfile,"\n");
}

//write f vs ky  
inline void kyWrite(float* f_ky, float* f_ky_h, char* filename, char* ext) 
{
  strcpy(filename,out_stem);
  strcat(filename, ext);
  FILE* out = fopen(filename,"w+");
  cudaMemcpy(f_ky_h, f_ky, sizeof(float)*(Ny/2+1),cudaMemcpyDeviceToHost);
  fprintf(out, "#\tky\t\t\t%s\n", ext);
  //for(int i=0; i<Ny/2+1; i++) {
  for(int i=0; i<(Ny-1)/3+1; i++) {
    fprintf(out, "\t%f\t\t%e\n", ky_h[i], f_ky_h[i]);
  }
  fclose(out);
}  

//write f vs ky  
inline void kyHistoryWrite(float* f_ky, float* f_ky_h, char* filename, char* ext, int counter, float runtime) 
{
  strcpy(filename,out_stem);
  strcat(filename, ext);
  FILE* out;
  if(counter==0 || counter==1) out = fopen(filename,"w+");
  else out = fopen(filename, "a");
  cudaMemcpy(f_ky_h, f_ky, sizeof(float)*(Ny/2+1),cudaMemcpyDeviceToHost);
  if(counter==0 || counter==1) fprintf(out, "#\tky\t\t\t%s\n", ext);
  //for(int i=0; i<Ny/2+1; i++) {
  fprintf(out,"t=%f\n", runtime);
  for(int i=0; i<(Ny-1)/3+1; i++) {
    fprintf(out, "\t%f\t\t%e\n", ky_h[i], f_ky_h[i]);
  }
  fprintf(out,"\n\n");
  fclose(out);
}  

inline void kykxWrite(float* f_kykx, float* f_kykx_h, char* filename, char* ext) 
{
  strcpy(filename, out_stem);
  strcat(filename, ext);
  FILE* out = fopen(filename,"w+");
  cudaMemcpy(f_kykx_h, f_kykx, sizeof(float)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
  fprintf(out, "#\tky\t\tkx\t\t%s\n", ext);
  for(int i=0; i<Ny/2+1; i++) {
    for(int j=0; j<Nx; j++) {
      fprintf(out, "\t%f\t\t%f\t\t%e\n", ky_h[i], kx_h[j], f_kykx_h[i+(Ny/2+1)*j]);
    }
  }
}

inline void kxkyWrite(float* f_kykx, float* f_kykx_h, char* filename, char* ext) 
{
  strcpy(filename, out_stem);
  strcat(filename, ext);
  FILE* out = fopen(filename,"w+");
  cudaMemcpy(f_kykx_h, f_kykx, sizeof(float)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
  fprintf(out, "#\tkx\t\tky\t\t%s\n", ext);
  for(int j=0; j<(Ny-1)/3+1; j++) {
    fprintf(out,"\n");
    for(int i=2*Nx/3+1; i<Nx; i++) {
      int idxy = j + (Ny/2+1)*i;
      fprintf(out,"%.4f\t\t%.4f\t\t%e\n", kx_h[i], ky_h[j], f_kykx_h[idxy]);
    }
    for(int i=0; i<((Nx-1)/3+1); i++) {
      int idxy = j + (Ny/2+1)*i;
      fprintf(out, "%.4f\t\t%.4f\t\t%e\n", kx_h[i], ky_h[j], f_kykx_h[idxy]);    
    }
  }
  
  fclose(out);
}

inline void zkyWrite(float* f_zky, float* f_zky_h, char* filename, char* ext)
{
  strcpy(filename, out_stem);
  strcat(filename,ext);
  FILE* out = fopen(filename,"w+");
  cudaMemcpy(f_zky_h, f_zky, sizeof(float)*(Ny/2+1)*Nz,cudaMemcpyDeviceToHost);
  fprintf(out, "#\tz (1)\t\t\tky (2)\t\t\t%s", ext);  
  fprintf(out, "\n");
  int blockid = 0;
  for(int j=0; j<(Ny-1)/3+1; j++) {
    fprintf(out,"\n");      
    for(int k=0; k<=Nz; k++) {
      int index = j+(Ny/2+1)*k;
      int index_z0 = j+(Ny/2+1)*(Nz/2);
      
      if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%e\n", -z_h[0], ky_h[j], f_zky_h[j]); 
      else fprintf(out, "\t%f\t\t%f\t\t%e\n", z_h[k], ky_h[j], f_zky_h[index]);    	  
      
    }     
  }  
  fclose(out);
}  

inline void zkyWriteNorm(float* f_zky, float* f_zky_h, char* filename, char* ext)
{
  strcpy(filename, out_stem);
  strcat(filename,ext);
  FILE* out = fopen(filename,"w+");
  cudaMemcpy(f_zky_h, f_zky, sizeof(float)*(Ny/2+1)*Nz,cudaMemcpyDeviceToHost);
  fprintf(out, "#\tz (1)\t\t\tky (2)\t\t\t%s", ext);  
  fprintf(out, "\n");
  int blockid = 0;
  for(int j=0; j<(Ny-1)/3+1; j++) {
    fprintf(out,"\n");      
    for(int k=0; k<=Nz; k++) {
      int index = j+(Ny/2+1)*k;
      int index_z0 = j+(Ny/2+1)*(Nz/2);
      
      if(k==Nz) fprintf(out, "\t%f\t\t%f\t\t%e\n", -z_h[0], ky_h[j], (double) f_zky_h[j]/f_zky_h[index_z0]); 
      else fprintf(out, "\t%f\t\t%f\t\t%e\n", z_h[k], ky_h[j], (double) f_zky_h[index]/f_zky_h[index_z0]);    	  
      
    }     
  }  
  fclose(out);
}  

inline void kxWrite(float* f_kx, float* f_kx_h, char* filename, char* ext)
{
  strcpy(filename, out_stem);
  strcat(filename,ext);
  FILE* out = fopen(filename,"w+");
  cudaMemcpy(f_kx_h, f_kx, sizeof(float)*Nx, cudaMemcpyDeviceToHost);
  fprintf(out, "#kx\t%s\n",ext);      
  for(int i=2*Nx/3+1; i<Nx; i++) {
    fprintf(out,"%.4f\t%e\n", kx_h[i], f_kx_h[i]);
  }
  for(int i=0; i<((Nx-1)/3+1); i++) {
    fprintf(out, "%.4f\t%e\n", kx_h[i], f_kx_h[i]);    
  }
  
  fclose(out);
}


//write final growth rates vs kx and ky. fast moving index is ky.
inline void omegakykxWrite(cuComplex* omegaAvg_h, char* filename, char* ext, float fac)
{
  strcpy(filename, out_stem);
  strcat(filename, ext);
  FILE* out = fopen(filename, "w+");
  fprintf(out, "#ky\tkx\t\tomega\t\tgamma\n");
  int blockid = 0;
  for(int i=0; i<((Nx-1)/3+1); i++) {
    fprintf(out, "\n#%d\n\n", blockid);
    blockid++;
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j + (Ny/2+1)*i;
      if(index!=0) {
	fprintf(out, "%.4f\t%.4f\t\t%.6f\t%.6f\n", fac*ky_h[j], fac*kx_h[i], omegaAvg_h[index].x/fac, omegaAvg_h[index].y/fac);
      }
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    fprintf(out, "\n#%d\n\n", blockid);
    blockid++;
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j + (Ny/2+1)*i;
      fprintf(out,"%.4f\t%.4f\t\t%.6f\t%.6f\n", fac*ky_h[j], fac*kx_h[i],omegaAvg_h[index].x/fac,omegaAvg_h[index].y/fac);
    }
  }	  
}

//same as above but with kx as fast moving index
inline void omegakxkyWrite(cuComplex* omegaAvg_h, char* filename, char* ext)
{
  strcpy(filename, out_stem);
  strcat(filename, ext);
  FILE* out = fopen(filename, "w+");
  fprintf(out, "#kx\tky\t\tomega\t\tgamma\n");
  int blockid = 0;
  for(int j=0; j<((Ny-1)/3+1); j++) {
    fprintf(out, "\n#%d\n\n", blockid);
    blockid++;
    for(int i=0; i<((Nx-1)/3+1); i++) {
      int index = j + (Ny/2+1)*i;
      if(index!=0) {
	fprintf(out, "%.4f\t%.4f\t\t%.6f\t%.6f\n", kx_h[i], ky_h[j], omegaAvg_h[index].x, omegaAvg_h[index].y);
      }
    }
    for(int i=2*Nx/3+1; i<Nx; i++) {
      int index = j + (Ny/2+1)*i;
      fprintf(out,"%.4f\t%.4f\t\t%.6f\t%.6f\n", kx_h[i], ky_h[j], omegaAvg_h[index].x, omegaAvg_h[index].y);
    }
  }
}

inline void omegaAbsDiffWrite(FILE* omegadiff_file, FILE* gammadiff_file, cuComplex* avg,cuComplex* omega, int counter)
{
  
  fprintf(omegadiff_file, "\t%d", counter);
  fprintf(gammadiff_file, "\t%d", counter);
  for(int i=0; i<((Nx-1)/3+1); i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j+(Ny/2+1)*i;
      if(index!=0) {
	if( isnan(abs((omega[index].x - avg[index].x))) || isinf(abs((omega[index].x - avg[index].x))) ) 
	  fprintf(omegadiff_file, "\t%f\t", abs((omega[index].x - avg[index].x)) ); 
	else
	  fprintf(omegadiff_file, "\t%f", abs((omega[index].x - avg[index].x)) ); 
	if( isnan(abs((omega[index].y - avg[index].y))) || isinf(abs((omega[index].y - avg[index].y))) )
	  fprintf(gammadiff_file, "\t%f\t", abs((omega[index].y - avg[index].y))); 
	else
	  fprintf(gammadiff_file, "\t%f", abs((omega[index].y - avg[index].y))); 
      }
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j+(Ny/2+1)*i;
      if( isnan(abs((omega[index].x - avg[index].x))) || isinf(abs((omega[index].x - avg[index].x))) ) 
	fprintf(omegadiff_file, "\t%f\t", abs((omega[index].x - avg[index].x)) ); 
      else
	fprintf(omegadiff_file, "\t%f", abs((omega[index].x - avg[index].x)) ); 
      if( isnan(abs((omega[index].y - avg[index].y))) || isinf(abs((omega[index].y - avg[index].y))) )
	fprintf(gammadiff_file, "\t%f\t", abs((omega[index].y - avg[index].y))); 
      else
	fprintf(gammadiff_file, "\t%f", abs((omega[index].y - avg[index].y))); 
    }
  }
  fprintf(omegadiff_file, "\n");
  fprintf(gammadiff_file, "\n");
}

inline void omegaPercentDiffWrite(FILE* omegadiff_file, FILE* gammadiff_file, cuComplex* avg,cuComplex* omega, int counter)
{
  
  fprintf(omegadiff_file, "\t%d", counter);
  fprintf(gammadiff_file, "\t%d", counter);
  for(int i=0; i<((Nx-1)/3+1); i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j+(Ny/2+1)*i;
      if(index!=0) {
	if( isnan( abs((omega[index].x - avg[index].x)/avg[index].x) ) || isinf( abs((omega[index].x - avg[index].x)/avg[index].x) ) )  
	  fprintf(omegadiff_file, "\t%f\t", abs((omega[index].x - avg[index].x)/avg[index].x) ); 
	else
	  fprintf(omegadiff_file, "\t%f", abs((omega[index].x - avg[index].x)/avg[index].x) ); 
	if( isnan( abs((omega[index].y - avg[index].y)/avg[index].y) ) || isinf( abs((omega[index].y - avg[index].y)/avg[index].y) ) )
	  fprintf(gammadiff_file, "\t%f\t", abs((omega[index].y - avg[index].y)/avg[index].y) ); 
	else
	  fprintf(gammadiff_file, "\t%f", abs((omega[index].y - avg[index].y)/avg[index].y) ); 
      }
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j+(Ny/2+1)*i;
      if( isnan( abs((omega[index].x - avg[index].x)/avg[index].x) ) || isinf( abs((omega[index].x - avg[index].x)/avg[index].x) ) )  
	fprintf(omegadiff_file, "\t%f\t", abs((omega[index].x - avg[index].x)/avg[index].x) ); 
      else
	fprintf(omegadiff_file, "\t%f", abs((omega[index].x - avg[index].x)/avg[index].x) ); 
      if( isnan( abs((omega[index].y - avg[index].y)/avg[index].y) ) || isinf( abs((omega[index].y - avg[index].y)/avg[index].y) ) )
	fprintf(gammadiff_file, "\t%f\t", abs((omega[index].y - avg[index].y)/avg[index].y) ); 
      else
	fprintf(gammadiff_file, "\t%f", abs((omega[index].y - avg[index].y)/avg[index].y) ); 
    }
  }
  fprintf(omegadiff_file, "\n");
  fprintf(gammadiff_file, "\n");
}



inline void omegaStability(cuComplex* omega, cuComplex* avg, cuComplex* stability, int* Stable, int stableMax)
{
  for(int i=0; i<(Nx-1)/3+1; i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) { 
      int index = j + (Ny/2+1)*i;
      if(index != 0) {
	if( abs((omega[index].x - avg[index].x)/avg[index].x) < .00001 && Stable[index] < stableMax) {
	  stability[index].x = omega[index].x;
	  Stable[index]++;
	  //if(Stable[index] == stableMax) printf("omega(%d,%d) converged: %f\n",j,i,stability[index].x);
	}
	else if( Stable[index] < stableMax ) Stable[index] = 0;
	if( abs((omega[index].y - avg[index].y)/avg[index].y) < .00001 && Stable[index + Nx*(Ny/2+1)] < stableMax) {
	  stability[index].y = omega[index].y;
	  Stable[index +Nx*(Ny/2+1)]++;
	  //if(Stable[index +Nx*(Ny/2+1)] == stableMax) printf("gamma(%d,%d) converged: %f\n",j,i,stability[index].y);
	}
	else if(Stable[index +Nx*(Ny/2+1)] < stableMax ) Stable[index +Nx*(Ny/2+1)] = 0;
      }
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      int index = j + (Ny/2+1)*i;
      if( abs((omega[index].x - avg[index].x)/avg[index].x) < .00001 && Stable[index] < stableMax) {
        stability[index].x = omega[index].x;
	Stable[index]++;
	//if(Stable[index] == stableMax) printf("omega(%d,%d) converged: %f\n",j,i,stability[index].x);
      }
      else if( Stable[index] < stableMax ) Stable[index] = 0;
      if( abs((omega[index].y - avg[index].y)/avg[index].y) < .00001 && Stable[index + Nx*(Ny/2+1)] < stableMax) {
        stability[index].y = omega[index].y;
	Stable[index +Nx*(Ny/2+1)]++;
	//if(Stable[index +Nx*(Ny/2+1)] == stableMax) printf("gamma(%d,%d) converged: %f\n",j,i,stability[index].y);
      }
      else if(Stable[index +Nx*(Ny/2+1)] < stableMax ) Stable[index +Nx*(Ny/2+1)] = 0;
    }
  }
  
  //set true for masked modes and zero mode
  if(Nx==1) {
    int i=0;
    for(int j=((Ny-1)/3+1); j<(Ny/2+1); j++) {
      int index = j + (Ny/2+1)*i;
      Stable[index] = stableMax;
      Stable[index+Nx*(Ny/2+1)] = stableMax;
      stability[index].x = 0;
      stability[index].y = 0;
    }
  }
  else {
    for(int i=(Nx-1)/3+1; i<2*Nx/3+1; i++) {
      for(int j=((Ny-1)/3+1); j<(Ny/2+1); j++) {
	int index = j + (Ny/2+1)*i;
	Stable[index] = stableMax;
	Stable[index+Nx*(Ny/2+1)] = stableMax;
	stability[index].x = 0;
	stability[index].y = 0;
      }
    }
  }
  Stable[0] = stableMax;
  Stable[0+Nx*(Ny/2+1)] = stableMax; 
  stability[0].x = 0;
  stability[0+Nx*(Ny/2+1)].y = 0;   
}

bool stabilityCheck(int* Stable, int stableMax)
{
  bool STOP = true;
  for(int i=0; i<Nx*(Ny/2+1); i++) {
    if(Stable[i] < stableMax)
      STOP = false;
    if(Stable[i +Nx*(Ny/2+1)] < stableMax)
      STOP = false;
  }
  return STOP;
}

inline void stabilityWrite(cuComplex* stability, int* Stable, int stableMax)
{
  char filename[200];
  sprintf(filename,"./scan/outputs/stability%g_f%g_t%g", dt, species[0].fprim, species[0].tprim);
  FILE* ofile = fopen(filename,"w+");
  fprintf(ofile, "Nx = %d  Ny = %d  Nz = %d  Boxsize = 2pi*(%g,%g,%g)\n\n", Nx, Ny, Nz, X0, Y0, Zp); 
  fprintf(ofile, "fprim= %g\ntprim= %g\n\n", species[0].fprim, species[0].tprim);
  for(int i=0; i<Nx; i++) {
    for(int j=0; j<Ny/2+1; j++) {
      int index = j+(Ny/2+1)*i;
      if(Stable[index] >= stableMax)
        fprintf(ofile,"omega(%d,%d)= %f\n", j,i, stability[index].x);
      else
        fprintf(ofile,"omega(%d,%d)= NOT STABLE\n", j, i);
      if(Stable[index +Nx*(Ny/2+1)] >= stableMax)
        fprintf(ofile,"gamma(%d,%d)= %f\n", j,i, stability[index].y);
      else
        fprintf(ofile,"gamma(%d,%d)= NOT STABLE\n", j, i);
      fprintf(ofile,"\n\n");
    }
  }
}    
  
inline void fieldNormalize(cuComplex** Dens, cuComplex** Upar, cuComplex** Tpar, cuComplex** Tprp,
                cuComplex** Qpar, cuComplex** Qprp, cuComplex* Phi,float norm)
{    
       
    for(int s=0; s<nSpecies; s++) {
      normalize<<<dimGrid,dimBlock>>>(Dens[s],Phi,norm);
      //scale<<<dimGrid,dimBlock>>>(Dens[s],Dens[s],norm); 
      mask<<<dimGrid,dimBlock>>>(Dens[s]);
      
      normalize<<<dimGrid,dimBlock>>>(Upar[s],Phi,norm);
      //scale<<<dimGrid,dimBlock>>>(Upar[s],Upar[s],norm); 
      mask<<<dimGrid,dimBlock>>>(Upar[s]);
      
      normalize<<<dimGrid,dimBlock>>>(Tpar[s],Phi,norm);
      //scale<<<dimGrid,dimBlock>>>(Tpar[s],Tpar[s],norm); 
      mask<<<dimGrid,dimBlock>>>(Tpar[s]);
      
      normalize<<<dimGrid,dimBlock>>>(Tprp[s],Phi,norm);
      //scale<<<dimGrid,dimBlock>>>(Tprp[s],Tprp[s],norm); 
      mask<<<dimGrid,dimBlock>>>(Tprp[s]);
      
      normalize<<<dimGrid,dimBlock>>>(Qpar[s],Phi,norm);
      //scale<<<dimGrid,dimBlock>>>(Qpar[s],Qpar[s],norm); 
      mask<<<dimGrid,dimBlock>>>(Qpar[s]);
      
      normalize<<<dimGrid,dimBlock>>>(Qprp[s],Phi,norm);
      //scale<<<dimGrid,dimBlock>>>(Qprp[s],Qprp[s],norm);       
      mask<<<dimGrid,dimBlock>>>(Qprp[s]);
      
    }
    normalize<<<dimGrid,dimBlock>>>(Phi,Phi,norm);
    //scale<<<dimGrid,dimBlock>>>(Phi,Phi,norm);  
    mask<<<dimGrid,dimBlock>>>(Phi);
      
  
}   


inline void restartWrite(cuComplex** Dens, cuComplex** Upar, cuComplex** Tpar, cuComplex** Tprp,
                cuComplex** Qpar, cuComplex** Qprp, cuComplex* Phi, float* pflxAvg, float* wpfxAvg, float* Phi2_kxky_sum, 
		float* Phi2_zonal_sum, float* zCorr_sum, float expectation_ky_sum, float expectation_kx_sum, float Phi_zf_kx1_avg, float dtSum,
		int counter, float runtime, float dt, float timer, char* restartfileName)
{
  //printf("restart file is\n%s\n", restartfileName);
  FILE *restart;
  restart = fopen(restartfileName, "wb");
  cuComplex *Dens_h[nSpecies];
  cuComplex *Upar_h[nSpecies];
  cuComplex *Tpar_h[nSpecies];
  cuComplex *Tprp_h[nSpecies];
  cuComplex *Qpar_h[nSpecies];
  cuComplex *Qprp_h[nSpecies];
  cuComplex *Phi_h;
  float* Phi2_kxky_sum_h;
  float* Phi2_zonal_sum_h;
  float* zCorr_sum_h;
  
  for(int s=0; s<nSpecies; s++) {
    Dens_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMemcpy(Dens_h[s], Dens[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
    
    Upar_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMemcpy(Upar_h[s], Upar[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
    
    Tpar_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMemcpy(Tpar_h[s], Tpar[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
    
    Tprp_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMemcpy(Tprp_h[s], Tprp[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
    
    Qpar_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMemcpy(Qpar_h[s], Qpar[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
    
    Qprp_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMemcpy(Qprp_h[s], Qprp[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
  }  
  Phi_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMemcpy(Phi_h, Phi, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
  
  Phi2_kxky_sum_h = (float*) malloc(sizeof(float)*Nx*(Ny/2+1));
  cudaMemcpy(Phi2_kxky_sum_h, Phi2_kxky_sum, sizeof(float)*Nx*(Ny/2+1), cudaMemcpyDeviceToHost);
  
  Phi2_zonal_sum_h = (float*) malloc(sizeof(float)*Nx);
  cudaMemcpy(Phi2_zonal_sum_h, Phi2_zonal_sum, sizeof(float)*Nx, cudaMemcpyDeviceToHost);
  
  zCorr_sum_h = (float*) malloc(sizeof(float)*(Ny/2+1)*Nz);
  cudaMemcpy(zCorr_sum_h, zCorr_sum, sizeof(float)*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
  
  fwrite(&counter,sizeof(int),1,restart);
  fwrite(&runtime,sizeof(float),1,restart);
  fwrite(&dt,sizeof(float),1,restart);
  fwrite(&timer,sizeof(float),1,restart);
  
  fwrite(wpfxAvg, sizeof(float)*nSpecies, 1, restart);
  fwrite(pflxAvg, sizeof(float)*nSpecies, 1, restart);
  fwrite(Phi2_kxky_sum_h,sizeof(float)*Nx*(Ny/2+1),1,restart);
  fwrite(&expectation_ky_sum, sizeof(float), 1, restart);
  fwrite(&expectation_kx_sum, sizeof(float), 1, restart);
  fwrite(&dtSum, sizeof(float), 1, restart);
  
  for(int s=0; s<nSpecies; s++) {
    fwrite(Dens_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    fwrite(Upar_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    fwrite(Tpar_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    fwrite(Tprp_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    fwrite(Qpar_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    fwrite(Qprp_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
  }
  fwrite(Phi_h,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,1,restart);
  
  fwrite(zCorr_sum_h, sizeof(float)*(Ny/2+1)*Nz,1,restart);
  fwrite(Phi2_zonal_sum_h, sizeof(float)*Nx,1,restart);

  fwrite(&Phi_zf_kx1_avg, sizeof(float), 1, restart);
  
  fclose(restart);
  
  for(int s=0; s<nSpecies; s++) {
    free(Dens_h[s]);
    free(Upar_h[s]);
    free(Tpar_h[s]);
    free(Tprp_h[s]);
    free(Qpar_h[s]);
    free(Qprp_h[s]);
  }
  free(Phi_h);
  free(Phi2_kxky_sum_h);
  free(Phi2_zonal_sum_h);
  
}

inline void restartRead(cuComplex** Dens, cuComplex** Upar, cuComplex** Tpar, cuComplex** Tprp,
                cuComplex** Qpar, cuComplex** Qprp, cuComplex* Phi, float* pflxAvg, float* wpfxAvg, float* Phi2_kxky_sum, 
		float* Phi2_zonal_sum, float* zCorr_sum, float* expectation_ky_sum, float* expectation_kx_sum, float* Phi_zf_kx1_avg, float* dtSum,
		int* counter, float* runtime, float* dt, float* timer, char* restartfileName)  
{
  FILE *restart;
  restart = fopen(restartfileName, "rb");
  cuComplex *Dens_h[nSpecies];
  cuComplex *Upar_h[nSpecies];
  cuComplex *Tpar_h[nSpecies];
  cuComplex *Tprp_h[nSpecies];
  cuComplex *Qpar_h[nSpecies];
  cuComplex *Qprp_h[nSpecies];
  cuComplex *Phi_h;
  float *Phi2_kxky_sum_h;
  float *Phi2_zonal_sum_h;
  float *zCorr_sum_h;
  
  for(int s=0; s<nSpecies; s++) {
    Dens_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    Upar_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);    
    Tpar_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);    
    Tprp_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    Qpar_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    Qprp_h[s] = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);    
  }  
  Phi_h = (cuComplex*) malloc(sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  
  Phi2_kxky_sum_h = (float*) malloc(sizeof(float)*Nx*(Ny/2+1));
  Phi2_zonal_sum_h = (float*) malloc(sizeof(float)*Nx);
  zCorr_sum_h = (float*) malloc(sizeof(float)*(Ny/2+1)*Nz);
    
  fread(counter,sizeof(int),1,restart);  
  fread(runtime,sizeof(float),1,restart); 
  fread(dt, sizeof(float),1,restart);
  fread(timer, sizeof(float),1,restart);
  
  fread(wpfxAvg, sizeof(float)*nSpecies, 1, restart);
  fread(pflxAvg, sizeof(float)*nSpecies, 1, restart);
  fread(Phi2_kxky_sum_h, sizeof(float)*Nx*(Ny/2+1),1,restart);
  cudaMemcpy(Phi2_kxky_sum, Phi2_kxky_sum_h, sizeof(float)*Nx*(Ny/2+1), cudaMemcpyHostToDevice);
  
  fread(expectation_ky_sum, sizeof(float),1,restart);
  fread(expectation_kx_sum, sizeof(float),1,restart);
  fread(dtSum,sizeof(float),1,restart); 
  
    
  for(int s=0; s<nSpecies; s++) {
    fread(Dens_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    cudaMemcpy(Dens[s], Dens_h[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
    
    fread(Upar_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    cudaMemcpy(Upar[s], Upar_h[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
    
    fread(Tpar_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    cudaMemcpy(Tpar[s], Tpar_h[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
    
    fread(Tprp_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    cudaMemcpy(Tprp[s], Tprp_h[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
    
    fread(Qpar_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    cudaMemcpy(Qpar[s], Qpar_h[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
    
    fread(Qprp_h[s],sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, 1, restart);
    cudaMemcpy(Qprp[s], Qprp_h[s], sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
  }
  fread(Phi_h,sizeof(cuComplex)*Nx*(Ny/2+1)*Nz,1,restart);
  cudaMemcpy(Phi, Phi_h, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);
  
  fread(zCorr_sum_h,sizeof(float)*(Ny/2+1)*Nz,1,restart);
  cudaMemcpy(zCorr_sum, zCorr_sum_h, sizeof(float)*(Ny/2+1)*Nz, cudaMemcpyHostToDevice);  
  
  fread(Phi2_zonal_sum_h, sizeof(float)*Nx,1,restart);
  cudaMemcpy(Phi2_zonal_sum, Phi2_zonal_sum_h, sizeof(float)*Nx, cudaMemcpyHostToDevice);
      
  fread(Phi_zf_kx1_avg, sizeof(float), 1, restart);

  fclose(restart);
  
  for(int s=0; s<nSpecies; s++) {
    free(Dens_h[s]);
    free(Upar_h[s]);
    free(Tpar_h[s]);
    free(Tprp_h[s]);
    free(Qpar_h[s]);
    free(Qprp_h[s]);
  }
  free(Phi_h);
  free(Phi2_kxky_sum_h);
  free(Phi2_zonal_sum_h);
  free(zCorr_sum_h);
  
}
/*
inline void check_b()
{
  FILE* b_file = fopen("./scan/outputs/b_check", "w++");
  float* b;
  cudaMalloc((inline void**) &b, sizeof(float)*Nx*(Ny/2+1)*Nz);
  
  float* b_h = (float*) malloc(sizeof(float)*Nx*(Ny/2+1)*Nz);
  
  bcheck<<<dimGrid,dimBlock>>>(b,species[ION].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
  
  cudaMemcpy(b_h, b, sizeof(float)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToHost);
  
  for(int i=0; i<Nx; i++) {
    for(int j=0; j<(Ny/2+1); j++) {
      for(int k=0; k<=Nz; k++) {
        int index = j+(Ny/2+1)*i+(Ny/2+1)*Nx*k;
	if(k==Nz) fprintf(b_file, "\t%f\t\t%f\t\t%f\t\t%f\t\n", -z_h[0], i/X0, j/Y0, b_h[j+(Ny/2+1)*i]); 
	else fprintf(b_file, "\t%f\t\t%f\t\t%f\t\t%f\t\n", z_h[k], i/X0, j/Y0, b_h[index]);    	  
      }
      fprintf(b_file, "\n\n");      
    }
  }
  
  free(b_h);
  cudaFree(b);
}
*/

inline void geoWrite(char* ext, char* filename)
{
  strcpy(filename,out_stem);
  strcat(filename,ext);
  FILE* out = fopen(filename,"w+");
  
  cudaMemcpy(gbdrift_h, gbdrift, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(grho_h, grho, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(z_h, z, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(cvdrift_h, cvdrift, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(gds2_h, gds2, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(bmag_h, bmag, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(bgrad_h, bgrad, sizeof(float)*Nz, cudaMemcpyDeviceToHost);    //
  cudaMemcpy(gds21_h, gds21, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(gds22_h, gds22, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(cvdrift0_h, cvdrift0, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(gbdrift0_h, gbdrift0, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(jacobian_h, jacobian, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  cudaMemcpy(kz_h, kz, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
  
  fprintf(out, "#\tz:1\t\tbmag:2\t\tbgrad:3\t\tgbd:4\t\tgbd0:5\t\tcvd:6\t\tcvd0:7\t\tgds2:8\t\tgds21:9\t\tgds22:10\tgrho:11\t\tjacobian:12\n");
  for(int i=0; i<Nz; i++) {
    fprintf(out, "\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", z_h[i], bmag_h[i], bgrad_h[i], gbdrift_h[i], gbdrift0_h[i], 
    									cvdrift_h[i], cvdrift0_h[i], gds2_h[i], gds21_h[i], gds22_h[i], grho_h[i], jacobian_h[i], kz_h[i]);
  }
  //periodic point
  int i=0;
  fprintf(out, "\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n", -z_h[i], bmag_h[i], bgrad_h[i], gbdrift_h[i], gbdrift0_h[i], 
    									cvdrift_h[i], cvdrift0_h[i], gds2_h[i], gds21_h[i], gds22_h[i], grho_h[i], jacobian_h[i]);
  
  
}

inline void gryfx_finish_diagnostics(cuComplex** Dens, cuComplex** Upar, cuComplex** Tpar, cuComplex** Tprp, cuComplex** Qpar, cuComplex** Qprp, 
	cuComplex* Phi,	cuComplex* tmp, cuComplex* corrNum_tmp, cuComplex* field, float* tmpZ, cuComplex* CtmpX, 
	float* tmpXY, float* phi2avg_tmpXY, float* corrDen_tmpXY, float* kphi2_tmpXY2, float* tmpXY3, float* tmpXY4, float* phi_corr_tmpYZ, float* phi_corr_J_tmpYZ,
	float* phi2avg_tmpX, float* phi2zonal_tmpX2, float* wpfx_over_phi2_ky_tmpY, float* wpfx_ky_tmpY, float* phi_corr_z0_tmpY, float* tmpY, float* tmpY2, float* phi2_ky_tmpY2, float* phi_corr_norm_tmpY2,
	int** kxCover, int** kyCover, float* tmpX_h, float* tmpY_h, float* tmpXY_h, float* tmpYZ_h, cuComplex* field_h, 
	int** kxCover_h, int** kyCover_h, cuComplex* omegaAvg_h, double* qflux, float* expectation_ky, float* expectation_kx,
	float* Phi2_kxky_sum, float* wpfxnorm_kxky_sum, float* Phi2_zonal_sum, float* zCorr_sum, float expectation_ky_sum, float expectation_kx_sum, //cuComplex* omegaAvg,// cuComplex* omega_sum,
	float dtSum, int counter, float runtime, bool end,
        float* Phi2, float* flux1_phase, float* flux2_phase, float* Dens_phase, float* Tpar_phase, float* Tprp_phase,
        float Phi2_sum, float flux1_phase_sum, float flux2_phase_sum, float Dens_phase_sum, float Tpar_phase_sum, float Tprp_phase_sum)
{
  char filename[200];  

  
  //scale<<<dimGrid,dimBlock>>>(omegaAvg, omega_sum, (float) 1./dtSum, Nx, Ny, 1
  
  //write final growth rates, with ky or kx as the fast index
  omegakykxWrite(omegaAvg_h, filename, "omega.kykx", 1.);
  omegakxkyWrite(omegaAvg_h, filename, "omega.kxky");
  
  //calculate and write wpfx(ky)
  if(end) {
    fluxes_k(wpfx_ky_tmpY, tmpY, tmpY2, tmpXY, tmpXY, kphi2_tmpXY2,
    	      Dens[ION], Tpar[ION], Tprp[ION], Phi, tmp, tmp, tmp, 
	      field, field, tmpZ, species[ION]);
    kyWrite(wpfx_ky_tmpY, tmpY_h, filename, "flux.ky");    
    kxkyWrite(tmpXY, tmpXY_h, filename, "flux.kykx");	      
    
    
  }
  
  
  
  //calculate time average of phi**2(ky,kx)
  scaleReal<<<dimGrid,dimBlock>>>(phi2avg_tmpXY, Phi2_kxky_sum, (float) 1./dtSum, Nx, Ny/2+1,1);
  scaleReal<<<dimGrid,dimBlock>>>(phi2zonal_tmpX2, Phi2_zonal_sum, (float) 1./dtSum, Nx, 1, 1);  
    
  //write zonal flow component of phi**2(kx)
  kxWrite(phi2zonal_tmpX2, tmpX_h, filename, "phi2_zonal.kx");
  
  //calculate and write non zonal component of phi**2(kx)
  /*
  // first calculate total phi**2(kx)
  sumY<<<dimGrid,dimBlock>>>(phi2avg_tmpX, phi2avg_tmpXY);
  // subtract zonal component to get non-zonal component
  add_scaled<<<dimGrid,dimBlock>>>(phi2avg_tmpX, 1., phi2avg_tmpX, -1., phi2zonal_tmpX2, Nx, 1, 1);
  */
  sumY_neq_0<<<dimGrid,dimBlock>>>(phi2avg_tmpX, phi2avg_tmpXY);  
  kxWrite(phi2avg_tmpX, tmpX_h, filename, "phi2.kx");
  
  //calculate and write phi**2(ky) == E(ky) -> electrostatic fluctuation spectra
  sumX<<<dimGrid,dimBlock>>>(phi2_ky_tmpY2, phi2avg_tmpXY);
  kyWrite(phi2_ky_tmpY2, tmpY_h, filename, "phi2.ky");
  kyHistoryWrite(phi2_ky_tmpY2, tmpY_h, filename, "phi2.ky.time", counter, runtime);  
  //write phi**2(kx,ky)
  kxkyWrite(phi2avg_tmpXY, tmpXY_h, filename, "phi2.kxky");
  
  if(end) {
    //calculate and write wpfx/phi**2(ky)
    multdiv<<<dimGrid,dimBlock>>>(wpfx_over_phi2_ky_tmpY, wpfx_ky_tmpY, phi2_ky_tmpY2, 1, Ny, 1, -1);
    kyWrite(wpfx_over_phi2_ky_tmpY, tmpY_h, filename, "flux_over_phi2.ky");
    
    //calculate kperp energy spectrum
    float dkperp = 1./X0;
    //E_kperp<<<dimGrid,dimBlock>>>(tmpY, phi2avg_tmpXY,species[ION].rho,kx,ky,shat,gds2,gds21,gds22,bmagInv);
  }
  
  //expectation value of ky and kx
  *expectation_ky = expectation_ky_sum/dtSum;
  *expectation_kx = expectation_kx_sum/dtSum;
  *Phi2 = Phi2_sum/dtSum;
  *flux1_phase = flux1_phase_sum/dtSum;
  *flux2_phase = flux2_phase_sum/dtSum;
  *Dens_phase = Dens_phase_sum/dtSum;
  *Tpar_phase = Tpar_phase_sum/dtSum;
  *Tprp_phase = Tprp_phase_sum/dtSum;
  
  if(end) printf("Phi2 = %f\nexpectation val of ky = %f\nexpectation val of kx = %f\n", Phi2, *expectation_ky, *expectation_kx);  

  if(end) printf("Q_i = %f\n\n", qflux[ION]);
  if(end) printf("flux1_phase = %f \t\t flux2_phase = %f\nDens_phase = %f \t\t Tpar_phase = %f \t\t Tprp_phase = %f\n", flux1_phase, flux2_phase, Dens_phase, Tpar_phase, Tprp_phase);
  
  //calculate and write time average of parallel correlation function, C(ky,z)
  scaleReal<<<dimGrid,dimBlock>>>(phi_corr_tmpYZ, zCorr_sum, (float) 1./dtSum, 1, Ny/2+1,Nz);
  zkyWriteNorm(phi_corr_tmpYZ, tmpYZ_h, filename, "phi_correlation.zky");
  
  //calculate several parallel correlation lengths as function of ky 
  //(correspond to correlation lengths calculated in gs2plots.f90)
  get_z0<<<dimGrid,dimBlock>>>(phi_corr_z0_tmpY, phi_corr_tmpYZ, 1, Ny, Nz);
  multZ<<<dimGrid,dimBlock>>>(phi_corr_J_tmpYZ, phi_corr_tmpYZ, jacobian, 1, Ny, Nz);   // phi_corr_J = corr(ky,z) * J(z) 
  //zkyWrite(phi_corr_J_tmpYZ, tmpYZ_h, filename, "phi_corr_J.zky");
  sumZ<<<dimGrid,dimBlock>>>(phi_corr_norm_tmpY2, phi_corr_tmpYZ, 1, Ny, Nz);  // phi_corr_norm = sum_z [ phi_corr_J ] 
  //kyWrite(tmpY2, tmpY_h, filename, "phi_corr_norm.ky");
  //kyWrite(tmpY, tmpY_h, filename, "phi_corr_z0.ky");
  //corr_length_3<<<dimGrid,dimBlock>>>(tmpY, phi_corr_norm_tmpY2, phi_corr_z0_tmpY, 1./fluxDen);
  multdiv<<<dimGrid,dimBlock>>>(tmpY, phi_corr_norm_tmpY2, phi_corr_z0_tmpY, 1, Ny, 1, -1);
  scaleReal<<<dimGrid,dimBlock>>>(tmpY, tmpY, (float) 1./fluxDen, 1, Ny/2+1, 1);
  kyWrite(tmpY, tmpY_h, filename, "corr_length_3.ky"); 
  if(end) 
  {
    corr_length_1<<<dimGrid,dimBlock>>>(tmpY, phi_corr_J_tmpYZ, phi_corr_norm_tmpY2, z);
    kyWrite(tmpY, tmpY_h, filename, "corr_length_1.ky"); 
    corr_length_4<<<dimGrid,dimBlock>>>(tmpY, phi_corr_J_tmpYZ, phi_corr_norm_tmpY2, z);
    kyWrite(tmpY, tmpY_h, filename, "corr_length_4.ky");  
  }

  if(end) {  
    //write fields(kx,ky) vs z


    //calculate and write zonal flow equilibrium Pfirsch-Schluter flows
    float ps_fac;
    if((varenna && varenna_fsa==true) || new_catto) {
      if(ivarenna>0) {
        volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Tpar[ION], jacobian, 1./fluxDen);
        ps_fac = 3.;
        PSdiagnostic_odd_fsa<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, CtmpX, shat, species[ION].rho);  //defined in operations_kernel.cu  
      }
      if(ivarenna<0 || new_catto) {
        volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Phi, jacobian, 1./fluxDen);
        ps_fac = -1.6*pow(eps,1.5)*3.;
        PSdiagnostic_odd_fsa<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, CtmpX, shat, species[ION].rho);  //defined in operations_kernel.cu  
      }
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "qpar_ps.field", filename);

      if(ivarenna>0) {
        volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Tprp[ION], jacobian, 1./fluxDen);
        ps_fac = 1.;
        PSdiagnostic_odd_fsa<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, CtmpX, shat, species[ION].rho);  //defined in operations_kernel.cu  
      }
      if(ivarenna<0 || new_catto) {
        volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Phi, jacobian, 1./fluxDen);
        ps_fac = -1.6*pow(eps,1.5)*1.;
        PSdiagnostic_odd_fsa<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, CtmpX, shat, species[ION].rho);  //defined in operations_kernel.cu  
      }
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "qprp_ps.field", filename);

      if(ivarenna<0 || new_catto) {        
        volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Phi, jacobian, 1./fluxDen);
        ps_fac = -1.6*pow(eps,1.5);
        PSdiagnostic_odd_fsa<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, CtmpX, shat, species[ION].rho); 
      }
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "upar_ps.field", filename);

      if(ivarenna<0 || new_catto) {        
        volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Phi, jacobian, 1./fluxDen);
        ps_fac = -1.6*pow(eps,1.5);
        PSdiagnostic_even_fsa<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, CtmpX, shat, species[ION].rho); 
      }
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "dens_ps.field", filename);

      if(ivarenna<0 || new_catto) {        
        volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Phi, jacobian, 1./fluxDen);
        ps_fac = -1.6*pow(eps,1.5)*2.;
        PSdiagnostic_even_fsa<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, CtmpX, shat, species[ION].rho); 
      }
      //add_scaled<<<dimGrid,dimBlock>>>(tmp, 1., tmp, -1., field); // Tpar = Ppar - Dens
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "tpar_ps.field", filename);
      
      if(ivarenna<0 || new_catto) {        
        volflux_zonal_complex<<<dimGrid,dimBlock>>>(CtmpX, Phi, jacobian, 1./fluxDen);
        ps_fac = 0.;
        PSdiagnostic_even_fsa<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, CtmpX, shat, species[ION].rho); 
      }
      //add_scaled<<<dimGrid,dimBlock>>>(tmp, 1., tmp, -1., field); // Tprp = Pprp - Dens
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "tprp_ps.field", filename);
    }

    if(varenna &&  varenna_fsa==false) {
      if(ivarenna>0) {
        ps_fac = 3.;
        PSdiagnostic_odd<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, Tpar[ION], shat, species[ION].rho);  //defined in operations_kernel.cu  
      }
      if(ivarenna<0) {
        ps_fac = -1.6*pow(eps,1.5)*3.;
        PSdiagnostic_odd<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, Phi, shat, species[ION].rho);  //defined in operations_kernel.cu  
      }
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "qpar_ps.field", filename);

      if(ivarenna>0) {
        ps_fac = 1.;
        PSdiagnostic_odd<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, Tprp[ION], shat, species[ION].rho);  //defined in operations_kernel.cu  
      }
      if(ivarenna<0) {
        ps_fac = -1.6*pow(eps,1.5)*1.;
        PSdiagnostic_odd<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, Phi, shat, species[ION].rho);  //defined in operations_kernel.cu  
      }
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "qprp_ps.field", filename);

      if(ivarenna<0) {        
        ps_fac = -1.6*pow(eps,1.5);
        PSdiagnostic_odd<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, Phi, shat, species[ION].rho); 
      }
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "upar_ps.field", filename);

      if(ivarenna<0) {        
        ps_fac = -1.6*pow(eps,1.5);
        PSdiagnostic_even<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, Phi, shat, species[ION].rho); 
      }
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(field, field_h, "dens_ps.field", filename);

      if(ivarenna<0) {        
        ps_fac = -1.6*pow(eps,1.5)*2.;
        PSdiagnostic_even<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, Phi, shat, species[ION].rho); //this is Ppar
      }
      //add_scaled<<<dimGrid,dimBlock>>>(tmp, 1., tmp, -1., field); // Tpar = Ppar - Dens
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "tpar_ps.field", filename);
      
      if(ivarenna<0) {        
        ps_fac = 0.;
        PSdiagnostic_even<<<dimGrid,dimBlock>>>(tmp, ps_fac, kx, gds22, qsf, eps, bmagInv, Phi, shat, species[ION].rho); //this is Pprp
      }
      //add_scaled<<<dimGrid,dimBlock>>>(tmp, 1., tmp, -1., field); // Tprp = Pprp - Dens
      //normalize<<<dimGrid,dimBlock>>>(tmp,Phi,1);
      fieldWrite(tmp, field_h, "tprp_ps.field", filename);
    }
      
    //normalize<<<dimGrid,dimBlock>>>(Dens[ION],Phi,1);
    fieldWrite(Dens[ION], field_h, "dens.field", filename);

    //normalize<<<dimGrid,dimBlock>>>(Upar[ION],Phi,1);
    fieldWrite(Upar[ION], field_h, "upar.field", filename);

    //normalize<<<dimGrid,dimBlock>>>(Qpar[ION],Phi,1);
    fieldWrite(Qpar[ION], field_h, "qpar.field", filename);

    //normalize<<<dimGrid,dimBlock>>>(Qprp[ION],Phi,1);
    fieldWrite(Qprp[ION], field_h, "qprp.field", filename);

    //normalize<<<dimGrid,dimBlock>>>(Tpar[ION],Phi,1);
    fieldWrite(Tpar[ION], field_h, "tpar.field", filename);

    //normalize<<<dimGrid,dimBlock>>>(Tprp[ION],Phi,1);
    fieldWrite(Tprp[ION], field_h, "tprp.field", filename);
            
    
    
    mask<<<dimGrid,dimBlock>>>(Phi);
    fieldWrite(Phi, field_h, "phi.field", filename);

    //normalizeCovering(Phi,kxCover,kyCover,kxCover_h,kyCover_h);   //<- need to write
    fieldWriteCovering(Phi,filename,kxCover,kyCover,kxCover_h,kyCover_h);

    squareComplex<<<dimGrid,dimBlock>>>(tmp, Phi);
    sumZ<<<dimGrid,dimBlock>>>(tmpXY4, tmp);
    squareComplex<<<dimGrid,dimBlock>>>(tmp, Dens[ION]);
    sumZ<<<dimGrid,dimBlock>>>(tmpXY, tmp);
    multdiv<<<dimGrid,dimBlock>>>(tmpXY, tmpXY, tmpXY4, Nx, Ny, 1, -1);
    kxkyWrite(tmpXY, tmpXY_h, filename, "dens_norm.kxky");
    squareComplex<<<dimGrid,dimBlock>>>(tmp, Upar[ION]);
    sumZ<<<dimGrid,dimBlock>>>(tmpXY, tmp);
    multdiv<<<dimGrid,dimBlock>>>(tmpXY, tmpXY, tmpXY4, Nx, Ny, 1, -1);
    kxkyWrite(tmpXY, tmpXY_h, filename, "upar_norm.kxky");
    squareComplex<<<dimGrid,dimBlock>>>(tmp, Tpar[ION]);
    sumZ<<<dimGrid,dimBlock>>>(tmpXY, tmp);
    multdiv<<<dimGrid,dimBlock>>>(tmpXY, tmpXY, tmpXY4, Nx, Ny, 1, -1);
    kxkyWrite(tmpXY, tmpXY_h, filename, "tpar_norm.kxky");
    squareComplex<<<dimGrid,dimBlock>>>(tmp, Tprp[ION]);
    sumZ<<<dimGrid,dimBlock>>>(tmpXY, tmp);
    multdiv<<<dimGrid,dimBlock>>>(tmpXY, tmpXY, tmpXY4, Nx, Ny, 1, -1);
    kxkyWrite(tmpXY, tmpXY_h, filename, "tprp_norm.kxky");
    squareComplex<<<dimGrid,dimBlock>>>(tmp, Qpar[ION]);
    sumZ<<<dimGrid,dimBlock>>>(tmpXY, tmp);
    multdiv<<<dimGrid,dimBlock>>>(tmpXY, tmpXY, tmpXY4, Nx, Ny, 1, -1);
    kxkyWrite(tmpXY, tmpXY_h, filename, "qpar_norm.kxky");
    squareComplex<<<dimGrid,dimBlock>>>(tmp, Qprp[ION]);
    sumZ<<<dimGrid,dimBlock>>>(tmpXY, tmp);
    multdiv<<<dimGrid,dimBlock>>>(tmpXY, tmpXY, tmpXY4, Nx, Ny, 1, -1);
    kxkyWrite(tmpXY, tmpXY_h, filename, "qprp_norm.kxky");


    //write out geometry arrays vs z (like gbd, bmag, etc)
    geoWrite("geo.z", filename);
    
    get_kperp<<<dimGrid,dimBlock>>>(tmpXY,0,species[ION].rho,kx,ky,shat,gds2,gds21,gds22,bmagInv);
    kxkyWrite(tmpXY, tmpXY_h, filename, "kperp.kxky");
    
  }

}


