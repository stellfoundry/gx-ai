void read_input(FILE* ifile) 
{    
  int nLines;
  fpos_t* lineStartPos;
  int ch;
  
  rewind(ifile);
  //find number of lines
  while( (ch = fgetc(ifile)) != EOF) 
  {
    if(ch == '\n') {
      nLines++;
    }  
  }
  lineStartPos = (fpos_t*) malloc(sizeof(fpos_t)*nLines);
  int i = 2;
  rewind(ifile);
  fgetpos(ifile, &lineStartPos[1]);
  while( (ch = fgetc(ifile)) != EOF) 
  {
    if(ch == '\n') {
      fgetpos(ifile, &lineStartPos[i]);
      i++;
    }
  }
  
  //lineStartPos[i] is start of line i in file... first line is i=1 (not 0)
  fsetpos(ifile, &lineStartPos[2]);
  fscanf(ifile, "%d %d %d %f %f %f %f %f", &ntgrid, &nperiod, &Nz, &drhodpsi, &rmaj, &shat, &kxfac, &qsf);

  
  gbdrift_h = (float*) malloc(sizeof(float)*Nz);
  grho_h = (float*) malloc(sizeof(float)*Nz);
  z_h = (float*) malloc(sizeof(float)*Nz);
  cvdrift_h = (float*) malloc(sizeof(float)*Nz);
  gds2_h = (float*) malloc(sizeof(float)*Nz);
  bmag_h = (float*) malloc(sizeof(float)*Nz);
  gds21_h = (float*) malloc(sizeof(float)*Nz);
  gds22_h = (float*) malloc(sizeof(float)*Nz);
  cvdrift0_h = (float*) malloc(sizeof(float)*Nz);
  gbdrift0_h = (float*) malloc(sizeof(float)*Nz); 
  
  /*cudaMallocHost((void**) &gbdrift, sizeof(float)*Nz);
  cudaMallocHost((void**) &grho, sizeof(float)*Nz);
  cudaMallocHost((void**) &z, sizeof(float)*Nz);
  cudaMallocHost((void**) &cvdrift, sizeof(float)*Nz);
  cudaMallocHost((void**) &gds2, sizeof(float)*Nz);
  cudaMallocHost((void**) &bmag, sizeof(float)*Nz);
  cudaMallocHost((void**) &gds21, sizeof(float)*Nz);
  cudaMallocHost((void**) &gds22, sizeof(float)*Nz);
  cudaMallocHost((void**) &cvdrift0, sizeof(float)*Nz);
  cudaMallocHost((void**) &gbdrift0, sizeof(float)*Nz);*/
    


  //first block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4]);
    fscanf(ifile, "%f %f %f %f", &gbdrift_h[i], &gradpar, &grho_h[i], &z_h[i]);
    gbdrift_h[i] = (rmaj/4)*gbdrift_h[i];    
//    printf("z: %f \n", z_h[i]);
  }
  gradpar = rmaj*gradpar;	

  //second block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4 +1*(Nz+2)]);
    fscanf(ifile, "%f %f %f", &cvdrift_h[i], &gds2_h[i], &bmag_h[i]);
    cvdrift_h[i] = (rmaj/4)*cvdrift_h[i];
  }

  //third block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4 +2*(Nz+2)]);
    fscanf(ifile, "%f %f", &gds21_h[i], &gds22_h[i]);
  }

  //fourth block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4 +3*(Nz+2)]);
    fscanf(ifile, "%f %f", &cvdrift0_h[i], &gbdrift0_h[i]);
    cvdrift0_h[i] = (rmaj/4)*cvdrift0_h[i];
    gbdrift0_h[i] = (rmaj/4)*gbdrift0_h[i];
    //    printf("z: %f \n", cvdrift0_h[i]);  
  }
  
}         
