void read_geo_input(input_parameters_struct * pars, grids_struct * grids, geometry_coefficents_struct * geo, FILE* ifile) 
{    
  int nLines;
  fpos_t* lineStartPos;
  int ch;

  int ntgrid;

  
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
  fscanf(ifile, "%d %d %d %f %f %f %f %f", &ntgrid, &pars->nperiod, &grids->Nz, &pars->drhodpsi, &pars->rmaj, &pars->shat, &pars->kxfac, &pars->qsf);
  if(pars->debug) printf("\n\nIN READ_GEO_INPUT:\nntgrid = %d, nperiod = %d, Nz = %d, rmaj = %f\n\n\n", ntgrid, pars->nperiod, grids->Nz, pars->rmaj);

	allocate_geo(ALLOCATE, ON_HOST, geo, &grids->z, &grids->Nz);

  //Local copy for convenience
  int Nz = grids->Nz;
  

  //first block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4]);
    fscanf(ifile, "%f %f %f %f", &geo->gbdrift[i], &geo->gradpar, &geo->grho[i], &grids->z[i]);
    geo->gbdrift[i] = (1./4.)*geo->gbdrift[i];    
//    printf("z: %f \n", z[i]);
  }

  //second block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4 +1*(Nz+2)]);
    fscanf(ifile, "%f %f %f", &geo->cvdrift[i], &geo->gds2[i], &geo->bmag[i]);
    geo->cvdrift[i] = (1./4.)*geo->cvdrift[i];
  }

  //third block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4 +2*(Nz+2)]);
    fscanf(ifile, "%f %f", &geo->gds21[i], &geo->gds22[i]);
  }

  //fourth block
  for(int i=0; i<Nz; i++) {
    fsetpos(ifile, &lineStartPos[i+4 +3*(Nz+2)]);
    fscanf(ifile, "%f %f", &geo->cvdrift0[i], &geo->gbdrift0[i]);
    geo->cvdrift0[i] = (1./4.)*geo->cvdrift0[i];
    geo->gbdrift0[i] = (1./4.)*geo->gbdrift0[i];
    //if(DEBUG) printf("z: %f \n", cvdrift0[i]);  
  }
  
}         
