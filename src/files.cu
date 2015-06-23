#define NO_GLOBALS true
#include "standard_headers.h"


FILE * open_output_file(char * run_name, char * extension, char * mode){
  char * filename = (char*)malloc(sizeof(char) * (strlen(run_name) + strlen(extension) + 2));
  strcpy(filename, run_name);
  strcat(filename, ".");
  strcat(filename, extension);
  FILE * output_file = fopen(filename, mode);
  free(filename);
  return output_file;
}

inline void phiWriteSetup(FILE* out)
{
  fprintf(out, "y\t\tx\t\tPhi(z=0)\n");
}

inline void omegaWriteSetup(grids_struct * grids, FILE* ofile, char* w)
{
  //Local copies for convenience
  int Nx = grids->Nx;
  int Ny = grids->Ny;
  //int Nz = grids->Nz;

  fprintf(ofile, "#\ttime(s)\t");
  int col = 2;
  for(int i=0; i<((Nx-1)/3+1); i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {      
      if(!(i==0 && j==0)) {
        fprintf(ofile, "\t\t\t%d:(%.3g,%.3g)", col, grids->ky[j],grids->kx[i]);
        col++;
      }
    }
  }
  for(int i=2*Nx/3+1; i<Nx; i++) {
    for(int j=0; j<((Ny-1)/3+1); j++) {
      fprintf(ofile, "\t\t\t%d:(%.3g,%.3g)", col, grids->ky[j],grids->kx[i]);
      col++;
    }
  }
  
  fprintf(ofile, "\n");
}

void setup_files(files_struct * files, input_parameters_struct * pars, grids_struct * grids, char * out_stem){
  ////////////////////////////////////////////////
  // set up some diagnostics/control flow files //
  ////////////////////////////////////////////////
  
  
  //set up stopfile
  //char stopfileName[60];
  strcpy(files->stopfileName, out_stem);
  strcat(files->stopfileName, ".stop");
  
  printf("stopfile = %s\n", files->stopfileName);

  //time histories (.time)
  
  if(!pars->restart || pars->secondary_test || pars->nlpm_test) {
    files->omegafile = open_output_file(out_stem, "omega.time", "w+");
    files->gammafile = open_output_file(out_stem, "gamma.time", "w+");
    files->fluxfile = open_output_file(out_stem, "flux.time", "w+");
    if(pars->write_omega) {
      //set up omega output file 
      omegaWriteSetup(grids, files->omegafile,"omega");
      //and gamma output file
      omegaWriteSetup(grids, files->gammafile,"gamma");  
    }
    files->phifile = open_output_file(out_stem, "phi.time", "w+");
    //phiWriteSetup(files->phifile);
  }
  else {
    files->omegafile = open_output_file(out_stem, "omega.time", "a");
    files->gammafile = open_output_file(out_stem, "gamma.time", "a");
    files->fluxfile = open_output_file(out_stem, "flux.time", "a");
    files->phifile = open_output_file(out_stem, "phi.time", "a");
  }
  
  ////////////////////////////////////////////
}

void close_files(files_struct  * files){
  fclose(files->omegafile);
  fclose(files->gammafile);
  fclose(files->fluxfile);
  fclose(files->phifile);
}
