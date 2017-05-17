#include <math.h>
#include <float.h>
#include "cufft.h"
#include "mpi.h"

extern void gryfx_main(int argc, char* argv[], int mpcom);

int mpcom_glob;
int argc_glob;
char** argv_glob;

int agrees_with_float(float * val, float * correct, const int size, const float eps){
  int result;

  result = 1;

  for (int i=0;i<size;i++) {
    if(
        (
          !((fabsf(correct[i]) < eps*1.e-5) && (fabsf(val[i]) < eps)) 
        ) && (
          (fabsf(correct[i]) > FLT_MIN) && 
          !( fabsf((val[i]-correct[i])/correct[i]) < eps) 
        ) 
      ) {
      result = 0;
      printf("Error in value %d: %e should be %e\n", i, val[i], correct[i]);
    }
//    else
      //printf("Value %e agrees with correct value %e\n", val[i], correct[i]);
  }
  return result;
}

int agrees_with_cuComplex_imag(cuComplex * val, cuComplex * correct, const int size, const float eps){
  int result;

  result = 1;

  for (int i=0;i<size;i++)
    result = agrees_with_float(&val[i].y, &correct[i].y, 1, eps) && result;
    printf("result = %d\n", result);
  return result;
}
 

int main(int argc, char* argv[])
{

  bool debug = false; 

  int proc;
  MPI_Init(&argc, &argv);
  mpcom_glob = MPI_Comm_c2f(MPI_COMM_WORLD);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  
  argc_glob = argc;
  argv_glob = argv;

  int size = strlen(argv[1])-3+1;
  char run_name[size];

  strncpy(run_name, argv[1], strlen(argv[1])-3);
  run_name[strlen(argv[1])-3] = '\0';
  printf("input file is %s\n", argv[1]);
  printf("Run name is %s\n", run_name);

  gryfx_main(argc_glob, argv_glob, mpcom_glob);

  //if(proc==0) {
  //if (agrees_with_cuComplex_imag(omega_out_h, omega_out_h_correct, 5, 1.0e-2)==0){
  //  printf("Growth rates don't match!\n");
  //  exit(1);
  // }
  // }
 
  if(proc==0) {
 
    char outfileName[200];
    strcpy(outfileName, run_name);
    strcat(outfileName, ".phi.time");

    char correctName[200];
    strcpy(correctName, run_name);
    strcat(correctName, ".phi.time.correct");
 
    FILE *outfile, *correctfile;
    outfile = fopen(outfileName, "r");
    if(debug) printf("successfully opened file %s\n", outfileName);
    correctfile = fopen(correctName, "r");
    if(debug) printf("successfully opened file %s\n", correctName);

    int ch;
    int nLines_out=0;
    fpos_t* lineStartPos_out;
  
    if(debug) printf("finding number of lines in %s\n", outfileName);
    rewind(outfile);
    //find number of lines
    while( (ch = fgetc(outfile)) != EOF)
    {
      if(ch == '\n') {
        nLines_out++;
      }
    }
    if(debug) printf("%s has %d lines\nFinding position of beginning of each line...\n", outfileName, nLines_out);
    //find position of beginning of each line
    lineStartPos_out = (fpos_t*) malloc(sizeof(fpos_t)*nLines_out);
    int i = 2;
    rewind(outfile);
    fgetpos(outfile, &lineStartPos_out[1]);
    while( (ch = fgetc(outfile)) != EOF)
    {
      if(ch == '\n') {
        fgetpos(outfile, &lineStartPos_out[i]);
        i++;
      }
    }

    if(debug) printf("finding number of lines in %s\n", correctName);
    int nLines_correct=0;
    fpos_t* lineStartPos_correct;
  
    rewind(correctfile);
    //find number of lines
    while( (ch = fgetc(correctfile)) != EOF)
    {
      if(ch == '\n') {
        nLines_correct++;
      }
    }
    if(debug) printf("%s has %d lines\nFinding position of beginning of each line...\n", correctName, nLines_correct);
    //find position of beginning of each line
    lineStartPos_correct = (fpos_t*) malloc(sizeof(fpos_t)*nLines_correct);
    i = 2;
    rewind(correctfile);
    fgetpos(correctfile, &lineStartPos_correct[1]);
    while( (ch = fgetc(correctfile)) != EOF)
    {
      if(ch == '\n') {
        fgetpos(correctfile, &lineStartPos_correct[i]);
        i++;
      }
    }
    
    float out[7];
    float correct[7];

    if(debug) printf("comparing data files %s & %s\n", outfileName, correctName);
    int ncheck = 20;  // number of lines to compare between the files
    for(int j=1; j<ncheck; j++) {
      int iline = j*(nLines_out/ncheck);
      fsetpos(outfile, &lineStartPos_out[iline]);
      fscanf(outfile, "\t%f\t\t\t\t%e\t\t\t%e\t\t\t%e\t\t\t%e\t\t\t%e\t\t\t%e", &out[0], &out[1], &out[2], &out[3], &out[4], &out[5], &out[6]);
      fsetpos(correctfile, &lineStartPos_correct[iline]);
      fscanf(correctfile, "\t%f\t\t\t\t%e\t\t\t%e\t\t\t%e\t\t\t%e\t\t\t%e\t\t\t%e", &correct[0], &correct[1], &correct[2], 
        									    &correct[3], &correct[4], &correct[5], &correct[6]);
      if (agrees_with_float(out, correct, 7, 1.0e-1)==0){
        printf("Values on line %d don't match!\n", iline);
        printf("===FAIL====%s FAILED!====FAIL===\n", outfileName);
        exit(1);
      } 
    }
   
    fclose(outfile);
    fclose(correctfile);

    printf("===========%s passed!=============\n", outfileName);
  }

        MPI_Finalize();
   
	return 0;

}
