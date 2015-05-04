#define NO_GLOBALS true
#include "standard_headers.h"
#include "allocations.h"

void setup_info(char * input_file, input_parameters_struct * pars, info_struct * info){
  /* Allocate and set run name */
  /* If input file is "myrun.in", run_name is "myrun"*/

	int run_name_size = strlen(input_file)-3+1;

  //Work out how long the restart file name can be
  int rfn_size, temp;
  rfn_size = run_name_size + 13;
  temp = strlen(pars->secondary_test_restartfileName) + 1;
  rfn_size = rfn_size > temp ? rfn_size : temp;

  rfn_size = rfn_size+200;

  allocate_info(ALLOCATE, ON_HOST, info, run_name_size, rfn_size);

	printf("Allocated run name pointer\n");

	strncpy(info->run_name, input_file, strlen(input_file)-3);
	printf("Copied run name\n");
  info->run_name[strlen(input_file)-3] = '\0';
	printf("Run name is %s\n", info->run_name);

  //set up restart file

  //Local pointer for convenience 
  char * restartfileName;
  restartfileName = info->restart_file_name ;

  
  strcpy(restartfileName, info->run_name);
  strcat(restartfileName, ".restart.bin");
 
  if(pars->secondary_test && !pars->linear) 
    strcpy(restartfileName, pars->secondary_test_restartfileName);
 
  if(pars->restart) {
    // check if restart file exists
    if( FILE* restartFile = fopen(restartfileName, "r") ) {
      printf("restart file found. restarting...\n");
    }
    else{
      printf("cannot restart because cannot find restart file. changing to no restart\n");
      // EGH Perhaps we should abort at this point?
      pars->restart = false;
      abort();
    }
  }			
  
  if(pars->check_for_restart) {
    printf("restart mode set to exist...\n");
    //check if restart file exists
    if(FILE* restartFile = fopen(restartfileName, "r") ) {
      fclose(restartFile);
      printf("restart file exists. restarting...\n");
      pars->restart = true;
    }
    else {
      printf("restart file does not exist. starting new run...\n");
      pars->restart = false;
    }
  }
}
