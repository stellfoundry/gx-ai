
typedef struct {
	char * run_name;
	char * restart_file_name;
  int gpuID;
} info_struct;
void setup_info(char * input_file, input_parameters_struct * pars, info_struct * info);

