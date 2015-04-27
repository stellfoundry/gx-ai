
typedef struct {
	FILE * fluxfile;
	FILE * omegafile; 
	FILE * gammafile;
	FILE * phifile;
	char stopfileName[2000];
} files_struct;

void setup_files(files_struct * files, input_parameters_struct * pars, grids_struct * grids, char * out_stem);
void close_files(files_struct  * files);
