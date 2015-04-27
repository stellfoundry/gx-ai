//Functions for writing gryfx output to a netcdf file.

//Set up the file and create dimensions and variables
//Write constants, i.e. geometric coefficients
void writedat_beginning(everything_struct * ev);

// Write data at the given timestep
void writedat_each(grids_struct * grids, outputs_struct * outs, fields_struct * flds, time_struct * time);

// Close the output file
void writedat_end(outputs_struct outs);
