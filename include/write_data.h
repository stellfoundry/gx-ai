//Functions for writing gryfx output to a netcdf file.

//Set up the file and create dimensions and variables
//Write constants, i.e. geometric coefficients
void writedat_beginning(everything_struct * ev);

// Write data at the given timestep
void writedat_each(grids_struct * grids, outputs_struct * outs, fields_struct * flds, time_struct * time);

// Close the output file
void writedat_end(outputs_struct * outs);

// Write the restart file .restart.cdf
void writedat_write_restart(everything_struct * ev_h, everything_struct * ev_hd);

// Read data from the netcdf restart file 
void writedat_read_restart(everything_struct * ev_h, everything_struct * ev_hd);
