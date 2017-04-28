#ifndef GRYFX_H_
#define GRYFX_H_

#include "fields.h"
#include "inputs.h"
#include "grids.h"
#include "structs.h"

class Gryfx {

public:
  Gryfx(int loc) : memory_location(loc);
  ~Gryfx();

  allocate();
  deallocate();
  
  Fields fields;
  Inputs input;
  Grids grids;
  Info info;
  time_struct time;
  cuda_dimensions_struct cdims;
  cuda_streams_struct streams;
  cuda_events_struct events;
  cuffts_struct ffts;
  mpi_info_struct mpi;
  
private:
  const int memory_location; 

}

#endif

