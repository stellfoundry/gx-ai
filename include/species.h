#pragma once
#define ION 0
 
typedef struct {
  float z; 
  float mass;
  float dens;
  float temp;
  float tprim;
  float fprim;
  float uprim;
  float zstm;
  float tz;
  float zt;
  float nu_ss;
  float rho;           
  float rho2;           
  float vt;
  float nt;
  float qneut;
  float nz;
  int type;
} specie;
