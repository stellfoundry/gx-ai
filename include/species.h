#pragma once
#define ION 0

// Defines the species identifiers and the specie struct
// which stores species properties
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
  // char type[100];        
} specie;
