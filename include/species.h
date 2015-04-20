#define ION 0
#define ELECTRON 1
#define PHI 0
#define DENS 1


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
  float vt;    
  char type[100];        
} specie;
