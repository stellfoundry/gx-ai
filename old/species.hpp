#pragma once
#define ION 0
#include <string>
#include <vector>

struct specie
{
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
  std::string type;

  // constructor
  specie(float s_z,
	 float s_mass,
	 float s_temp,
	 float s_tprim,
	 float s_fprim,
	 float s_uprim,
	 float s_zstm;
	 float s_tz;
	 float s_zt;
	 float s_nu_ss;
	 float s_rho;
	 float s_rho2;
	 float s_vt;
	 std::string s_type) :
    z(std::move(s_z)),
    mass(std::move(s_mass)),
    temp(std::move(s_temp)),
    tprim(std::move(s_tprim)),
    fprim(std::move(s_fprim)),
    uprim(std::move(s_uprim)),
    zstm(std::move(s_zstm)),
    tz(std::move(s_tz)),
    zt(std::move(s_tz)),
    nu_ss(std::move(s_nu_ss)),
    rho(std::move(s_rho)),
    rho2(std::move(s_rho2)),
    vt(std::move(s_vt)),
    type(std:move(s_type)) 
};
