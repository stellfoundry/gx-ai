#define BOOL_TO_STR ?"true":"false"
#include "cufft.h"
#include "float.h"
#include "unit_tests.h"


int agrees_with_bool(bool * val, bool * correct, const int size){
  int result;

  result = 1;

  for (int i=0;i<size;i++) {
    if(val[i] != correct[i]){
      result = 0;
      printf("Error: %s should be %s\n", val[i] BOOL_TO_STR, correct[i] BOOL_TO_STR);
    }
    else
      printf("Value %s agrees with correct value %s\n", val[i] BOOL_TO_STR, correct[i] BOOL_TO_STR);
  }
  return result;
}

int agrees_with_float(float * val, float * correct, const int size, const float eps){
  int result;

  result = 1;

  for (int i=0;i<size;i++) {
    if(
        (
          (fabsf(correct[i]) < FLT_MIN) && !(fabsf(val[i]) < eps) 
        ) || (
          (fabsf(correct[i]) > FLT_MIN) && 
          !( fabsf((val[i]-correct[i])/correct[i]) < eps) 
        ) 
      ) {
      result = 0;
      printf("Error: %e should be %e\n", val[i], correct[i]);
    }
    else
      printf("Value %e agrees with correct value %e\n", val[i], correct[i]);
  }
  return result;
}

int agrees_with_cuComplex_imag(cuComplex * val, cuComplex * correct, const int size, const float eps){
  int result;

  result = 1;

  for (int i=0;i<size;i++)
    result = agrees_with_float(&val[i].y, &correct[i].y, 1, eps) && result;
    printf("result = %d\n", result);
  return result;
}
