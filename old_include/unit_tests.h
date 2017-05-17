// A series of functions to help with performing unit tests.
int agrees_with_bool(bool * val, bool * correct, const int size);
int agrees_with_float(float * val, float * correct, const int size, const float eps);
int agrees_with_cuComplex_imag(cuComplex * val, cuComplex * correct, const int size, const float eps);
