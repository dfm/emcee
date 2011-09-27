%module acor

%{
#define SWIG_FILE_WITH_INIT
#include "acor.h"
double acor_dfm(double *X, int L);
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double *X, int L)};

double acor_dfm(double *X, int L);

