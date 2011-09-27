#include "acor.h"

double acor_dfm(double *X, int L){
    double tau = 0.0, mean = 0.0, sigma = 0.0;
    acor(&mean, &sigma, &tau, X, L);
    return tau;
}

