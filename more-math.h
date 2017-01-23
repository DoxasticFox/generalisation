#ifndef MORE_MATH_H
#define MORE_MATH_H

#include <float.h>
#include <math.h>
#include <cstddef>

double sgn(double x);
int clamp(ptrdiff_t x, ptrdiff_t min, ptrdiff_t max);
int clamp(int x, int min, int max);
double clamp(double x, double min, double max);
int count(bool *x, int n);
void maskedMoments(double *x, bool *mask, int n, double &avg, double &var);
double maskedAvg(double *x, bool *mask, int n);
double maskedMin(double *x, bool *mask, int n);
double maskedMax(double *x, bool *mask, int n);
void moments(double *x, int n, double &avg, double &var);
double max(double x, double y);
double min(double x, double y);
double max(double *x, int n);
double min(double *x, int n);
void add(double *x, int n, double c);
void mul(double *x, int n, double c);
double mean(double* x, int n);
double var(double* x, int n);
void hadamard(double* x, double* y, int n);
double l2Norm(double* x, int n);

#endif
